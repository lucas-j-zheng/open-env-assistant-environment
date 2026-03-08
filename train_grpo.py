"""
GRPO Training — Calendar Personal Assistant (Qwen2.5-3B-Instruct)

Trains with QLoRA via Unsloth on H100. Designed for a 2-hour hackathon window.
Targets the current 18-task environment with inbox, personal events,
negotiation, schema drift, and weighted rewards.

Usage:
    python train_grpo.py
"""

import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, "personal_assistant")
from server.personal_assistant_environment import PersonalAssistantEnvironment
from models import CalendarAction


@dataclass
class Config:
    model_name: str = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
    lora_r: int = 32
    lora_alpha: int = 32
    max_seq_len: int = 2048

    # GRPO (same seed per group — proper GRPO)
    group_size: int = 4
    max_steps_per_episode: int = 40  # longer episodes for behavioral divergence

    # Training
    num_iterations: int = 15
    lr: float = 5e-5
    max_grad_norm: float = 1.0
    ppo_epochs: int = 1

    # Generation
    temperature: float = 0.9
    max_new_tokens: int = 128

    # Checkpointing
    save_every: int = 10
    output_dir: str = "/home/jovyan/checkpoints"
    log_file: str = "/home/jovyan/training_log.jsonl"


SYSTEM_PROMPT = """\
You are a calendar personal assistant. Respond ONLY with a JSON tool call.

Format: {"tool": "tool_name", "args": {"key": "value"}}

Available tools:
- get_task_list: See tasks to complete (no args)
- list_events: List events for a date (date: "today"/"tomorrow"/day name/YYYY-MM-DD)
- create_event: Create event (title, date, start_time "HH:MM", duration_minutes, attendees "Alice,Bob", description)
- delete_event: Delete event by title (title)
- edit_event: Edit event (title, optional: new_title/new_date/new_start_time/new_duration_minutes/new_attendees/new_description)
- find_free_slots: Find open time slots (date, duration_minutes)
- check_conflicts: Check for overlapping events (date)
- resolve_conflict: Move an event to new time (event_title, new_start_time "HH:MM")
- send_notification: Notify someone (to, message)
- check_availability: Check person's busy/free times (person, date)
- get_constraints: Get public scheduling rules (no args)
- get_contact_preferences: Get person's private preferences and constraints (person)
- check_constraint_violations: Audit calendar for all violations (no args)
- read_inbox: Read inbox messages (status: "all"/"unread"/"unreplied")
- reply_message: Reply to inbox message (message_id, reply — must be 20+ chars and address the concern)
- check_personal_calendar: Show personal/immovable events (no args)

Strategy:
1. Call get_task_list to see what needs doing
2. Call get_constraints for public rules, then get_contact_preferences for each person
3. Call read_inbox to check for messages that need replies
4. Call check_personal_calendar to see immovable personal events
5. Call check_availability before scheduling any meeting
6. Handle INTERRUPT messages immediately when they appear
7. When someone DECLINES a meeting, adjust and re-create with their feedback
8. After changes, call check_constraint_violations to verify
9. Meetings >30 min need a description (if policy is active, use edit_event to add)

IMPORTANT:
- Personal events CANNOT be moved — work events must accommodate them
- Some tasks are hidden in inbox messages — read your inbox
- Attendees may negotiate/decline — adapt to their feedback
- Availability can change mid-episode — re-check after updates

Output ONLY the JSON tool call, nothing else."""


def extract_tool_call(text: str) -> str:
    """Pull a JSON tool call from model output."""
    match = re.search(r'\{"tool"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{[^}]*\}\s*\}', text)
    if match:
        try:
            parsed = json.loads(match.group())
            if "tool" in parsed:
                return match.group()
        except json.JSONDecodeError:
            pass
    match = re.search(r'\{[^{}]*"tool"[^{}]*\}', text)
    if match:
        try:
            parsed = json.loads(match.group())
            if "tool" in parsed:
                return match.group()
        except json.JSONDecodeError:
            pass
    return text.strip()


def run_episode(model, tokenizer, seed: int, cfg: Config, device: str):
    """Run one episode. Returns (reward, trajectory, num_steps)."""
    env = PersonalAssistantEnvironment()
    obs = env.reset(seed=seed)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": obs.output},
    ]
    trajectory = []

    for step in range(cfg.max_steps_per_episode):
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                        max_length=cfg.max_seq_len - cfg.max_new_tokens)
        input_ids = enc.input_ids.to(device)

        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        gen_ids = out[0, input_ids.shape[1]:]
        resp = tokenizer.decode(gen_ids, skip_special_tokens=True)

        trajectory.append({
            "input_ids": input_ids[0].cpu(),
            "generated_ids": gen_ids.cpu(),
            "full_ids": out[0].cpu(),
        })

        tool_json = extract_tool_call(resp)
        obs = env.step(CalendarAction(instruction=tool_json))

        messages.append({"role": "assistant", "content": resp})
        messages.append({"role": "user", "content": obs.output})

        if obs.done:
            break

    return obs.reward, trajectory, step + 1


def trajectory_log_probs(model, trajectory, device):
    """Sum of per-token log-probs for generated tokens in the trajectory."""
    total_lp = torch.tensor(0.0, device=device)
    n_tokens = 0

    for step_data in trajectory:
        gen_len = step_data["generated_ids"].shape[0]
        if gen_len == 0:
            continue

        full = step_data["full_ids"].unsqueeze(0).to(device)
        prompt_len = step_data["input_ids"].shape[0]

        logits = model(full).logits
        shift_logits = logits[0, prompt_len - 1 : prompt_len + gen_len - 1]
        targets = full[0, prompt_len : prompt_len + gen_len]

        lp = F.log_softmax(shift_logits, dim=-1)
        token_lp = lp.gather(1, targets.unsqueeze(1)).squeeze(1)

        total_lp = total_lp + token_lp.sum()
        n_tokens += gen_len

    return total_lp, n_tokens


def main():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 60)
    print("GRPO Training — Qwen2.5-3B Calendar Assistant")
    print("18 tasks | 15 tools | 7 interrupts | weighted rewards")
    print("=" * 60)

    # --- Load model ---
    from unsloth import FastLanguageModel

    print(f"\nLoading {cfg.model_name} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_len,
        load_in_4bit=True,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)\n")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr
    )

    print(f"Training: {cfg.num_iterations} iters x {cfg.group_size} episodes/iter (same seed)")
    print(f"Group size: {cfg.group_size}, Max steps: {cfg.max_steps_per_episode}\n")

    best_reward = 0.0
    reward_log = []
    log_f = open(cfg.log_file, "w")

    # --- Training loop ---
    for it in range(1, cfg.num_iterations + 1):
        t0 = time.time()

        # Phase 1: Rollout — same seed (proper GRPO), fallback to mixed seeds
        FastLanguageModel.for_inference(model)
        seed = random.randint(0, 100_000)
        group_rewards, group_trajs = [], []

        for _ in range(cfg.group_size):
            r, traj, n_steps = run_episode(model, tokenizer, seed, cfg, device)
            group_rewards.append(r)
            group_trajs.append(traj)

        # If zero variance (all same reward), retry with different seeds
        if np.std(group_rewards) < 1e-6:
            print(f"    (same-seed gave identical rewards {group_rewards[0]:.3f}, retrying mixed seeds)")
            group_rewards, group_trajs = [], []
            for _ in range(cfg.group_size):
                seed = random.randint(0, 100_000)
                r, traj, n_steps = run_episode(model, tokenizer, seed, cfg, device)
                group_rewards.append(r)
                group_trajs.append(traj)

        # GRPO advantages
        mu = np.mean(group_rewards)
        sigma = np.std(group_rewards) + 1e-8
        advs = [(r - mu) / sigma for r in group_rewards]

        rollout_data = []
        for g in range(cfg.group_size):
            rollout_data.append({
                "trajectory": group_trajs[g],
                "advantage": advs[g],
                "reward": group_rewards[g],
            })

        all_rewards = [d["reward"] for d in rollout_data]
        avg_r = np.mean(all_rewards)
        max_r = np.max(all_rewards)
        reward_log.append(avg_r)

        # Phase 2: GRPO update
        FastLanguageModel.for_training(model)
        total_loss = 0.0
        n_updates = 0

        for _epoch in range(cfg.ppo_epochs):
            random.shuffle(rollout_data)
            for item in rollout_data:
                adv = item["advantage"]
                if abs(adv) < 1e-6:
                    continue

                lp, n_tok = trajectory_log_probs(model, item["trajectory"], device)
                if n_tok == 0:
                    continue

                loss = -(adv * lp / n_tok)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.max_grad_norm,
                )
                optimizer.step()

                total_loss += loss.item()
                n_updates += 1

        avg_loss = total_loss / max(n_updates, 1)
        elapsed = time.time() - t0

        # Logging
        log_entry = {
            "iter": it, "avg_reward": round(avg_r, 4),
            "max_reward": round(max_r, 4), "loss": round(avg_loss, 4),
            "time_s": round(elapsed, 1),
        }
        log_f.write(json.dumps(log_entry) + "\n")
        log_f.flush()

        print(
            f"[{it:4d}/{cfg.num_iterations}]  "
            f"reward {avg_r:.3f} (max {max_r:.3f})  "
            f"loss {avg_loss:.4f}  "
            f"{elapsed:.0f}s"
        )

        if max_r > best_reward:
            best_reward = max_r
            print(f"  ** new best: {best_reward:.3f}")
            model.save_pretrained(f"{cfg.output_dir}/best")
            tokenizer.save_pretrained(f"{cfg.output_dir}/best")

        if it % cfg.save_every == 0:
            path = f"{cfg.output_dir}/iter-{it}"
            model.save_pretrained(path)
            tokenizer.save_pretrained(path)
            print(f"  checkpoint -> {path}")

    # Done
    log_f.close()
    final = f"{cfg.output_dir}/final"
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"\nDone! Best reward: {best_reward:.3f}")
    print(f"Last 10 avg rewards: {[f'{r:.3f}' for r in reward_log[-10:]]}")
    print(f"Model: {final}")
    print(f"Log: {cfg.log_file}")


if __name__ == "__main__":
    main()
