"""
Evaluate a trained model (or baseline) on fixed seeds.
Targets the current 18-task environment with weighted rewards.

Usage:
    # Baseline eval (before training)
    python eval_model.py --baseline

    # Evaluate trained checkpoint
    python eval_model.py --checkpoint /home/jovyan/checkpoints/best

    # Compare baseline vs trained + generate charts
    python eval_model.py --compare --checkpoint /home/jovyan/checkpoints/best

    # Plot training curve only
    python eval_model.py --plot-training
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

sys.path.insert(0, "personal_assistant")
from server.personal_assistant_environment import PersonalAssistantEnvironment
from models import CalendarAction

EVAL_SEEDS = list(range(20))

ALL_FLAGS = [
    "standup_scheduled", "focus_time_booked", "conflicts_resolved",
    "reminder_set", "meeting_cancelled", "ceo_sync_accommodated",
    "cancellation_handled", "reschedule_handled", "kickoff_scheduled",
    "hard_constraints_clear", "preferences_optimized",
    "availability_drift_handled", "description_policy_met",
    "inbox_cleared", "diplomatic_reply_sent", "client_request_updated",
    "work_life_conflicts_resolved", "personal_update_handled",
]

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


@dataclass
class EvalConfig:
    max_seq_len: int = 4096
    max_steps_per_episode: int = 50
    temperature: float = 0.3
    max_new_tokens: int = 256


def extract_tool_call(text: str) -> str:
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


def run_eval_episode(model, tokenizer, seed: int, cfg: EvalConfig, device: str, verbose: bool = False):
    """Run one eval episode. Returns (reward, flags_found, num_steps, tool_calls)."""
    env = PersonalAssistantEnvironment()
    obs = env.reset(seed=seed)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": obs.output},
    ]
    tool_calls = []

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

        tool_json = extract_tool_call(resp)
        tool_calls.append(tool_json)

        if verbose:
            print(f"  Step {step+1}: {tool_json[:120]}")

        obs = env.step(CalendarAction(instruction=tool_json))

        if verbose and obs.flags_found:
            print(f"         Flags: {obs.flags_found} | Reward: {obs.reward:.3f}")

        messages.append({"role": "assistant", "content": resp})
        messages.append({"role": "user", "content": obs.output})

        if obs.done:
            break

    return obs.reward, list(obs.flags_found), step + 1, tool_calls


def load_model(model_name: str, checkpoint: str = None, max_seq_len: int = 4096):
    from unsloth import FastLanguageModel

    print(f"Loading {model_name} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_len,
        load_in_4bit=True,
        dtype=None,
    )

    if checkpoint:
        print(f"Loading LoRA weights from {checkpoint} ...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    FastLanguageModel.for_inference(model)
    return model, tokenizer


def evaluate(model, tokenizer, label: str, device: str, verbose: bool = False):
    cfg = EvalConfig()
    results = []

    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"{'='*60}")
    print(f"Seeds: {len(EVAL_SEEDS)} | Max steps: {cfg.max_steps_per_episode} | 18 tasks (weighted)")
    print()

    for seed in EVAL_SEEDS:
        t0 = time.time()
        reward, flags, n_steps, tools = run_eval_episode(
            model, tokenizer, seed, cfg, device, verbose=verbose
        )
        elapsed = time.time() - t0

        results.append({
            "seed": seed, "reward": reward, "flags": flags,
            "num_flags": len(flags), "steps": n_steps, "time_s": round(elapsed, 1),
        })

        flag_str = ", ".join(sorted(flags)) if flags else "(none)"
        print(f"  Seed {seed:3d}: reward={reward:.3f} | {len(flags):2d}/18 flags | {n_steps:2d} steps | {elapsed:.0f}s")
        print(f"            {flag_str}")

    # Summary
    rewards = [r["reward"] for r in results]
    flag_counts = [r["num_flags"] for r in results]

    print(f"\n--- {label} Summary ---")
    print(f"  Avg reward:  {np.mean(rewards):.3f} +/- {np.std(rewards):.3f}")
    print(f"  Avg flags:   {np.mean(flag_counts):.1f} / 18")
    print(f"  Max reward:  {np.max(rewards):.3f}")
    print(f"  Min reward:  {np.min(rewards):.3f}")
    print(f"  Avg steps:   {np.mean([r['steps'] for r in results]):.1f}")

    # Per-flag success rate
    print(f"\n  Per-flag success rate:")
    for flag in ALL_FLAGS:
        count = sum(1 for r in results if flag in r["flags"])
        bar = "#" * count + "." * (len(EVAL_SEEDS) - count)
        print(f"    {flag:35s} {count:2d}/{len(EVAL_SEEDS)} ({100*count/len(EVAL_SEEDS):3.0f}%) |{bar}|")

    return results


def plot_comparison(baseline_results, trained_results, output_path="/home/jovyan/eval_comparison.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Reward distribution
        ax = axes[0]
        b_rewards = [r["reward"] for r in baseline_results]
        t_rewards = [r["reward"] for r in trained_results]
        ax.boxplot([b_rewards, t_rewards], labels=["Baseline", "Trained"])
        ax.set_title("Reward Distribution")
        ax.set_ylabel("Reward (weighted, 0-1)")
        ax.grid(True, alpha=0.3)

        # Flags per seed
        ax = axes[1]
        seeds = [r["seed"] for r in baseline_results]
        b_flags = [r["num_flags"] for r in baseline_results]
        t_flags = [r["num_flags"] for r in trained_results]
        x = np.arange(len(seeds))
        ax.bar(x - 0.2, b_flags, 0.4, label="Baseline", alpha=0.7)
        ax.bar(x + 0.2, t_flags, 0.4, label="Trained", alpha=0.7)
        ax.set_xlabel("Seed")
        ax.set_ylabel("Flags completed (out of 18)")
        ax.set_title("Flags per Seed")
        ax.set_xticks(x)
        ax.set_xticklabels(seeds, fontsize=7)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Per-flag success rate
        ax = axes[2]
        b_rates = [sum(1 for r in baseline_results if f in r["flags"]) / len(baseline_results) for f in ALL_FLAGS]
        t_rates = [sum(1 for r in trained_results if f in r["flags"]) / len(trained_results) for f in ALL_FLAGS]
        y = np.arange(len(ALL_FLAGS))
        ax.barh(y - 0.2, b_rates, 0.4, label="Baseline", alpha=0.7, color="#6366f1")
        ax.barh(y + 0.2, t_rates, 0.4, label="Trained", alpha=0.7, color="#22c55e")
        ax.set_yticks(y)
        ax.set_yticklabels([f.replace("_", " ") for f in ALL_FLAGS], fontsize=7)
        ax.set_xlabel("Success Rate")
        ax.set_title("Per-Flag Success Rate")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"\nComparison chart saved to {output_path}")
        plt.close()
    except ImportError:
        print("\nmatplotlib not available — pip install matplotlib")


def plot_training_curve(log_file="/home/jovyan/training_log.jsonl", output_path="/home/jovyan/training_curve.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        entries = []
        with open(log_file) as f:
            for line in f:
                entries.append(json.loads(line))

        if not entries:
            print("No training log entries found")
            return

        iters = [e["iter"] for e in entries]
        avg_rewards = [e["avg_reward"] for e in entries]
        max_rewards = [e["max_reward"] for e in entries]
        losses = [e["loss"] for e in entries]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        ax1.plot(iters, avg_rewards, label="Avg Reward", alpha=0.5, linewidth=1)
        ax1.plot(iters, max_rewards, label="Max Reward", alpha=0.5, linewidth=1)
        if len(avg_rewards) > 10:
            w = min(10, len(avg_rewards) // 3)
            smoothed = np.convolve(avg_rewards, np.ones(w)/w, mode="valid")
            ax1.plot(iters[w-1:], smoothed, label=f"Avg (smoothed {w})", linewidth=2, color="#22c55e")
        ax1.set_ylabel("Reward (weighted)")
        ax1.set_title("GRPO Training — Reward Curve (18 tasks, weighted)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(iters, losses, alpha=0.7, color="red")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Loss")
        ax2.set_title("Training Loss")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"Training curve saved to {output_path}")
        plt.close()
    except (ImportError, FileNotFoundError) as e:
        print(f"Could not plot training curve: {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate calendar assistant model")
    parser.add_argument("--baseline", action="store_true", help="Evaluate base model only")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained LoRA checkpoint")
    parser.add_argument("--compare", action="store_true", help="Run both baseline and trained, then compare")
    parser.add_argument("--verbose", action="store_true", help="Print each tool call")
    parser.add_argument("--model", type=str, default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    parser.add_argument("--plot-training", action="store_true", help="Plot training curve from log")
    parser.add_argument("--output-dir", type=str, default="/home/jovyan", help="Where to save plots")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.plot_training:
        plot_training_curve(output_path=f"{args.output_dir}/training_curve.png")
        return

    if args.compare:
        if not args.checkpoint:
            args.checkpoint = "/home/jovyan/checkpoints/best"

        model, tokenizer = load_model(args.model)
        baseline_results = evaluate(model, tokenizer, "Baseline (no training)", device, args.verbose)
        del model
        torch.cuda.empty_cache()

        model, tokenizer = load_model(args.model, checkpoint=args.checkpoint)
        trained_results = evaluate(model, tokenizer, f"Trained ({args.checkpoint})", device, args.verbose)

        b_avg = np.mean([r["reward"] for r in baseline_results])
        t_avg = np.mean([r["reward"] for r in trained_results])
        b_flags = np.mean([r["num_flags"] for r in baseline_results])
        t_flags = np.mean([r["num_flags"] for r in trained_results])

        print(f"\n{'='*60}")
        print(f"COMPARISON")
        print(f"{'='*60}")
        print(f"  Avg Reward:  {b_avg:.3f} -> {t_avg:.3f}  ({t_avg-b_avg:+.3f})")
        print(f"  Avg Flags:   {b_flags:.1f} -> {t_flags:.1f}  ({t_flags-b_flags:+.1f})")
        print(f"  Improvement: {100*(t_avg-b_avg)/max(b_avg,0.001):.0f}%")

        plot_comparison(baseline_results, trained_results, f"{args.output_dir}/eval_comparison.png")
        plot_training_curve(output_path=f"{args.output_dir}/training_curve.png")

        with open(f"{args.output_dir}/eval_results.json", "w") as f:
            json.dump({"baseline": baseline_results, "trained": trained_results}, f, indent=2)
        print(f"Results saved to {args.output_dir}/eval_results.json")

    elif args.baseline:
        model, tokenizer = load_model(args.model)
        evaluate(model, tokenizer, "Baseline (no training)", device, args.verbose)

    elif args.checkpoint:
        model, tokenizer = load_model(args.model, checkpoint=args.checkpoint)
        evaluate(model, tokenizer, f"Trained ({args.checkpoint})", device, args.verbose)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
