# Calendar Personal Assistant — OpenEnv RL Environment

An RL environment where an agent manages a team calendar under **shifting rules, evolving constraints, and ambiguous inbox requests** — built for the [OpenEnv Hackathon](https://openenv.org) (Patronus AI track: *Consumer Workflows with Schema Drift*).

## Why This Environment Is Hard

Unlike toy scheduling puzzles, the agent must:

- **Discover tasks from its inbox** — no checklist is provided. Boss and team messages describe what needs doing in natural language.
- **Handle schema drift mid-episode** — attendee availability changes, new policies activate, meetings get cancelled, personal events shift. The same tool call returns different data before and after an interrupt.
- **Negotiate with attendees** — creating a meeting can trigger a multi-round rejection/counter-proposal flow. Bob wants standups under 20 min. Eve needs buffer time.
- **Respect immovable personal events** — the agent must schedule work around gym sessions, school pickups, and dinner reservations.
- **Adapt to 7 randomized interrupts** — CEO syncs, policy changes, availability drift, and family texts fire at different steps per seed.

## Architecture

```
┌─────────────────────────────────────────────────┐
│              OpenEnv Environment                 │
│                                                  │
│  reset(seed) ──► Initial calendar + inbox        │
│       │          (4-7 events, 11 messages,       │
│       │           2 personal events)             │
│       ▼                                          │
│  step(action) ──► Fire interrupts (if due)       │
│       │           ──► Dispatch tool call          │
│       │           ──► Recompute 18 flags          │
│       │           ──► Return Markov observation   │
│       ▼                                          │
│  reward = weighted_flags / total_weight           │
│  done = all 18 flags earned                      │
└─────────────────────────────────────────────────┘
```

**Action format:** `{"tool": "create_event", "args": {"title": "Standup", "date": "today", ...}}`

**Observation:** Self-contained Markov snapshot with calendar state, constraint status, inbox, negotiations, and unhandled interrupts — no conversation history needed.

## Schema Drift (Patronus AI Track)

| Drift Type | When | What Changes |
|---|---|---|
| **Availability change** | Step 6-9 | Bob's Wednesday schedule gets new blocked hours. `check_availability` returns different data. |
| **Description policy** | Step 11-14 | HR activates a new hard constraint: meetings >30 min need descriptions. Constraint count goes 3→4. |
| **Meeting cancellation** | Step 5-8 | Dave cancels lunch. The 12:00-13:00 slot opens up. Tool output changes. |
| **Reschedule request** | Step 8-11 | Alice asks to move a meeting. Agent must re-validate conflicts. |
| **Personal event shift** | Step 9-12 | Family texts that dinner moved earlier. New work-life conflicts appear. |
| **Inbox contradiction** | Step 4-7 | Client changes their earlier request (different duration, time constraint). |
| **CEO emergency** | Step 2-5 | Urgent meeting injected at 3 PM. Existing events may now conflict. |

All interrupt steps are **randomized per seed** within ranges — agents can't memorize timing.

## 18 Tasks (Weighted Rewards)

| Weight | Tasks |
|---|---|
| **1.5x** (hard) | Schedule standup (negotiation), kickoff (negotiation), clear hard constraints, handle availability drift, handle client contradiction, handle personal event update |
| **1.0x** (medium) | Book focus time, resolve conflicts, cancel meeting, optimize preferences, clear inbox, diplomatic reply, description policy, work-life conflicts |
| **0.5x** (easy) | Set reminder, acknowledge cancellation, acknowledge reschedule |

Three tasks are **one-shot**: once earned, reward is locked in permanently even if later actions would revoke them.

## Tools (15)

| Category | Tools |
|---|---|
| **Calendar** | `list_events`, `create_event`, `delete_event`, `edit_event` |
| **Scheduling** | `find_free_slots`, `check_conflicts`, `resolve_conflict`, `check_availability` |
| **Constraints** | `get_constraints`, `get_contact_preferences`, `check_constraint_violations` |
| **Communication** | `read_inbox`, `reply_message`, `send_notification` |
| **Personal** | `check_personal_calendar` |

## Quick Start

### 1. Setup

```bash
cd personal_assistant
uv venv && source .venv/bin/activate
uv sync
```

### 2. Run the Environment Server

```bash
cd personal_assistant
uvicorn server.app:app --port 8000
```

### 3. Test with Random Actions

```bash
python run_random.py
```

### 4. Run LLM Agent (requires HF_TOKEN)

```bash
# Basic agent (Qwen2.5-7B via HuggingFace Router)
python run_agent.py

# Agent with live web dashboard
python run_agent_live.py
# Open http://localhost:8001 in browser
```

### 5. Batch Evaluation

```bash
# Run across multiple seeds (requires GROQ_API_KEY)
python run_batch_test.py
```

### 6. Docker

```bash
cd personal_assistant
docker build -t personal-assistant:latest -f server/Dockerfile .
docker run -p 8000:8000 personal-assistant:latest
```

## Training (GRPO)

Train a Qwen2.5-3B-Instruct model with QLoRA + GRPO on GPU:

```bash
# From repo root (requires GPU + unsloth)
python train_grpo.py
```

**Config:** 4-bit QLoRA, LoRA r=32, group_size=3, 10 iterations, lr=5e-5. Designed to run in ~2 hours on an H100.

**Evaluate trained model:**

```bash
# Baseline only
python eval_model.py --baseline

# Compare baseline vs trained
python eval_model.py --compare --checkpoint /path/to/checkpoints/best

# Plot training reward curve
python eval_model.py --plot-training
```

## Tests

```bash
pytest tests/ -v
```

| Test File | Coverage |
|---|---|
| `test_seed_determinism.py` | Same seed = same initial state |
| `test_completion_criteria.py` | All 18 flag conditions |
| `test_inbox.py` | Message read/reply mechanics |
| `test_negotiation.py` | Multi-round standup/kickoff negotiation |
| `test_schema_drift.py` | Availability change, policy activation, constraint count changes |
| `test_personal_calendar.py` | Immovable events, work-life conflict resolution |

## Project Structure

```
personal-assistant/
├── personal_assistant/
│   ├── server/
│   │   ├── app.py                          # FastAPI (REST + WebSocket)
│   │   ├── personal_assistant_environment.py  # Core env (1500 lines)
│   │   ├── seed_generator.py               # Episode randomization
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── models.py                           # CalendarAction/Observation/State
│   ├── client.py                           # EnvClient
│   ├── run_random.py                       # Random interaction test
│   ├── run_agent.py                        # LLM agent (headless)
│   ├── run_agent_live.py                   # LLM agent + live dashboard
│   └── run_batch_test.py                   # Multi-seed batch eval
├── train_grpo.py                           # GRPO training script
├── eval_model.py                           # Baseline/trained evaluation + charts
├── tests/                                  # pytest suite (6 files)
├── ENV_BEHAVIOR.md                         # Full environment specification
└── state_diagram.md                        # State machine documentation
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/reset` | POST | Reset environment (optional `seed`) |
| `/step` | POST | Take action (`{"instruction": "..."}`) |
| `/ws` | WebSocket | Persistent session (reset/step/state) |
| `/web` | GET | Built-in web interface |

## Judging Criteria Alignment

| Criterion | What We Built |
|---|---|
| **Environment Innovation (40%)** | 18-task calendar env with schema drift (availability, policies, API contracts change mid-episode), multi-round negotiation, randomized interrupts, inbox-driven task discovery |
| **Storytelling (30%)** | Live dashboard (`run_agent_live.py`) showing real-time tool calls, interrupt timeline, reward progression, task completion |
| **Training Progress (20%)** | `train_grpo.py` + `eval_model.py` with reward curves, per-flag success rates, baseline vs trained comparison charts |
| **Reward Pipeline (10%)** | Weighted rewards (0.5x/1.0x/1.5x), one-shot locking, 18 outcome-based flags, constraint-aware evaluation |
