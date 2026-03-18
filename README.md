# Calendar Personal Assistant — OpenEnv RL Environment

An RL environment where an agent manages a team calendar under shifting rules, evolving constraints, and ambiguous inbox requests. Built with the [OpenEnv](https://openenv.org) framework.

The agent must discover tasks from natural-language inbox messages, handle schema drift mid-episode (availability changes, new policies, meeting cancellations), negotiate with attendees who push back on proposals, and schedule around immovable personal events — all while 7 randomized interrupts fire at unpredictable steps.

## Quick Start

```bash
# Setup
cd personal_assistant
uv venv && source .venv/bin/activate
uv sync

# Start the environment server
uvicorn server.app:app --port 8000

# Test with random actions
python run_random.py
```

## Running the Agent

```bash
# Headless agent (Qwen2.5-7B via HuggingFace, requires HF_TOKEN)
python run_agent.py

# Agent with live web dashboard
python run_agent_live.py
# Then open http://localhost:8001

# Batch evaluation across multiple seeds (requires GROQ_API_KEY)
python run_batch_test.py
```

## Docker

```bash
cd personal_assistant
docker build -t personal-assistant:latest -f server/Dockerfile .
docker run -p 8000:8000 personal-assistant:latest
```

## How It Works

```
reset(seed) ──► Initial calendar + inbox (4-7 events, 11 messages, 2 personal events)
     │
     ▼
step(action) ──► Fire interrupts (if due) ──► Dispatch tool call ──► Recompute 18 flags
     │
     ▼
reward = flags_earned / 18       done = (all 18 flags earned)
```

**Actions** are JSON tool calls: `{"tool": "create_event", "args": {"title": "Standup", ...}}`

**Observations** are self-contained Markov snapshots — calendar state, constraints, inbox, negotiations, and pending interrupts. No conversation history needed.

### Schema Drift

The environment changes mid-episode through 7 randomized interrupts:

| Interrupt | Effect |
|---|---|
| CEO emergency | Urgent meeting injected at 3 PM, may conflict with existing events |
| Meeting cancellation | A lunch meeting is cancelled, opening a slot |
| Reschedule request | Attendee asks to move a meeting to a new time |
| Availability change | Bob's Wednesday schedule gets new blocked hours |
| Inbox contradiction | A client changes their earlier request |
| Personal event shift | Family texts that dinner moved earlier, creating new conflicts |
| Description policy | HR activates a rule: meetings >30 min need descriptions |

Interrupt timing is randomized per seed — agents can't memorize when they fire.

### Tools (15)

| Category | Tools |
|---|---|
| Calendar | `list_events`, `create_event`, `delete_event`, `edit_event` |
| Scheduling | `find_free_slots`, `check_conflicts`, `resolve_conflict`, `check_availability` |
| Constraints | `get_constraints`, `get_contact_preferences`, `check_constraint_violations` |
| Communication | `read_inbox`, `reply_message`, `send_notification` |
| Personal | `check_personal_calendar` |

### Tasks (18)

Tasks are weighted by difficulty:

- **Hard (1.5x):** Schedule standup with negotiation, schedule kickoff with negotiation, clear hard constraints, handle availability drift, handle client contradiction, handle personal event update
- **Medium (1.0x):** Book focus time, resolve conflicts, cancel meeting, accommodate CEO sync, optimize preferences, meet description policy, clear inbox, send diplomatic reply, resolve work-life conflicts
- **Easy (0.5x):** Set reminder, handle cancellation, handle reschedule

Three reactive tasks use **one-shot locking** — once earned, the reward is kept even if later actions would invalidate the condition.

## Training (GRPO)

Train a Qwen2.5-3B-Instruct model with QLoRA + GRPO:

```bash
python train_grpo.py
```

Uses 4-bit QLoRA (rank 32), group size 3 with cross-seed advantages, 40 steps/episode, 10 iterations, lr 5e-5. Designed for ~2 hours on an H100.

### Evaluation

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

| Test | Coverage |
|---|---|
| `test_seed_determinism.py` | Same seed = same initial state |
| `test_completion_criteria.py` | All 18 flag conditions |
| `test_inbox.py` | Message read/reply mechanics |
| `test_negotiation.py` | Multi-round standup/kickoff negotiation |
| `test_schema_drift.py` | Availability change, policy activation, constraint changes |
| `test_personal_calendar.py` | Immovable events, work-life conflict resolution |

## API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/reset` | POST | Reset environment (optional `seed`) |
| `/step` | POST | Take action (`{"instruction": "..."}`) |
| `/ws` | WebSocket | Persistent session |
| `/web` | GET | Built-in web interface |

## Project Structure

```
personal-assistant/
├── personal_assistant/
│   ├── server/
│   │   ├── app.py                             # FastAPI entry point
│   │   ├── personal_assistant_environment.py  # Core environment (~1500 lines)
│   │   ├── seed_generator.py                  # Deterministic episode generation
│   │   └── Dockerfile
│   ├── models.py              # Pydantic schemas (Action/Observation/State)
│   ├── client.py              # Type-safe EnvClient wrapper
│   ├── run_random.py          # Random action test
│   ├── run_agent.py           # Headless LLM agent
│   ├── run_agent_live.py      # Agent + live web dashboard
│   └── run_batch_test.py      # Multi-seed evaluation
├── train_grpo.py              # GRPO training script
├── eval_model.py              # Evaluation + comparison charts
├── tests/                     # Test suite
├── ENV_BEHAVIOR.md            # Full environment specification
└── PROJECT_DESCRIPTION.md     # Detailed technical description
```
