"""Step 6b: LLM agent with live web dashboard."""

import asyncio
import json
import os
import random
import threading
import time
from datetime import date, timedelta
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from openai import OpenAI
import uvicorn
import websockets

load_dotenv()

WS_URL = "ws://localhost:8000/ws"
MODEL = "Qwen/Qwen2.5-72B-Instruct"
AGENT_SEED = int(os.getenv("AGENT_SEED", "7"))
EPISODE_BASE_DATE = date(2026, 1, 5)  # Keep in sync with environment.
EPISODE_WEEKS = 3  # Keep in sync with environment.

# --- Dashboard Server ---

dashboard_app = FastAPI()
dashboard_clients: List[WebSocket] = []


async def broadcast(event: dict):
    """Send event to all connected dashboard clients."""
    dead = []
    for client in dashboard_clients:
        try:
            await client.send_json(event)
        except Exception:
            dead.append(client)
    for d in dead:
        dashboard_clients.remove(d)


@dashboard_app.get("/", response_class=HTMLResponse)
async def dashboard_page():
    return DASHBOARD_HTML


@dashboard_app.websocket("/ws/feed")
async def dashboard_feed(websocket: WebSocket):
    await websocket.accept()
    dashboard_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        dashboard_clients.remove(websocket)


# --- Tool Definitions ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_events",
            "description": "List all calendar events for a given date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date string: 'today', 'tomorrow', 'next monday', or YYYY-MM-DD"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_event",
            "description": "Create a calendar event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "date": {"type": "string", "description": "'today', 'tomorrow', 'next monday', or YYYY-MM-DD"},
                    "start_time": {"type": "string", "description": "HH:MM format"},
                    "duration_minutes": {"type": "integer", "default": 60},
                    "attendees": {"type": "string", "description": "Comma-separated names"},
                    "description": {"type": "string", "description": "Meeting agenda/description (required for meetings >30 min after policy update)"},
                },
                "required": ["title", "date", "start_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_event",
            "description": "Delete a calendar event by title.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"}
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_event",
            "description": "Edit an existing event. Only provided fields are changed. Use empty string or 0 to keep current value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Current title of the event to edit"},
                    "new_title": {"type": "string", "description": "New title (optional)"},
                    "new_date": {"type": "string", "description": "New date (optional)"},
                    "new_start_time": {"type": "string", "description": "New start time HH:MM (optional)"},
                    "new_duration_minutes": {"type": "integer", "description": "New duration in minutes (optional)"},
                    "new_attendees": {"type": "string", "description": "New comma-separated attendees list — replaces all current attendees (optional)"},
                    "new_description": {"type": "string", "description": "New description/agenda for the event"},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_free_slots",
            "description": "Find available time slots on a given date (8:00-18:00).",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "default": "today"},
                    "duration_minutes": {"type": "integer", "default": 60},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_conflicts",
            "description": "Check for scheduling conflicts on a date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "default": "today"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_conflict",
            "description": "Resolve a conflict by moving an event to a new time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_title": {"type": "string"},
                    "new_start_time": {"type": "string", "description": "HH:MM format"},
                },
                "required": ["event_title", "new_start_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_notification",
            "description": "Send a notification to a person.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["to", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_task_list",
            "description": "Get the list of tasks to complete.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "Check a person's availability on a given date. Shows their busy times from external commitments and free windows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "person": {"type": "string", "description": "Person's name (e.g. Alice, Bob, Charlie, Dave, Eve)"},
                    "date": {"type": "string", "description": "Date: 'today', 'tomorrow', day name like 'tuesday', 'next wednesday', or YYYY-MM-DD", "default": "today"},
                },
                "required": ["person"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_constraints",
            "description": "Get scheduling constraints (hard and soft) that apply to the calendar. Note: individual people may have additional private constraints — use get_contact_preferences to discover them.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_contact_preferences",
            "description": "Get a person's scheduling preferences, private constraints, role, and preferred notification method. Some constraints are only visible through this tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "person": {"type": "string", "description": "Person's name (e.g. Alice, Bob, Charlie, Dave, Eve)"},
                },
                "required": ["person"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_constraint_violations",
            "description": "Check the current calendar for all constraint violations.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a calendar personal assistant. You have access to tools to manage a calendar.

Your goal is to complete ALL tasks on the task list. Start by calling get_task_list to see what needs to be done, then use the available tools to complete each task.

IMPORTANT workflow:
1. First call get_task_list and get_constraints to understand the general rules.
2. BEFORE scheduling any meeting with someone, call get_contact_preferences(person) to discover their private constraints and preferences. Not all constraints are visible in get_constraints.
3. Use check_availability before scheduling — don't guess times.
4. When scheduling, respect HARD constraints (must obey) and SOFT constraints (preferences).
5. After making changes, call check_constraint_violations to catch any violations.
6. Periodically call get_task_list to check which tasks are still TODO.
7. The "preferences_optimized" task requires soft constraints to be satisfied. If you see soft violations, fix them.
8. Think step by step about what tools to call and in what order.
9. When creating meetings, attendees may decline with feedback. Read their response carefully and adjust your next attempt (different duration, time, or format) to address their specific concern. Do not just retry the same parameters.
10. Pay attention to policy changes announced via interrupts. Rules may change mid-session — tools and constraints that worked before may behave differently after an update. Re-check constraints after any policy announcement.
11. If someone's availability changes, re-validate any meetings you already scheduled with that person."""


# --- Agent Logic ---


def _seed_to_episode_today(seed: int) -> str:
    rng = random.Random(seed)
    weekdays = []
    for week in range(EPISODE_WEEKS):
        for day in range(5):
            weekdays.append(EPISODE_BASE_DATE + timedelta(weeks=week, days=day))
    return rng.choice(weekdays).isoformat()


def _extract_interrupt(step: int, output: str) -> dict | None:
    if "--- INTERRUPT ---" not in output:
        return None

    interrupt_text = output.split("--- INTERRUPT ---", 1)[1].strip()
    if step == 3 or "CEO" in interrupt_text:
        key = "ceo_sync"
    elif step == 7 or "unavailable Wednesday" in interrupt_text:
        key = "availability_change"
    elif step == 6 or "Lunch with Client" in interrupt_text:
        key = "lunch_cancellation"
    elif step == 9 or "Morning Sync" in interrupt_text:
        key = "morning_reschedule"
    elif step == 12 or "description/agenda" in interrupt_text:
        key = "description_policy"
    else:
        key = f"interrupt_step_{step}"

    return {"key": key, "message": interrupt_text}


async def _fetch_final_calendar(ws):
    """Fetch calendar for this week and next week (all weekdays)."""
    episode_today = _seed_to_episode_today(AGENT_SEED)
    today_dt = date.fromisoformat(episode_today)
    days_since_mon = today_dt.weekday()
    week_start = today_dt - timedelta(days=days_since_mon)

    dates_to_check = []
    for week in range(2):
        for day in range(5):  # Mon-Fri
            d = week_start + timedelta(weeks=week, days=day)
            dates_to_check.append(d.isoformat())

    results = []
    for d in dates_to_check:
        action = {"type": "step", "data": {"instruction": json.dumps({"tool": "list_events", "args": {"date": d}})}}
        await ws.send(json.dumps(action))
        resp = json.loads(await ws.recv())
        output = resp["data"]["observation"]["output"]
        if "No events" not in output:
            results.append(output)
    return "\n\n".join(results) if results else "No events on any day."


async def run_agent():
    # Wait for a browser client to connect before starting
    print("Waiting for a browser client to connect to the dashboard...")
    while not dashboard_clients:
        await asyncio.sleep(0.5)
    print("Browser connected! Starting agent in 2 seconds...")
    await asyncio.sleep(2)

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.getenv("HF_TOKEN"),
    )

    async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=120) as ws:
        # Reset
        await ws.send(json.dumps({"type": "reset", "data": {"seed": AGENT_SEED}}))
        reset_resp = json.loads(await ws.recv())
        obs = reset_resp["data"]["observation"]
        await ws.send(json.dumps({"type": "state"}))
        state_resp = json.loads(await ws.recv())
        episode_id = state_resp.get("data", {}).get("episode_id", "unknown")
        episode_today = _seed_to_episode_today(AGENT_SEED)

        await broadcast({
            "type": "reset",
            "observation": obs,
            "state": {"step_count": 0},
            "context": {
                "seed": AGENT_SEED,
                "episode_id": episode_id,
                "episode_today": episode_today,
            },
        })

        # Sliding window: keep system prompt + state summary + last N exchanges
        SLIDING_WINDOW = 6  # keep last 6 messages (3 tool call/result pairs)
        history = []  # full history for sliding window
        latest_state_summary = obs.get("state_summary", obs["output"])

        for step in range(60):
            await broadcast({"type": "thinking", "step": step + 1})

            # Build messages: system + state summary + recent history
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": latest_state_summary},
            ] + history[-SLIDING_WINDOW:]

            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )

            msg = response.choices[0].message

            if msg.tool_calls:
                history.append(msg)

                for tool_call in msg.tool_calls:
                    tool_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments or "{}")

                    await broadcast({
                        "type": "tool_call",
                        "step": step + 1,
                        "tool": tool_name,
                        "args": args,
                    })

                    action = {"type": "step", "data": {"instruction": json.dumps({"tool": tool_name, "args": args})}}
                    await ws.send(json.dumps(action))
                    step_resp = json.loads(await ws.recv())

                    data = step_resp["data"]
                    obs = data["observation"]
                    reward = data.get("reward", 0)
                    done = data.get("done", False)

                    # Update state summary from latest observation
                    latest_state_summary = obs.get("state_summary", latest_state_summary)

                    await broadcast({
                        "type": "observation",
                        "step": step + 1,
                        "tool": tool_name,
                        "output": obs["output"],
                        "events_today": obs.get("events_today", 0),
                        "pending_tasks": obs.get("pending_tasks", 0),
                        "flags_found": obs.get("flags_found", []),
                        "reward": reward,
                        "done": done,
                    })
                    interrupt = _extract_interrupt(step + 1, obs["output"])
                    if interrupt:
                        await broadcast({
                            "type": "interrupt",
                            "step": step + 1,
                            "interrupt_key": interrupt["key"],
                            "message": interrupt["message"],
                        })

                    history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": obs["output"],
                    })

                    if done:
                        calendar = await _fetch_final_calendar(ws)
                        await broadcast({"type": "complete", "reward": reward, "calendar": calendar})
                        return
            else:
                await broadcast({
                    "type": "llm_text",
                    "step": step + 1,
                    "content": msg.content,
                })
                history.append(msg)
                history.append({"role": "user", "content": "Continue completing the remaining tasks. Use the tools available to you."})

        calendar = await _fetch_final_calendar(ws)
        await broadcast({"type": "max_steps_reached", "calendar": calendar})


def start_dashboard():
    uvicorn.run(dashboard_app, host="0.0.0.0", port=8001, log_level="warning")


async def main():
    # Start dashboard in background thread
    thread = threading.Thread(target=start_dashboard, daemon=True)
    thread.start()

    print("Dashboard running at http://localhost:8001")
    print("Open it in your browser, then the agent will start in 2 seconds...")

    await run_agent()

    print("\nAgent finished. Dashboard still running — refresh browser to review.")
    print("Press Ctrl+C to exit.")
    # Keep alive so dashboard stays up
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass


# --- Dashboard HTML ---

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Calendar Agent - Live Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    background: #0a0a0f;
    color: #e0e0e0;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }
  header {
    background: #12121a;
    border-bottom: 1px solid #2a2a3a;
    padding: 12px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  header h1 {
    font-size: 16px;
    color: #7c8aff;
    font-weight: 600;
  }
  #status {
    font-size: 12px;
    padding: 4px 10px;
    border-radius: 12px;
    background: #1a1a2e;
  }
  #status.connected { color: #4ade80; border: 1px solid #4ade8040; }
  #status.disconnected { color: #f87171; border: 1px solid #f8717140; }
  .main {
    display: flex;
    flex: 1;
    overflow: hidden;
  }
  .left-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    border-right: 1px solid #2a2a3a;
    min-width: 0;
  }
  .right-panel {
    width: 320px;
    display: flex;
    flex-direction: column;
    background: #0d0d14;
  }
  .panel-header {
    padding: 10px 16px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #666;
    border-bottom: 1px solid #2a2a3a;
    background: #10101a;
  }
  #feed {
    flex: 1;
    overflow-y: auto;
    padding: 12px;
  }
  .event {
    margin-bottom: 8px;
    padding: 10px 12px;
    border-radius: 6px;
    font-size: 13px;
    line-height: 1.5;
    animation: fadeIn 0.3s ease;
  }
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
  }
  .event.reset { background: #1a1a2e; border-left: 3px solid #7c8aff; }
  .event.thinking { background: #1a1a20; border-left: 3px solid #fbbf24; color: #fbbf24; }
  .event.tool-call { background: #0f1a15; border-left: 3px solid #4ade80; }
  .event.observation { background: #12121a; border-left: 3px solid #60a5fa; }
  .event.llm-text { background: #1a1518; border-left: 3px solid #f472b6; }
  .event.complete { background: #0f1a15; border-left: 3px solid #4ade80; color: #4ade80; font-weight: bold; }
  .event.calendar { background: #12121a; border-left: 3px solid #a78bfa; }
  .event.calendar strong { color: #a78bfa; }
  .event.interrupt { background: #21180f; border-left: 3px solid #f59e0b; color: #f8d18d; }
  .event.error { background: #1a1012; border-left: 3px solid #f87171; }
  .step-badge {
    display: inline-block;
    background: #2a2a3a;
    color: #aaa;
    font-size: 10px;
    padding: 1px 6px;
    border-radius: 3px;
    margin-right: 6px;
  }
  .tool-name {
    color: #4ade80;
    font-weight: 600;
  }
  .reward-bar-container {
    padding: 16px;
    border-bottom: 1px solid #2a2a3a;
  }
  .reward-label {
    font-size: 11px;
    color: #888;
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
  }
  .reward-bar {
    height: 20px;
    background: #1a1a2e;
    border-radius: 10px;
    overflow: hidden;
  }
  .reward-fill {
    height: 100%;
    background: linear-gradient(90deg, #7c8aff, #4ade80);
    border-radius: 10px;
    transition: width 0.5s ease;
    width: 0%;
  }
  .stats {
    padding: 16px;
    border-bottom: 1px solid #2a2a3a;
  }
  .stat-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
    font-size: 13px;
  }
  .stat-label { color: #666; }
  .stat-value { color: #e0e0e0; font-weight: 600; }
  .episode-context {
    padding: 12px 16px;
    border-bottom: 1px solid #2a2a3a;
    background: #10101a;
  }
  .context-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
    margin-bottom: 8px;
  }
  .context-row:last-child { margin-bottom: 0; }
  .context-key { color: #7a7a93; text-transform: uppercase; letter-spacing: 0.5px; }
  .context-value {
    color: #d5d8ff;
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    max-width: 190px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .interrupts {
    padding: 12px 16px;
    border-bottom: 1px solid #2a2a3a;
    background: #0f1017;
  }
  .interrupt-item {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    font-size: 12px;
    padding: 7px 8px;
    border-radius: 4px;
    margin-bottom: 6px;
    border: 1px solid #2a2a3a;
  }
  .interrupt-item:last-child { margin-bottom: 0; }
  .interrupt-item.pending { color: #8a8aa3; background: #141624; }
  .interrupt-item.fired { color: #ffd38a; background: #221a10; border-color: #5a4520; }
  .interrupt-item.current { box-shadow: inset 0 0 0 1px #7c8aff80; }
  .interrupt-step {
    min-width: 44px;
    color: #98a2ff;
    font-size: 11px;
    text-transform: uppercase;
  }
  .interrupt-label { flex: 1; line-height: 1.3; }
  .interrupt-history {
    margin: 0;
    padding: 8px 16px 10px;
    border-bottom: 1px solid #2a2a3a;
    background: #0f1017;
  }
  .interrupt-history summary {
    cursor: pointer;
    color: #b4bbff;
    font-size: 12px;
    user-select: none;
    list-style: none;
  }
  .interrupt-history summary::-webkit-details-marker { display: none; }
  .interrupt-history summary::before {
    content: '▸';
    display: inline-block;
    margin-right: 6px;
    transition: transform 0.2s ease;
  }
  .interrupt-history[open] summary::before { transform: rotate(90deg); }
  .interrupt-log {
    margin-top: 8px;
    max-height: 170px;
    overflow-y: auto;
    border: 1px solid #2a2a3a;
    border-radius: 4px;
    background: #141624;
    padding: 8px;
  }
  .interrupt-log-item {
    border-bottom: 1px solid #2a2a3a;
    padding-bottom: 8px;
    margin-bottom: 8px;
  }
  .interrupt-log-item:last-child {
    border-bottom: none;
    margin-bottom: 0;
    padding-bottom: 0;
  }
  .interrupt-log-header {
    color: #d7dbff;
    font-size: 11px;
    margin-bottom: 4px;
  }
  .interrupt-log-empty {
    color: #7a7a93;
    font-size: 11px;
    font-style: italic;
  }
  .flags-section {
    padding: 16px;
    flex: 1;
    overflow-y: auto;
  }
  .task-item {
    display: flex;
    align-items: flex-start;
    margin-bottom: 10px;
    font-size: 12px;
    padding: 8px;
    border-radius: 4px;
    transition: all 0.3s ease;
  }
  .task-item.done { background: #0f1a1540; }
  .task-item.todo { background: transparent; }
  .task-desc {
    font-size: 11px;
    color: #666;
    margin-top: 2px;
  }
  .task-item.done .task-desc { color: #4ade8080; }
  .flag-icon { margin-right: 8px; font-size: 16px; flex-shrink: 0; margin-top: 1px; }
  .task-item.done .flag-icon { color: #4ade80; }
  .task-item.todo .flag-icon { color: #555; }
  .task-item.done strong { color: #4ade80; }
  .task-item.todo strong { color: #888; }
  pre {
    white-space: pre-wrap;
    word-break: break-word;
    font-family: inherit;
    margin: 4px 0 0 0;
    color: #bbb;
  }
</style>
</head>
<body>

<header>
  <h1>Calendar Agent - Live Dashboard</h1>
  <span id="status" class="disconnected">disconnected</span>
</header>

<div class="main">
  <div class="left-panel">
    <div class="panel-header">Agent Feed</div>
    <div id="feed"></div>
  </div>
  <div class="right-panel">
    <div class="panel-header">Reward</div>
    <div class="reward-bar-container">
      <div class="reward-label">
        <span>Progress</span>
        <span id="reward-pct">0%</span>
      </div>
      <div class="reward-bar">
        <div class="reward-fill" id="reward-fill"></div>
      </div>
    </div>
    <div class="panel-header">Stats</div>
    <div class="stats">
      <div class="stat-row"><span class="stat-label">Step</span><span class="stat-value" id="stat-step">0</span></div>
      <div class="stat-row"><span class="stat-label">Events Today</span><span class="stat-value" id="stat-events">0</span></div>
      <div class="stat-row"><span class="stat-label">Pending Tasks</span><span class="stat-value" id="stat-pending">0</span></div>
      <div class="stat-row"><span class="stat-label">Reward</span><span class="stat-value" id="stat-reward">0.0</span></div>
    </div>
    <div class="panel-header">Episode</div>
    <div class="episode-context">
      <div class="context-row"><span class="context-key">Seed</span><span class="context-value" id="ctx-seed">-</span></div>
      <div class="context-row"><span class="context-key">Date</span><span class="context-value" id="ctx-date">-</span></div>
      <div class="context-row"><span class="context-key">Episode ID</span><span class="context-value" id="ctx-episode-id">-</span></div>
    </div>
    <div class="panel-header">Interrupt Timeline</div>
    <div class="interrupts" id="interrupts"></div>
    <details class="interrupt-history" id="interrupt-history">
      <summary id="interrupt-history-summary">Interrupt details (0)</summary>
      <div class="interrupt-log" id="interrupt-log">
        <div class="interrupt-log-empty">No interrupts fired yet.</div>
      </div>
    </details>
    <div class="panel-header">Tasks</div>
    <div class="flags-section" id="flags">
      <div class="task-item todo"><span class="flag-icon">○</span><div><strong>standup_scheduled</strong><div class="task-desc">Find time for Alice & Bob this week for standup</div></div></div>
      <div class="task-item todo"><span class="flag-icon">○</span><div><strong>focus_time_booked</strong><div class="task-desc">Book 1hr+ focus time today with no overlaps</div></div></div>
      <div class="task-item todo"><span class="flag-icon">○</span><div><strong>conflicts_resolved</strong><div class="task-desc">Clean up overlapping meetings today</div></div></div>
      <div class="task-item todo"><span class="flag-icon">○</span><div><strong>reminder_set</strong><div class="task-desc">Dentist appointment next week, afternoon</div></div></div>
      <div class="task-item todo"><span class="flag-icon">○</span><div><strong>meeting_cancelled</strong><div class="task-desc">Handle Old Project Review cancellation + notify</div></div></div>
      <div class="task-item todo"><span class="flag-icon">○</span><div><strong>ceo_sync_accommodated</strong><div class="task-desc">[DYNAMIC] CEO Sync at 3PM, no conflicts</div></div></div>
      <div class="task-item todo"><span class="flag-icon">○</span><div><strong>cancellation_handled</strong><div class="task-desc">[DYNAMIC] Handle Lunch with Client cancellation</div></div></div>
      <div class="task-item todo"><span class="flag-icon">○</span><div><strong>reschedule_handled</strong><div class="task-desc">[DYNAMIC] Move Morning Sync to 11:00 AM</div></div></div>
      <div class="task-item todo"><span class="flag-icon">○</span><div><strong>kickoff_scheduled</strong><div class="task-desc">Kickoff with Alice, Bob, Eve respecting constraints</div></div></div>
      <div class="task-item todo"><span class="flag-icon">○</span><div><strong>hard_constraints_clear</strong><div class="task-desc">Zero hard-constraint violations</div></div></div>
      <div class="task-item todo"><span class="flag-icon">○</span><div><strong>preferences_optimized</strong><div class="task-desc">Satisfy soft constraints/preferences</div></div></div>
      <div class="task-item todo"><span class="flag-icon">○</span><div><strong>availability_drift_handled</strong><div class="task-desc">[DYNAMIC] Re-validate after Bob availability change</div></div></div>
      <div class="task-item todo"><span class="flag-icon">○</span><div><strong>description_policy_met</strong><div class="task-desc">[DYNAMIC] All meetings comply with description policy</div></div></div>
    </div>
  </div>
</div>

<script>
const feed = document.getElementById('feed');
const statusEl = document.getElementById('status');

const TASKS = [
  { flag: 'standup_scheduled', desc: 'Find time for Alice & Bob this week for standup' },
  { flag: 'focus_time_booked', desc: 'Book 1hr+ focus time today with no overlaps' },
  { flag: 'conflicts_resolved', desc: 'Clean up overlapping meetings today' },
  { flag: 'reminder_set', desc: 'Dentist appointment next week, afternoon' },
  { flag: 'meeting_cancelled', desc: 'Handle Old Project Review cancellation + notify' },
  { flag: 'ceo_sync_accommodated', desc: '[DYNAMIC] CEO Sync at 3PM, no conflicts' },
  { flag: 'cancellation_handled', desc: '[DYNAMIC] Handle Lunch with Client cancellation' },
  { flag: 'reschedule_handled', desc: '[DYNAMIC] Move Morning Sync to 11:00 AM' },
  { flag: 'kickoff_scheduled', desc: 'Kickoff with Alice, Bob, Eve respecting constraints' },
  { flag: 'hard_constraints_clear', desc: 'Zero hard-constraint violations' },
  { flag: 'preferences_optimized', desc: 'Satisfy soft constraints/preferences' },
  { flag: 'availability_drift_handled', desc: '[DYNAMIC] Re-validate after Bob availability change' },
  { flag: 'description_policy_met', desc: '[DYNAMIC] All meetings comply with description policy' },
];
const INTERRUPTS = [
  { key: 'ceo_sync', step: 3, label: 'Inject urgent CEO Sync at 15:00 today' },
  { key: 'availability_change', step: 7, label: "Bob's Wednesday afternoons now blocked" },
  { key: 'lunch_cancellation', step: 6, label: "Cancel 'Lunch with Client' and free 12:00-13:00 slot" },
  { key: 'morning_reschedule', step: 9, label: "Request move: 'Morning Sync' to 11:00" },
  { key: 'description_policy', step: 12, label: "HR: meetings >30min need description" },
];
let firedInterrupts = new Set();
const interruptMessages = new Map();
let interruptHistory = [];

function addEvent(html, cls) {
  const div = document.createElement('div');
  div.className = 'event ' + cls;
  div.innerHTML = html;
  feed.appendChild(div);
  feed.scrollTop = feed.scrollHeight;
}

function updateReward(r) {
  const pct = Math.round(r * 100);
  document.getElementById('reward-fill').style.width = pct + '%';
  document.getElementById('reward-pct').textContent = pct + '%';
  document.getElementById('stat-reward').textContent = r.toFixed(2);
}

function updateFlags(flags) {
  const container = document.getElementById('flags');
  container.innerHTML = TASKS.map(t => {
    const done = flags.includes(t.flag);
    return `<div class="task-item ${done ? 'done' : 'todo'}"><span class="flag-icon">${done ? '●' : '○'}</span><div><strong>${t.flag}</strong><div class="task-desc">${t.desc}</div></div></div>`;
  }).join('');
}

function setEpisodeContext(context) {
  document.getElementById('ctx-seed').textContent = context.seed ?? '-';
  document.getElementById('ctx-date').textContent = context.episode_today ?? '-';
  document.getElementById('ctx-episode-id').textContent = context.episode_id ?? '-';
}

function escapeHtml(str) {
  return String(str)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function renderInterrupts(currentStep = 0) {
  const container = document.getElementById('interrupts');
  container.innerHTML = INTERRUPTS.map(item => {
    const fired = firedInterrupts.has(item.key);
    const isCurrent = currentStep === item.step;
    const cls = `${fired ? 'fired' : 'pending'} ${isCurrent ? 'current' : ''}`;
    const hoverText = interruptMessages.get(item.key) || '';
    const hoverAttr = hoverText ? ` title="${escapeHtml(hoverText)}"` : '';
    return `<div class="interrupt-item ${cls}">` +
      `<span class="interrupt-step">Step ${item.step}</span>` +
      `<span class="interrupt-label"${hoverAttr}>${item.label}</span>` +
      `</div>`;
  }).join('');
}

function renderInterruptHistory() {
  const summary = document.getElementById('interrupt-history-summary');
  const log = document.getElementById('interrupt-log');
  summary.textContent = `Interrupt details (${interruptHistory.length})`;

  if (!interruptHistory.length) {
    log.innerHTML = '<div class="interrupt-log-empty">No interrupts fired yet.</div>';
    return;
  }

  log.innerHTML = interruptHistory.map(item => {
    return `<div class="interrupt-log-item">` +
      `<div class="interrupt-log-header">Step ${item.step} • ${escapeHtml(item.label)}</div>` +
      `<pre>${escapeHtml(item.message)}</pre>` +
      `</div>`;
  }).join('');
}

function connect() {
  const ws = new WebSocket('ws://localhost:8001/ws/feed');

  ws.onopen = () => {
    statusEl.textContent = 'connected';
    statusEl.className = 'connected';
  };

  ws.onclose = () => {
    statusEl.textContent = 'disconnected';
    statusEl.className = 'disconnected';
    setTimeout(connect, 2000);
  };

  ws.onmessage = (e) => {
    const data = JSON.parse(e.data);

    switch (data.type) {
      case 'reset':
        feed.innerHTML = '';
        firedInterrupts = new Set();
        interruptMessages.clear();
        interruptHistory = [];
        const taskListHtml = TASKS.map((t, i) => `  ${i+1}. ${t.desc}`).join('\\n');
        addEvent(
          `<strong>Environment Reset</strong><pre>${data.observation.output}</pre>` +
          `<br><strong>Tasks to Complete:</strong><pre>${taskListHtml}</pre>`,
          'reset'
        );
        document.getElementById('stat-events').textContent = data.observation.events_today;
        document.getElementById('stat-pending').textContent = data.observation.pending_tasks;
        document.getElementById('stat-step').textContent = '0';
        updateReward(0);
        updateFlags([]);
        setEpisodeContext(data.context || {});
        renderInterrupts(0);
        renderInterruptHistory();
        break;

      case 'thinking':
        addEvent(`<span class="step-badge">Step ${data.step}</span> Thinking...`, 'thinking');
        document.getElementById('stat-step').textContent = data.step;
        renderInterrupts(data.step);
        break;

      case 'tool_call':
        addEvent(
          `<span class="step-badge">Step ${data.step}</span> <span class="tool-name">${data.tool}</span>` +
          `<pre>${JSON.stringify(data.args, null, 2)}</pre>`,
          'tool-call'
        );
        break;

      case 'observation':
        addEvent(
          `<span class="step-badge">Step ${data.step}</span> Result from <span class="tool-name">${data.tool}</span>` +
          `<pre>${data.output}</pre>`,
          'observation'
        );
        document.getElementById('stat-events').textContent = data.events_today;
        document.getElementById('stat-pending').textContent = data.pending_tasks;
        updateReward(data.reward);
        updateFlags(data.flags_found);
        renderInterrupts(data.step);
        break;

      case 'interrupt':
        firedInterrupts.add(data.interrupt_key);
        interruptMessages.set(data.interrupt_key, data.message);
        const meta = INTERRUPTS.find(x => x.key === data.interrupt_key);
        interruptHistory.push({
          step: data.step,
          key: data.interrupt_key,
          label: meta ? meta.label : data.interrupt_key,
          message: data.message,
        });
        addEvent(
          `<span class="step-badge">Step ${data.step}</span> <strong>Interrupt Fired</strong>` +
          `<pre>${data.message}</pre>`,
          'interrupt'
        );
        renderInterrupts(data.step);
        renderInterruptHistory();
        break;

      case 'llm_text':
        addEvent(
          `<span class="step-badge">Step ${data.step}</span> LLM Response` +
          `<pre>${data.content}</pre>`,
          'llm-text'
        );
        break;

      case 'complete':
        addEvent(`All tasks completed! Final reward: ${data.reward}`, 'complete');
        if (data.calendar) {
          addEvent(`<strong>Final Calendar</strong><pre>${data.calendar}</pre>`, 'calendar');
        }
        break;

      case 'max_steps_reached':
        addEvent('Reached max steps without completing all tasks.', 'error');
        if (data.calendar) {
          addEvent(`<strong>Final Calendar</strong><pre>${data.calendar}</pre>`, 'calendar');
        }
        break;
    }
  };
}

renderInterrupts(0);
renderInterruptHistory();
connect();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    asyncio.run(main())
