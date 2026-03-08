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


# --- Tool Definitions (prompt-based, no OpenAI tools parameter) ---

TOOL_DESCRIPTIONS = """## Available Tools

Call tools by outputting a JSON block. You may call ONE tool per response.
Format: ```json
{"tool": "tool_name", "args": {"param": "value"}}
```

### Tools:

1. **list_events** — List all calendar events for a given date.
   Args: date (string, optional): 'today', 'tomorrow', 'next monday', or YYYY-MM-DD

2. **create_event** — Create a calendar event.
   Args: title (string, REQUIRED), date (string, REQUIRED), start_time (string HH:MM, REQUIRED),
         duration_minutes (int, default 60), attendees (string, comma-separated),
         description (string, meeting agenda — required for meetings >30 min after policy update)

3. **delete_event** — Delete a calendar event by title.
   Args: title (string, REQUIRED)

4. **edit_event** — Edit an existing event. Only provided fields are changed.
   Args: title (string, REQUIRED — current title), new_title, new_date, new_start_time (HH:MM),
         new_duration_minutes (int), new_attendees (comma-separated, replaces all), new_description

5. **find_free_slots** — Find available time slots on a given date (8:00-18:00).
   Args: date (string, default 'today'), duration_minutes (int, default 60)

6. **check_conflicts** — Check for scheduling conflicts on a date.
   Args: date (string, default 'today')

7. **resolve_conflict** — Resolve a conflict by moving an event to a new time.
   Args: event_title (string, REQUIRED), new_start_time (string HH:MM, REQUIRED)

8. **send_notification** — Send a notification to a person.
   Args: to (string, REQUIRED), message (string, REQUIRED)

9. **check_availability** — Check a person's availability on a given date.
   Args: person (string, REQUIRED), date (string, default 'today')

10. **get_constraints** — Get scheduling constraints (hard and soft). People may have additional private constraints — use get_contact_preferences.
    Args: (none)

11. **get_contact_preferences** — Get a person's scheduling preferences, private constraints, role, notification method.
    Args: person (string, REQUIRED)

12. **check_constraint_violations** — Check the current calendar for all constraint violations.
    Args: (none)

13. **read_inbox** — List inbox messages.
    Args: status (string: 'all', 'unread', or 'unreplied', default 'all')

14. **reply_message** — Reply to an inbox message. Reply must address the sender's concern.
    Args: message_id (string, REQUIRED), reply (string, REQUIRED, min 20 chars)

15. **check_personal_calendar** — Show personal (immovable) events.
    Args: (none)"""

SYSTEM_PROMPT = """You are a calendar personal assistant managing a team's calendar.

Start by reading your inbox (read_inbox) and reviewing the calendar (list_events) to understand what needs to be done today.

IMPORTANT workflow:
1. Read your inbox to see pending requests from your boss and team.
2. Review the calendar to understand the current schedule.
3. BEFORE scheduling any meeting, call get_contact_preferences(person) to learn their constraints.
4. Use check_availability before scheduling — don't guess times.
5. Respect HARD constraints (must obey) and SOFT constraints (preferences).
6. After making changes, call check_constraint_violations to verify.
7. Keep checking your inbox — new messages may arrive while you work.
8. When attendees decline a meeting, read their feedback and adjust your proposal.
9. Personal events on the calendar are immovable — schedule work around them.
10. Reply to messages that need responses.
11. Inbox-driven requests are tracked in the inbox; if any task-style view omits them, use read_inbox as the source of truth.
12. Think step by step about what tools to call and in what order.
13. Pay attention to policy changes announced via interrupts. Rules may change mid-session.
14. If someone's availability changes, re-validate any meetings you already scheduled with that person.
15. Family may text mid-session about personal event changes. Re-check for new conflicts.

RESPONSE FORMAT: Think briefly, then output exactly ONE tool call as a JSON code block:
```json
{"tool": "tool_name", "args": {"param": "value"}}
```
Do NOT output multiple tool calls. Do NOT skip the JSON block — every response MUST contain one tool call.

""" + TOOL_DESCRIPTIONS


def _parse_tool_call(text: str):
    """Extract a JSON tool call from LLM text output. Returns (tool_name, args) or None."""
    import re
    # Try ```json ... ``` blocks first
    for m in re.finditer(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL):
        try:
            obj = json.loads(m.group(1))
            if "tool" in obj:
                return obj["tool"], obj.get("args", {})
        except json.JSONDecodeError:
            continue
    # Try bare JSON objects
    for m in re.finditer(r'\{[^{}]*"tool"\s*:\s*"[^"]+?"[^{}]*\}', text, re.DOTALL):
        try:
            obj = json.loads(m.group(0))
            if "tool" in obj:
                return obj["tool"], obj.get("args", {})
        except json.JSONDecodeError:
            continue
    return None


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
    if "CEO" in interrupt_text:
        key = "ceo_sync"
    elif "unavailable Wednesday" in interrupt_text or "unavailable" in interrupt_text.lower() and "afternoon" in interrupt_text.lower():
        key = "availability_change"
    elif "cancelled" in interrupt_text and ("Lunch" in interrupt_text or "slot is now free" in interrupt_text):
        key = "lunch_cancellation"
    elif "move" in interrupt_text.lower() and ("reschedule" in interrupt_text.lower() or "11:00" in interrupt_text):
        key = "morning_reschedule"
    elif "description" in interrupt_text.lower() and "agenda" in interrupt_text.lower():
        key = "description_policy"
    elif "inbox" in interrupt_text.lower() or "follow-up" in interrupt_text.lower():
        key = "inbox_update"
    elif "partner" in interrupt_text.lower() or "family" in interrupt_text.lower() or "dinner" in interrupt_text.lower():
        key = "personal_event_change"
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
        SLIDING_WINDOW = 8  # keep last 8 messages (4 assistant/user pairs)
        history = []
        latest_state_summary = obs.get("state_summary", obs["output"])
        consecutive_no_tool = 0

        for step in range(120):
            await broadcast({"type": "thinking", "step": step + 1})

            # Build messages: system + state summary + recent history
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": latest_state_summary},
            ] + history[-SLIDING_WINDOW:]

            try:
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=MODEL,
                    messages=messages,
                )
            except Exception as e:
                await broadcast({"type": "llm_text", "step": step + 1, "content": f"API error: {e}"})
                await asyncio.sleep(2)
                continue

            content = response.choices[0].message.content or ""
            parsed = _parse_tool_call(content)

            if parsed:
                consecutive_no_tool = 0
                tool_name, args = parsed

                # Add assistant message to history
                history.append({"role": "assistant", "content": content})

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
                    "calendar_snapshot": obs.get("calendar_snapshot", []),
                })
                interrupt = _extract_interrupt(step + 1, obs["output"])
                if interrupt:
                    await broadcast({
                        "type": "interrupt",
                        "step": step + 1,
                        "interrupt_key": interrupt["key"],
                        "message": interrupt["message"],
                    })

                # Add tool result as user message
                history.append({
                    "role": "user",
                    "content": f"Tool result from {tool_name}:\n{obs['output']}",
                })

                if done:
                    calendar = await _fetch_final_calendar(ws)
                    await broadcast({"type": "complete", "reward": reward, "calendar": calendar})
                    return
            else:
                # LLM responded with text but no tool call
                consecutive_no_tool += 1
                await broadcast({
                    "type": "llm_text",
                    "step": step + 1,
                    "content": content,
                })
                history.append({"role": "assistant", "content": content})
                if consecutive_no_tool >= 3:
                    history.append({"role": "user", "content": "You MUST call a tool. Output a JSON block like: ```json\n{\"tool\": \"read_inbox\", \"args\": {}}\n``` Do it now."})
                else:
                    history.append({"role": "user", "content": "Continue. Call a tool by outputting a JSON block: ```json\n{\"tool\": \"tool_name\", \"args\": {...}}\n```"})

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
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0a0a0f;
    color: #e0e0e0;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }
  header {
    background: #12121a;
    border-bottom: 1px solid #2a2a3a;
    padding: 14px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  header h1 {
    font-size: 15px;
    color: #e0e0e0;
    font-weight: 700;
    letter-spacing: -0.3px;
  }
  header h1 span { color: #7c8aff; }
  .header-right { display: flex; align-items: center; gap: 12px; }
  #status {
    font-size: 11px;
    padding: 4px 10px;
    border-radius: 12px;
    background: #1a1a2e;
    font-weight: 500;
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
    flex-direction: row;
    border-right: 1px solid #2a2a3a;
    min-width: 0;
  }
  .right-panel {
    width: 360px;
    display: flex;
    flex-direction: column;
    background: #0d0d14;
    overflow-y: auto;
  }
  .panel-header {
    padding: 10px 16px;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #555;
    border-bottom: 1px solid #1e1e2e;
    background: #0e0e16;
    position: sticky;
    top: 0;
    z-index: 1;
  }
  #feed-tab { overflow-y: auto; }
  #feed {
    padding: 12px;
  }
  .event {
    margin-bottom: 6px;
    padding: 10px 12px;
    border-radius: 8px;
    font-size: 12.5px;
    line-height: 1.5;
    animation: fadeIn 0.25s ease;
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
  }
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
  }
  .event.reset { background: #141428; border-left: 3px solid #7c8aff; }
  .event.thinking { background: #18170e; border-left: 3px solid #fbbf24; color: #fbbf24; }
  .event.tool-call { background: #0e1a12; border-left: 3px solid #4ade80; }
  .event.observation { background: #0e1220; border-left: 3px solid #60a5fa; }
  .event.llm-text { background: #1a1218; border-left: 3px solid #f472b6; }
  .event.complete { background: #0e1a12; border-left: 3px solid #4ade80; color: #4ade80; font-weight: bold; }
  .event.calendar { background: #12121a; border-left: 3px solid #a78bfa; }
  .event.calendar strong { color: #a78bfa; }
  .event.interrupt { background: #1e170c; border-left: 3px solid #f59e0b; color: #f8d18d; }
  .event.error { background: #1a0e10; border-left: 3px solid #f87171; }
  .step-badge {
    display: inline-block;
    background: #2a2a3a;
    color: #999;
    font-size: 10px;
    padding: 2px 7px;
    border-radius: 4px;
    margin-right: 6px;
    font-weight: 600;
  }
  .tool-name { color: #4ade80; font-weight: 700; }

  /* --- Reward Section --- */
  .reward-section { padding: 16px; border-bottom: 1px solid #1e1e2e; }
  .reward-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 10px;
  }
  .reward-pct {
    font-size: 28px;
    font-weight: 800;
    color: #e0e0e0;
    letter-spacing: -1px;
    line-height: 1;
  }
  .reward-detail {
    font-size: 11px;
    color: #666;
    text-align: right;
    line-height: 1.4;
  }
  .reward-bar {
    height: 8px;
    background: #1a1a2e;
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 12px;
  }
  .reward-fill {
    height: 100%;
    background: linear-gradient(90deg, #7c8aff, #4ade80);
    border-radius: 4px;
    transition: width 0.5s ease;
    width: 0%;
  }
  .reward-breakdown {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 8px;
  }
  .breakdown-card {
    background: #141420;
    border-radius: 6px;
    padding: 8px 10px;
    text-align: center;
  }
  .breakdown-value {
    font-size: 16px;
    font-weight: 700;
    color: #e0e0e0;
  }
  .breakdown-label {
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #666;
    margin-top: 2px;
  }

  /* --- Stats --- */
  .stats {
    padding: 12px 16px;
    border-bottom: 1px solid #1e1e2e;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }
  .stat-card {
    background: #141420;
    border-radius: 6px;
    padding: 8px 10px;
  }
  .stat-value { font-size: 18px; font-weight: 700; color: #e0e0e0; }
  .stat-label { font-size: 9px; text-transform: uppercase; letter-spacing: 0.8px; color: #555; margin-top: 1px; }

  /* --- Episode Context --- */
  .episode-context {
    padding: 12px 16px;
    border-bottom: 1px solid #1e1e2e;
  }
  .context-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11px;
    margin-bottom: 6px;
  }
  .context-row:last-child { margin-bottom: 0; }
  .context-key { color: #555; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; font-size: 10px; }
  .context-value {
    color: #b0b4e0;
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 11px;
  }

  /* --- Interrupts --- */
  .interrupts {
    padding: 10px 16px;
    border-bottom: 1px solid #1e1e2e;
  }
  .interrupt-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    padding: 6px 8px;
    border-radius: 5px;
    margin-bottom: 4px;
    border: 1px solid #1e1e2e;
  }
  .interrupt-item:last-child { margin-bottom: 0; }
  .interrupt-item.pending { color: #666; background: #111118; }
  .interrupt-item.fired { color: #ffd38a; background: #1a1508; border-color: #3a3018; }
  .interrupt-step {
    min-width: 48px;
    color: #7c8aff;
    font-size: 10px;
    font-weight: 600;
  }
  .interrupt-label { flex: 1; line-height: 1.3; }
  .interrupt-history {
    margin: 0;
    padding: 8px 16px 10px;
    border-bottom: 1px solid #1e1e2e;
  }
  .interrupt-history summary {
    cursor: pointer;
    color: #7c8aff;
    font-size: 11px;
    user-select: none;
    list-style: none;
    font-weight: 500;
  }
  .interrupt-history summary::-webkit-details-marker { display: none; }
  .interrupt-history summary::before {
    content: '>';
    display: inline-block;
    margin-right: 6px;
    transition: transform 0.2s ease;
    font-family: monospace;
  }
  .interrupt-history[open] summary::before { transform: rotate(90deg); }
  .interrupt-log {
    margin-top: 8px;
    max-height: 170px;
    overflow-y: auto;
    border: 1px solid #1e1e2e;
    border-radius: 6px;
    background: #111118;
    padding: 8px;
    font-family: 'SF Mono', 'Fira Code', monospace;
  }
  .interrupt-log-item {
    border-bottom: 1px solid #1e1e2e;
    padding-bottom: 8px;
    margin-bottom: 8px;
  }
  .interrupt-log-item:last-child {
    border-bottom: none;
    margin-bottom: 0;
    padding-bottom: 0;
  }
  .interrupt-log-header { color: #b0b4e0; font-size: 10px; margin-bottom: 4px; font-weight: 600; }
  .interrupt-log-empty { color: #555; font-size: 11px; font-style: italic; }

  /* --- Tasks --- */
  .flags-section {
    padding: 12px 16px;
    flex: 1;
    overflow-y: auto;
  }
  .task-group-header {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 8px 0 6px;
    border-bottom: 1px solid #1e1e2e;
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .task-group-header.hard { color: #f59e0b; }
  .task-group-header.medium { color: #60a5fa; }
  .task-group-header.easy { color: #4ade80; }
  .task-group-count { font-size: 10px; font-weight: 500; }
  .task-item {
    display: flex;
    align-items: flex-start;
    margin-bottom: 4px;
    font-size: 11px;
    padding: 7px 8px;
    border-radius: 5px;
    transition: all 0.3s ease;
    gap: 8px;
  }
  .task-item.done { background: #0e1a1230; }
  .task-item.todo { background: transparent; }
  .task-item.locked { background: #14180e30; }
  .task-content { flex: 1; min-width: 0; }
  .task-name {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
  }
  .task-desc {
    font-size: 10px;
    color: #555;
    margin-top: 2px;
    line-height: 1.3;
  }
  .task-item.done .task-desc { color: #4ade8060; }
  .flag-icon { font-size: 14px; flex-shrink: 0; margin-top: 2px; line-height: 1; }
  .task-item.done .flag-icon { color: #4ade80; }
  .task-item.locked .flag-icon { color: #f59e0b; }
  .task-item.todo .flag-icon { color: #333; }
  .task-item.done strong { color: #4ade80; font-weight: 600; }
  .task-item.locked strong { color: #f59e0b; font-weight: 600; }
  .task-item.todo strong { color: #777; font-weight: 500; }
  .weight-badge {
    display: inline-block;
    font-size: 9px;
    padding: 1px 5px;
    border-radius: 3px;
    font-weight: 700;
    letter-spacing: 0.3px;
    flex-shrink: 0;
  }
  .weight-badge.w-hard { background: #f59e0b20; color: #f59e0b; }
  .weight-badge.w-medium { background: #60a5fa20; color: #60a5fa; }
  .weight-badge.w-easy { background: #4ade8020; color: #4ade80; }
  .one-shot-badge {
    display: inline-block;
    font-size: 8px;
    padding: 1px 5px;
    border-radius: 3px;
    background: #a78bfa20;
    color: #a78bfa;
    font-weight: 700;
    letter-spacing: 0.3px;
  }
  .lock-badge {
    display: inline-block;
    font-size: 9px;
    padding: 1px 5px;
    border-radius: 3px;
    background: #f59e0b20;
    color: #f59e0b;
    font-weight: 700;
  }
  pre {
    white-space: pre-wrap;
    word-break: break-word;
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    margin: 4px 0 0 0;
    color: #999;
    font-size: 11.5px;
  }

  /* --- Info Accessed --- */
  .info-section { padding: 12px 16px; border-bottom: 1px solid #1e1e2e; }
  .info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4px;
  }
  .info-tool {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 10px;
    padding: 4px 8px;
    border-radius: 4px;
    background: #111118;
  }
  .info-tool-name { color: #888; font-family: 'SF Mono', monospace; }
  .info-tool-count { color: #e0e0e0; font-weight: 700; font-family: 'SF Mono', monospace; }
  .info-people {
    margin-top: 8px;
  }
  .info-sub-header {
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #555;
    margin-bottom: 4px;
    margin-top: 8px;
  }
  .info-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }
  .info-chip {
    font-size: 10px;
    padding: 3px 8px;
    border-radius: 4px;
    background: #141420;
    font-weight: 500;
  }
  .info-chip.prefs { color: #a78bfa; border: 1px solid #a78bfa30; }
  .info-chip.avail { color: #60a5fa; border: 1px solid #60a5fa30; }
  .info-chip.notified { color: #f59e0b; border: 1px solid #f59e0b30; }
  .info-chip.discovery { color: #4ade80; border: 1px solid #4ade8030; }
  .info-chip.unchecked { color: #555; border: 1px solid #333; opacity: 0.5; }

  /* --- Split Panels --- */
  .feed-pane {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 200px;
    overflow: hidden;
  }
  .drag-handle {
    width: 6px;
    cursor: col-resize;
    background: #1e1e2e;
    flex-shrink: 0;
    transition: background 0.2s;
    position: relative;
  }
  .drag-handle:hover, .drag-handle.dragging {
    background: #7c8aff;
  }
  .drag-handle::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 1px;
    width: 4px;
    height: 30px;
    transform: translateY(-50%);
    border-left: 1px solid #444;
    border-right: 1px solid #444;
  }
  .calendar-pane {
    width: 45%;
    min-width: 200px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* --- Calendar Timeline --- */
  #calendar-view {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    display: flex;
    gap: 12px;
  }
  .cal-day {
    flex: 1;
    min-width: 140px;
  }
  .cal-day-header {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #7c8aff;
    padding: 6px 0;
    border-bottom: 1px solid #1e1e2e;
    margin-bottom: 4px;
    text-align: center;
  }
  .cal-day-header.today { color: #4ade80; }
  .cal-timeline {
    position: relative;
    height: 600px; /* 10hrs * 60px */
  }
  .cal-hour-line {
    position: absolute;
    left: 0;
    right: 0;
    border-top: 1px solid #1a1a25;
    font-size: 9px;
    color: #444;
    padding-left: 2px;
    pointer-events: none;
  }
  .cal-event {
    position: absolute;
    left: 4px;
    right: 4px;
    border-radius: 4px;
    padding: 3px 6px;
    font-size: 10px;
    line-height: 1.3;
    overflow: hidden;
    cursor: default;
    border-left: 3px solid;
    transition: all 0.3s ease;
  }
  .cal-event.work {
    background: #141e30;
    border-left-color: #60a5fa;
    color: #a0c4ff;
  }
  .cal-event.personal {
    background: #1e1420;
    border-left-color: #f472b6;
    color: #f4a0cc;
  }
  .cal-event-title {
    font-weight: 700;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .cal-event-time {
    font-size: 9px;
    opacity: 0.7;
  }
  .cal-event-attendees {
    font-size: 9px;
    opacity: 0.6;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .cal-no-events {
    text-align: center;
    color: #444;
    font-size: 11px;
    padding: 20px 0;
    font-style: italic;
  }
</style>
</head>
<body>

<header>
  <h1><span>Calendar Agent</span> Live Dashboard</h1>
  <div class="header-right">
    <span id="status" class="disconnected">disconnected</span>
  </div>
</header>

<div class="main">
  <div class="left-panel">
    <div class="feed-pane">
      <div class="panel-header">Agent Feed</div>
      <div id="feed-tab" style="flex:1;overflow-y:auto;">
        <div id="feed"></div>
      </div>
    </div>
    <div class="drag-handle" id="drag-handle"></div>
    <div class="calendar-pane" id="calendar-pane">
      <div class="panel-header">Calendar</div>
      <div id="calendar-view"></div>
    </div>
  </div>
  <div class="right-panel">
    <div class="panel-header">Reward</div>
    <div class="reward-section">
      <div class="reward-header">
        <span class="reward-pct" id="reward-pct">0%</span>
        <div class="reward-detail">
          <div id="reward-weight">0.0 / 19.5 weight</div>
          <div id="reward-flags">0 / 18 tasks</div>
        </div>
      </div>
      <div class="reward-bar">
        <div class="reward-fill" id="reward-fill"></div>
      </div>
      <div class="reward-breakdown">
        <div class="breakdown-card">
          <div class="breakdown-value" id="bd-hard" style="color:#f59e0b">0</div>
          <div class="breakdown-label">Hard 1.5x</div>
        </div>
        <div class="breakdown-card">
          <div class="breakdown-value" id="bd-medium" style="color:#60a5fa">0</div>
          <div class="breakdown-label">Medium 1.0x</div>
        </div>
        <div class="breakdown-card">
          <div class="breakdown-value" id="bd-easy" style="color:#4ade80">0</div>
          <div class="breakdown-label">Easy 0.5x</div>
        </div>
      </div>
    </div>

    <div class="panel-header">Stats</div>
    <div class="stats">
      <div class="stat-card"><div class="stat-value" id="stat-step">0</div><div class="stat-label">Step</div></div>
      <div class="stat-card"><div class="stat-value" id="stat-events">0</div><div class="stat-label">Events Today</div></div>
      <div class="stat-card"><div class="stat-value" id="stat-pending">0</div><div class="stat-label">Pending Tasks</div></div>
      <div class="stat-card"><div class="stat-value" id="stat-locked">0</div><div class="stat-label">Locked (1-shot)</div></div>
    </div>

    <div class="panel-header">Info Accessed</div>
    <div class="info-section" id="info-accessed">
      <div class="info-sub-header">Tool Calls</div>
      <div class="info-grid" id="info-tools"></div>
      <div class="info-sub-header">Preferences Checked</div>
      <div class="info-chips" id="info-prefs"></div>
      <div class="info-sub-header">Availability Checked</div>
      <div class="info-chips" id="info-avail"></div>
      <div class="info-sub-header">Notifications Sent</div>
      <div class="info-chips" id="info-notified"></div>
      <div class="info-sub-header">Discovery Actions</div>
      <div class="info-chips" id="info-discovery"></div>
    </div>

    <div class="panel-header">Episode</div>
    <div class="episode-context">
      <div class="context-row"><span class="context-key">Seed</span><span class="context-value" id="ctx-seed">-</span></div>
      <div class="context-row"><span class="context-key">Date</span><span class="context-value" id="ctx-date">-</span></div>
      <div class="context-row"><span class="context-key">Episode</span><span class="context-value" id="ctx-episode-id">-</span></div>
    </div>

    <div class="panel-header">Interrupts</div>
    <div class="interrupts" id="interrupts"></div>
    <details class="interrupt-history" id="interrupt-history">
      <summary id="interrupt-history-summary">Interrupt details (0)</summary>
      <div class="interrupt-log" id="interrupt-log">
        <div class="interrupt-log-empty">No interrupts fired yet.</div>
      </div>
    </details>

    <div class="panel-header">Tasks</div>
    <div class="flags-section" id="flags"></div>
  </div>
</div>

<script>
const feed = document.getElementById('feed');
const statusEl = document.getElementById('status');

const TASKS = [
  { flag: 'standup_scheduled', desc: 'Find time for Alice & Bob this week for standup', weight: 1.5 },
  { flag: 'focus_time_booked', desc: 'Book 1hr+ focus time today with no overlaps', weight: 1.0 },
  { flag: 'conflicts_resolved', desc: 'Clean up overlapping meetings today', weight: 1.0 },
  { flag: 'reminder_set', desc: 'Dentist appointment next week, afternoon', weight: 0.5 },
  { flag: 'meeting_cancelled', desc: 'Handle Old Project Review cancellation + notify', weight: 1.0 },
  { flag: 'ceo_sync_accommodated', desc: 'CEO Sync at 3PM, no conflicts', weight: 1.0 },
  { flag: 'cancellation_handled', desc: 'Handle Lunch with Client cancellation', weight: 0.5 },
  { flag: 'reschedule_handled', desc: 'Move Morning Sync to 11:00 AM', weight: 0.5 },
  { flag: 'kickoff_scheduled', desc: 'Kickoff with Alice, Bob, Eve respecting constraints', weight: 1.5 },
  { flag: 'hard_constraints_clear', desc: 'Zero hard-constraint violations', weight: 1.5 },
  { flag: 'preferences_optimized', desc: 'Satisfy soft constraints/preferences', weight: 1.0 },
  { flag: 'availability_drift_handled', desc: 'Re-validate after Bob availability change', weight: 1.5, oneShot: true },
  { flag: 'description_policy_met', desc: 'All meetings comply with description policy', weight: 1.0 },
  { flag: 'inbox_cleared', desc: 'Reply to all inbox messages', weight: 1.0 },
  { flag: 'diplomatic_reply_sent', desc: 'Reply diplomatically to tough email', weight: 1.0 },
  { flag: 'client_request_updated', desc: 'Handle contradicting client message', weight: 1.5, oneShot: true },
  { flag: 'work_life_conflicts_resolved', desc: 'Resolve personal vs work event conflicts', weight: 1.0 },
  { flag: 'personal_update_handled', desc: 'Adjust after personal event time change', weight: 1.5, oneShot: true },
];

const TOTAL_WEIGHT = TASKS.reduce((s, t) => s + t.weight, 0);
const lockedFlags = new Set();

// --- Info tracking ---
const toolCallCounts = {};
const prefsChecked = new Set();
const availChecked = new Set();
const notifiedPeople = new Set();
const discoveryActions = new Set(); // constraints, inbox, personal_calendar, constraint_violations
const ALL_PEOPLE = ['Alice', 'Bob', 'Charlie', 'Dave', 'Eve'];
const DISCOVERY_ITEMS = [
  { key: 'get_constraints', label: 'Constraints' },
  { key: 'read_inbox', label: 'Inbox' },
  { key: 'check_personal_calendar', label: 'Personal Cal' },
  { key: 'check_constraint_violations', label: 'Violations' },
  { key: 'check_conflicts', label: 'Conflicts' },
  { key: 'find_free_slots', label: 'Free Slots' },
];

function trackToolCall(tool, args) {
  toolCallCounts[tool] = (toolCallCounts[tool] || 0) + 1;
  if (tool === 'get_contact_preferences' && args.person) prefsChecked.add(args.person);
  if (tool === 'check_availability' && args.person) availChecked.add(args.person);
  if (tool === 'send_notification' && args.to) notifiedPeople.add(args.to);
  if (DISCOVERY_ITEMS.some(d => d.key === tool)) discoveryActions.add(tool);
  renderInfoAccessed();
}

function renderInfoAccessed() {
  // Tool call grid (sorted by count desc)
  const sorted = Object.entries(toolCallCounts).sort((a, b) => b[1] - a[1]);
  document.getElementById('info-tools').innerHTML = sorted.map(([name, count]) =>
    `<div class="info-tool"><span class="info-tool-name">${name}</span><span class="info-tool-count">${count}</span></div>`
  ).join('');

  // Prefs chips
  document.getElementById('info-prefs').innerHTML = ALL_PEOPLE.map(p =>
    `<span class="info-chip ${prefsChecked.has(p) ? 'prefs' : 'unchecked'}">${p}</span>`
  ).join('');

  // Avail chips
  document.getElementById('info-avail').innerHTML = ALL_PEOPLE.map(p =>
    `<span class="info-chip ${availChecked.has(p) ? 'avail' : 'unchecked'}">${p}</span>`
  ).join('');

  // Notified chips
  document.getElementById('info-notified').innerHTML = notifiedPeople.size
    ? [...notifiedPeople].map(p => `<span class="info-chip notified">${p}</span>`).join('')
    : '<span class="info-chip unchecked">none yet</span>';

  // Discovery chips
  document.getElementById('info-discovery').innerHTML = DISCOVERY_ITEMS.map(d =>
    `<span class="info-chip ${discoveryActions.has(d.key) ? 'discovery' : 'unchecked'}">${d.label}</span>`
  ).join('');
}

function resetInfoTracking() {
  Object.keys(toolCallCounts).forEach(k => delete toolCallCounts[k]);
  prefsChecked.clear();
  availChecked.clear();
  notifiedPeople.clear();
  discoveryActions.clear();
  renderInfoAccessed();
}

const INTERRUPTS = [
  { key: 'ceo_sync', label: 'CEO Sync injected at 15:00' },
  { key: 'inbox_update', label: 'Contradicting follow-up arrives' },
  { key: 'availability_change', label: "Bob's Wed afternoons blocked" },
  { key: 'lunch_cancellation', label: "Lunch with Client cancelled" },
  { key: 'morning_reschedule', label: "Morning Sync -> 11:00" },
  { key: 'personal_event_change', label: 'Personal event time changed' },
  { key: 'description_policy', label: "Description policy enacted" },
];
let firedInterrupts = new Set();
const interruptMessages = new Map();
const interruptFiredSteps = new Map();
let interruptHistory = [];

function addEvent(html, cls) {
  const div = document.createElement('div');
  div.className = 'event ' + cls;
  div.innerHTML = html;
  feed.appendChild(div);
  feed.scrollTop = feed.scrollHeight;
}

function weightClass(w) {
  if (w >= 1.5) return 'hard';
  if (w >= 1.0) return 'medium';
  return 'easy';
}

function updateReward(r, flags) {
  const pct = Math.round(r * 100);
  document.getElementById('reward-fill').style.width = pct + '%';
  document.getElementById('reward-pct').textContent = pct + '%';

  const earned = TASKS.filter(t => flags.includes(t.flag)).reduce((s, t) => s + t.weight, 0);
  document.getElementById('reward-weight').textContent = `${earned.toFixed(1)} / ${TOTAL_WEIGHT.toFixed(1)} weight`;
  document.getElementById('reward-flags').textContent = `${flags.length} / ${TASKS.length} tasks`;

  // Breakdown by tier
  const hardDone = TASKS.filter(t => t.weight >= 1.5 && flags.includes(t.flag)).length;
  const hardTotal = TASKS.filter(t => t.weight >= 1.5).length;
  const medDone = TASKS.filter(t => t.weight === 1.0 && flags.includes(t.flag)).length;
  const medTotal = TASKS.filter(t => t.weight === 1.0).length;
  const easyDone = TASKS.filter(t => t.weight < 1.0 && flags.includes(t.flag)).length;
  const easyTotal = TASKS.filter(t => t.weight < 1.0).length;

  document.getElementById('bd-hard').textContent = `${hardDone}/${hardTotal}`;
  document.getElementById('bd-medium').textContent = `${medDone}/${medTotal}`;
  document.getElementById('bd-easy').textContent = `${easyDone}/${easyTotal}`;

  // Track locked one-shot flags
  TASKS.filter(t => t.oneShot && flags.includes(t.flag)).forEach(t => lockedFlags.add(t.flag));
  document.getElementById('stat-locked').textContent = lockedFlags.size;
}

function updateFlags(flags) {
  const container = document.getElementById('flags');

  // Track one-shot locks
  TASKS.filter(t => t.oneShot && flags.includes(t.flag)).forEach(t => lockedFlags.add(t.flag));

  const groups = [
    { label: 'Hard / Reactive', cls: 'hard', tasks: TASKS.filter(t => t.weight >= 1.5) },
    { label: 'Medium', cls: 'medium', tasks: TASKS.filter(t => t.weight === 1.0) },
    { label: 'Easy', cls: 'easy', tasks: TASKS.filter(t => t.weight < 1.0) },
  ];

  container.innerHTML = groups.map(g => {
    const done = g.tasks.filter(t => flags.includes(t.flag)).length;
    const header = `<div class="task-group-header ${g.cls}"><span>${g.label}</span><span class="task-group-count">${done}/${g.tasks.length}</span></div>`;
    const items = g.tasks.map(t => {
      const isDone = flags.includes(t.flag);
      const isLocked = lockedFlags.has(t.flag);
      const cls = isLocked ? 'locked' : (isDone ? 'done' : 'todo');
      const icon = isLocked ? '&#x1f512;' : (isDone ? '&#x25CF;' : '&#x25CB;');
      const wCls = 'w-' + weightClass(t.weight);
      const badges = [
        `<span class="weight-badge ${wCls}">${t.weight}x</span>`,
        t.oneShot ? `<span class="one-shot-badge">1-SHOT</span>` : '',
        isLocked ? `<span class="lock-badge">LOCKED</span>` : '',
      ].filter(Boolean).join('');
      return `<div class="task-item ${cls}">` +
        `<span class="flag-icon">${icon}</span>` +
        `<div class="task-content"><div class="task-name"><strong>${t.flag}</strong>${badges}</div>` +
        `<div class="task-desc">${t.desc}</div></div></div>`;
    }).join('');
    return header + items;
  }).join('');
}

function setEpisodeContext(context) {
  document.getElementById('ctx-seed').textContent = context.seed ?? '-';
  document.getElementById('ctx-date').textContent = context.episode_today ?? '-';
  document.getElementById('ctx-episode-id').textContent = context.episode_id ?? '-';
  episodeToday = context.episode_today || null;
}

function escapeHtml(str) {
  return String(str).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#39;');
}

function renderInterrupts(currentStep = 0) {
  const container = document.getElementById('interrupts');
  container.innerHTML = INTERRUPTS.map(item => {
    const fired = firedInterrupts.has(item.key);
    const firedStep = interruptFiredSteps.get(item.key);
    const cls = fired ? 'fired' : 'pending';
    const hoverText = interruptMessages.get(item.key) || '';
    const hoverAttr = hoverText ? ` title="${escapeHtml(hoverText)}"` : '';
    const stepLabel = fired ? `Step ${firedStep}` : 'Pending';
    return `<div class="interrupt-item ${cls}"><span class="interrupt-step">${stepLabel}</span><span class="interrupt-label"${hoverAttr}>${item.label}</span></div>`;
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
  log.innerHTML = interruptHistory.map(item =>
    `<div class="interrupt-log-item"><div class="interrupt-log-header">Step ${item.step} - ${escapeHtml(item.label)}</div><pre>${escapeHtml(item.message)}</pre></div>`
  ).join('');
}

function connect() {
  const ws = new WebSocket('ws://localhost:8001/ws/feed');

  ws.onopen = () => { statusEl.textContent = 'connected'; statusEl.className = 'connected'; };
  ws.onclose = () => { statusEl.textContent = 'disconnected'; statusEl.className = 'disconnected'; setTimeout(connect, 2000); };

  ws.onmessage = (e) => {
    const data = JSON.parse(e.data);

    switch (data.type) {
      case 'reset':
        feed.innerHTML = '';
        firedInterrupts = new Set();
        interruptMessages.clear();
        interruptFiredSteps.clear();
        interruptHistory = [];
        lockedFlags.clear();
        resetInfoTracking();
        const taskListHtml = TASKS.map((t, i) => `  ${i+1}. [${t.weight}x] ${t.desc}${t.oneShot ? ' (1-shot)' : ''}`).join('\\n');
        addEvent(
          `<strong>Environment Reset</strong><pre>${data.observation.output}</pre><br><strong>Tasks (weighted):</strong><pre>${taskListHtml}</pre>`,
          'reset'
        );
        document.getElementById('stat-events').textContent = data.observation.events_today;
        document.getElementById('stat-pending').textContent = data.observation.pending_tasks;
        document.getElementById('stat-step').textContent = '0';
        document.getElementById('stat-locked').textContent = '0';
        updateReward(0, []);
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
        trackToolCall(data.tool, data.args || {});
        addEvent(
          `<span class="step-badge">Step ${data.step}</span> <span class="tool-name">${data.tool}</span><pre>${JSON.stringify(data.args, null, 2)}</pre>`,
          'tool-call'
        );
        break;

      case 'observation':
        addEvent(
          `<span class="step-badge">Step ${data.step}</span> <span class="tool-name">${data.tool}</span><pre>${data.output}</pre>`,
          'observation'
        );
        document.getElementById('stat-events').textContent = data.events_today;
        document.getElementById('stat-pending').textContent = data.pending_tasks;
        updateReward(data.reward, data.flags_found || []);
        updateFlags(data.flags_found || []);
        renderInterrupts(data.step);
        if (data.calendar_snapshot && data.calendar_snapshot.length) {
          renderCalendar(data.calendar_snapshot);
        }
        break;

      case 'interrupt':
        firedInterrupts.add(data.interrupt_key);
        interruptMessages.set(data.interrupt_key, data.message);
        interruptFiredSteps.set(data.interrupt_key, data.step);
        const meta = INTERRUPTS.find(x => x.key === data.interrupt_key);
        interruptHistory.push({ step: data.step, key: data.interrupt_key, label: meta ? meta.label : data.interrupt_key, message: data.message });
        addEvent(`<span class="step-badge">Step ${data.step}</span> <strong>INTERRUPT</strong><pre>${data.message}</pre>`, 'interrupt');
        renderInterrupts(data.step);
        renderInterruptHistory();
        break;

      case 'llm_text':
        addEvent(`<span class="step-badge">Step ${data.step}</span> LLM<pre>${data.content}</pre>`, 'llm-text');
        break;

      case 'complete':
        addEvent(`ALL TASKS COMPLETED — Final reward: ${(data.reward * 100).toFixed(1)}%`, 'complete');
        if (data.calendar) addEvent(`<strong>Final Calendar</strong><pre>${data.calendar}</pre>`, 'calendar');
        break;

      case 'max_steps_reached':
        addEvent('Max steps reached.', 'error');
        if (data.calendar) addEvent(`<strong>Final Calendar</strong><pre>${data.calendar}</pre>`, 'calendar');
        break;
    }
  };
}

// --- Drag handle for split panes ---
(function() {
  const handle = document.getElementById('drag-handle');
  const calPane = document.getElementById('calendar-pane');
  const leftPanel = document.querySelector('.left-panel');
  let dragging = false;

  handle.addEventListener('mousedown', (e) => {
    dragging = true;
    handle.classList.add('dragging');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    e.preventDefault();
  });

  document.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const rect = leftPanel.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const totalW = rect.width;
    const calW = Math.max(200, Math.min(totalW - 200, totalW - x));
    calPane.style.width = calW + 'px';
    calPane.style.flex = 'none';
  });

  document.addEventListener('mouseup', () => {
    if (dragging) {
      dragging = false;
      handle.classList.remove('dragging');
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    }
  });
})();

// --- Calendar rendering ---
let currentCalendarSnapshot = [];
let episodeToday = null;

const HOUR_HEIGHT = 60; // px per hour
const START_HOUR = 8;
const END_HOUR = 18;
const TOTAL_HOURS = END_HOUR - START_HOUR;

function timeToMinutes(t) {
  const [h, m] = t.split(':').map(Number);
  return h * 60 + m;
}

function renderCalendar(snapshot) {
  currentCalendarSnapshot = snapshot;
  const container = document.getElementById('calendar-view');
  if (!snapshot || !snapshot.length) {
    container.innerHTML = '<div class="cal-no-events">No events to display</div>';
    return;
  }

  // Group events by date
  const byDate = {};
  snapshot.forEach(e => {
    if (!byDate[e.date]) byDate[e.date] = [];
    byDate[e.date].push(e);
  });

  // Sort dates
  const dates = Object.keys(byDate).sort();
  const todayStr = episodeToday || '';

  container.innerHTML = dates.map(d => {
    const isToday = d === todayStr;
    const dayName = new Date(d + 'T12:00:00').toLocaleDateString('en-US', { weekday: 'short' });
    const dateShort = d.slice(5); // MM-DD

    // Hour lines
    let hourLines = '';
    for (let h = START_HOUR; h <= END_HOUR; h++) {
      const top = (h - START_HOUR) * HOUR_HEIGHT;
      hourLines += `<div class="cal-hour-line" style="top:${top}px">${String(h).padStart(2,'0')}:00</div>`;
    }

    // Events
    const events = byDate[d].map(e => {
      const startMin = timeToMinutes(e.start_time || '08:00');
      const endMin = timeToMinutes(e.end_time || e.start_time || '09:00');
      const top = ((startMin - START_HOUR * 60) / 60) * HOUR_HEIGHT;
      const height = Math.max(((endMin - startMin) / 60) * HOUR_HEIGHT, 18);
      const cls = (e.type === 'personal') ? 'personal' : 'work';
      const attendees = (e.attendees || []).join(', ');
      const timeStr = `${e.start_time}-${e.end_time}`;
      return `<div class="cal-event ${cls}" style="top:${top}px;height:${height}px" title="${escapeHtml(e.title)}\\n${timeStr}\\n${attendees}">` +
        `<div class="cal-event-title">${escapeHtml(e.title)}</div>` +
        (height > 28 ? `<div class="cal-event-time">${timeStr}</div>` : '') +
        (height > 42 && attendees ? `<div class="cal-event-attendees">${escapeHtml(attendees)}</div>` : '') +
        `</div>`;
    }).join('');

    return `<div class="cal-day">` +
      `<div class="cal-day-header ${isToday ? 'today' : ''}">${dayName} ${dateShort}${isToday ? ' (today)' : ''}</div>` +
      `<div class="cal-timeline" style="height:${TOTAL_HOURS * HOUR_HEIGHT}px">` +
      hourLines + events +
      `</div></div>`;
  }).join('');
}

renderInterrupts(0);
renderInterruptHistory();
updateFlags([]);
renderInfoAccessed();
connect();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    asyncio.run(main())
