"""Step 6b: LLM agent with live web dashboard."""

import asyncio
import json
import os
import threading
import time
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from openai import OpenAI
import uvicorn
import websockets

load_dotenv()

SERVER_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws"
MODEL = "Qwen/Qwen2.5-72B-Instruct"

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
]

SYSTEM_PROMPT = """You are a calendar personal assistant. You have access to tools to manage a calendar.

Your goal is to complete ALL tasks on the task list. Start by calling get_task_list to see what needs to be done, then use the available tools to complete each task.

Think step by step about what tools to call and in what order. After completing actions, check the task list again to verify progress."""


# --- Agent Logic ---

async def _fetch_final_calendar(ws):
    """Fetch today's and tomorrow's calendar after agent finishes."""
    results = []
    for date in ["today", "tomorrow", "next monday"]:
        action = {"type": "step", "data": {"instruction": json.dumps({"tool": "list_events", "args": {"date": date}})}}
        await ws.send(json.dumps(action))
        resp = json.loads(await ws.recv())
        output = resp["data"]["observation"]["output"]
        results.append(output)
    return "\n\n".join(results)


async def run_agent():
    # Wait for dashboard to be ready
    await asyncio.sleep(2)

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.getenv("HF_TOKEN"),
    )

    async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=120) as ws:
        # Reset
        await ws.send(json.dumps({"type": "reset"}))
        reset_resp = json.loads(await ws.recv())
        obs = reset_resp["data"]["observation"]

        await broadcast({
            "type": "reset",
            "observation": obs,
            "state": {"step_count": 0},
        })

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": obs["output"]},
        ]

        for step in range(30):
            await broadcast({"type": "thinking", "step": step + 1})

            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )

            msg = response.choices[0].message

            if msg.tool_calls:
                messages.append(msg)

                for tool_call in msg.tool_calls:
                    tool_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

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

                    messages.append({
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
                messages.append(msg)
                messages.append({"role": "user", "content": "Continue completing the remaining tasks. Use the tools available to you."})

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
    <div class="panel-header">Tasks</div>
    <div class="flags-section" id="flags">
      <div class="task-item todo"><span class="flag-icon">○</span><div><strong>standup_scheduled</strong><div class="task-desc">Schedule standup tomorrow 9AM, 30min, Alice & Bob</div></div></div>
      <div class="task-item todo"><span class="flag-icon">○</span><div><strong>focus_time_booked</strong><div class="task-desc">Find free 1hr slot this afternoon, book focus time</div></div></div>
      <div class="task-item todo"><span class="flag-icon">○</span><div><strong>conflicts_resolved</strong><div class="task-desc">Check conflicts today and resolve them</div></div></div>
      <div class="task-item todo"><span class="flag-icon">○</span><div><strong>reminder_set</strong><div class="task-desc">Set reminder for dentist next Monday 2PM</div></div></div>
      <div class="task-item todo"><span class="flag-icon">○</span><div><strong>meeting_cancelled</strong><div class="task-desc">Cancel 'Old Project Review' and notify attendees</div></div></div>
    </div>
  </div>
</div>

<script>
const feed = document.getElementById('feed');
const statusEl = document.getElementById('status');

const TASKS = [
  { flag: 'standup_scheduled', desc: 'Schedule standup tomorrow 9AM, 30min, Alice & Bob' },
  { flag: 'focus_time_booked', desc: 'Find free 1hr slot this afternoon, book focus time' },
  { flag: 'conflicts_resolved', desc: 'Check conflicts today and resolve them' },
  { flag: 'reminder_set', desc: 'Set reminder for dentist next Monday 2PM' },
  { flag: 'meeting_cancelled', desc: "Cancel 'Old Project Review' and notify attendees" },
];

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
        break;

      case 'thinking':
        addEvent(`<span class="step-badge">Step ${data.step}</span> Thinking...`, 'thinking');
        document.getElementById('stat-step').textContent = data.step;
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

connect();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    asyncio.run(main())
