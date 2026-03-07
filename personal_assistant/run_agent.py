"""Step 6: LLM agent that solves calendar tasks via WebSocket."""

import asyncio
import json
import os

from dotenv import load_dotenv
from openai import OpenAI
import requests
import websockets

load_dotenv()

SERVER_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws"
MODEL = "Qwen/Qwen2.5-7B-Instruct"


def print_separator():
    print("─" * 70)


def print_full_state():
    """Fetch and print the full environment state from REST."""
    try:
        state = requests.get(f"{SERVER_URL}/state").json()
        print("  ┌── ENV STATE ──────────────────────────────────────")
        for k, v in state.items():
            print(f"  │ {k}: {v}")
        print("  └──────────────────────────────────────────────────")
    except Exception as e:
        print(f"  (could not fetch state: {e})")

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


async def run_agent():
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.getenv("HF_TOKEN"),
    )

    async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=120) as ws:
        # Reset environment
        await ws.send(json.dumps({"type": "reset"}))
        reset_resp = json.loads(await ws.recv())
        obs = reset_resp["data"]["observation"]
        print_separator()
        print("🔄 RESET")
        print_separator()
        print(f"  Output: {obs['output']}")
        print(f"  Events today: {obs['events_today']}")
        print(f"  Pending tasks: {obs['pending_tasks']}")
        print_full_state()
        print()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": obs["output"]},
        ]

        for step in range(30):
            print_separator()
            print(f"📍 STEP {step + 1}")
            print_separator()

            # Ask LLM for next action
            print("  ⏳ Thinking...")
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )

            msg = response.choices[0].message

            # If LLM wants to call tools
            if msg.tool_calls:
                messages.append(msg)
                print(f"  🔧 LLM chose {len(msg.tool_calls)} tool call(s):")

                for i, tool_call in enumerate(msg.tool_calls):
                    tool_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    print()
                    print(f"  ── Tool Call {i+1}: {tool_name} ──")
                    print(f"  Args: {json.dumps(args, indent=4)}")

                    # Send to environment via WebSocket
                    action = {"type": "step", "data": {"instruction": json.dumps({"tool": tool_name, "args": args})}}
                    await ws.send(json.dumps(action))
                    step_resp = json.loads(await ws.recv())

                    data = step_resp["data"]
                    obs = data["observation"]
                    reward = data.get("reward", 0)
                    done = data.get("done", False)

                    print(f"  ┌── OBSERVATION ──")
                    print(f"  │ Output: {obs['output']}")
                    print(f"  │ Events today: {obs.get('events_today', '?')}")
                    print(f"  │ Pending tasks: {obs.get('pending_tasks', '?')}")
                    print(f"  │ Flags: {obs.get('flags_found', [])}")
                    print(f"  │ Reward: {reward}")
                    print(f"  │ Done: {done}")
                    print(f"  └──────────────────")

                    print_full_state()

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": obs["output"],
                    })

                    if done:
                        print()
                        print_separator()
                        print("🎉 ALL TASKS COMPLETED!")
                        print_separator()
                        return

            else:
                # LLM responded with text (no tool call)
                print(f"  💬 LLM says: {msg.content}")
                messages.append(msg)
                # Prompt it to keep going
                messages.append({"role": "user", "content": "Continue completing the remaining tasks. Use the tools available to you."})

            print()

        print_separator()
        print("⚠️  Reached max steps without completing all tasks.")
        print_separator()


if __name__ == "__main__":
    asyncio.run(run_agent())
