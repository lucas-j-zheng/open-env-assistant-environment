"""Step 6: LLM agent that solves calendar tasks via WebSocket."""

import asyncio
import json
import os

from dotenv import load_dotenv
from openai import OpenAI
import websockets

load_dotenv()

WS_URL = "ws://localhost:8000/ws"
MODEL = "Qwen/Qwen2.5-7B-Instruct"


def print_separator():
    print("─" * 70)


async def print_full_state(ws):
    """Fetch and print the full environment state via WebSocket."""
    try:
        await ws.send(json.dumps({"type": "state"}))
        resp = json.loads(await ws.recv())
        state = resp.get("data", resp)
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
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "Check a person's availability on a given date. Shows their busy times from external commitments and free windows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "person": {"type": "string", "description": "Person's name (e.g. Alice, Bob, Charlie, Dave, Eve)"},
                    "date": {"type": "string", "description": "Date: 'today', 'tomorrow', day name, or YYYY-MM-DD", "default": "today"},
                },
                "required": ["person"],
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
                    "new_attendees": {"type": "string", "description": "New comma-separated attendees list (optional)"},
                },
                "required": ["title"],
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
            "name": "get_constraints",
            "description": "Get scheduling constraints (hard and soft) that apply to the calendar.",
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
1. First call get_task_list and get_constraints to understand the rules.
2. BEFORE scheduling any meeting with someone, call get_contact_preferences(person) to discover their private constraints and preferences.
3. Use check_availability before scheduling — don't guess times.
4. When scheduling, respect HARD constraints (must obey) and SOFT constraints (preferences).
5. After making changes, call check_constraint_violations to catch any violations.
6. Periodically call get_task_list to check which tasks are still TODO.
7. Think step by step about what tools to call and in what order."""


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
        await print_full_state(ws)
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

                    await print_full_state(ws)

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
