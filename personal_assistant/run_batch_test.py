"""Headless batch test — run the agent on multiple seeds and print results."""

import asyncio
import json
import os
import random
from datetime import date, timedelta

from dotenv import load_dotenv
from openai import OpenAI
import websockets

load_dotenv()

WS_URL = "ws://localhost:8000/ws"
MODEL = "llama-3.3-70b-versatile"
EPISODE_BASE_DATE = date(2026, 1, 5)
EPISODE_WEEKS = 3
MAX_STEPS = 60

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
9. When creating meetings, attendees may decline with feedback. Read their response carefully and adjust your next attempt (different duration, time, or format) to address their specific concern. Do not just retry the same parameters."""

TOOLS = [
    {"type": "function", "function": {"name": "list_events", "description": "List all calendar events for a given date.", "parameters": {"type": "object", "properties": {"date": {"type": "string", "description": "Date string: 'today', 'tomorrow', 'next monday', or YYYY-MM-DD"}}, "required": []}}},
    {"type": "function", "function": {"name": "create_event", "description": "Create a calendar event.", "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "date": {"type": "string", "description": "'today', 'tomorrow', 'next monday', or YYYY-MM-DD"}, "start_time": {"type": "string", "description": "HH:MM format"}, "duration_minutes": {"type": "integer", "default": 60}, "attendees": {"type": "string", "description": "Comma-separated names"}}, "required": ["title", "date", "start_time"]}}},
    {"type": "function", "function": {"name": "delete_event", "description": "Delete a calendar event by title.", "parameters": {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}}},
    {"type": "function", "function": {"name": "edit_event", "description": "Edit an existing event. Only provided fields are changed.", "parameters": {"type": "object", "properties": {"title": {"type": "string", "description": "Current title of the event to edit"}, "new_title": {"type": "string"}, "new_date": {"type": "string"}, "new_start_time": {"type": "string", "description": "HH:MM"}, "new_duration_minutes": {"type": "integer"}, "new_attendees": {"type": "string", "description": "Comma-separated, replaces all"}}, "required": ["title"]}}},
    {"type": "function", "function": {"name": "find_free_slots", "description": "Find available time slots on a given date (8:00-18:00).", "parameters": {"type": "object", "properties": {"date": {"type": "string", "default": "today"}, "duration_minutes": {"type": "integer", "default": 60}}, "required": []}}},
    {"type": "function", "function": {"name": "check_conflicts", "description": "Check for scheduling conflicts on a date.", "parameters": {"type": "object", "properties": {"date": {"type": "string", "default": "today"}}, "required": []}}},
    {"type": "function", "function": {"name": "resolve_conflict", "description": "Resolve a conflict by moving an event to a new time.", "parameters": {"type": "object", "properties": {"event_title": {"type": "string"}, "new_start_time": {"type": "string", "description": "HH:MM format"}}, "required": ["event_title", "new_start_time"]}}},
    {"type": "function", "function": {"name": "send_notification", "description": "Send a notification to a person.", "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "message": {"type": "string"}}, "required": ["to", "message"]}}},
    {"type": "function", "function": {"name": "get_task_list", "description": "Get the list of tasks to complete.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "check_availability", "description": "Check a person's availability on a given date.", "parameters": {"type": "object", "properties": {"person": {"type": "string"}, "date": {"type": "string", "default": "today"}}, "required": ["person"]}}},
    {"type": "function", "function": {"name": "get_constraints", "description": "Get scheduling constraints (hard and soft) that apply to the calendar. Note: individual people may have additional private constraints — use get_contact_preferences to discover them.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "get_contact_preferences", "description": "Get a person's scheduling preferences, private constraints, role, and preferred notification method. Some constraints are only visible through this tool.", "parameters": {"type": "object", "properties": {"person": {"type": "string", "description": "Person's name (e.g. Alice, Bob, Charlie, Dave, Eve)"}}, "required": ["person"]}}},
    {"type": "function", "function": {"name": "check_constraint_violations", "description": "Check the current calendar for all constraint violations.", "parameters": {"type": "object", "properties": {}, "required": []}}},
]


def _seed_to_episode_today(seed: int) -> str:
    rng = random.Random(seed)
    weekdays = []
    for week in range(EPISODE_WEEKS):
        for day in range(5):
            weekdays.append(EPISODE_BASE_DATE + timedelta(weeks=week, days=day))
    return rng.choice(weekdays).isoformat()


async def run_seed(client: OpenAI, seed: int) -> dict:
    episode_today = _seed_to_episode_today(seed)
    print(f"\n{'='*60}")
    print(f"SEED {seed} | Episode date: {episode_today}")
    print(f"{'='*60}")

    async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=120) as ws:
        # Reset
        await ws.send(json.dumps({"type": "reset", "data": {"seed": seed}}))
        reset_resp = json.loads(await ws.recv())
        obs = reset_resp["data"]["observation"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": obs["output"]},
        ]

        final_reward = 0
        final_flags = []
        steps_used = 0

        for step in range(MAX_STEPS):
            steps_used = step + 1
            try:
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto",
                )
            except Exception as e:
                print(f"  Step {step+1}: API error: {e}")
                break

            msg = response.choices[0].message

            if msg.tool_calls:
                messages.append(msg)
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    print(f"  Step {step+1}: {tool_name}({json.dumps(args, separators=(',',':'))})")

                    action = {"type": "step", "data": {"instruction": json.dumps({"tool": tool_name, "args": args})}}
                    await ws.send(json.dumps(action))
                    step_resp = json.loads(await ws.recv())

                    data = step_resp["data"]
                    obs = data["observation"]
                    final_reward = data.get("reward", 0)
                    final_flags = obs.get("flags_found", [])

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": obs["output"],
                    })

                    if data.get("done", False):
                        print(f"  >>> DONE at step {step+1}")
                        break

                if data.get("done", False):
                    break
            else:
                content = msg.content or ""
                print(f"  Step {step+1}: [text] {content[:80]}...")
                messages.append(msg)
                messages.append({"role": "user", "content": "Continue completing the remaining tasks. Use the tools available to you."})

        # Fetch final calendar
        calendar_lines = []
        today_dt = date.fromisoformat(episode_today)
        days_since_mon = today_dt.weekday()
        week_start = today_dt - timedelta(days=days_since_mon)
        for week in range(2):
            for day in range(5):
                d = (week_start + timedelta(weeks=week, days=day)).isoformat()
                action = {"type": "step", "data": {"instruction": json.dumps({"tool": "list_events", "args": {"date": d}})}}
                await ws.send(json.dumps(action))
                resp = json.loads(await ws.recv())
                output = resp["data"]["observation"]["output"]
                if "No events" not in output:
                    calendar_lines.append(output)

        calendar = "\n".join(calendar_lines) if calendar_lines else "No events."

    print(f"\n  REWARD: {final_reward:.2f} ({len(final_flags)}/11)")
    print(f"  FLAGS: {sorted(final_flags)}")
    print(f"  STEPS: {steps_used}")
    print(f"  CALENDAR:\n{calendar}")

    return {"seed": seed, "date": episode_today, "reward": final_reward, "flags": sorted(final_flags), "steps": steps_used}


async def main():
    seeds = [1, 3, 5, 9, 12]
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
    )

    results = []
    for seed in seeds:
        try:
            result = await run_seed(client, seed)
            results.append(result)
        except Exception as e:
            print(f"\nSEED {seed} FAILED: {e}")
            results.append({"seed": seed, "reward": 0, "flags": [], "error": str(e)})

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        flags_count = len(r.get("flags", []))
        print(f"  Seed {r['seed']:3d} | {r.get('date','?'):>10s} | Reward: {r.get('reward',0):.2f} ({flags_count}/11) | Steps: {r.get('steps','?')}")

    rewards = [r.get("reward", 0) for r in results]
    print(f"\n  Average reward: {sum(rewards)/len(rewards):.2f}")
    print(f"  Min: {min(rewards):.2f} | Max: {max(rewards):.2f}")


if __name__ == "__main__":
    asyncio.run(main())
