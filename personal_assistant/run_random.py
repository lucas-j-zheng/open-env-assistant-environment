"""Step 5: Random interaction script to test the environment."""

import asyncio
import json
import random

import websockets

WS_URL = "ws://localhost:8000/ws"

RANDOM_ACTIONS = [
    {"tool": "get_task_list", "args": {}},
    {"tool": "list_events", "args": {"date": "today"}},
    {"tool": "list_events", "args": {"date": "tomorrow"}},
    {"tool": "find_free_slots", "args": {"date": "today", "duration_minutes": 60}},
    {"tool": "check_conflicts", "args": {"date": "today"}},
    {
        "tool": "create_event",
        "args": {
            "title": "Team Standup",
            "date": "tomorrow",
            "start_time": "09:00",
            "duration_minutes": 30,
            "attendees": "Alice,Bob",
        },
    },
    {
        "tool": "create_event",
        "args": {
            "title": "Focus Time",
            "date": "today",
            "start_time": "16:00",
            "duration_minutes": 60,
        },
    },
    {
        "tool": "create_event",
        "args": {
            "title": "Dentist Appointment",
            "date": "next monday",
            "start_time": "14:00",
            "duration_minutes": 60,
        },
    },
    {
        "tool": "resolve_conflict",
        "args": {"event_title": "Design Review", "new_start_time": "15:30"},
    },
    {"tool": "delete_event", "args": {"title": "Old Project Review"}},
    {
        "tool": "send_notification",
        "args": {"to": "Alice", "message": "Old Project Review has been cancelled"},
    },
    {
        "tool": "send_notification",
        "args": {"to": "Bob", "message": "Old Project Review has been cancelled"},
    },
    {
        "tool": "send_notification",
        "args": {"to": "Charlie", "message": "Old Project Review has been cancelled"},
    },
]


def _parse_ws_response(raw_message: str) -> dict:
    """Unwrap OpenEnv WebSocket response payloads and surface errors."""
    message = json.loads(raw_message)
    if message.get("type") == "error":
        error = message.get("data", {}).get("message", "Unknown WebSocket error")
        raise RuntimeError(error)
    return message["data"]


async def main() -> None:
    async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=120) as ws:
        # Reset once per session so subsequent steps share state.
        await ws.send(json.dumps({"type": "reset", "data": {}}))
        reset_data = _parse_ws_response(await ws.recv())
        obs = reset_data["observation"]
        print("=== RESET ===")
        print(f"Output: {obs['output']}")
        print(f"Events today: {obs['events_today']}")
        print(f"Pending tasks: {obs['pending_tasks']}")
        print()

        for i in range(12):
            action_data = random.choice(RANDOM_ACTIONS)
            print(f"=== Step {i+1}: {action_data['tool']} ===")

            await ws.send(
                json.dumps(
                    {
                        "type": "step",
                        "data": {"instruction": json.dumps(action_data)},
                    }
                )
            )
            step_data = _parse_ws_response(await ws.recv())
            obs = step_data["observation"]

            print(f"Output: {obs['output']}")
            print(f"Reward: {step_data['reward']}")
            print(f"Done: {step_data['done']}")
            print(f"Flags: {obs['flags_found']}")
            print(f"Pending: {obs['pending_tasks']}")
            print()

            if step_data["done"]:
                print("ALL TASKS COMPLETED!")
                break

        await ws.send(json.dumps({"type": "state"}))
        state_data = _parse_ws_response(await ws.recv())
        print("=== Final State ===")
        print(json.dumps(state_data, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
