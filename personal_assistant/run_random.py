"""Step 5: Random interaction script to test the environment."""

import json
import random
import requests

BASE = "http://localhost:8000"

RANDOM_ACTIONS = [
    {"tool": "get_task_list", "args": {}},
    {"tool": "list_events", "args": {"date": "today"}},
    {"tool": "list_events", "args": {"date": "tomorrow"}},
    {"tool": "find_free_slots", "args": {"date": "today", "duration_minutes": 60}},
    {"tool": "check_conflicts", "args": {"date": "today"}},
    {"tool": "create_event", "args": {"title": "Team Standup", "date": "tomorrow", "start_time": "09:00", "duration_minutes": 30, "attendees": "Alice,Bob"}},
    {"tool": "create_event", "args": {"title": "Focus Time", "date": "today", "start_time": "16:00", "duration_minutes": 60}},
    {"tool": "create_event", "args": {"title": "Dentist Appointment", "date": "next monday", "start_time": "14:00", "duration_minutes": 60}},
    {"tool": "resolve_conflict", "args": {"event_title": "Design Review", "new_start_time": "15:30"}},
    {"tool": "delete_event", "args": {"title": "Old Project Review"}},
    {"tool": "send_notification", "args": {"to": "Alice", "message": "Old Project Review has been cancelled"}},
    {"tool": "send_notification", "args": {"to": "Bob", "message": "Old Project Review has been cancelled"}},
    {"tool": "send_notification", "args": {"to": "Charlie", "message": "Old Project Review has been cancelled"}},
]


def main():
    # Reset
    r = requests.post(f"{BASE}/reset").json()
    obs = r["observation"]
    print("=== RESET ===")
    print(f"Output: {obs['output']}")
    print(f"Events today: {obs['events_today']}")
    print(f"Pending tasks: {obs['pending_tasks']}")
    print()

    # Random interactions
    for i in range(12):
        action_data = random.choice(RANDOM_ACTIONS)
        payload = {"action": {"instruction": json.dumps(action_data)}}

        print(f"=== Step {i+1}: {action_data['tool']} ===")
        r = requests.post(f"{BASE}/step", json=payload).json()
        obs = r["observation"]
        print(f"Output: {obs['output']}")
        print(f"Reward: {r['reward']}")
        print(f"Done: {r['done']}")
        print(f"Flags: {obs['flags_found']}")
        print(f"Pending: {obs['pending_tasks']}")
        print()

        if r["done"]:
            print("ALL TASKS COMPLETED!")
            break

    # Final state
    state = requests.get(f"{BASE}/state").json()
    print("=== Final State ===")
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
