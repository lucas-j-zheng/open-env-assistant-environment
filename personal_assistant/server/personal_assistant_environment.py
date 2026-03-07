# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Personal Assistant Calendar Environment.

An environment where an agent manages a calendar: creating events,
checking schedules, resolving conflicts, and completing user requests.
The agent gets rewarded for successfully completing calendar tasks.
"""

import json
from datetime import datetime, timedelta
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import CalendarAction, CalendarObservation, CalendarState


class PersonalAssistantEnvironment(Environment):
    """
    Calendar personal assistant environment.

    The agent sends natural language instructions. The environment parses
    them into tool calls internally via step(). MCP tools are also exposed
    for direct tool-call interaction.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    TASKS = [
        {
            "description": "Schedule a team standup meeting tomorrow at 9:00 AM for 30 minutes with Alice and Bob",
            "flag": "standup_scheduled",
        },
        {
            "description": "Find a free 1-hour slot this afternoon and book a focus time block",
            "flag": "focus_time_booked",
        },
        {
            "description": "Check if there are any scheduling conflicts today and resolve them",
            "flag": "conflicts_resolved",
        },
        {
            "description": "Set a reminder for the dentist appointment next Monday at 2:00 PM",
            "flag": "reminder_set",
        },
        {
            "description": "Cancel the meeting titled 'Old Project Review' and notify attendees",
            "flag": "meeting_cancelled",
        },
    ]

    def __init__(self):
        self._events: list[dict] = []
        self._notifications: list[str] = []
        self._found: set[str] = set()
        self._conflicts_resolved = False
        self._old_meeting_cancelled = False
        self._state = CalendarState(
            episode_id=str(uuid4()),
            step_count=0,
            total_tasks=len(self.TASKS),
        )

    # --- Tool implementations ---

    def _resolve_date(self, date_str: str) -> str:
        today = datetime.now()
        if date_str == "today":
            return today.strftime("%Y-%m-%d")
        elif date_str == "tomorrow":
            return (today + timedelta(days=1)).strftime("%Y-%m-%d")
        elif date_str == "next monday":
            days_ahead = 7 - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        return date_str

    def tool_list_events(self, date: str = "today") -> str:
        """List all calendar events for a given date."""
        target = self._resolve_date(date)
        day_events = [e for e in self._events if e["date"] == target]
        if not day_events:
            return f"No events scheduled for {target}."
        lines = [f"Events for {target}:"]
        for e in sorted(day_events, key=lambda x: x["start_time"]):
            attendees = ", ".join(e.get("attendees", []))
            lines.append(f"  - {e['title']} | {e['start_time']}-{e['end_time']} | Attendees: {attendees}")
        return "\n".join(lines)

    def tool_create_event(self, title: str, date: str, start_time: str,
                          duration_minutes: int = 60, attendees: str = "") -> str:
        """Create a calendar event."""
        target = self._resolve_date(date)
        start_dt = datetime.strptime(f"{target} {start_time}", "%Y-%m-%d %H:%M")
        end_dt = start_dt + timedelta(minutes=duration_minutes)
        end_time = end_dt.strftime("%H:%M")

        event = {
            "id": str(uuid4())[:8],
            "title": title,
            "date": target,
            "start_time": start_time,
            "end_time": end_time,
            "duration_minutes": duration_minutes,
            "attendees": [a.strip() for a in attendees.split(",") if a.strip()],
        }
        self._events.append(event)
        self._check_completions()
        return f"Event '{title}' created on {target} from {start_time} to {end_time}."

    def tool_delete_event(self, title: str) -> str:
        """Delete a calendar event by title."""
        matches = [e for e in self._events if e["title"].lower() == title.lower()]
        if not matches:
            return f"No event found with title '{title}'."
        for m in matches:
            self._events.remove(m)
        if "old project review" in title.lower():
            self._old_meeting_cancelled = True
            self._check_completions()
        return f"Deleted event '{title}'."

    def tool_find_free_slots(self, date: str = "today", duration_minutes: int = 60) -> str:
        """Find available time slots on a given date (8:00-18:00)."""
        target = self._resolve_date(date)
        day_events = sorted(
            [e for e in self._events if e["date"] == target],
            key=lambda x: x["start_time"],
        )
        slots = []
        current = datetime.strptime(f"{target} 08:00", "%Y-%m-%d %H:%M")
        end_of_day = datetime.strptime(f"{target} 18:00", "%Y-%m-%d %H:%M")

        for event in day_events:
            event_start = datetime.strptime(f"{target} {event['start_time']}", "%Y-%m-%d %H:%M")
            if (event_start - current).total_seconds() / 60 >= duration_minutes:
                slots.append(f"  {current.strftime('%H:%M')} - {event_start.strftime('%H:%M')}")
            event_end = datetime.strptime(f"{target} {event['end_time']}", "%Y-%m-%d %H:%M")
            current = max(current, event_end)

        if (end_of_day - current).total_seconds() / 60 >= duration_minutes:
            slots.append(f"  {current.strftime('%H:%M')} - {end_of_day.strftime('%H:%M')}")

        if not slots:
            return f"No free {duration_minutes}-minute slots available on {target}."
        return f"Free slots on {target} ({duration_minutes}+ min):\n" + "\n".join(slots)

    def tool_check_conflicts(self, date: str = "today") -> str:
        """Check for scheduling conflicts on a date."""
        target = self._resolve_date(date)
        day_events = sorted(
            [e for e in self._events if e["date"] == target],
            key=lambda x: x["start_time"],
        )
        conflicts = []
        for i in range(len(day_events) - 1):
            if day_events[i]["end_time"] > day_events[i + 1]["start_time"]:
                conflicts.append(
                    f"  CONFLICT: '{day_events[i]['title']}' (ends {day_events[i]['end_time']}) "
                    f"overlaps with '{day_events[i+1]['title']}' (starts {day_events[i+1]['start_time']})"
                )
        if not conflicts:
            return f"No conflicts found on {target}."
        return f"Conflicts on {target}:\n" + "\n".join(conflicts)

    def tool_resolve_conflict(self, event_title: str, new_start_time: str) -> str:
        """Resolve a conflict by moving an event to a new time."""
        matches = [e for e in self._events if e["title"].lower() == event_title.lower()]
        if not matches:
            return f"No event found with title '{event_title}'."
        event = matches[0]
        start_dt = datetime.strptime(f"{event['date']} {new_start_time}", "%Y-%m-%d %H:%M")
        end_dt = start_dt + timedelta(minutes=event["duration_minutes"])
        event["start_time"] = new_start_time
        event["end_time"] = end_dt.strftime("%H:%M")
        self._conflicts_resolved = True
        self._check_completions()
        return f"Moved '{event_title}' to {new_start_time}-{event['end_time']}."

    def tool_send_notification(self, to: str, message: str) -> str:
        """Send a notification to a person."""
        self._notifications.append({"to": to, "message": message})
        return f"Notification sent to {to}: {message}"

    def tool_get_task_list(self) -> str:
        """Get the list of tasks to complete."""
        lines = ["Tasks to complete:"]
        for i, t in enumerate(self.TASKS, 1):
            status = "DONE" if t["flag"] in self._found else "TODO"
            lines.append(f"  {i}. [{status}] {t['description']}")
        return "\n".join(lines)

    # --- Tool dispatch ---

    TOOL_MAP = {
        "list_events": "tool_list_events",
        "create_event": "tool_create_event",
        "delete_event": "tool_delete_event",
        "find_free_slots": "tool_find_free_slots",
        "check_conflicts": "tool_check_conflicts",
        "resolve_conflict": "tool_resolve_conflict",
        "send_notification": "tool_send_notification",
        "get_task_list": "tool_get_task_list",
    }

    def _dispatch_tool(self, tool_name: str, arguments: dict) -> str:
        method_name = self.TOOL_MAP.get(tool_name)
        if not method_name:
            available = ", ".join(self.TOOL_MAP.keys())
            return f"Unknown tool '{tool_name}'. Available tools: {available}"
        method = getattr(self, method_name)
        try:
            return method(**arguments)
        except TypeError as e:
            return f"Error calling {tool_name}: {e}"

    # --- Completion checking ---

    def _check_completions(self):
        # standup_scheduled
        if any(
            "standup" in e.get("title", "").lower()
            and "09:00" in e.get("start_time", "")
            for e in self._events
        ):
            self._found.add("standup_scheduled")

        # focus_time_booked
        if any("focus" in e.get("title", "").lower() for e in self._events):
            self._found.add("focus_time_booked")

        # conflicts_resolved
        if self._conflicts_resolved:
            self._found.add("conflicts_resolved")

        # reminder_set
        if any("dentist" in e.get("title", "").lower() for e in self._events):
            self._found.add("reminder_set")

        # meeting_cancelled
        if self._old_meeting_cancelled:
            self._found.add("meeting_cancelled")

    # --- Seed data ---

    def _seed_initial_events(self):
        today = self._resolve_date("today")
        self._events = [
            {
                "id": "seed1",
                "title": "Morning Sync",
                "date": today,
                "start_time": "10:00",
                "end_time": "10:30",
                "duration_minutes": 30,
                "attendees": ["Alice", "Charlie"],
            },
            {
                "id": "seed2",
                "title": "Lunch with Client",
                "date": today,
                "start_time": "12:00",
                "end_time": "13:00",
                "duration_minutes": 60,
                "attendees": ["Dave"],
            },
            {
                "id": "seed3",
                "title": "Old Project Review",
                "date": today,
                "start_time": "14:00",
                "end_time": "15:00",
                "duration_minutes": 60,
                "attendees": ["Alice", "Bob", "Charlie"],
            },
            {
                "id": "seed4",
                "title": "Design Review",
                "date": today,
                "start_time": "14:30",
                "end_time": "15:30",
                "duration_minutes": 60,
                "attendees": ["Eve"],
            },
        ]

    # --- Core Environment API ---

    def reset(self, **kwargs) -> CalendarObservation:
        self._found = set()
        self._conflicts_resolved = False
        self._old_meeting_cancelled = False
        self._notifications = []
        self._state = CalendarState(
            episode_id=str(uuid4()),
            step_count=0,
            total_tasks=len(self.TASKS),
        )
        self._seed_initial_events()

        return CalendarObservation(
            output=(
                "Calendar assistant ready. You have 4 events today including a scheduling conflict. "
                "Use get_task_list to see what needs to be done."
            ),
            pending_tasks=len(self.TASKS),
            events_today=len([e for e in self._events if e["date"] == self._resolve_date("today")]),
            done=False,
            reward=0.0,
        )

    def step(self, action: CalendarAction, **kwargs) -> CalendarObservation:
        self._state.step_count += 1

        # Parse instruction as a tool call: try JSON first, then plain text
        instruction = action.instruction.strip()
        tool_name = None
        arguments = {}

        # Try JSON tool call format: {"tool": "name", "args": {...}}
        try:
            parsed = json.loads(instruction)
            if isinstance(parsed, dict) and "tool" in parsed:
                tool_name = parsed["tool"]
                arguments = parsed.get("args", parsed.get("arguments", {}))
        except (json.JSONDecodeError, KeyError):
            pass

        if tool_name:
            output = self._dispatch_tool(tool_name, arguments)
        else:
            # Plain text - show available tools
            output = (
                f"Received: '{instruction}'\n"
                f"To interact, send a JSON tool call like:\n"
                f'{{"tool": "get_task_list", "args": {{}}}}\n'
                f'{{"tool": "list_events", "args": {{"date": "today"}}}}\n'
                f'{{"tool": "create_event", "args": {{"title": "Meeting", "date": "tomorrow", "start_time": "09:00", "duration_minutes": 30, "attendees": "Alice,Bob"}}}}\n'
                f"Available tools: {', '.join(self.TOOL_MAP.keys())}"
            )

        new_flags = len(self._found)
        reward = new_flags / len(self.TASKS)
        done = new_flags == len(self.TASKS)
        self._state.tasks_completed = new_flags

        return CalendarObservation(
            output=output,
            pending_tasks=len(self.TASKS) - new_flags,
            events_today=len([e for e in self._events if e["date"] == self._resolve_date("today")]),
            flags_found=list(self._found),
            reward=reward,
            done=done,
        )

    @property
    def state(self) -> CalendarState:
        return self._state
