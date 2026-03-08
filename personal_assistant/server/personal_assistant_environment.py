# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Personal Assistant Calendar Environment.

Tasks are intentionally ambiguous — the agent must reason about attendee
availability, preferences, and constraints rather than follow explicit
instructions.
"""

import json
import random
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import CalendarAction, CalendarObservation, CalendarState
except ImportError:
    from models import CalendarAction, CalendarObservation, CalendarState


class PersonalAssistantEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    DEFAULT_EPISODE_BASE_DATE: date = date(2026, 1, 5)  # Monday
    EPISODE_WEEKDAYS: int = 5  # Only Mon-Fri episodes
    EPISODE_WEEKS: int = 3  # Pick from 3 weeks of weekdays

    # --- Attendee schedules (busy blocks per day-of-week) ---
    ATTENDEE_SCHEDULES = {
        "Alice": {
            "monday": [("09:00", "10:00"), ("14:00", "15:00")],
            "tuesday": [("08:00", "09:30")],
            "wednesday": [("13:00", "14:00")],
            "thursday": [],
            "friday": [("09:00", "10:00"), ("15:00", "16:00")],
            "saturday": [("08:00", "18:00")],
            "sunday": [("08:00", "18:00")],
        },
        "Bob": {
            "monday": [("08:00", "18:00")],
            "tuesday": [("10:00", "11:00")],
            "wednesday": [("09:00", "10:00"), ("14:00", "15:30")],
            "thursday": [("11:00", "12:00")],
            "friday": [],
            "saturday": [("08:00", "18:00")],
            "sunday": [("08:00", "18:00")],
        },
        "Charlie": {
            "monday": [("08:00", "12:00")],
            "tuesday": [("08:00", "12:00")],
            "wednesday": [("08:00", "12:00")],
            "thursday": [("08:00", "12:00")],
            "friday": [("08:00", "12:00")],
            "saturday": [("08:00", "18:00")],
            "sunday": [("08:00", "18:00")],
        },
        "Dave": {
            "monday": [("12:00", "13:00")],
            "tuesday": [("12:00", "13:00")],
            "wednesday": [("12:00", "13:00")],
            "thursday": [("12:00", "13:00")],
            "friday": [("12:00", "13:00")],
            "saturday": [("08:00", "18:00")],
            "sunday": [("08:00", "18:00")],
        },
        "Eve": {
            "monday": [("08:00", "10:00")],
            "tuesday": [("08:00", "10:00")],
            "wednesday": [("08:00", "10:00"), ("16:00", "18:00")],
            "thursday": [("08:00", "10:00")],
            "friday": [("08:00", "10:00"), ("13:00", "14:00")],
            "saturday": [("08:00", "18:00")],
            "sunday": [("08:00", "18:00")],
        },
    }

    CONSTRAINTS = [
        # Public — visible via get_constraints
        {"name": "no_meetings_during_lunch", "type": "hard", "visibility": "public", "description": "No meetings should overlap with the 12:00-13:00 lunch block (except pre-existing client lunches)."},
        {"name": "max_3_meetings_per_day", "type": "soft", "visibility": "public", "description": "No person should have more than 3 meetings in a single day."},
        # Private — only discoverable via get_contact_preferences(person)
        {"name": "bob_no_mondays", "type": "hard", "visibility": "private", "person": "Bob", "description": "Bob cannot attend meetings on Mondays."},
        {"name": "eve_not_before_10", "type": "hard", "visibility": "private", "person": "Eve", "description": "Eve is unavailable before 10:00 AM."},
        {"name": "alice_prefers_mornings", "type": "soft", "visibility": "private", "person": "Alice", "description": "Alice prefers meetings in the morning (before 12:00)."},
        {"name": "charlie_prefers_afternoons", "type": "soft", "visibility": "private", "person": "Charlie", "description": "Charlie prefers meetings in the afternoon (after 13:00)."},
    ]

    # Per-person hidden info — only revealed by get_contact_preferences
    CONTACT_PREFERENCES = {
        "Alice": {
            "role": "Engineering Lead",
            "notification_preference": "Slack DM",
            "notes": "Prefers meetings before noon so she can code in the afternoon. Likes agendas shared in advance.",
        },
        "Bob": {
            "role": "Product Manager",
            "notification_preference": "Email",
            "notes": "Works remotely on Mondays and is completely unavailable for meetings that day. Prefers meetings kept to 30 minutes or under when possible.",
        },
        "Charlie": {
            "role": "QA Engineer",
            "notification_preference": "Slack channel #team",
            "notes": "Mornings are reserved for deep testing work. Strongly prefers meetings in the afternoon (after 1 PM).",
        },
        "Dave": {
            "role": "Account Manager",
            "notification_preference": "Email",
            "notes": "Lunch hours are often booked with client meetings. Generally flexible otherwise.",
        },
        "Eve": {
            "role": "Designer",
            "notification_preference": "Slack DM",
            "notes": "Not a morning person — unavailable before 10:00 AM (hard requirement). Prefers some buffer between back-to-back meetings.",
        },
    }

    _LUNCH_EXEMPT_IDS: set = {"seed2"}

    # --- Ambiguous tasks ---
    TASKS = [
        {"description": "Find a time that works for Alice and Bob this week for a 30-minute team standup", "flag": "standup_scheduled"},
        {"description": "I need a block of focused work time today — at least an hour with no interruptions", "flag": "focus_time_booked"},
        {"description": "My calendar looks messy today. Clean up any overlapping meetings", "flag": "conflicts_resolved"},
        {"description": "I have a dentist appointment sometime next week, probably Monday afternoon. Make sure it's on my calendar", "flag": "reminder_set"},
        {"description": "The 'Old Project Review' is no longer needed. Handle it and let the team know", "flag": "meeting_cancelled"},
        {"description": "[DYNAMIC] Accommodate the urgent CEO Sync at 3:00 PM today without conflicts", "flag": "ceo_sync_accommodated"},
        {"description": "[DYNAMIC] Handle Dave's cancellation of 'Lunch with Client' — re-use the freed slot or acknowledge", "flag": "cancellation_handled"},
        {"description": "[DYNAMIC] Reschedule 'Morning Sync' to 11:00 AM as Alice requested", "flag": "reschedule_handled"},
        {"description": "Schedule a project kickoff meeting with Alice, Bob, and Eve — find a slot that respects everyone's constraints", "flag": "kickoff_scheduled"},
        {"description": "Ensure the calendar has zero hard-constraint violations (use check_constraint_violations to verify)", "flag": "hard_constraints_clear"},
        {"description": "Optimize the schedule so that as many soft constraints (preferences) as possible are satisfied", "flag": "preferences_optimized"},
    ]

    INTERRUPTS = [
        {
            "at_step": 3, "type": "new_meeting",
            "event": {"title": "CEO Sync", "date": "today", "start_time": "15:00", "end_time": "15:30", "duration_minutes": 30, "attendees": ["CEO"], "priority": "high"},
            "message": "URGENT: The CEO just scheduled a mandatory sync at 3:00 PM today. This has been added to your calendar. Check for new conflicts and resolve them. This is high priority — other meetings should move if needed.",
        },
        {
            "at_step": 6, "type": "cancellation", "cancel_title": "Lunch with Client",
            "message": "UPDATE: Dave just cancelled 'Lunch with Client'. The 12:00-13:00 slot is now free. Consider using it for any pending tasks that need a time slot.",
        },
        {
            "at_step": 9, "type": "reschedule_request", "event_title": "Morning Sync", "new_time": "11:00",
            "message": "REQUEST: Alice asks to move 'Morning Sync' from 10:00 AM to 11:00 AM. Please reschedule it and confirm there are no new conflicts.",
        },
    ]

    def __init__(self):
        self._events: list[dict] = []
        self._notifications: list[str] = []
        self._found: set[str] = set()
        self._had_initial_conflicts = False
        self._fired_interrupts: set[int] = set()
        self._pending_interrupt_msg: str = ""
        self._seed: int = 0
        self._rng = random.Random(self._seed)
        self._episode_today: date = self.DEFAULT_EPISODE_BASE_DATE
        self._state = CalendarState(episode_id=self._make_episode_id(), step_count=0, total_tasks=len(self.TASKS))

    DAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

    def _make_episode_id(self) -> str:
        return f"ep-{self._seed}-{self._rng.getrandbits(32):08x}"

    def _next_event_id(self) -> str:
        return f"evt{self._rng.getrandbits(32):08x}"

    def _set_episode_context(self, seed: Optional[int], episode_id: Optional[str]) -> None:
        # Freeze episode-relative dates so all "today/tomorrow/next monday" calls
        # remain deterministic during the episode and replayable with a seed.
        self._seed = 0 if seed is None else int(seed)
        self._rng = random.Random(self._seed)
        # Only pick weekdays (Mon-Fri) across EPISODE_WEEKS weeks
        weekdays = []
        for week in range(self.EPISODE_WEEKS):
            for day in range(5):  # Mon=0..Fri=4
                weekdays.append(self.DEFAULT_EPISODE_BASE_DATE + timedelta(weeks=week, days=day))
        self._episode_today = self._rng.choice(weekdays)
        self._state = CalendarState(
            episode_id=episode_id or self._make_episode_id(),
            step_count=0,
            total_tasks=len(self.TASKS),
        )

    def _resolve_date(self, date_str: str) -> str:
        today = self._episode_today
        low = date_str.lower().strip()
        if low == "today":
            return today.isoformat()
        if low == "tomorrow":
            return (today + timedelta(days=1)).isoformat()
        if low.startswith("next "):
            day_name = low.replace("next ", "")
            if day_name in self.DAY_NAMES:
                target_weekday = self.DAY_NAMES.index(day_name)
                days_ahead = (target_weekday - today.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7
                return (today + timedelta(days=days_ahead)).isoformat()
        if low in self.DAY_NAMES:
            target_weekday = self.DAY_NAMES.index(low)
            days_ahead = (target_weekday - today.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            return (today + timedelta(days=days_ahead)).isoformat()
        return date_str

    def _get_day_name(self, date_str: str) -> str:
        try:
            return self.DAY_NAMES[datetime.strptime(date_str, "%Y-%m-%d").weekday()]
        except ValueError:
            return "unknown"

    # --- Constraint evaluation ---

    def _evaluate_constraints(self) -> dict:
        hard_violations, soft_violations = [], []
        for constraint in self.CONSTRAINTS:
            violations = self._check_single_constraint(constraint)
            (hard_violations if constraint["type"] == "hard" else soft_violations).extend(violations)
        hard_total = sum(1 for c in self.CONSTRAINTS if c["type"] == "hard")
        soft_total = sum(1 for c in self.CONSTRAINTS if c["type"] == "soft")
        return {
            "hard_violations": hard_violations, "soft_violations": soft_violations,
            "hard_satisfied": hard_total - len({v["constraint"] for v in hard_violations}),
            "soft_satisfied": soft_total - len({v["constraint"] for v in soft_violations}),
            "hard_total": hard_total, "soft_total": soft_total,
        }

    def _check_single_constraint(self, constraint: dict) -> list[dict]:
        name = constraint["name"]
        violations = []
        if name == "bob_no_mondays":
            for e in self._events:
                if "Bob" in e.get("attendees", []):
                    try:
                        if datetime.strptime(e["date"], "%Y-%m-%d").weekday() == 0:
                            violations.append({"constraint": name, "event": e["title"], "detail": f"Bob is scheduled for '{e['title']}' on a Monday ({e['date']})"})
                    except ValueError:
                        pass
        elif name == "no_meetings_during_lunch":
            for e in self._events:
                if e.get("id") in self._LUNCH_EXEMPT_IDS:
                    continue
                if e.get("start_time", "") < "13:00" and e.get("end_time", "") > "12:00":
                    violations.append({"constraint": name, "event": e["title"], "detail": f"'{e['title']}' ({e['start_time']}-{e['end_time']}) overlaps with lunch block (12:00-13:00)"})
        elif name == "eve_not_before_10":
            for e in self._events:
                if "Eve" in e.get("attendees", []) and e.get("start_time", "") < "10:00":
                    violations.append({"constraint": name, "event": e["title"], "detail": f"Eve is scheduled for '{e['title']}' at {e['start_time']}, before 10:00 AM"})
        elif name == "alice_prefers_mornings":
            for e in self._events:
                if "Alice" in e.get("attendees", []) and e.get("start_time", "") >= "12:00":
                    violations.append({"constraint": name, "event": e["title"], "detail": f"Alice's meeting '{e['title']}' starts at {e['start_time']} (prefers before 12:00)"})
        elif name == "charlie_prefers_afternoons":
            for e in self._events:
                if "Charlie" in e.get("attendees", []) and e.get("start_time", "") < "13:00":
                    violations.append({"constraint": name, "event": e["title"], "detail": f"Charlie's meeting '{e['title']}' starts at {e['start_time']} (prefers after 13:00)"})
        elif name == "max_3_meetings_per_day":
            counts: dict[tuple[str, str], int] = defaultdict(int)
            for e in self._events:
                for a in e.get("attendees", []):
                    counts[(a, e["date"])] += 1
            for (person, date), count in counts.items():
                if count > 3:
                    violations.append({"constraint": name, "event": f"(multiple on {date})", "detail": f"{person} has {count} meetings on {date} (max preferred: 3)"})
        return violations

    # --- Tool implementations ---

    def tool_list_events(self, date: str = "today") -> str:
        target = self._resolve_date(date)
        day_events = [e for e in self._events if e["date"] == target]
        if not day_events:
            return f"No events scheduled for {target}."
        lines = [f"Events for {target}:"]
        for e in sorted(day_events, key=lambda x: x["start_time"]):
            lines.append(f"  - {e['title']} | {e['start_time']}-{e['end_time']} | Attendees: {', '.join(e.get('attendees', []))}")
        return "\n".join(lines)

    def tool_create_event(self, title: str, date: str, start_time: str, duration_minutes: int = 60, attendees: str = "") -> str:
        target = self._resolve_date(date)
        end_time = (datetime.strptime(f"{target} {start_time}", "%Y-%m-%d %H:%M") + timedelta(minutes=duration_minutes)).strftime("%H:%M")
        event = {"id": self._next_event_id(), "title": title, "date": target, "start_time": start_time, "end_time": end_time, "duration_minutes": duration_minutes, "attendees": [a.strip() for a in attendees.split(",") if a.strip()]}
        self._events.append(event)
        warnings = []
        for c in self.CONSTRAINTS:
            for v in self._check_single_constraint(c):
                if v["event"] == title:
                    warnings.append(f"  WARNING [{'HARD' if c['type'] == 'hard' else 'SOFT'}]: {v['detail']}")
        result = f"Event '{title}' created on {target} from {start_time} to {end_time}."
        if warnings:
            result += "\n\nConstraint warnings for this event:\n" + "\n".join(warnings)
        return result

    def tool_delete_event(self, title: str) -> str:
        matches = [e for e in self._events if e["title"].lower() == title.lower()]
        if not matches:
            return f"No event found with title '{title}'."
        for m in matches:
            self._events.remove(m)
        return f"Deleted event '{title}'."

    def tool_find_free_slots(self, date: str = "today", duration_minutes: int = 60) -> str:
        target = self._resolve_date(date)
        day_events = sorted([e for e in self._events if e["date"] == target], key=lambda x: x["start_time"])
        slots = []
        current = datetime.strptime(f"{target} 08:00", "%Y-%m-%d %H:%M")
        end_of_day = datetime.strptime(f"{target} 18:00", "%Y-%m-%d %H:%M")
        for event in day_events:
            event_start = datetime.strptime(f"{target} {event['start_time']}", "%Y-%m-%d %H:%M")
            if (event_start - current).total_seconds() / 60 >= duration_minutes:
                slots.append(f"  {current.strftime('%H:%M')} - {event_start.strftime('%H:%M')}")
            current = max(current, datetime.strptime(f"{target} {event['end_time']}", "%Y-%m-%d %H:%M"))
        if (end_of_day - current).total_seconds() / 60 >= duration_minutes:
            slots.append(f"  {current.strftime('%H:%M')} - {end_of_day.strftime('%H:%M')}")
        if not slots:
            return f"No free {duration_minutes}-minute slots available on {target}."
        return f"Free slots on {target} ({duration_minutes}+ min):\n" + "\n".join(slots)

    def tool_check_conflicts(self, date: str = "today") -> str:
        target = self._resolve_date(date)
        day_events = sorted([e for e in self._events if e["date"] == target], key=lambda x: x["start_time"])
        conflicts = []
        for i in range(len(day_events) - 1):
            if day_events[i]["end_time"] > day_events[i + 1]["start_time"]:
                conflicts.append(f"  CONFLICT: '{day_events[i]['title']}' (ends {day_events[i]['end_time']}) overlaps with '{day_events[i+1]['title']}' (starts {day_events[i+1]['start_time']})")
        return f"No conflicts found on {target}." if not conflicts else f"Conflicts on {target}:\n" + "\n".join(conflicts)

    def tool_resolve_conflict(self, event_title: str, new_start_time: str) -> str:
        matches = [e for e in self._events if e["title"].lower() == event_title.lower()]
        if not matches:
            return f"No event found with title '{event_title}'."
        event = matches[0]
        event["start_time"] = new_start_time
        event["end_time"] = (datetime.strptime(f"{event['date']} {new_start_time}", "%Y-%m-%d %H:%M") + timedelta(minutes=event["duration_minutes"])).strftime("%H:%M")
        warnings = []
        for c in self.CONSTRAINTS:
            for v in self._check_single_constraint(c):
                if v["event"] == event["title"]:
                    warnings.append(f"  WARNING [{'HARD' if c['type'] == 'hard' else 'SOFT'}]: {v['detail']}")
        result = f"Moved '{event_title}' to {new_start_time}-{event['end_time']}."
        if warnings:
            result += "\n\nConstraint warnings:\n" + "\n".join(warnings)
        return result

    def tool_send_notification(self, to: str, message: str) -> str:
        self._notifications.append({"to": to, "message": message})
        return f"Notification sent to {to}: {message}"

    def tool_get_task_list(self) -> str:
        lines = ["Tasks to complete:"]
        for i, t in enumerate(self.TASKS, 1):
            status = "DONE" if t["flag"] in self._found else "TODO"
            lines.append(f"  {i}. [{status}] {t['description']}")
        return "\n".join(lines)

    def tool_check_availability(self, person: str, date: str = "today") -> str:
        """Check a person's availability on a given date. Shows busy times and free windows."""
        target = self._resolve_date(date)
        day_name = self._get_day_name(target)
        person_key = person.strip().title()
        lines = [f"Availability for {person_key} on {target} ({day_name}):"]

        schedule = self.ATTENDEE_SCHEDULES.get(person_key, {})
        busy_blocks = schedule.get(day_name, [])
        if busy_blocks:
            lines.append("  External commitments (busy):")
            for s, e in busy_blocks:
                lines.append(f"    - {s} to {e}")
        else:
            lines.append("  No external commitments.")

        cal_events = [e for e in self._events if e["date"] == target and person_key in e.get("attendees", [])]
        if cal_events:
            lines.append("  Calendar events:")
            for e in sorted(cal_events, key=lambda x: x["start_time"]):
                lines.append(f"    - {e['title']} ({e['start_time']}-{e['end_time']})")

        all_busy = sorted(list(busy_blocks) + [(e["start_time"], e["end_time"]) for e in cal_events])
        merged = []
        for s, e in all_busy:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        free, cursor = [], "08:00"
        for s, e in merged:
            if cursor < s:
                free.append((cursor, s))
            cursor = max(cursor, e)
        if cursor < "18:00":
            free.append((cursor, "18:00"))

        if free:
            lines.append("  Free windows (8:00-18:00):")
            for s, e in free:
                lines.append(f"    - {s} to {e}")
        else:
            lines.append("  No free windows on this date.")
        return "\n".join(lines)

    def tool_edit_event(self, title: str, new_title: str = "", new_date: str = "", new_start_time: str = "", new_duration_minutes: int = 0, new_attendees: str = "") -> str:
        """Edit an existing event. Only provided fields are changed. Use empty string or 0 to keep current value."""
        matches = [e for e in self._events if e["title"].lower() == title.lower()]
        if not matches:
            return f"No event found with title '{title}'."
        event = matches[0]
        changes = []
        if new_title:
            changes.append(f"title: '{event['title']}' → '{new_title}'")
            event["title"] = new_title
        if new_date:
            target = self._resolve_date(new_date)
            changes.append(f"date: {event['date']} → {target}")
            event["date"] = target
        if new_start_time:
            dur = new_duration_minutes if new_duration_minutes > 0 else event["duration_minutes"]
            end_time = (datetime.strptime(f"{event['date']} {new_start_time}", "%Y-%m-%d %H:%M") + timedelta(minutes=dur)).strftime("%H:%M")
            changes.append(f"time: {event['start_time']}-{event['end_time']} → {new_start_time}-{end_time}")
            event["start_time"] = new_start_time
            event["end_time"] = end_time
            if new_duration_minutes > 0:
                event["duration_minutes"] = new_duration_minutes
        elif new_duration_minutes > 0:
            end_time = (datetime.strptime(f"{event['date']} {event['start_time']}", "%Y-%m-%d %H:%M") + timedelta(minutes=new_duration_minutes)).strftime("%H:%M")
            changes.append(f"duration: {event['duration_minutes']}min → {new_duration_minutes}min ({event['start_time']}-{end_time})")
            event["end_time"] = end_time
            event["duration_minutes"] = new_duration_minutes
        if new_attendees:
            old_att = ", ".join(event.get("attendees", []))
            event["attendees"] = [a.strip() for a in new_attendees.split(",") if a.strip()]
            changes.append(f"attendees: [{old_att}] → [{new_attendees}]")
        if not changes:
            return f"No changes specified for '{title}'."
        # Check constraint warnings
        warnings = []
        for c in self.CONSTRAINTS:
            for v in self._check_single_constraint(c):
                if v["event"] == event["title"]:
                    warnings.append(f"  WARNING [{'HARD' if c['type'] == 'hard' else 'SOFT'}]: {v['detail']}")
        result = f"Updated '{title}':\n" + "\n".join(f"  - {c}" for c in changes)
        if warnings:
            result += "\n\nConstraint warnings:\n" + "\n".join(warnings)
        return result

    def tool_get_constraints(self) -> str:
        lines = ["Scheduling Constraints:", "", "HARD constraints (must be satisfied):"]
        for c in self.CONSTRAINTS:
            if c["type"] == "hard" and c.get("visibility") == "public":
                lines.append(f"  - {c['description']}")
        lines += ["", "SOFT constraints (preferences, partial credit):"]
        for c in self.CONSTRAINTS:
            if c["type"] == "soft" and c.get("visibility") == "public":
                lines.append(f"  - {c['description']}")
        lines += ["", "NOTE: Individual team members may have additional constraints and preferences.",
                  "Use get_contact_preferences(person) to learn about a specific person's scheduling needs."]
        return "\n".join(lines)

    def tool_get_contact_preferences(self, person: str) -> str:
        """Get a person's scheduling preferences, constraints, and contact info."""
        person_key = person.strip().title()
        prefs = self.CONTACT_PREFERENCES.get(person_key)
        if not prefs:
            return f"No contact info found for '{person}'. Known contacts: {', '.join(self.CONTACT_PREFERENCES.keys())}"
        lines = [f"Contact Preferences for {person_key}:", f"  Role: {prefs['role']}", f"  Notify via: {prefs['notification_preference']}", f"  Notes: {prefs['notes']}", ""]
        # Reveal private constraints for this person
        person_constraints = [c for c in self.CONSTRAINTS if c.get("person") == person_key and c.get("visibility") == "private"]
        if person_constraints:
            lines.append("  Scheduling rules:")
            for c in person_constraints:
                label = "HARD" if c["type"] == "hard" else "SOFT"
                lines.append(f"    - [{label}] {c['description']}")
        return "\n".join(lines)

    def tool_check_constraint_violations(self) -> str:
        result = self._evaluate_constraints()
        lines = ["Constraint Violation Report:", f"  Hard constraints: {result['hard_satisfied']}/{result['hard_total']} satisfied", f"  Soft constraints: {result['soft_satisfied']}/{result['soft_total']} satisfied"]
        if result["hard_violations"]:
            lines += ["", "HARD CONSTRAINT VIOLATIONS (must fix):"] + [f"  - [{v['constraint']}] {v['detail']}" for v in result["hard_violations"]]
        if result["soft_violations"]:
            lines += ["", "SOFT CONSTRAINT VIOLATIONS (preferences not met):"] + [f"  - [{v['constraint']}] {v['detail']}" for v in result["soft_violations"]]
        if not result["hard_violations"] and not result["soft_violations"]:
            lines += ["", "All constraints satisfied!"]
        return "\n".join(lines)

    TOOL_MAP = {
        "list_events": "tool_list_events",
        "create_event": "tool_create_event",
        "delete_event": "tool_delete_event",
        "edit_event": "tool_edit_event",
        "find_free_slots": "tool_find_free_slots",
        "check_conflicts": "tool_check_conflicts",
        "resolve_conflict": "tool_resolve_conflict",
        "send_notification": "tool_send_notification",
        "get_task_list": "tool_get_task_list",
        "check_availability": "tool_check_availability",
        "get_constraints": "tool_get_constraints",
        "get_contact_preferences": "tool_get_contact_preferences",
        "check_constraint_violations": "tool_check_constraint_violations",
    }

    def _dispatch_tool(self, tool_name: str, arguments: dict) -> str:
        method_name = self.TOOL_MAP.get(tool_name)
        if not method_name:
            return f"Unknown tool '{tool_name}'. Available tools: {', '.join(self.TOOL_MAP.keys())}"
        try:
            return getattr(self, method_name)(**arguments)
        except TypeError as e:
            return f"Error calling {tool_name}: {e}"

    def _process_interrupts(self):
        step = self._state.step_count
        for i, intr in enumerate(self.INTERRUPTS):
            if intr["at_step"] == step and i not in self._fired_interrupts:
                self._fired_interrupts.add(i)
                if intr["type"] == "new_meeting":
                    event_data = intr["event"].copy()
                    event_data["date"] = self._resolve_date(event_data["date"])
                    event_data["id"] = self._next_event_id()
                    self._events.append(event_data)
                    self._pending_interrupt_msg = f"\n\n--- INTERRUPT ---\n{intr['message']}"
                elif intr["type"] == "cancellation":
                    for e in self._events:
                        if e["title"].lower() == intr["cancel_title"].lower():
                            e["cancelled"] = True
                    self._pending_interrupt_msg = f"\n\n--- INTERRUPT ---\n{intr['message']}\nPlease delete '{intr['cancel_title']}' from the calendar."
                elif intr["type"] == "reschedule_request":
                    self._pending_interrupt_msg = f"\n\n--- INTERRUPT ---\n{intr['message']}"

    # --- Completion checking (outcome-based) ---

    def _check_completions(self):
        today = self._resolve_date("today")
        today_dt = self._episode_today
        days_since_monday = today_dt.weekday()
        week_start = today_dt - timedelta(days=days_since_monday)
        week_dates = [(week_start + timedelta(days=i)).isoformat() for i in range(7)]
        next_week_dates = [(week_start + timedelta(days=i)).isoformat() for i in range(7, 14)]

        # standup_scheduled: event this week, "standup" in title, Alice+Bob, exactly 30min,
        # no external schedule conflicts AND no calendar event overlaps
        standup_scheduled = False
        for e in self._events:
            if "standup" not in e.get("title", "").lower():
                continue
            if e["date"] not in week_dates:
                continue
            if not {"Alice", "Bob"}.issubset(set(e.get("attendees", []))):
                continue
            if e.get("duration_minutes", 60) != 30:
                continue
            day_name = self._get_day_name(e["date"])
            bad = False
            for person in ["Alice", "Bob"]:
                for bs, be in self.ATTENDEE_SCHEDULES.get(person, {}).get(day_name, []):
                    if e["start_time"] < be and e["end_time"] > bs:
                        bad = True
                        break
                if bad:
                    break
            # Also check for calendar event overlaps on that day
            if not bad:
                for other in self._events:
                    if other.get("id") == e.get("id") or other["date"] != e["date"]:
                        continue
                    if e["start_time"] < other["end_time"] and e["end_time"] > other["start_time"]:
                        bad = True
                        break
            if not bad:
                standup_scheduled = True
                break
        if standup_scheduled:
            self._found.add("standup_scheduled")
        else:
            self._found.discard("standup_scheduled")

        # focus_time_booked: event today, "focus" in title, >=60min, no overlaps, within 08:00-18:00
        focus_time_booked = False
        for e in self._events:
            if e["date"] != today or "focus" not in e.get("title", "").lower() or e.get("duration_minutes", 0) < 60:
                continue
            if e.get("start_time", "") < "08:00" or e.get("end_time", "") > "18:00":
                continue
            if not any(o.get("id") != e.get("id") and o["date"] == today and e["start_time"] < o["end_time"] and e["end_time"] > o["start_time"] for o in self._events):
                focus_time_booked = True
                break
        if focus_time_booked:
            self._found.add("focus_time_booked")
        else:
            self._found.discard("focus_time_booked")

        # conflicts_resolved: no overlapping events today (revocable — new overlaps lose the flag)
        today_events = sorted([e for e in self._events if e["date"] == today], key=lambda x: x["start_time"])
        has_conflicts = any(today_events[i]["end_time"] > today_events[i + 1]["start_time"] for i in range(len(today_events) - 1))
        if not has_conflicts and self._had_initial_conflicts:
            self._found.add("conflicts_resolved")
        elif has_conflicts:
            self._found.discard("conflicts_resolved")

        # reminder_set: dentist event next week, afternoon (>=12:00), within working hours
        reminder_set = False
        for e in self._events:
            if "dentist" not in e.get("title", "").lower():
                continue
            if e["date"] not in next_week_dates:
                continue
            if e.get("start_time", "") < "12:00" or e.get("end_time", "") > "18:00":
                continue
            reminder_set = True
            break
        if reminder_set:
            self._found.add("reminder_set")
        else:
            self._found.discard("reminder_set")

        # meeting_cancelled: Old Project Review gone AND attendees notified
        # Must notify at least one of the actual attendees (Alice, Bob, Charlie)
        old_exists = any("old project review" in e.get("title", "").lower() for e in self._events)
        attendees_notified = any(
            n.get("to", "").strip().title() in ("Alice", "Bob", "Charlie")
            and ("old project" in n.get("message", "").lower() or "cancel" in n.get("message", "").lower())
            for n in self._notifications
        )
        if not old_exists and attendees_notified:
            self._found.add("meeting_cancelled")
        else:
            self._found.discard("meeting_cancelled")

        # ceo_sync_accommodated: CEO sync still at 15:00-15:30 with no conflicts
        ceo_sync_accommodated = False
        if 0 in self._fired_interrupts:
            ceo_events = [e for e in self._events if "ceo" in e.get("title", "").lower() and e.get("date") == today]
            for ceo in ceo_events:
                if ceo.get("start_time") == "15:00" and ceo.get("end_time") == "15:30":
                    if not any(e.get("id") != ceo.get("id") and e["date"] == today and e["start_time"] < ceo["end_time"] and e["end_time"] > ceo["start_time"] for e in self._events):
                        ceo_sync_accommodated = True
                        break
        if ceo_sync_accommodated:
            self._found.add("ceo_sync_accommodated")
        else:
            self._found.discard("ceo_sync_accommodated")

        # cancellation_handled
        if 1 in self._fired_interrupts and not any("lunch with client" in e.get("title", "").lower() for e in self._events):
            self._found.add("cancellation_handled")
        else:
            self._found.discard("cancellation_handled")

        # reschedule_handled: Morning Sync moved to 11:00 (direct request, no schedule check)
        if 2 in self._fired_interrupts and any("morning sync" in e.get("title", "").lower() and e.get("start_time") == "11:00" for e in self._events):
            self._found.add("reschedule_handled")
        else:
            self._found.discard("reschedule_handled")

        # kickoff_scheduled: Alice+Bob+Eve, no hard constraint violations,
        # no attendee schedule conflicts, AND no calendar event overlaps
        kickoff_scheduled = False
        for ke in self._events:
            if "kickoff" not in ke.get("title", "").lower():
                continue
            if not {"Alice", "Bob", "Eve"}.issubset(set(ke.get("attendees", []))):
                continue
            # Check hard constraints
            hard_ok = True
            for c in self.CONSTRAINTS:
                if c["type"] == "hard" and any(v["event"] == ke["title"] for v in self._check_single_constraint(c)):
                    hard_ok = False
                    break
            if not hard_ok:
                continue
            # Check attendee external schedule conflicts
            day_name = self._get_day_name(ke["date"])
            schedule_ok = True
            for person in ke.get("attendees", []):
                for bs, be in self.ATTENDEE_SCHEDULES.get(person, {}).get(day_name, []):
                    if ke["start_time"] < be and ke["end_time"] > bs:
                        schedule_ok = False
                        break
                if not schedule_ok:
                    break
            if not schedule_ok:
                continue
            # Check no calendar event overlaps on that day
            cal_ok = True
            for other in self._events:
                if other.get("id") == ke.get("id") or other["date"] != ke["date"]:
                    continue
                if ke["start_time"] < other["end_time"] and ke["end_time"] > other["start_time"]:
                    cal_ok = False
                    break
            if cal_ok:
                kickoff_scheduled = True
                break
        if kickoff_scheduled:
            self._found.add("kickoff_scheduled")
        else:
            self._found.discard("kickoff_scheduled")

        # hard_constraints_clear (revocable)
        result = self._evaluate_constraints()
        if result["hard_satisfied"] == result["hard_total"]:
            self._found.add("hard_constraints_clear")
        else:
            self._found.discard("hard_constraints_clear")

        # preferences_optimized (revocable)
        if result["soft_satisfied"] >= 2:
            self._found.add("preferences_optimized")
        else:
            self._found.discard("preferences_optimized")

    def _seed_initial_events(self):
        today = self._resolve_date("today")
        self._events = [
            {"id": "seed1", "title": "Morning Sync", "date": today, "start_time": "10:00", "end_time": "10:30", "duration_minutes": 30, "attendees": ["Alice", "Charlie"]},
            {"id": "seed2", "title": "Lunch with Client", "date": today, "start_time": "12:00", "end_time": "13:00", "duration_minutes": 60, "attendees": ["Dave"]},
            {"id": "seed3", "title": "Old Project Review", "date": today, "start_time": "14:00", "end_time": "15:00", "duration_minutes": 60, "attendees": ["Alice", "Bob", "Charlie"]},
            {"id": "seed4", "title": "Design Review", "date": today, "start_time": "14:30", "end_time": "15:30", "duration_minutes": 60, "attendees": ["Eve"]},
        ]

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> CalendarObservation:
        self._found = set()
        self._had_initial_conflicts = False
        self._notifications = []
        self._fired_interrupts = set()
        self._pending_interrupt_msg = ""
        self._set_episode_context(seed=seed, episode_id=episode_id)
        self._seed_initial_events()
        self._had_initial_conflicts = True
        self._check_completions()

        new_flags = len(self._found)
        reward = new_flags / len(self.TASKS)
        done = new_flags == len(self.TASKS)
        self._state.tasks_completed = new_flags

        return CalendarObservation(
            output="Calendar assistant ready. You have 4 events today including a scheduling conflict. "
                   "There are also person-specific constraints and preferences to satisfy. "
                   "Use get_task_list to see what needs to be done, get_constraints for general rules, "
                   "get_contact_preferences to learn about individual people's needs, "
                   "and check_availability to look up people's schedules before booking. "
                   "Warning: new requests may arrive while you work — stay alert and adapt.",
            pending_tasks=len(self.TASKS) - new_flags,
            events_today=len([e for e in self._events if e["date"] == self._resolve_date("today")]),
            metadata={"seed": self._seed, "episode_today": self._resolve_date("today")},
            flags_found=list(self._found), done=done, reward=reward,
        )

    def step(self, action: CalendarAction, **kwargs) -> CalendarObservation:
        self._state.step_count += 1
        self._pending_interrupt_msg = ""
        self._process_interrupts()

        instruction = action.instruction.strip()
        tool_name, arguments = None, {}
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
            output = (
                f"Received: '{instruction}'\nTo interact, send a JSON tool call like:\n"
                f'{{"tool": "get_task_list", "args": {{}}}}\n'
                f'{{"tool": "check_availability", "args": {{"person": "Alice", "date": "tuesday"}}}}\n'
                f'{{"tool": "create_event", "args": {{"title": "Meeting", "date": "tomorrow", "start_time": "09:00", "duration_minutes": 30, "attendees": "Alice,Bob"}}}}\n'
                f"Available tools: {', '.join(self.TOOL_MAP.keys())}"
            )

        self._check_completions()

        if self._pending_interrupt_msg:
            output += self._pending_interrupt_msg

        new_flags = len(self._found)
        reward = new_flags / len(self.TASKS)
        done = new_flags == len(self.TASKS)
        self._state.tasks_completed = new_flags

        return CalendarObservation(
            output=output, pending_tasks=len(self.TASKS) - new_flags,
            events_today=len([e for e in self._events if e["date"] == self._resolve_date("today")]),
            metadata={"seed": self._seed, "episode_today": self._resolve_date("today")},
            flags_found=list(self._found), reward=reward, done=done,
        )

    @property
    def state(self) -> CalendarState:
        return self._state
