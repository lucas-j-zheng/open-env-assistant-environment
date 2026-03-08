"""
Seed-based episode config generator for diverse calendar scenarios.

Each seed produces an EpisodeConfig that defines variable event names,
times, and attendees. The completion checker, tasks, and interrupts
all reference the config instead of hardcoded strings.
"""

import random
from dataclasses import dataclass, field


MORNING_NAMES = [
    "Morning Sync", "Daily Standup", "AM Check-in",
    "Morning Huddle", "Sunrise Sync",
]
LUNCH_NAMES = [
    "Lunch with Client", "Client Lunch", "Working Lunch",
    "Lunch & Learn", "Midday Meeting",
]
CANCEL_NAMES = [
    "Old Project Review", "Legacy Sprint Retro", "Q3 Budget Debrief",
    "Deprecated Feature Review", "Sunset Planning",
]
CONFLICT_NAMES = [
    "Design Review", "UX Walkthrough", "Wireframe Review",
    "Design Critique", "Visual QA",
]
NOISE_TEMPLATES = [
    {"title": "Sprint Planning", "duration_range": (30, 60), "attendee_count": (1, 3)},
    {"title": "1:1 with Manager", "duration_range": (20, 30), "attendee_count": (1, 1)},
    {"title": "Code Review Session", "duration_range": (30, 45), "attendee_count": (1, 2)},
    {"title": "Client Demo", "duration_range": (30, 60), "attendee_count": (1, 3)},
    {"title": "Architecture Discussion", "duration_range": (45, 60), "attendee_count": (2, 3)},
    {"title": "Retrospective", "duration_range": (30, 45), "attendee_count": (2, 4)},
    {"title": "Onboarding Walkthrough", "duration_range": (30, 60), "attendee_count": (1, 2)},
    {"title": "Security Review", "duration_range": (30, 45), "attendee_count": (1, 2)},
    {"title": "Metrics Review", "duration_range": (20, 30), "attendee_count": (1, 3)},
    {"title": "Team Social", "duration_range": (30, 60), "attendee_count": (2, 4)},
]

ALL_ATTENDEES = ["Alice", "Bob", "Charlie", "Dave", "Eve"]

MESSAGE_TEMPLATES = [
    {"type": "meeting_request", "from": "Client", "subject": "Meeting Request: Q2 Planning",
     "body": "I'd like a 45-minute meeting this week to discuss the Q2 roadmap. Can you find a time?",
     "core_ask": "schedule_meeting", "reply_keywords": ["scheduled", "booked", "confirmed", "time", "slot"]},
    {"type": "diplomatic", "from": "Client", "subject": "Concern about project timeline",
     "body": "I'm worried the current timeline is too aggressive. We've missed two milestones. Can we discuss adjustments?",
     "core_ask": "acknowledge_concern", "reply_keywords": ["understand", "address", "discuss", "adjust", "appreciate", "concern"]},
    {"type": "fyi", "from": "HR", "subject": "Office closed next Friday",
     "body": "The office will be closed next Friday for maintenance. Work from home. No action needed unless it affects meetings.",
     "core_ask": "acknowledge", "reply_keywords": ["noted", "thanks", "acknowledged", "got it", "understand", "thank"]},
    {"type": "conflict_notification", "from": "Eve", "subject": "Schedule conflict notice",
     "body": "I have a conflict with the design review this week. Can you reschedule or find an alternative time?",
     "core_ask": "resolve_conflict", "reply_keywords": ["reschedule", "moved", "alternative", "updated", "new time", "changed"]},
]

CONTRADICTION_TEMPLATES = [
    {"type": "meeting_request", "from": "Client", "subject": "RE: Meeting Request: Q2 Planning",
     "body": "Actually, I need to change the Q2 planning meeting — make it 30 minutes instead and it needs to be in the afternoon (after 2 PM). Morning doesn't work anymore.",
     "core_ask": "update_meeting", "reply_keywords": ["updated", "changed", "afternoon", "rescheduled", "moved", "30"]},
]

PERSONAL_TEMPLATES = [
    {"title": "Pick up kids from school", "time_range": (15*60, 16*60), "duration": 30},
    {"title": "Dinner reservation", "time_range": (17*60, 18*60), "duration": 60},
    {"title": "Gym session", "time_range": (7*60, 8*60), "duration": 60},
    {"title": "Doctor appointment", "time_range": (14*60, 16*60), "duration": 45},
    {"title": "School recital", "time_range": (15*60, 16*60), "duration": 60},
]

BOSS_REQUEST_TEMPLATES = [
    {
        "driven_flag": "standup_scheduled",
        "type": "boss_request", "from": "Manager",
        "subject": "Team standup",
        "body": "Hey, can you find a time for Alice and Bob to do a quick standup this week? Keep it short — 30 min max.",
        "core_ask": "schedule_standup",
        "reply_keywords": ["scheduled", "booked", "standup", "time", "slot"],
    },
    {
        "driven_flag": "focus_time_booked",
        "type": "boss_request", "from": "Boss",
        "subject": "Need focus time",
        "body": "I need some focused work time today — block off at least an hour where nothing's scheduled.",
        "core_ask": "book_focus_time",
        "reply_keywords": ["blocked", "focus", "booked", "hour", "scheduled"],
    },
    {
        "driven_flag": "conflicts_resolved",
        "type": "boss_request", "from": "Boss",
        "subject": "Calendar looks messy",
        "body": "I glanced at my calendar and it looks like there are overlapping meetings today. Can you clean that up?",
        "core_ask": "resolve_conflicts",
        "reply_keywords": ["resolved", "cleaned", "fixed", "moved", "conflict"],
    },
    {
        "driven_flag": "reminder_set",
        "type": "boss_request", "from": "Boss",
        "subject": "Dentist next week",
        "body": "I have a dentist appointment sometime next week, probably Monday afternoon. Make sure it's on my calendar.",
        "core_ask": "set_reminder",
        "reply_keywords": ["added", "calendar", "dentist", "appointment", "scheduled"],
    },
    {
        "driven_flag": "meeting_cancelled",
        "type": "boss_request", "from": "Boss",
        "subject": "Cancel {cancel_title}",
        "body": "The '{cancel_title}' isn't needed anymore — cancel it and let everyone know.",
        "core_ask": "cancel_meeting",
        "reply_keywords": ["cancelled", "canceled", "notified", "removed", "deleted"],
    },
    {
        "driven_flag": "kickoff_scheduled",
        "type": "boss_request", "from": "Manager",
        "subject": "Project kickoff",
        "body": "We need to schedule a project kickoff with Alice, Bob, and Eve. Find a time that works for everyone.",
        "core_ask": "schedule_kickoff",
        "reply_keywords": ["scheduled", "kickoff", "booked", "time", "slot"],
    },
    {
        "driven_flag": "hard_constraints_clear",
        "type": "boss_request", "from": "Boss",
        "subject": "Respect the rules",
        "body": "Make sure the schedule respects everyone's availability and preferences. No one should be double-booked or scheduled when they can't attend.",
        "core_ask": "check_constraints",
        "reply_keywords": ["checked", "verified", "constraints", "availability", "compliant"],
    },
    {
        "driven_flag": "work_life_conflicts_resolved",
        "type": "boss_request", "from": "Boss",
        "subject": "Personal events",
        "body": "My personal stuff is on the calendar too — those can't move. If anything overlaps, move the work meetings.",
        "core_ask": "resolve_personal_conflicts",
        "reply_keywords": ["moved", "resolved", "personal", "adjusted", "conflict"],
    },
]



@dataclass
class EpisodeConfig:
    initial_events: list[dict] = field(default_factory=list)
    lunch_exempt_ids: set[str] = field(default_factory=set)
    # Role -> actual title mapping
    morning_meeting_title: str = "Morning Sync"
    lunch_meeting_title: str = "Lunch with Client"
    cancellable_title: str = "Old Project Review"
    conflicting_title: str = "Design Review"
    cancellable_attendees: list[str] = field(
        default_factory=lambda: ["Alice", "Bob", "Charlie"]
    )
    initial_messages: list[dict] = field(default_factory=list)
    mid_episode_message: dict = field(default_factory=dict)
    personal_events: list[dict] = field(default_factory=list)
    personal_event_update: dict = field(default_factory=dict)
    interrupt_steps: dict[str, int] = field(default_factory=dict)


def _round_time(minutes: int) -> str:
    """Convert minutes-since-midnight to HH:MM, floored to 15-minute increments."""
    minutes = max(0, min(minutes, 24 * 60 - 1))
    rounded = (minutes // 15) * 15
    return f"{rounded // 60:02d}:{rounded % 60:02d}"


def _add_minutes(time_str: str, minutes: int) -> str:
    h, m = map(int, time_str.split(":"))
    total = h * 60 + m + minutes
    return f"{total // 60:02d}:{total % 60:02d}"


def _time_to_minutes(time_str: str) -> int:
    h, m = map(int, time_str.split(":"))
    return h * 60 + m


def _events_overlap(e1_start: str, e1_end: str, e2_start: str, e2_end: str) -> bool:
    return e1_start < e2_end and e1_end > e2_start


def generate_episode_config(rng: random.Random, today: str) -> EpisodeConfig:
    """Generate a diverse episode config from a seeded RNG."""

    # Pick names from pools
    morning_title = rng.choice(MORNING_NAMES)
    lunch_title = rng.choice(LUNCH_NAMES)
    cancel_title = rng.choice(CANCEL_NAMES)
    conflict_title = rng.choice(CONFLICT_NAMES)

    events = []
    used_slots: list[tuple[str, str]] = []  # (start, end) pairs for overlap checking

    def _find_slot(earliest_min: int, latest_min: int, duration: int) -> str | None:
        """Find a non-overlapping slot within the given range."""
        candidates = list(range(earliest_min, latest_min + 1, 15))
        rng.shuffle(candidates)
        for start_min in candidates:
            start = _round_time(start_min)
            end = _add_minutes(start, duration)
            if _time_to_minutes(end) > 18 * 60:
                continue
            if not any(_events_overlap(start, end, s, e) for s, e in used_slots):
                return start
        return None

    # 1. Morning meeting: 08:00-10:00 range, 20-30min
    morning_duration = rng.choice([20, 25, 30])
    morning_start_min = rng.randint(8 * 60, 10 * 60 - morning_duration)
    morning_start = _round_time(morning_start_min)
    morning_end = _add_minutes(morning_start, morning_duration)
    # Pick 1 additional attendee beyond Alice
    morning_extra = rng.choice([a for a in ALL_ATTENDEES if a != "Alice"])
    events.append({
        "id": "seed1", "title": morning_title,
        "date": today, "start_time": morning_start, "end_time": morning_end,
        "duration_minutes": morning_duration,
        "attendees": ["Alice", morning_extra],
    })
    used_slots.append((morning_start, morning_end))

    # 2. Lunch meeting: always 12:00-13:00 (fixed for lunch constraint logic)
    events.append({
        "id": "seed2", "title": lunch_title,
        "date": today, "start_time": "12:00", "end_time": "13:00",
        "duration_minutes": 60, "attendees": ["Dave"],
    })
    used_slots.append(("12:00", "13:00"))

    # 3. Cancellable review: PM slot 13:00-15:30, 45-60min
    cancel_duration = rng.choice([45, 60])
    cancel_start = _find_slot(13 * 60, 15 * 60 + 30 - cancel_duration, cancel_duration)
    if cancel_start is None:
        cancel_start = "14:00"
    cancel_end = _add_minutes(cancel_start, cancel_duration)
    # Attendees: Alice + Bob + Charlie (always, for notification check)
    cancel_attendees = ["Alice", "Bob", "Charlie"]
    events.append({
        "id": "seed3", "title": cancel_title,
        "date": today, "start_time": cancel_start, "end_time": cancel_end,
        "duration_minutes": cancel_duration,
        "attendees": cancel_attendees,
    })
    used_slots.append((cancel_start, cancel_end))

    # 4. Conflicting event: intentionally overlaps with cancellable review
    conflict_duration = rng.choice([45, 60])
    # Offset overlap by 15-30 minutes into the cancellable
    overlap_offset = rng.choice([15, 30])
    conflict_start = _add_minutes(cancel_start, overlap_offset)
    conflict_end = _add_minutes(conflict_start, conflict_duration)
    # Eve + 0-1 others (not from cancel_attendees to keep it interesting)
    conflict_extra = []
    extras = [a for a in ALL_ATTENDEES if a not in cancel_attendees and a != "Eve"]
    if extras and rng.random() < 0.5:
        conflict_extra = [rng.choice(extras)]
    events.append({
        "id": "seed4", "title": conflict_title,
        "date": today, "start_time": conflict_start, "end_time": conflict_end,
        "duration_minutes": conflict_duration,
        "attendees": ["Eve"] + conflict_extra,
    })
    # Keep this overlapping with seed3, but prevent additional noise overlaps.
    used_slots.append((conflict_start, conflict_end))

    # 5. Noise events: 0-3 additional meetings
    noise_count = rng.randint(0, 3)
    available_noise = rng.sample(NOISE_TEMPLATES, min(noise_count, len(NOISE_TEMPLATES)))
    for i, template in enumerate(available_noise):
        dur = rng.randint(*template["duration_range"])
        dur = (dur // 15) * 15 or 15  # round to 15min
        slot = _find_slot(8 * 60, 17 * 60, dur)
        if slot is None:
            continue
        end = _add_minutes(slot, dur)
        n_attendees = rng.randint(*template["attendee_count"])
        attendees = rng.sample(ALL_ATTENDEES, min(n_attendees, len(ALL_ATTENDEES)))
        events.append({
            "id": f"noise{i}", "title": template["title"],
            "date": today, "start_time": slot, "end_time": end,
            "duration_minutes": dur, "attendees": attendees,
        })
        used_slots.append((slot, end))

    # --- Messages / Inbox ---
    picked_messages = rng.sample(MESSAGE_TEMPLATES, 3)
    # Ensure the "diplomatic" type is always included
    if not any(m["type"] == "diplomatic" for m in picked_messages):
        diplomatic = [m for m in MESSAGE_TEMPLATES if m["type"] == "diplomatic"][0]
        picked_messages[-1] = diplomatic
    initial_messages = []
    for idx, msg in enumerate(picked_messages):
        initial_messages.append({
            **msg,
            "id": f"msg{idx}",
            "received_at_step": 0,
            "read": False,
            "replied": False,
            "reply_text": "",
        })

    # Pick contradiction template (used later for mid-episode message)
    contra = rng.choice(CONTRADICTION_TEMPLATES)

    # --- Personal events ---
    picked_personal = rng.sample(PERSONAL_TEMPLATES, 2)
    personal_events = []

    # First personal event: try to overlap with seed3 or seed4
    p0 = picked_personal[0]
    # Attempt to place it overlapping with seed3 (cancellable) or seed4 (conflict)
    seed3_start_min = _time_to_minutes(cancel_start)
    seed4_start_min = _time_to_minutes(conflict_start)
    # Pick the target that falls closest to the personal event's time_range
    target_start = seed3_start_min if abs(seed3_start_min - p0["time_range"][0]) <= abs(seed4_start_min - p0["time_range"][0]) else seed4_start_min
    # Offset slightly so it overlaps (start 15min before the target ends or at target start)
    p0_start_min = max(p0["time_range"][0], target_start)
    p0_start_min = (p0_start_min // 15) * 15
    p0_start = _round_time(p0_start_min)
    p0_end = _add_minutes(p0_start, p0["duration"])
    personal_events.append({
        "id": "personal0",
        "title": p0["title"],
        "date": today,
        "start_time": p0_start,
        "end_time": p0_end,
        "duration_minutes": p0["duration"],
        "type": "personal",
        "immovable": True,
        "attendees": [],
    })

    # Second personal event: place randomly within its time_range
    p1 = picked_personal[1]
    p1_start_min = rng.randint(p1["time_range"][0], p1["time_range"][1] - p1["duration"])
    p1_start_min = (p1_start_min // 15) * 15
    p1_start = _round_time(p1_start_min)
    p1_end = _add_minutes(p1_start, p1["duration"])
    personal_events.append({
        "id": "personal1",
        "title": p1["title"],
        "date": today,
        "start_time": p1_start,
        "end_time": p1_end,
        "duration_minutes": p1["duration"],
        "type": "personal",
        "immovable": True,
        "attendees": [],
    })

    # Personal event update: shift first personal event 1 hour earlier
    p0_update_start_min = p0_start_min - 60
    p0_update_start = _round_time(p0_update_start_min)
    p0_update_end = _add_minutes(p0_update_start, p0["duration"])
    personal_event_update = {
        "event_id": "personal0",
        "new_start_time": p0_update_start,
        "new_end_time": p0_update_end,
    }

    # --- Randomized interrupt steps ---
    interrupt_ranges = {
        "new_meeting": (2, 5),
        "inbox_update": (4, 7),
        "cancellation": (5, 8),
        "availability_change": (6, 9),
        "reschedule_request": (8, 11),
        "personal_event_change": (9, 12),
        "policy_change": (11, 14),
    }
    interrupt_steps: dict[str, int] = {}
    used_steps: set[int] = set()
    for itype, (lo, hi) in interrupt_ranges.items():
        step = rng.randint(lo, hi)
        while step in used_steps:
            step += 1
        used_steps.add(step)
        interrupt_steps[itype] = step

    # Contradiction / mid-episode message (uses randomized inbox_update step)
    mid_episode_message = {
        **contra,
        "id": "msg_contra",
        "received_at_step": interrupt_steps.get("inbox_update", 5),
        "read": False,
        "replied": False,
        "reply_text": "",
    }

    # --- Boss request messages (inbox-driven tasks) ---
    for idx, tmpl in enumerate(BOSS_REQUEST_TEMPLATES):
        initial_messages.append({
            "type": tmpl["type"],
            "from": tmpl["from"],
            "subject": tmpl["subject"].format(cancel_title=cancel_title),
            "body": tmpl["body"].format(cancel_title=cancel_title),
            "core_ask": tmpl["core_ask"],
            "reply_keywords": tmpl["reply_keywords"],
            "driven_flag": tmpl["driven_flag"],
            "id": f"boss{idx}",
            "received_at_step": 0,
            "read": False,
            "replied": False,
            "reply_text": "",
        })

    return EpisodeConfig(
        initial_events=events,
        lunch_exempt_ids={"seed2"},
        morning_meeting_title=morning_title,
        lunch_meeting_title=lunch_title,
        cancellable_title=cancel_title,
        conflicting_title=conflict_title,
        cancellable_attendees=cancel_attendees,
        initial_messages=initial_messages,
        mid_episode_message=mid_episode_message,
        personal_events=personal_events,
        personal_event_update=personal_event_update,
        interrupt_steps=interrupt_steps,
    )
