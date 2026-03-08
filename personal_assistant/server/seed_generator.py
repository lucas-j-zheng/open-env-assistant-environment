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

    return EpisodeConfig(
        initial_events=events,
        lunch_exempt_ids={"seed2"},
        morning_meeting_title=morning_title,
        lunch_meeting_title=lunch_title,
        cancellable_title=cancel_title,
        conflicting_title=conflict_title,
        cancellable_attendees=cancel_attendees,
    )
