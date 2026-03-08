import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "personal_assistant"))

from models import CalendarAction  # noqa: E402
from server.personal_assistant_environment import PersonalAssistantEnvironment  # noqa: E402


def _step(env, tool=None, args=None, instruction=None):
    if instruction is None:
        instruction = json.dumps({"tool": tool, "args": args or {}})
    return env.step(CalendarAction(instruction=instruction))


def _advance_to_step(env, target_step):
    """Advance the environment to a target step by calling check_conflicts repeatedly."""
    obs = None
    while env._state.step_count < target_step:
        obs = _step(env, tool="check_conflicts")
    return obs


def _get_interrupt_step(env, interrupt_type):
    """Get the actual step number for a given interrupt type."""
    return env._config.interrupt_steps.get(interrupt_type)


def _find_personal_events(env):
    return [e for e in env._events if e.get("type") == "personal"]


def _find_work_events(env):
    return [e for e in env._events if e.get("type") != "personal"]


def _find_overlapping_pairs(personal_events, work_events):
    """Find (personal, work) pairs that overlap on the same date."""
    pairs = []
    for pe in personal_events:
        for we in work_events:
            if (pe["date"] == we["date"]
                    and pe["start_time"] < we["end_time"]
                    and pe["end_time"] > we["start_time"]):
                pairs.append((pe, we))
    return pairs


def _find_free_slot(env, date, duration_minutes, exclude_ranges, exclude_event_id=None):
    """Find a free time slot on the given date that avoids all exclude_ranges and existing events.

    exclude_event_id: skip this event when building the busy list (the event being moved).
    """
    from datetime import datetime, timedelta

    day_events = [e for e in env._events if e["date"] == date and e.get("id") != exclude_event_id]
    busy = [(e["start_time"], e["end_time"]) for e in day_events]
    busy.extend(exclude_ranges)
    busy.sort()

    for start_min in range(8 * 60, 18 * 60 - duration_minutes + 1, 15):
        start = f"{start_min // 60:02d}:{start_min % 60:02d}"
        end_dt = datetime.strptime(f"{date} {start}", "%Y-%m-%d %H:%M") + timedelta(minutes=duration_minutes)
        end = end_dt.strftime("%H:%M")
        if end > "18:00":
            continue
        conflict = False
        for bs, be in busy:
            if start < be and end > bs:
                conflict = True
                break
        if not conflict:
            return start
    return None


# ── 1. Personal events are seeded ────────────────────────────────────


class TestPersonalEventsSeeded:

    def test_personal_events_seeded(self):
        """Reset with seed=0 should produce at least 2 personal events."""
        env = PersonalAssistantEnvironment()
        env.reset(seed=0)
        personal = _find_personal_events(env)
        assert len(personal) >= 2, f"Expected >= 2 personal events, got {len(personal)}: {[e['title'] for e in personal]}"
        for pe in personal:
            assert pe.get("type") == "personal"
            assert pe.get("immovable") is True


# ── 2. check_personal_calendar tool ──────────────────────────────────


class TestCheckPersonalCalendar:

    def test_check_personal_calendar_tool(self):
        """The check_personal_calendar tool should show IMMOVABLE personal events."""
        env = PersonalAssistantEnvironment()
        env.reset(seed=0)
        obs = _step(env, tool="check_personal_calendar")
        assert "IMMOVABLE" in obs.output
        personal = _find_personal_events(env)
        for pe in personal:
            assert pe["title"] in obs.output, f"Personal event '{pe['title']}' not shown in output"


# ── 3. Personal event cannot be edited ───────────────────────────────


class TestPersonalEventCannotBeEdited:

    def test_personal_event_cannot_be_edited(self):
        """Editing a personal/immovable event should be rejected."""
        env = PersonalAssistantEnvironment()
        env.reset(seed=0)
        personal = _find_personal_events(env)
        assert personal, "No personal events found"
        target = personal[0]
        obs = _step(env, tool="edit_event", args={
            "title": target["title"],
            "new_start_time": "08:00",
        })
        assert "immovable" in obs.output.lower() or "cannot" in obs.output.lower(), \
            f"Expected rejection for editing immovable event, got: {obs.output}"


# ── 4. Personal event cannot be deleted ──────────────────────────────


class TestPersonalEventCannotBeDeleted:

    def test_personal_event_cannot_be_deleted(self):
        """Deleting a personal/immovable event should be rejected."""
        env = PersonalAssistantEnvironment()
        env.reset(seed=0)
        personal = _find_personal_events(env)
        assert personal, "No personal events found"
        target = personal[0]
        obs = _step(env, tool="delete_event", args={"title": target["title"]})
        assert "immovable" in obs.output.lower() or "cannot" in obs.output.lower(), \
            f"Expected rejection for deleting immovable event, got: {obs.output}"
        # Event should still exist
        still_exists = any(e["id"] == target["id"] for e in env._events)
        assert still_exists, "Personal event was deleted despite being immovable"

    def test_duplicate_title_deletes_only_movable_events(self):
        """If title matches both personal and work events, only work events should be deleted."""
        env = PersonalAssistantEnvironment()
        env.reset(seed=0)
        personal = _find_personal_events(env)
        assert personal, "No personal events found"
        target = personal[0]

        _step(env, tool="create_event", args={
            "title": target["title"],
            "date": "today",
            "start_time": "17:00",
            "duration_minutes": 30,
            "attendees": "Alice",
        })

        obs = _step(env, tool="delete_event", args={"title": target["title"]})
        assert "Deleted" in obs.output
        assert "Kept" in obs.output

        # Personal copy remains.
        assert any(e.get("id") == target["id"] for e in env._events)
        # Any movable copies with the same title are removed.
        assert not any(
            e["title"].lower() == target["title"].lower() and not e.get("immovable")
            for e in env._events
        )


# ── 5. Personal event cannot be resolve_conflicted ───────────────────


class TestPersonalEventCannotBeResolveConflicted:

    def test_personal_event_cannot_be_resolve_conflicted(self):
        """Using resolve_conflict on a personal event should be rejected."""
        env = PersonalAssistantEnvironment()
        env.reset(seed=0)
        personal = _find_personal_events(env)
        assert personal, "No personal events found"
        target = personal[0]
        obs = _step(env, tool="resolve_conflict", args={
            "event_title": target["title"],
            "new_start_time": "08:00",
        })
        assert "immovable" in obs.output.lower() or "cannot" in obs.output.lower(), \
            f"Expected rejection for moving immovable event, got: {obs.output}"
        # Event time should be unchanged
        event = next(e for e in env._events if e["id"] == target["id"])
        assert event["start_time"] == target["start_time"], "Personal event was moved despite being immovable"


# ── 6. Work/life conflict resolved ──────────────────────────────────


class TestWorkLifeConflictResolved:

    def test_work_life_conflict_resolved(self):
        """Resolving all personal/work overlaps should set the work_life_conflicts_resolved flag."""
        env = PersonalAssistantEnvironment()
        env.reset(seed=0)

        personal = _find_personal_events(env)
        work = _find_work_events(env)
        overlaps = _find_overlapping_pairs(personal, work)
        assert overlaps, "Seed 0 should produce at least one personal/work overlap"

        # Resolve overlaps iteratively. Interrupts (e.g. CEO Sync at step 3) may
        # inject new events that create additional overlaps, so we keep resolving
        # until a check_conflicts call shows the flag set (meaning no overlaps exist
        # at the point when completions are checked, after interrupts fire).
        obs = None
        for _ in range(15):  # safety bound
            personal = _find_personal_events(env)
            work = _find_work_events(env)
            overlaps = _find_overlapping_pairs(personal, work)
            if not overlaps:
                # No overlaps visible — do a step to check flag (may trigger interrupt)
                obs = _step(env, tool="check_conflicts")
                if "work_life_conflicts_resolved" in obs.flags_found:
                    break
                # Interrupt may have injected a new event; loop will re-check
                continue
            pe, we = overlaps[0]
            personal_ranges = [(p["start_time"], p["end_time"]) for p in personal]
            free_start = _find_free_slot(env, we["date"], we["duration_minutes"],
                                         personal_ranges, exclude_event_id=we["id"])
            assert free_start is not None, f"Could not find a free slot for '{we['title']}'"
            obs = _step(env, tool="resolve_conflict", args={
                "event_title": we["title"],
                "new_start_time": free_start,
            })
            assert "immovable" not in obs.output.lower(), \
                f"resolve_conflict should work on work events, got: {obs.output}"

        assert obs is not None
        # Final verification
        if "work_life_conflicts_resolved" not in obs.flags_found:
            obs = _step(env, tool="check_conflicts")
        assert "work_life_conflicts_resolved" in obs.flags_found, \
            f"Expected work_life_conflicts_resolved flag, got: {obs.flags_found}"


# ── 7. Work/life conflict flag is revocable ──────────────────────────


class TestWorkLifeConflictFlagRevocable:

    def test_work_life_conflict_flag_revocable(self):
        """Creating a new work/personal overlap should revoke the flag; removing it should restore it."""
        env = PersonalAssistantEnvironment()
        env.reset(seed=0)

        personal = _find_personal_events(env)
        work = _find_work_events(env)
        overlaps = _find_overlapping_pairs(personal, work)

        personal_ranges = [(pe["start_time"], pe["end_time"]) for pe in personal]

        # Step 1: Resolve all existing overlaps iteratively (interrupts may inject new events).
        # After each step (resolve or check), re-check for new overlaps since interrupts
        # can inject events that create new personal/work conflicts.
        obs = None
        for _ in range(20):
            personal = _find_personal_events(env)
            work = _find_work_events(env)
            overlaps = _find_overlapping_pairs(personal, work)
            if overlaps:
                pe, we = overlaps[0]
                personal_ranges = [(p["start_time"], p["end_time"]) for p in personal]
                free_start = _find_free_slot(env, we["date"], we["duration_minutes"],
                                             personal_ranges, exclude_event_id=we["id"])
                assert free_start is not None
                obs = _step(env, tool="resolve_conflict", args={
                    "event_title": we["title"],
                    "new_start_time": free_start,
                })
            else:
                # No overlaps: check if the flag is already set
                if obs is not None and "work_life_conflicts_resolved" in obs.flags_found:
                    break
                # Do a step to potentially trigger interrupts and re-evaluate
                obs = _step(env, tool="check_conflicts")
                # If the step introduced no new overlaps and flag is set, we're done
                personal = _find_personal_events(env)
                work = _find_work_events(env)
                if not _find_overlapping_pairs(personal, work) and "work_life_conflicts_resolved" in obs.flags_found:
                    break

        assert obs is not None
        assert "work_life_conflicts_resolved" in obs.flags_found, \
            "Flag should be set after resolving overlaps"

        # Step 2: Create a new work event that overlaps a personal event
        personal = _find_personal_events(env)
        target_pe = personal[0]
        obs = _step(env, tool="create_event", args={
            "title": "Overlapping Work Meeting",
            "date": target_pe["date"],
            "start_time": target_pe["start_time"],
            "duration_minutes": 30,
            "attendees": "Alice",
        })
        assert "work_life_conflicts_resolved" not in obs.flags_found, \
            "Flag should be revoked after creating overlapping work event"

        # Step 3: Delete the overlapping event
        _step(env, tool="delete_event", args={"title": "Overlapping Work Meeting"})

        # Resolve any interrupt-injected overlaps (e.g. CEO sync may have been added)
        for _ in range(10):
            personal = _find_personal_events(env)
            work = _find_work_events(env)
            overlaps = _find_overlapping_pairs(personal, work)
            if not overlaps:
                obs = _step(env, tool="check_conflicts")
                if "work_life_conflicts_resolved" in obs.flags_found:
                    break
            else:
                pe, we = overlaps[0]
                personal_ranges = [(p["start_time"], p["end_time"]) for p in personal]
                free_start = _find_free_slot(env, we["date"], we["duration_minutes"],
                                             personal_ranges, exclude_event_id=we["id"])
                assert free_start is not None
                obs = _step(env, tool="resolve_conflict", args={
                    "event_title": we["title"],
                    "new_start_time": free_start,
                })

        assert "work_life_conflicts_resolved" in obs.flags_found, \
            "Flag should be restored after removing overlapping work event"


# ── 8. Personal update at step 10 ───────────────────────────────────


class TestPersonalUpdateAtStep10:

    def test_personal_update_at_step_10(self):
        """At step 10, a personal event shifts time. Resolve new overlaps to earn personal_update_handled."""
        env = PersonalAssistantEnvironment()
        env.reset(seed=0)

        # Record personal0's original time
        personal0_before = next(e for e in env._events if e.get("id") == "personal0")
        original_start = personal0_before["start_time"]
        original_end = personal0_before["end_time"]

        # First resolve all initial work/life conflicts so we start clean
        for _ in range(10):
            personal = _find_personal_events(env)
            work = _find_work_events(env)
            overlaps = _find_overlapping_pairs(personal, work)
            if not overlaps:
                break
            pe, we = overlaps[0]
            personal_ranges = [(p["start_time"], p["end_time"]) for p in personal]
            free_start = _find_free_slot(env, we["date"], we["duration_minutes"],
                                         personal_ranges, exclude_event_id=we["id"])
            if free_start is not None:
                _step(env, tool="resolve_conflict", args={
                    "event_title": we["title"],
                    "new_start_time": free_start,
                })

        # Advance to the personal_event_change interrupt step
        personal_step = _get_interrupt_step(env, "personal_event_change")
        _advance_to_step(env, personal_step)

        # Verify personal0's time has changed
        personal0_after = next(e for e in env._events if e.get("id") == "personal0")
        time_changed = (personal0_after["start_time"] != original_start
                        or personal0_after["end_time"] != original_end)
        assert time_changed, \
            f"personal0 time should have changed at step 10. Before: {original_start}-{original_end}, After: {personal0_after['start_time']}-{personal0_after['end_time']}"

        # Resolve any new work/personal overlaps by moving work events
        for _ in range(10):
            updated_personal = _find_personal_events(env)
            updated_work = _find_work_events(env)
            new_overlaps = _find_overlapping_pairs(updated_personal, updated_work)
            if not new_overlaps:
                break
            pe, we = new_overlaps[0]
            updated_personal_ranges = [(p["start_time"], p["end_time"]) for p in updated_personal]
            free_start = _find_free_slot(env, we["date"], we["duration_minutes"],
                                         updated_personal_ranges, exclude_event_id=we["id"])
            if free_start is not None:
                _step(env, tool="resolve_conflict", args={
                    "event_title": we["title"],
                    "new_start_time": free_start,
                })

        # Trigger completion check
        obs = _step(env, tool="check_conflicts")
        assert "personal_update_handled" in obs.flags_found, \
            f"Expected personal_update_handled flag after resolving post-update overlaps, got: {obs.flags_found}"
