"""Tests for schema drift features: availability change + description policy."""

import json
import re
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


def _advance_to(env, target_step):
    """Advance env to target_step by issuing no-op get_task_list calls."""
    while env._state.step_count < target_step:
        _step(env, tool="get_task_list")


# ── Availability drift tests ──


def test_bob_availability_changes_at_step_7():
    env = PersonalAssistantEnvironment()
    env.reset(seed=0)

    # Before step 7, Bob's Wednesday has limited blocks
    before = list(env._attendee_schedules["Bob"]["wednesday"])

    _advance_to(env, 7)

    after = env._attendee_schedules["Bob"]["wednesday"]
    assert len(after) > len(before), "Bob's Wednesday blocks should have grown"
    # The new block (13:00, 17:00) should be present
    assert ("13:00", "17:00") in after


def test_check_availability_returns_different_data_after_drift():
    env = PersonalAssistantEnvironment()
    env.reset(seed=0)

    obs_before = _step(env, tool="check_availability", args={"person": "Bob", "date": "wednesday"})
    output_before = obs_before.output

    _advance_to(env, 7)

    obs_after = _step(env, tool="check_availability", args={"person": "Bob", "date": "wednesday"})
    output_after = obs_after.output

    assert output_before != output_after, "Availability output should differ after drift"


def test_existing_bob_wednesday_event_invalidated():
    """Place an event with Bob on Wednesday afternoon, trigger drift, flag should NOT be set."""
    env = PersonalAssistantEnvironment()
    env.reset(seed=0)

    # Create a Bob event on Wednesday at 14:00 (will conflict after drift)
    _step(env, tool="create_event", args={
        "title": "Bob Wednesday Meeting",
        "date": "wednesday",
        "start_time": "14:00",
        "duration_minutes": 60,
        "attendees": "Bob",
    })

    _advance_to(env, 7)

    obs = _step(env, tool="get_task_list")
    assert "availability_drift_handled" not in obs.flags_found


def test_availability_drift_handled_flag():
    """Fix conflicting event after drift → flag sets."""
    env = PersonalAssistantEnvironment()
    env.reset(seed=0)

    # Create a Bob event on Wednesday afternoon
    _step(env, tool="create_event", args={
        "title": "Bob Wednesday Meeting",
        "date": "wednesday",
        "start_time": "14:00",
        "duration_minutes": 60,
        "attendees": "Bob",
    })

    _advance_to(env, 7)

    # Flag should not be set yet
    obs = _step(env, tool="get_task_list")
    assert "availability_drift_handled" not in obs.flags_found

    # Fix: move event to morning
    obs = _step(env, tool="edit_event", args={
        "title": "Bob Wednesday Meeting",
        "new_start_time": "10:00",
    })
    assert "availability_drift_handled" in obs.flags_found


def test_availability_drift_flag_revocable():
    """Add new conflicting event after fix → flag flips back."""
    env = PersonalAssistantEnvironment()
    env.reset(seed=0)

    _advance_to(env, 7)

    # No Bob Wednesday events → flag should be set
    obs = _step(env, tool="get_task_list")
    assert "availability_drift_handled" in obs.flags_found

    # Create conflicting event
    obs = _step(env, tool="create_event", args={
        "title": "Late Bob Meeting",
        "date": "wednesday",
        "start_time": "15:00",
        "duration_minutes": 60,
        "attendees": "Bob",
    })
    assert "availability_drift_handled" not in obs.flags_found


# ── Description policy tests ──


def test_create_event_before_policy_no_warning():
    """Before step 12, creating a >30min event without description → no policy warning."""
    env = PersonalAssistantEnvironment()
    env.reset(seed=0)

    obs = _step(env, tool="create_event", args={
        "title": "Long Meeting",
        "date": "today",
        "start_time": "16:00",
        "duration_minutes": 60,
    })
    assert "description/agenda" not in obs.output.lower()
    assert "no description" not in obs.output.lower()


def test_constraint_totals_only_count_active_rules():
    env = PersonalAssistantEnvironment()
    env.reset(seed=0)

    before = _step(env, tool="check_constraint_violations")
    assert re.search(r"Hard constraints:\s+\d+/3 satisfied", before.output), before.output

    _advance_to(env, 12)
    after = _step(env, tool="check_constraint_violations")
    assert re.search(r"Hard constraints:\s+\d+/4 satisfied", after.output), after.output


def test_unhandled_interrupt_clears_after_handling():
    env = PersonalAssistantEnvironment()
    env.reset(seed=0)
    lunch_title = env._config.lunch_meeting_title

    # Advance to step 6 so cancellation interrupt fires.
    for _ in range(5):
        _step(env, tool="get_task_list")
    obs = _step(env, tool="get_task_list")  # step 6
    assert any(lunch_title in msg for msg in obs.unhandled_interrupts)

    # Handle interruption by deleting the cancelled lunch meeting.
    obs = _step(env, tool="delete_event", args={"title": lunch_title})
    assert "cancellation_handled" in obs.flags_found
    assert not any(lunch_title in msg for msg in obs.unhandled_interrupts)


def test_hard_constraints_flip_after_policy():
    """After step 12, >30min events without description cause hard constraint violation."""
    env = PersonalAssistantEnvironment()
    env.reset(seed=0)

    # Create a long event without description
    _step(env, tool="create_event", args={
        "title": "Long Planning",
        "date": "today",
        "start_time": "16:00",
        "duration_minutes": 60,
    })

    # Advance past step 12
    _advance_to(env, 12)

    obs = _step(env, tool="check_constraint_violations")
    assert "long_meetings_need_description" in obs.output or "no description" in obs.output.lower()


def test_create_event_after_policy_with_description_ok():
    """After step 12, creating >30min event WITH description → no violation for that event."""
    env = PersonalAssistantEnvironment()
    env.reset(seed=0)

    _advance_to(env, 12)

    obs = _step(env, tool="create_event", args={
        "title": "Described Meeting",
        "date": "today",
        "start_time": "16:00",
        "duration_minutes": 60,
        "description": "Discuss Q2 roadmap",
    })
    # The event itself should not trigger a description warning
    assert "no description" not in obs.output.lower()


def test_edit_event_adds_description():
    """Verify new_description param works on edit_event."""
    env = PersonalAssistantEnvironment()
    env.reset(seed=0)

    _step(env, tool="create_event", args={
        "title": "Undescribed",
        "date": "today",
        "start_time": "16:00",
        "duration_minutes": 60,
    })

    obs = _step(env, tool="edit_event", args={
        "title": "Undescribed",
        "new_description": "Added agenda",
    })
    assert "description: added" in obs.output

    event = next(e for e in env._events if e["title"] == "Undescribed")
    assert event["description"] == "Added agenda"


def test_description_policy_met_flag():
    """All long meetings have descriptions after policy → flag sets."""
    env = PersonalAssistantEnvironment()
    env.reset(seed=0)

    _advance_to(env, 12)

    # Add descriptions to all existing >30min events
    for e in env._events:
        if e.get("duration_minutes", 0) > 30 and not e.get("description", ""):
            _step(env, tool="edit_event", args={
                "title": e["title"],
                "new_description": "Agenda added",
            })

    obs = _step(env, tool="get_task_list")
    assert "description_policy_met" in obs.flags_found


def test_description_policy_met_revocable():
    """Creating a new long undescribed event after flag is set → flag revoked."""
    env = PersonalAssistantEnvironment()
    env.reset(seed=0)

    _advance_to(env, 12)

    # Fix all existing events
    for e in list(env._events):
        if e.get("duration_minutes", 0) > 30 and not e.get("description", ""):
            _step(env, tool="edit_event", args={
                "title": e["title"],
                "new_description": "Agenda",
            })

    obs = _step(env, tool="get_task_list")
    assert "description_policy_met" in obs.flags_found

    # Create new long event without description
    obs = _step(env, tool="create_event", args={
        "title": "New Long Meeting",
        "date": "today",
        "start_time": "17:00",
        "duration_minutes": 45,
    })
    assert "description_policy_met" not in obs.flags_found


def test_get_constraints_shows_new_rule_after_policy():
    """get_constraints output differs before vs after step 12."""
    env = PersonalAssistantEnvironment()
    env.reset(seed=0)

    obs_before = _step(env, tool="get_constraints")

    _advance_to(env, 12)

    obs_after = _step(env, tool="get_constraints")
    assert "description" not in obs_before.output.lower() or "agenda" not in obs_before.output.lower()
    assert "description" in obs_after.output.lower() or "agenda" in obs_after.output.lower()


def test_interrupt_messages_appear():
    """Both interrupt messages appear at the right steps."""
    env = PersonalAssistantEnvironment()
    env.reset(seed=0)

    # Step to 7
    _advance_to(env, 6)
    obs = _step(env, tool="get_task_list")  # step 7
    assert "INTERRUPT" in obs.output
    assert "Bob" in obs.output

    # Step to 12
    _advance_to(env, 11)
    obs = _step(env, tool="get_task_list")  # step 12
    assert "INTERRUPT" in obs.output
    assert "description" in obs.output.lower() or "agenda" in obs.output.lower()
