import json
import sys
from pathlib import Path
from datetime import datetime, timedelta


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "personal_assistant"))

from models import CalendarAction  # noqa: E402
from server.personal_assistant_environment import PersonalAssistantEnvironment  # noqa: E402


def _step(env: PersonalAssistantEnvironment, tool: str | None = None, args: dict | None = None, instruction: str | None = None):
    if instruction is None:
        instruction = json.dumps({"tool": tool, "args": args or {}})
    return env.step(CalendarAction(instruction=instruction))


def _advance_to(env, target_step):
    """Advance env to target_step by issuing no-op check_conflicts calls."""
    while env._state.step_count < target_step:
        _step(env, tool="check_conflicts")


def _get_interrupt_step(env, interrupt_type):
    """Get the actual step number for a given interrupt type."""
    return env._config.interrupt_steps.get(interrupt_type)


def _first_free_start(env: PersonalAssistantEnvironment, target_date: str, duration_minutes: int) -> str:
    day_events = sorted([e for e in env._events if e["date"] == target_date], key=lambda x: x["start_time"])
    current = datetime.strptime(f"{target_date} 08:00", "%Y-%m-%d %H:%M")
    end_of_day = datetime.strptime(f"{target_date} 18:00", "%Y-%m-%d %H:%M")
    for event in day_events:
        event_start = datetime.strptime(f"{target_date} {event['start_time']}", "%Y-%m-%d %H:%M")
        if (event_start - current).total_seconds() / 60 >= duration_minutes:
            return current.strftime("%H:%M")
        current = max(current, datetime.strptime(f"{target_date} {event['end_time']}", "%Y-%m-%d %H:%M"))
    if (end_of_day - current).total_seconds() / 60 >= duration_minutes:
        return current.strftime("%H:%M")
    raise AssertionError(f"No free slot of {duration_minutes} minutes found on {target_date}")


def test_standup_requires_this_week_negotiation_and_is_revocable():
    env = PersonalAssistantEnvironment()
    env.reset(seed=0)

    # First attempt with 30min triggers negotiation rejection from Bob.
    obs = _step(
        env,
        tool="create_event",
        args={
            "title": "Team Standup Next Week",
            "date": "next tuesday",
            "start_time": "11:30",
            "duration_minutes": 30,
            "attendees": "Alice,Bob",
        },
    )
    assert "standup_scheduled" not in obs.flags_found
    assert "NOT created" in obs.output

    # Retry with <=20 min resolves negotiation, but next week still shouldn't count.
    obs = _step(
        env,
        tool="create_event",
        args={
            "title": "Team Standup Next Week",
            "date": "next tuesday",
            "start_time": "11:30",
            "duration_minutes": 20,
            "attendees": "Alice,Bob",
        },
    )
    assert "standup_scheduled" not in obs.flags_found  # next week doesn't count

    # Now create one this week — negotiation already resolved so it goes through.
    obs = _step(
        env,
        tool="create_event",
        args={
            "title": "Team Standup This Week",
            "date": "friday",
            "start_time": "10:30",
            "duration_minutes": 20,
            "attendees": "Alice,Bob",
        },
    )
    assert "standup_scheduled" in obs.flags_found

    # Must be revocable if the qualifying event is removed.
    obs = _step(env, tool="delete_event", args={"title": "Team Standup This Week"})
    assert "standup_scheduled" not in obs.flags_found


def test_reward_decreases_when_focus_time_becomes_invalid():
    env = PersonalAssistantEnvironment()
    env.reset(seed=0)
    today = env._resolve_date("today")
    start_time = _first_free_start(env, today, 60)

    obs = _step(
        env,
        tool="create_event",
        args={
            "title": "Focus Time",
            "date": "today",
            "start_time": start_time,
            "duration_minutes": 60,
            "attendees": "",
        },
    )
    assert "focus_time_booked" in obs.flags_found
    reward_before = obs.reward

    # Move focus block into an overlap so completion and reward drop.
    focus_event = next(e for e in env._events if e["title"] == "Focus Time")
    overlap_target = next(
        e for e in env._events
        if e.get("id") != focus_event.get("id") and e["date"] == today
    )
    obs = _step(
        env,
        tool="edit_event",
        args={
            "title": "Focus Time",
            "new_start_time": overlap_target["start_time"],
        },
    )
    assert "focus_time_booked" not in obs.flags_found
    assert obs.reward < reward_before


def test_interrupts_can_revoke_flags_on_non_tool_steps():
    env = PersonalAssistantEnvironment()
    env.reset(seed=0)
    today = env._resolve_date("today")

    # Resolve ALL overlapping events today (work-work and work-personal).
    for _ in range(10):
        today_events = sorted(
            [e for e in env._events if e["date"] == today],
            key=lambda x: x["start_time"],
        )
        overlap_found = False
        for i in range(len(today_events) - 1):
            if today_events[i]["end_time"] > today_events[i + 1]["start_time"]:
                # Move the work event (personal events are immovable)
                to_move = today_events[i + 1] if today_events[i].get("type") == "personal" else today_events[i]
                if to_move.get("immovable"):
                    to_move = today_events[i] if today_events[i + 1].get("type") == "personal" else today_events[i + 1]
                move_to = _first_free_start(env, today, to_move["duration_minutes"])
                _step(env, tool="resolve_conflict", args={
                    "event_title": to_move["title"],
                    "new_start_time": move_to,
                })
                overlap_found = True
                break
        if not overlap_found:
            break

    obs = _step(env, tool="check_conflicts")
    assert "conflicts_resolved" in obs.flags_found

    # Advance to just before the new_meeting (CEO sync) interrupt step.
    ceo_step = _get_interrupt_step(env, "new_meeting")
    _advance_to(env, ceo_step - 1)

    # The next step triggers the CEO sync interrupt which injects a new event that
    # can re-introduce a conflict. Plain text instruction still runs completions.
    obs = _step(env, instruction="hello")
    assert "conflicts_resolved" not in obs.flags_found


def test_reset_initializes_rewards_from_current_state():
    env = PersonalAssistantEnvironment()
    obs = env.reset(seed=0)

    expected_flags = len(obs.flags_found)
    expected_reward = expected_flags / len(env.TASKS)
    expected_pending = len(env.TASKS) - expected_flags

    assert obs.reward == expected_reward
    assert obs.pending_tasks == expected_pending
    assert env.state.tasks_completed == expected_flags

    # A non-mutating first step should preserve reward/flags when no interrupts fire.
    step_obs = _step(env, tool="check_conflicts", args={})
    assert sorted(step_obs.flags_found) == sorted(obs.flags_found)
    assert step_obs.reward == obs.reward
    assert step_obs.pending_tasks == obs.pending_tasks


def test_get_task_list_still_supported_for_compat():
    env = PersonalAssistantEnvironment()
    env.reset(seed=0)

    obs = _step(env, tool="get_task_list", args={})
    assert "Tasks to complete:" in obs.output
    assert "[TODO]" in obs.output or "[DONE]" in obs.output


def test_ceo_sync_accommodated_uses_any_qualifying_ceo_event():
    env = PersonalAssistantEnvironment()
    env.reset(seed=0)

    # Advance to the new_meeting interrupt step so the CEO sync is injected.
    ceo_step = _get_interrupt_step(env, "new_meeting")
    _advance_to(env, ceo_step)
    obs = _step(env, tool="check_conflicts", args={})
    assert "ceo_sync_accommodated" not in obs.flags_found

    # Make the original CEO event non-qualifying (16:00) and add another at 15:00.
    _step(env, tool="edit_event", args={"title": "CEO Sync", "new_start_time": "16:00"})
    _step(
        env,
        tool="create_event",
        args={
            "title": "CEO Sync Backup",
            "date": "today",
            "start_time": "15:00",
            "duration_minutes": 30,
            "attendees": "CEO",
        },
    )

    # Remove all overlaps against the 15:00-15:30 CEO slot so the backup can qualify.
    today = env._resolve_date("today")
    target_start, target_end = "15:00", "15:30"
    for _ in range(8):
        blockers = [
            e for e in env._events
            if e["date"] == today
            and "ceo" not in e.get("title", "").lower()
            and e["start_time"] < target_end
            and e["end_time"] > target_start
        ]
        if not blockers:
            break
        blocker = blockers[0]
        move_to = _first_free_start(env, today, blocker["duration_minutes"])
        _step(env, tool="resolve_conflict", args={"event_title": blocker["title"], "new_start_time": move_to})

    obs = _step(env, tool="check_conflicts", args={})
    assert "ceo_sync_accommodated" in obs.flags_found
