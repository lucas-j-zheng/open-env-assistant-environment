import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "personal_assistant"))

from models import CalendarAction  # noqa: E402
from server.personal_assistant_environment import PersonalAssistantEnvironment  # noqa: E402


def _step(env: PersonalAssistantEnvironment, tool: str | None = None, args: dict | None = None, instruction: str | None = None):
    if instruction is None:
        instruction = json.dumps({"tool": tool, "args": args or {}})
    return env.step(CalendarAction(instruction=instruction))


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

    obs = _step(
        env,
        tool="create_event",
        args={
            "title": "Focus Time",
            "date": "today",
            "start_time": "16:00",
            "duration_minutes": 60,
            "attendees": "",
        },
    )
    assert "focus_time_booked" in obs.flags_found
    reward_before = obs.reward

    # Move focus block into an overlap so completion and reward drop.
    obs = _step(
        env,
        tool="edit_event",
        args={
            "title": "Focus Time",
            "new_start_time": "14:45",
        },
    )
    assert "focus_time_booked" not in obs.flags_found
    assert obs.reward < reward_before


def test_interrupts_can_revoke_flags_on_non_tool_steps():
    env = PersonalAssistantEnvironment()
    env.reset(seed=0)

    # Resolve initial overlap first.
    obs = _step(env, tool="resolve_conflict", args={"event_title": "Old Project Review", "new_start_time": "13:00"})
    assert "conflicts_resolved" in obs.flags_found

    # Step 2 with a tool call keeps the flag.
    obs = _step(env, tool="get_task_list", args={})
    assert "conflicts_resolved" in obs.flags_found

    # Step 3 with plain text triggers CEO interrupt and can re-introduce a conflict.
    # Completion checks must still run and revoke the flag.
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
    step_obs = _step(env, tool="get_task_list", args={})
    assert sorted(step_obs.flags_found) == sorted(obs.flags_found)
    assert step_obs.reward == obs.reward
    assert step_obs.pending_tasks == obs.pending_tasks


def test_ceo_sync_accommodated_uses_any_qualifying_ceo_event():
    env = PersonalAssistantEnvironment()
    env.reset(seed=0)

    # Advance to step 3 so the CEO interrupt fires.
    _step(env, tool="get_task_list", args={})
    _step(env, tool="get_task_list", args={})
    obs = _step(env, tool="get_task_list", args={})
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

    # Remove overlap so the new 15:00 event qualifies.
    obs = _step(env, tool="resolve_conflict", args={"event_title": "Design Review", "new_start_time": "15:30"})
    assert "ceo_sync_accommodated" in obs.flags_found
