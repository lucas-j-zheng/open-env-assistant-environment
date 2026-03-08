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


def _create(env, title, date, start_time, duration_minutes=60, attendees=""):
    return _step(env, tool="create_event", args={
        "title": title, "date": date, "start_time": start_time,
        "duration_minutes": duration_minutes, "attendees": attendees,
    })


# ── Standup negotiation ──────────────────────────────────────────────


class TestStandupNegotiation:

    def test_first_attempt_rejected(self):
        env = PersonalAssistantEnvironment()
        env.reset(seed=9)
        obs = _create(env, "Team Standup", "today", "11:00", 60, "Alice,Bob")
        assert "NOT created" in obs.output
        assert "DECLINED by Bob" in obs.output

    def test_too_long_still_rejected(self):
        env = PersonalAssistantEnvironment()
        env.reset(seed=9)
        _create(env, "Team Standup", "today", "11:00", 60, "Alice,Bob")
        obs = _create(env, "Team Standup", "today", "11:00", 25, "Alice,Bob")
        assert "NOT created" in obs.output
        assert "Bob" in obs.output

    def test_short_enough_accepted(self):
        env = PersonalAssistantEnvironment()
        env.reset(seed=9)
        _create(env, "Team Standup", "today", "11:00", 60, "Alice,Bob")
        obs = _create(env, "Team Standup", "today", "11:00", 20, "Alice,Bob")
        assert "created" in obs.output.lower()
        assert "NOT created" not in obs.output

    def test_exact_boundary_20min_accepted(self):
        env = PersonalAssistantEnvironment()
        env.reset(seed=9)
        _create(env, "Team Standup", "today", "11:00", 60, "Alice,Bob")
        obs = _create(env, "Team Standup", "today", "11:00", 20, "Alice,Bob")
        assert "NOT created" not in obs.output
        assert env._negotiation_resolved.get("standup_negotiation") is True

    def test_15min_accepted(self):
        env = PersonalAssistantEnvironment()
        env.reset(seed=9)
        _create(env, "Team Standup", "today", "11:00", 60, "Alice,Bob")
        obs = _create(env, "Team Standup", "today", "11:00", 15, "Alice,Bob")
        assert "NOT created" not in obs.output

    def test_sets_flag_after_negotiation(self):
        env = PersonalAssistantEnvironment()
        env.reset(seed=9)
        _create(env, "Team Standup", "today", "11:00", 60, "Alice,Bob")
        obs = _create(env, "Team Standup", "today", "11:00", 15, "Alice,Bob")
        assert "standup_scheduled" in obs.flags_found

    def test_flag_not_set_without_negotiation(self):
        """Even if event params are perfect, flag requires negotiation resolved."""
        env = PersonalAssistantEnvironment()
        env.reset(seed=9)
        # Bypass negotiation by creating a non-matching title first, then one that matches
        obs = _create(env, "Daily Sync", "today", "11:00", 20, "Alice,Bob")
        # "standup" not in title so no negotiation triggered, but also no flag
        assert "standup_scheduled" not in obs.flags_found

    def test_max_rounds_exceeded(self):
        env = PersonalAssistantEnvironment()
        env.reset(seed=9)
        # 3 attempts all too long -> negotiation fails on 4th
        _create(env, "Team Standup", "today", "11:00", 60, "Alice,Bob")
        _create(env, "Team Standup", "today", "11:00", 45, "Alice,Bob")
        _create(env, "Team Standup", "today", "11:00", 30, "Alice,Bob")
        obs = _create(env, "Team Standup", "today", "11:00", 25, "Alice,Bob")
        assert "NEGOTIATION FAILED" in obs.output
        assert env._negotiation_resolved.get("standup_negotiation") is False

    def test_hint_on_second_rejection(self):
        env = PersonalAssistantEnvironment()
        env.reset(seed=9)
        _create(env, "Team Standup", "today", "11:00", 60, "Alice,Bob")
        obs = _create(env, "Team Standup", "today", "11:00", 45, "Alice,Bob")
        assert "get_contact_preferences" in obs.output

    def test_non_matching_event_bypasses_negotiation(self):
        env = PersonalAssistantEnvironment()
        env.reset(seed=9)
        obs = _create(env, "Regular Meeting", "today", "11:00", 60, "Alice,Bob")
        assert "NOT created" not in obs.output
        assert "created" in obs.output.lower()

    def test_missing_attendee_bypasses_negotiation(self):
        env = PersonalAssistantEnvironment()
        env.reset(seed=9)
        obs = _create(env, "Team Standup", "today", "11:00", 60, "Alice")
        assert "NOT created" not in obs.output


# ── Kickoff negotiation ──────────────────────────────────────────────


class TestKickoffNegotiation:

    def _make_env(self, seed=9):
        env = PersonalAssistantEnvironment()
        env.reset(seed=seed)
        return env

    def test_first_attempt_rejected_by_eve(self):
        env = self._make_env()
        obs = _create(env, "Project Kickoff", "tuesday", "11:00", 60, "Alice,Bob,Eve")
        assert "NOT created" in obs.output
        assert "DECLINED by Eve" in obs.output

    def test_duration_45_with_buffer_accepted_round1(self):
        """45 min + morning + buffer should pass Eve's round and skip Alice's."""
        env = self._make_env()
        _create(env, "Project Kickoff", "tuesday", "11:00", 60, "Alice,Bob,Eve")
        obs = _create(env, "Project Kickoff", "tuesday", "11:00", 45, "Alice,Bob,Eve")
        assert "NOT created" not in obs.output
        assert env._negotiation_resolved.get("kickoff_negotiation") is True

    def test_duration_still_too_long_rejected(self):
        env = self._make_env()
        _create(env, "Project Kickoff", "tuesday", "11:00", 60, "Alice,Bob,Eve")
        obs = _create(env, "Project Kickoff", "tuesday", "11:00", 50, "Alice,Bob,Eve")
        assert "NOT created" in obs.output

    def test_afternoon_triggers_alice_rejection(self):
        """45 min afternoon should pass Eve's check but trigger Alice's morning preference."""
        env = self._make_env()
        _create(env, "Project Kickoff", "thursday", "14:00", 60, "Alice,Bob,Eve")
        obs = _create(env, "Project Kickoff", "thursday", "14:00", 45, "Alice,Bob,Eve")
        assert "NOT created" in obs.output
        assert "DECLINED by Alice" in obs.output

    def test_afternoon_then_morning_accepted(self):
        """After Alice rejects afternoon, moving to morning should succeed."""
        env = self._make_env()
        _create(env, "Project Kickoff", "thursday", "14:00", 60, "Alice,Bob,Eve")
        _create(env, "Project Kickoff", "thursday", "14:00", 45, "Alice,Bob,Eve")
        obs = _create(env, "Project Kickoff", "thursday", "11:00", 45, "Alice,Bob,Eve")
        assert "NOT created" not in obs.output
        assert env._negotiation_resolved.get("kickoff_negotiation") is True

    def test_no_buffer_rejected(self):
        """Eve needs 30 min buffer — placing kickoff adjacent to her busy block should fail."""
        env = self._make_env()
        # Eve is busy 08:00-10:00 on tuesday; placing at 10:00 means 0 buffer before
        obs = _create(env, "Project Kickoff", "tuesday", "10:00", 45, "Alice,Bob,Eve")
        assert "NOT created" in obs.output

    def test_no_buffer_retry_with_buffer_accepted(self):
        env = self._make_env()
        _create(env, "Project Kickoff", "tuesday", "10:00", 45, "Alice,Bob,Eve")
        # 10:30 gives 30 min buffer from Eve's 08:00-10:00
        obs = _create(env, "Project Kickoff", "tuesday", "10:30", 45, "Alice,Bob,Eve")
        assert "NOT created" not in obs.output

    def test_sets_flag_after_negotiation(self):
        env = self._make_env()
        _create(env, "Project Kickoff", "tuesday", "11:00", 60, "Alice,Bob,Eve")
        obs = _create(env, "Project Kickoff", "tuesday", "11:00", 45, "Alice,Bob,Eve")
        assert "kickoff_scheduled" in obs.flags_found

    def test_flag_not_set_without_negotiation(self):
        env = self._make_env()
        obs = _create(env, "Project Launch", "tuesday", "11:00", 45, "Alice,Bob,Eve")
        assert "kickoff_scheduled" not in obs.flags_found

    def test_max_rounds_exceeded(self):
        env = self._make_env()
        _create(env, "Project Kickoff", "tuesday", "11:00", 60, "Alice,Bob,Eve")
        _create(env, "Project Kickoff", "tuesday", "11:00", 55, "Alice,Bob,Eve")
        _create(env, "Project Kickoff", "tuesday", "11:00", 50, "Alice,Bob,Eve")
        _create(env, "Project Kickoff", "tuesday", "11:00", 48, "Alice,Bob,Eve")
        obs = _create(env, "Project Kickoff", "tuesday", "11:00", 46, "Alice,Bob,Eve")
        assert "NEGOTIATION FAILED" in obs.output


# ── Cross-scenario isolation ─────────────────────────────────────────


class TestNegotiationIsolation:

    def test_standup_and_kickoff_independent(self):
        """Resolving standup negotiation does not affect kickoff."""
        env = PersonalAssistantEnvironment()
        env.reset(seed=9)
        _create(env, "Team Standup", "today", "11:00", 60, "Alice,Bob")
        _create(env, "Team Standup", "today", "11:00", 15, "Alice,Bob")
        assert env._negotiation_resolved.get("standup_negotiation") is True
        assert "kickoff_negotiation" not in env._negotiation_resolved

    def test_resolved_scenario_not_retriggered(self):
        """Once standup negotiation is resolved, new standup events skip negotiation."""
        env = PersonalAssistantEnvironment()
        env.reset(seed=9)
        _create(env, "Team Standup", "today", "11:00", 60, "Alice,Bob")
        _create(env, "Team Standup", "today", "11:00", 15, "Alice,Bob")
        obs = _create(env, "Another Standup", "tomorrow", "11:00", 60, "Alice,Bob")
        assert "NOT created" not in obs.output

    def test_failed_scenario_can_be_retried(self):
        """After failure, standup negotiation should be triggerable again."""
        env = PersonalAssistantEnvironment()
        env.reset(seed=9)

        # Exhaust attempts (max_rounds=3, fails on the 4th call).
        _create(env, "Team Standup", "today", "11:00", 60, "Alice,Bob")
        _create(env, "Team Standup", "today", "11:00", 45, "Alice,Bob")
        _create(env, "Team Standup", "today", "11:00", 30, "Alice,Bob")
        failed = _create(env, "Team Standup", "today", "11:00", 25, "Alice,Bob")
        assert "NEGOTIATION FAILED" in failed.output
        assert env._negotiation_resolved.get("standup_negotiation") is False

        # Retry should start a fresh negotiation cycle (first attempt rejected again).
        retry_1 = _create(env, "Team Standup Retry", "today", "11:00", 20, "Alice,Bob")
        assert "NOT created" in retry_1.output
        assert "DECLINED by Bob" in retry_1.output

        # Second retry attempt with valid duration can then resolve successfully.
        retry_2 = _create(env, "Team Standup Retry", "today", "11:00", 20, "Alice,Bob")
        assert "NOT created" not in retry_2.output
        assert env._negotiation_resolved.get("standup_negotiation") is True

    def test_reset_clears_negotiation_state(self):
        env = PersonalAssistantEnvironment()
        env.reset(seed=9)
        _create(env, "Team Standup", "today", "11:00", 60, "Alice,Bob")
        _create(env, "Team Standup", "today", "11:00", 15, "Alice,Bob")
        assert env._negotiation_resolved.get("standup_negotiation") is True

        env.reset(seed=9)
        assert env._negotiation_resolved == {}
        assert env._active_negotiations == {}

    def test_different_seeds_same_behavior(self):
        """Negotiation triggers are deterministic regardless of seed."""
        for seed in [0, 5, 9, 12]:
            env = PersonalAssistantEnvironment()
            env.reset(seed=seed)
            obs = _create(env, "Team Standup", "today", "11:00", 60, "Alice,Bob")
            assert "NOT created" in obs.output, f"seed={seed} didn't reject"
            assert "Bob" in obs.output, f"seed={seed} missing Bob"
