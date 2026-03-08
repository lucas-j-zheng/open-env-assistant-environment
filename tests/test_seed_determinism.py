import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "personal_assistant"))

from server.personal_assistant_environment import PersonalAssistantEnvironment  # noqa: E402


def _snapshot(env: PersonalAssistantEnvironment) -> dict:
    return {
        "today": env._resolve_date("today"),
        "events": [
            (
                e["id"],
                e["title"],
                e["date"],
                e["start_time"],
                e["end_time"],
                e["duration_minutes"],
                tuple(e.get("attendees", [])),
            )
            for e in env._events
        ],
        "tasks": [t["description"] for t in env._tasks],
        "interrupts": [dict(i) for i in env._interrupts],
    }


def test_seed_reproducible_even_with_custom_episode_id():
    seed = 5

    env_default = PersonalAssistantEnvironment()
    env_default.reset(seed=seed)
    snap_default = _snapshot(env_default)

    env_custom = PersonalAssistantEnvironment()
    env_custom.reset(seed=seed, episode_id="custom-episode-id")
    snap_custom = _snapshot(env_custom)

    assert snap_custom == snap_default

