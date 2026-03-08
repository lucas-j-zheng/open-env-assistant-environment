import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "personal_assistant"))

from models import CalendarAction
from server.personal_assistant_environment import PersonalAssistantEnvironment


def _step(env, tool=None, args=None, instruction=None):
    if instruction is None:
        instruction = json.dumps({"tool": tool, "args": args or {}})
    return env.step(CalendarAction(instruction=instruction))


def _advance_to_step(env, target_step):
    """Advance the environment to a given step by calling check_conflicts repeatedly."""
    while env._state.step_count < target_step:
        _step(env, tool="check_conflicts")


def _get_interrupt_step(env, interrupt_type):
    """Get the actual step number for a given interrupt type."""
    return env._config.interrupt_steps.get(interrupt_type)


def _reply_to_msg(env, msg, text=None):
    """Reply to a message dict with appropriate keyword-containing text."""
    if text is None:
        # Build a reply that includes the first keyword and is >= 20 chars
        kw = msg["reply_keywords"][0]
        text = f"I have {kw} the request as discussed with the team today"
    return _step(env, tool="reply_message", args={"message_id": msg["id"], "reply": text})


# ── Tests ────────────────────────────────────────────────────────────


class TestInbox:

    def test_inbox_seeded_on_reset(self):
        """Reset with seed=0, call read_inbox, verify at least 3 messages with From:/Subject:."""
        env = PersonalAssistantEnvironment()
        env.reset(seed=0)
        obs = _step(env, tool="read_inbox")
        assert "From:" in obs.output
        assert "Subject:" in obs.output
        # At least 3 messages should be present
        assert obs.output.count("From:") >= 3

    def test_read_inbox_marks_read(self):
        """First read_inbox shows messages; second call with status=unread shows none."""
        env = PersonalAssistantEnvironment()
        env.reset(seed=0)
        obs1 = _step(env, tool="read_inbox")
        assert "From:" in obs1.output

        obs2 = _step(env, tool="read_inbox", args={"status": "unread"})
        assert "No unread messages" in obs2.output

    def test_reply_validates_keywords(self):
        """Reply without relevant keywords is rejected; reply with keywords succeeds."""
        env = PersonalAssistantEnvironment()
        env.reset(seed=0)
        _step(env, tool="read_inbox")

        msg = env._inbox[0]
        # Reply without any keyword — should be rejected
        bad_reply = "Hello there this is my reply message without any relevant words at all"
        obs_bad = _step(env, tool="reply_message", args={"message_id": msg["id"], "reply": bad_reply})
        assert "doesn't address" in obs_bad.output or "concern" in obs_bad.output.lower()

        # Reply with a proper keyword — should succeed
        kw = msg["reply_keywords"][0]
        good_reply = f"Thank you, I have {kw} everything as requested by your team"
        obs_good = _step(env, tool="reply_message", args={"message_id": msg["id"], "reply": good_reply})
        assert "Reply sent" in obs_good.output

    def test_reply_too_short_rejected(self):
        """A reply under 20 characters should be rejected."""
        env = PersonalAssistantEnvironment()
        env.reset(seed=0)
        _step(env, tool="read_inbox")

        msg = env._inbox[0]
        obs = _step(env, tool="reply_message", args={"message_id": msg["id"], "reply": "ok"})
        assert "too short" in obs.output.lower() or "minimum 20" in obs.output

    def test_inbox_cleared_flag(self):
        """Replying to all initial messages should set the inbox_cleared flag."""
        env = PersonalAssistantEnvironment()
        env.reset(seed=0)
        _step(env, tool="read_inbox")

        for msg in env._inbox:
            _reply_to_msg(env, msg)

        assert "inbox_cleared" in env._found

    def test_inbox_cleared_revocable(self):
        """After clearing inbox, the inbox_update interrupt adds a new message and revokes the flag."""
        env = PersonalAssistantEnvironment()
        env.reset(seed=0)
        inbox_step = _get_interrupt_step(env, "inbox_update")

        _step(env, tool="read_inbox")

        # Reply to all currently visible messages. As we reply, the step count
        # increases and msg_contra may become visible (its received_at_step is
        # the inbox_update step). We reply to everything visible so far, then
        # check if more messages became visible and reply to those too.
        replied_ids = set()
        for _ in range(20):  # safety bound
            visible = [m for m in env._inbox
                       if m.get("received_at_step", 0) <= env._state.step_count
                       and not m.get("replied")
                       and m["id"] not in replied_ids]
            if not visible:
                break
            msg = visible[0]
            _reply_to_msg(env, msg)
            replied_ids.add(msg["id"])

        assert "inbox_cleared" in env._found, \
            f"Expected inbox_cleared after replying to all visible messages, got: {env._found}"

        # Now advance past the inbox_update step if we haven't already.
        # If msg_contra already arrived and was replied to, we need to un-reply
        # it to test revocability. Instead, verify the flag was set and
        # test revocability by checking that if msg_contra were unreplied, the
        # flag would be revoked. We do this by verifying the flag IS set,
        # then we know the completion logic works; the revocability of inbox_cleared
        # is tested implicitly because we had to reply to msg_contra to get it set.
        # For an explicit revocability test: mark msg_contra as unreplied and re-check.
        contra_msg = next((m for m in env._inbox if m["id"] == "msg_contra"), None)
        if contra_msg and contra_msg.get("replied"):
            # msg_contra was already replied to. Un-reply it to test revocability.
            contra_msg["replied"] = False
            contra_msg["reply_text"] = ""
            # Do a step to trigger completion re-check
            obs = _step(env, tool="check_conflicts")
            assert "inbox_cleared" not in obs.flags_found, \
                "Flag should be revoked when msg_contra is unreplied"

            # Re-reply to restore the flag
            _reply_to_msg(env, contra_msg)
            assert "inbox_cleared" in env._found, \
                "Flag should be restored after replying to msg_contra"
        else:
            # msg_contra hasn't arrived yet, advance to trigger it
            _advance_to_step(env, inbox_step)
            assert "inbox_cleared" not in env._found, \
                "Flag should be revoked after msg_contra arrives"

            contra_msg = next(m for m in env._inbox if m["id"] == "msg_contra")
            _reply_to_msg(env, contra_msg)
            assert "inbox_cleared" in env._found, \
                "Flag should be restored after replying to msg_contra"

    def test_diplomatic_reply_flag(self):
        """Replying to the diplomatic message with 'understand' and 'discuss' sets the flag."""
        env = PersonalAssistantEnvironment()
        env.reset(seed=0)
        _step(env, tool="read_inbox")

        diplomatic_msg = next(m for m in env._inbox if m["type"] == "diplomatic")
        reply_text = "I understand your concerns and would like to discuss adjustments to the timeline"
        obs = _step(env, tool="reply_message", args={
            "message_id": diplomatic_msg["id"],
            "reply": reply_text,
        })
        assert "Reply sent" in obs.output
        assert "diplomatic_reply_sent" in env._found

    def test_client_request_updated_flag(self):
        """After inbox_update interrupt, replying to msg_contra sets client_request_updated."""
        env = PersonalAssistantEnvironment()
        env.reset(seed=0)

        # Advance to the inbox_update step to trigger the interrupt
        inbox_step = _get_interrupt_step(env, "inbox_update")
        _advance_to_step(env, inbox_step)

        # Read inbox to see the new message
        obs = _step(env, tool="read_inbox")
        assert "msg_contra" in obs.output or any(m["id"] == "msg_contra" for m in env._inbox)

        contra_msg = next(m for m in env._inbox if m["id"] == "msg_contra")
        kw = contra_msg["reply_keywords"][0]
        reply_text = f"The meeting has been {kw} per your latest request and confirmed"
        _step(env, tool="reply_message", args={
            "message_id": "msg_contra",
            "reply": reply_text,
        })

        assert "client_request_updated" in env._found

    def test_get_task_list_hides_inbox_driven_duplicates(self):
        """get_task_list should not repeat objectives already represented by inbox boss requests."""
        env = PersonalAssistantEnvironment()
        env.reset(seed=0)

        obs = _step(env, tool="get_task_list")
        visible_inbox_flags = {
            m.get("driven_flag")
            for m in env._inbox
            if m.get("received_at_step", 0) <= env._state.step_count and m.get("driven_flag")
        }
        assert visible_inbox_flags, "Expected inbox-driven flags to be present in seeded inbox messages"

        for flag in visible_inbox_flags:
            task_desc = next(t["description"] for t in env._tasks if t["flag"] == flag)
            assert task_desc not in obs.output

        assert "Inbox-driven requests hidden from this list" in obs.output
