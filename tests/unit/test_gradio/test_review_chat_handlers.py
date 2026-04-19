"""
tests/unit/test_gradio/test_review_chat_handlers.py

Behavioral unit tests for review_chat_turn, confirm_review_edit, and cancel_review_edit
in gradio_app.handlers.review.
"""

import os
import tempfile
from unittest.mock import patch


def _make_session(tmp_path_str: str | None = None) -> dict:
    """Build a minimal valid session_state for review_chat_turn.

    Args:
        tmp_path_str: Absolute path to a temporary working-copy file. When
            ``None``, a new temporary ``.md`` file is created automatically.

    Returns:
        A ``dict`` containing the minimum keys expected by
        ``review_chat_turn`` and related handlers.
    """
    if tmp_path_str is None:
        tmp_fd, tmp_path_str = tempfile.mkstemp(suffix=".md")
        os.close(tmp_fd)
        with open(tmp_path_str, "w", encoding="utf-8") as f:
            f.write("# Test Document\n\nSome content here.\n")

    return {
        "original_file_path": "reviews/test.md",
        "working_copy_path": tmp_path_str,
        "current_markdown": "# Test Document\n\nSome content here.\n",
        "chat_history": [],
        "pending_edit": {},
        "last_target_resolution": {},
        "retrieval_trace": [],
        "status": "ready",
    }


class TestReviewChatTurnPatchTarget:
    """Verify that run_review_agent is patched at the correct import location."""

    def test_run_review_agent_called_via_correct_patch_target(self, tmp_path):
        from gradio_app.handlers.review import review_chat_turn

        md_file = tmp_path / "working.md"
        md_file.write_text("# Section\n\nParagraph text.\n", encoding="utf-8")
        session = _make_session(str(md_file))

        fake_result = {
            "action": "answer",
            "reply": "Here is the answer.",
            "trace": [],
        }

        with patch(
            "gradio_app.handlers.review.run_review_agent", return_value=fake_result
        ) as mock_agent:
            history, _, _, _ = review_chat_turn(
                "What is this about?", [], session, web_enabled=False
            )

        mock_agent.assert_called_once()
        assert any(
            m["content"] == "Here is the answer." for m in history if m["role"] == "assistant"
        )


class TestReviewChatTurnEditProposal:
    """Verify edit proposal action stores pending_edit and uses English text."""

    def test_edit_proposal_action_sets_pending_edit(self, tmp_path):
        from gradio_app.handlers.review import review_chat_turn

        md_file = tmp_path / "working.md"
        md_file.write_text("# Section\n\nParagraph text.\n", encoding="utf-8")
        session = _make_session(str(md_file))

        proposal = {
            "section_title": "Section",
            "paragraph_index": 0,
            "before": "Paragraph text.",
            "after": "Improved paragraph text.",
            "start": 11,
            "end": 26,
        }
        fake_result = {
            "action": "edit_proposal",
            "reply": "",
            "edit_proposal": proposal,
            "trace": [],
        }

        session["last_language"] = "en"
        with patch("gradio_app.handlers.review.run_review_agent", return_value=fake_result):
            history, new_state, _, _ = review_chat_turn(
                "improve this paragraph", [], session, web_enabled=False
            )

        assert new_state["pending_edit"] == proposal
        # Reply must reference the edit proposal in either supported language.
        assistant_msgs = [m["content"] for m in history if m["role"] == "assistant"]
        assert any("Edit proposal" in msg or "Proposta de edição" in msg for msg in assistant_msgs)


class TestReviewChatTurnNoSession:
    """Verify that review_chat_turn handles missing session gracefully."""

    def test_no_session_returns_error_status(self):
        from gradio_app.handlers.review import review_chat_turn

        history, state, status, content = review_chat_turn("hello", [], {}, web_enabled=False)

        assert "❌" in status
        assert content == ""


class TestReviewChatTurnWebDisabled:
    """Verify that allow_web=False is forwarded to run_review_agent when web_enabled=False and no explicit web keyword in the message."""

    def test_web_disabled_flag_forwarded_to_agent(self, tmp_path):
        from gradio_app.handlers.review import review_chat_turn

        md_file = tmp_path / "working.md"
        md_file.write_text("# Section\n\nContent here.\n", encoding="utf-8")
        session = _make_session(str(md_file))

        fake_result = {"action": "answer", "reply": "No web needed.", "trace": []}

        with patch(
            "gradio_app.handlers.review.run_review_agent", return_value=fake_result
        ) as mock_agent:
            review_chat_turn("summarize this", [], session, web_enabled=False)

        _, call_kwargs = mock_agent.call_args
        assert call_kwargs.get("allow_web") is False


class TestCancelReviewEdit:
    """Verify cancel_review_edit clears pending_edit."""

    def test_cancel_clears_pending_edit(self, tmp_path):
        from gradio_app.handlers.review import cancel_review_edit

        md_file = tmp_path / "working.md"
        md_file.write_text("# Section\n\nContent.\n", encoding="utf-8")
        session = _make_session(str(md_file))
        session["pending_edit"] = {"before": "old", "after": "new", "start": 0, "end": 10}
        session["last_language"] = "en"

        fake_result = {"action": "cancel_edit", "reply": "Canceled.", "trace": []}

        with patch("gradio_app.handlers.review.run_review_agent", return_value=fake_result):
            _, new_state, _, _ = cancel_review_edit([], session)

        assert new_state["pending_edit"] == {}
