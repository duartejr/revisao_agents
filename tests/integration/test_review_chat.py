"""
Integration tests for the interactive review chat flow through handlers.py.

These tests exercise start_review_session → review_chat_turn → confirm/cancel
with a mocked LLM + tools so no external services are needed.
"""

from __future__ import annotations

import os

# We need the handlers on sys.path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


SAMPLE_REVIEW = """\
## 1. Introduction

Climate models have evolved significantly in recent years.

### Referências desta seção
[1] Smith et al. (2024) Climate model evaluation.

## 2. Results

Chronos-2 outperforms LSTM in 7 out of 10 benchmarks.

The model shows particular strength in long-horizon forecasting.

### Referências desta seção
[1] Amazon (2024) Chronos-2 report.
[2] Brown (2023) Benchmark suite.
"""


@pytest.fixture
def review_dir(tmp_path):
    """Create a temporary reviews/ directory with a sample review file.
    Returns relative path 'reviews/test_review.md' (handler requires reviews/ prefix).
    """
    reviews = tmp_path / "reviews"
    reviews.mkdir()
    f = reviews / "test_review.md"
    f.write_text(SAMPLE_REVIEW, encoding="utf-8")
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    yield "reviews/test_review.md"  # relative path
    os.chdir(old_cwd)


def _fake_llm_response(content, tool_calls=None):
    """Create a lightweight fake LLM response object.

    Args:
        content: The text content of the response.
        tool_calls: Optional list of tool-call objects. Defaults to an empty list.

    Returns:
        A ``SimpleNamespace`` with ``content`` and ``tool_calls`` attributes.
    """
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])


def _mock_agent_answer(reply_text: str):
    """Create a mock for run_review_agent that returns an 'answer' action."""
    return {
        "reply": reply_text,
        "edit_proposal": None,
        "action": "answer",
        "trace": [],
    }


def _mock_agent_edit_proposal(section_title, para_idx, before, after):
    """Create a mock for run_review_agent that returns an edit proposal."""
    return {
        "reply": "Here is my edit proposal.",
        "edit_proposal": {
            "section_title": section_title,
            "paragraph_index": para_idx,
            "start": 0,
            "end": 10,
            "before": before,
            "after": after,
            "created_at": "2026-03-15T12:00:00",
        },
        "action": "edit_proposal",
        "trace": [],
    }


def _mock_agent_apply():
    return {
        "reply": "ACTION: APPLY_EDIT",
        "edit_proposal": None,
        "action": "apply_edit",
        "trace": [],
    }


def _mock_agent_cancel():
    return {
        "reply": "ACTION: CANCEL_EDIT",
        "edit_proposal": None,
        "action": "cancel_edit",
        "trace": [],
    }


# ── Session lifecycle ─────────────────────────────────────────────────────


class TestSessionLifecycle:
    def test_start_session_creates_working_copy(self, review_dir):
        from gradio_app.handlers import start_review_session

        history, state, status, content = start_review_session(
            review_dir,
            [],
            {},
        )
        assert "✅" in status
        assert state["working_copy_path"]
        assert os.path.exists(state["working_copy_path"])
        assert state["original_file_path"] == os.path.normpath(review_dir)
        assert "Introduction" in content

    def test_start_session_rejects_non_reviews_path(self, tmp_path):
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            from gradio_app.handlers import start_review_session

            # Create file outside reviews/
            f = tmp_path / "evil.md"
            f.write_text("bad")
            history, state, status, _ = start_review_session(
                "evil.md",
                [],
                {},
            )
            assert "❌" in status
        finally:
            os.chdir(old_cwd)


# ── Chat turn with agent ─────────────────────────────────────────────────


class TestChatTurn:
    def _start_session(self, review_dir):
        from gradio_app.handlers import start_review_session

        history, state, status, content = start_review_session(
            review_dir,
            [],
            {},
        )
        return history, state

    def test_answer_turn(self, review_dir: str) -> None:
        """Test that a simple question to the agent returns an answer and updates the chat history.
        This test verifies that when a user asks a question, the review_chat_turn handler correctly invokes
        the agent, receives a response, and updates the chat history with the assistant's reply. The test uses a mock agent response to simulate the LLM's behavior.

        Args:
            review_dir (str): The path to the review document used to initialize the session.

        Asserts:
            The chat history is updated with the assistant's reply, and the status indicates success.
        """
        from gradio_app.handlers import review_chat_turn

        history, state = self._start_session(review_dir)

        with patch(
            "gradio_app.handlers.review.run_review_agent",
            return_value=_mock_agent_answer("Here are the references:\n[1] Smith"),
        ):
            hist, st, status, md = review_chat_turn(
                "liste as referencias",
                history,
                state,
            )

        assert len(hist) > len(history)
        assert hist[-1]["role"] == "assistant"
        assert "Smith" in hist[-1]["content"]
        assert "✅" in status

    def test_edit_proposal_sets_pending(self, review_dir: str) -> None:
        """Test that when the agent returns an edit proposal, the pending_edit state is set and the proposal is included in the chat history.
        This test ensures that when the agent generates an edit proposal in response to a user message,
        the review_chat_turn handler correctly updates the session state to indicate that there is a pending edit, and that the content of the edit proposal is included in the assistant's reply in the chat history.

        Args:
            review_dir (str): The path to the review document used to initialize the session.

        Asserts:
            The session state has pending_edit set to True, the status indicates a pending edit, and
            the assistant's reply in the chat history contains the edit proposal details.
        S"""
        from gradio_app.handlers import review_chat_turn

        history, state = self._start_session(review_dir)

        with patch(
            "gradio_app.handlers.review.run_review_agent",
            return_value=_mock_agent_edit_proposal(
                "1. Introduction",
                0,
                "Climate models have evolved significantly in recent years.",
                "Climate models have evolved dramatically in the last decade.",
            ),
        ):
            hist, st, status, md = review_chat_turn(
                "edit first paragraph of section 1",
                history,
                state,
            )

        assert st.get("pending_edit")
        assert "🟡" in status
        assert "Edit proposal" in hist[-1]["content"]

    def test_confirm_edit_applies_and_clears_pending(self, review_dir):
        """
        Test that confirming an edit proposal successfully applies changes to
        the working copy and clears the pending edit state.

        Args:
            review_dir (str): Path to the original document directory used
                to initialize the review session.

        Returns:
            None: The test asserts that the pending edit flag is cleared and
            the modified text is present in the updated working copy.
        """
        from gradio_app.handlers import confirm_review_edit, review_chat_turn

        history, state = self._start_session(review_dir)

        # First: set a pending edit
        # Compute actual start/end from the working copy
        working = state["working_copy_path"]

        # Using context handler to safely read the working copy
        with open(working, encoding="utf-8") as f:
            md = f.read()

        # Find the introduction paragraph
        intro_start = md.index("Climate models have evolved")
        intro_end = intro_start + len("Climate models have evolved significantly in recent years.")

        with patch(
            "gradio_app.handlers.review.run_review_agent",
            return_value={
                "reply": "proposal",
                "edit_proposal": {
                    "section_title": "1. Introduction",
                    "paragraph_index": 0,
                    "start": intro_start,
                    "end": intro_end,
                    "before": "Climate models have evolved significantly in recent years.",
                    "after": "Climate models have evolved DRAMATICALLY.",
                    "created_at": "2026-03-15T12:00:00",
                },
                "action": "edit_proposal",
                "trace": [],
            },
        ):
            hist, st, _, _ = review_chat_turn("edit intro", history, state)

        assert st["pending_edit"]

        # Confirm
        with patch(
            "gradio_app.handlers.review.run_review_agent",
            return_value=_mock_agent_apply(),
        ):
            hist2, st2, status2, md2 = confirm_review_edit(hist, st)

        assert not st2.get("pending_edit")
        assert "✅" in status2 or "Edição aplicada" in hist2[-1]["content"]
        # Working copy should have the new text
        assert "DRAMATICALLY" in md2

    def test_original_file_unchanged_after_edit(self, review_dir):
        """
        Verify that the original source file remains unmodified even after
        an edit proposal is generated and confirmed in the working copy.

        Args:
            review_dir (str): Path to the original document being reviewed.

        Returns:
            None: The test asserts that the content of the original file
            matches its initial state at the end of the process.
        """
        from gradio_app.handlers import confirm_review_edit, review_chat_turn

        history, state = self._start_session(review_dir)

        # Using context handlers to ensure files are closed properly
        with open(review_dir, encoding="utf-8") as f:
            original_content = f.read()

        with open(state["working_copy_path"], encoding="utf-8") as f:
            md = f.read()

        intro_start = md.index("Climate models have evolved")
        intro_end = intro_start + len("Climate models have evolved significantly in recent years.")

        with patch(
            "gradio_app.handlers.review.run_review_agent",
            return_value={
                "reply": "proposal",
                "edit_proposal": {
                    "section_title": "1. Introduction",
                    "paragraph_index": 0,
                    "start": intro_start,
                    "end": intro_end,
                    "before": "old",
                    "after": "CHANGED_TEXT_HERE",
                    "created_at": "2026-03-15T12:00:00",
                },
                "action": "edit_proposal",
                "trace": [],
            },
        ):
            hist, st, _, _ = review_chat_turn("edit", history, state)

        with patch(
            "gradio_app.handlers.review.run_review_agent",
            return_value=_mock_agent_apply(),
        ):
            confirm_review_edit(hist, st)

        # Re-check original file integrity using context handler
        with open(review_dir, encoding="utf-8") as f:
            assert f.read() == original_content


# ── Web gate in handler ───────────────────────────────────────────────────


class TestWebGateHandler:
    def test_web_enabled_via_toggle(self, review_dir: str) -> None:
        """Test that enabling the web toggle allows the agent to use web tools, even without keywords
        This test verifies that when the web_enabled flag is set to True (simulating the user toggling on web access), the review_chat_turn handler correctly allows the agent to use web tools, regardless of whether the user's message contains specific keywords. The test uses a mock agent response to check that the allow_web parameter is set to True when the toggle is enabled.

        Args:
            review_dir (str): The path to the review document used to initialize the session.

        Asserts:
            The mock agent is called with allow_web set to True when the web toggle is enabled.
        """
        from gradio_app.handlers import review_chat_turn, start_review_session

        history, state, _, _ = start_review_session(review_dir, [], {})

        with patch(
            "gradio_app.handlers.review.run_review_agent",
            return_value=_mock_agent_answer("searched the web"),
        ) as mock_agent:
            review_chat_turn(
                "find more sources about climate",
                history,
                state,
                web_enabled=True,
            )

        # allow_web should be True because toggle is on
        assert mock_agent.called
        call_kwargs = mock_agent.call_args.kwargs
        assert call_kwargs["allow_web"] is True

    def test_web_enabled_via_keyword_even_toggle_off(self, review_dir: str) -> None:
        """Test that certain keywords in the user's message can enable web access for the agent, even if the web toggle is off.
        This test ensures that the review_chat_turn handler correctly detects specific keywords in the user's message that indicate a need for web access (e.g., "search on the internet") and allows the agent to use web tools accordingly, even when the web_enabled flag is set to False. The test uses a mock agent response to verify that allow_web is set to True when such keywords are present.

        Args:
            review_dir (str): The path to the review document used to initialize the session.

        Asserts:
            The mock agent is called with allow_web set to True when the user's message contains keywords indicating a need for web access, even if the web toggle is off.
        """
        from gradio_app.handlers import review_chat_turn, start_review_session

        history, state, _, _ = start_review_session(review_dir, [], {})

        with patch(
            "gradio_app.handlers.review.run_review_agent",
            return_value=_mock_agent_answer("searched the web"),
        ) as mock_agent:
            review_chat_turn(
                "search on the internet for sources",
                history,
                state,
                web_enabled=False,
            )

        # allow_web should be True because keyword detected
        assert mock_agent.called
        call_kwargs = mock_agent.call_args.kwargs
        assert call_kwargs["allow_web"] is True

    def test_no_web_without_toggle_or_keyword(self, review_dir: str) -> None:
        """Test that if the web toggle is off and no keywords are present, the agent does not have web access.
        This test verifies that when the web_enabled flag is set to False and the user's message does not contain any keywords indicating a need for web access, the review_chat_turn handler correctly prevents the agent from using web tools. The test uses a mock agent response to check that allow_web is set to False in this scenario.

        Args:
            review_dir (str): The path to the review document used to initialize the session.

        Asserts:
            The mock agent is called with allow_web set to False when the web toggle is off and no keywords are present in the user's message.
        """
        from gradio_app.handlers import review_chat_turn, start_review_session

        history, state, _, _ = start_review_session(review_dir, [], {})

        with patch(
            "gradio_app.handlers.review.run_review_agent",
            return_value=_mock_agent_answer("local only"),
        ) as mock_agent:
            review_chat_turn(
                "find more sources about climate",
                history,
                state,
                web_enabled=False,
            )

        assert mock_agent.called
        call_kwargs = mock_agent.call_args.kwargs
        assert call_kwargs["allow_web"] is False


# ── Empty / error handling ────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_message_returns_warning(self, review_dir):
        from gradio_app.handlers import review_chat_turn, start_review_session

        history, state, _, _ = start_review_session(review_dir, [], {})
        hist, st, status, _ = review_chat_turn("", history, state)
        assert "vazia" in status.lower() or "⚠️" in status

    def test_no_session_returns_error(self):
        from gradio_app.handlers import review_chat_turn

        hist, st, status, _ = review_chat_turn("hello", [], {})
        assert "❌" in status

    def test_agent_exception_returns_error_message(self, review_dir: str) -> None:
        """Test that if the agent raises an exception during processing, the review_chat_turn handler returns
        an appropriate error message in the status and does not crash.
        This test ensures that the review_chat_turn handler has proper error handling in place to catch exceptions raised by the agent (e.g., due to LLM issues) and returns a user-friendly error message in the status, while also ensuring that the chat history and session state remain intact.

        Args:
            review_dir (str): The path to the review document used to initialize the session.

        Asserts:
            When the agent raises a RuntimeError, the status message includes an error indication, and the chat history is updated with an error message without crashing the handler.
        """
        from gradio_app.handlers import review_chat_turn, start_review_session

        history, state, _, _ = start_review_session(review_dir, [], {})

        with patch(
            "gradio_app.handlers.review.run_review_agent",
            side_effect=RuntimeError("LLM down"),
        ):
            hist, st, status, _ = review_chat_turn("hello", history, state)

        assert "❌" in status or "Erro" in hist[-1]["content"]
