"""
Integration tests for the interactive review chat flow through handlers.py.

These tests exercise start_review_session → review_chat_turn → confirm/cancel
with a mocked LLM + tools so no external services are needed.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# We need the handlers on sys.path
import sys

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

    def test_answer_turn(self, review_dir):
        from gradio_app.handlers import review_chat_turn

        history, state = self._start_session(review_dir)

        with patch(
            "gradio_app.handlers.run_review_agent",
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

    def test_edit_proposal_sets_pending(self, review_dir):
        from gradio_app.handlers import review_chat_turn

        history, state = self._start_session(review_dir)

        with patch(
            "gradio_app.handlers.run_review_agent",
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
        assert "Proposta de edição" in hist[-1]["content"]

    def test_confirm_edit_applies_and_clears_pending(self, review_dir):
        from gradio_app.handlers import review_chat_turn, confirm_review_edit

        history, state = self._start_session(review_dir)

        # First: set a pending edit
        # Compute actual start/end from the working copy
        working = state["working_copy_path"]
        md = open(working, encoding="utf-8").read()
        # Find the introduction paragraph
        intro_start = md.index("Climate models have evolved")
        intro_end = intro_start + len(
            "Climate models have evolved significantly in recent years."
        )

        with patch(
            "gradio_app.handlers.run_review_agent",
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
            "gradio_app.handlers.run_review_agent",
            return_value=_mock_agent_apply(),
        ):
            hist2, st2, status2, md2 = confirm_review_edit(hist, st)

        assert not st2.get("pending_edit")
        assert "✅" in status2 or "Edição aplicada" in hist2[-1]["content"]
        # Working copy should have the new text
        assert "DRAMATICALLY" in md2

    def test_cancel_edit_clears_pending(self, review_dir):
        from gradio_app.handlers import review_chat_turn, cancel_review_edit

        history, state = self._start_session(review_dir)

        # Manually inject a pending edit
        state["pending_edit"] = {
            "section_title": "test",
            "paragraph_index": 0,
            "start": 0,
            "end": 5,
            "before": "old",
            "after": "new",
        }

        with patch(
            "gradio_app.handlers.run_review_agent",
            return_value=_mock_agent_cancel(),
        ):
            hist, st, status, _ = cancel_review_edit(history, state)

        assert not st.get("pending_edit")
        assert (
            "cancelada" in hist[-1]["content"].lower()
            or "cancel" in hist[-1]["content"].lower()
        )

    def test_original_file_unchanged_after_edit(self, review_dir):
        from gradio_app.handlers import review_chat_turn, confirm_review_edit

        history, state = self._start_session(review_dir)
        original_content = open(review_dir, encoding="utf-8").read()

        md = open(state["working_copy_path"], encoding="utf-8").read()
        intro_start = md.index("Climate models have evolved")
        intro_end = intro_start + len(
            "Climate models have evolved significantly in recent years."
        )

        with patch(
            "gradio_app.handlers.run_review_agent",
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
            "gradio_app.handlers.run_review_agent",
            return_value=_mock_agent_apply(),
        ):
            confirm_review_edit(hist, st)

        # Original must remain unchanged
        assert open(review_dir, encoding="utf-8").read() == original_content


# ── Web gate in handler ───────────────────────────────────────────────────


class TestWebGateHandler:
    def test_web_enabled_via_toggle(self, review_dir):
        from gradio_app.handlers import start_review_session, review_chat_turn

        history, state, _, _ = start_review_session(review_dir, [], {})

        with patch(
            "gradio_app.handlers.run_review_agent",
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

    def test_web_enabled_via_keyword_even_toggle_off(self, review_dir):
        from gradio_app.handlers import start_review_session, review_chat_turn

        history, state, _, _ = start_review_session(review_dir, [], {})

        with patch(
            "gradio_app.handlers.run_review_agent",
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

    def test_no_web_without_toggle_or_keyword(self, review_dir):
        from gradio_app.handlers import start_review_session, review_chat_turn

        history, state, _, _ = start_review_session(review_dir, [], {})

        with patch(
            "gradio_app.handlers.run_review_agent",
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
        from gradio_app.handlers import start_review_session, review_chat_turn

        history, state, _, _ = start_review_session(review_dir, [], {})
        hist, st, status, _ = review_chat_turn("", history, state)
        assert "vazia" in status.lower() or "⚠️" in status

    def test_no_session_returns_error(self):
        from gradio_app.handlers import review_chat_turn

        hist, st, status, _ = review_chat_turn("hello", [], {})
        assert "❌" in status

    def test_agent_exception_returns_error_message(self, review_dir):
        from gradio_app.handlers import start_review_session, review_chat_turn

        history, state, _, _ = start_review_session(review_dir, [], {})

        with patch(
            "gradio_app.handlers.run_review_agent",
            side_effect=RuntimeError("LLM down"),
        ):
            hist, st, status, _ = review_chat_turn("hello", history, state)

        assert "❌" in status or "Erro" in hist[-1]["content"]
