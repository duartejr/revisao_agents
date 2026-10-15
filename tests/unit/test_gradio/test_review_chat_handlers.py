"""
tests/unit/test_gradio/test_review_chat_handlers.py

Behavioral unit tests for review_chat_turn, confirm_review_edit, and cancel_review_edit
in gradio_app.handlers.review.
"""

import os
import tempfile
from unittest.mock import patch

import pytest


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


# ═══════════════════════════════════════════════════════════════════════════
# Journal Reviewer Mode (W11) — new tests only; everything above is unchanged.
# ═══════════════════════════════════════════════════════════════════════════

REVIEWER_DOC = "# Title\n\n## Intro\n\nFirst paragraph.\n\n## Methods\n\nSecond paragraph.\n"
RUN_REVIEW = "gradio_app.handlers.review_parts.reviewer.run_journal_review"
RUN_AGENT_IN_REVIEWER = "gradio_app.handlers.review_parts.reviewer.run_review_agent"


def _reviewer_report():
    from revisao_agents.core.schemas.journal_review import (
        CriterionAssessment,
        JournalReviewReport,
    )

    crit = CriterionAssessment(score=6, strengths=["s"], weaknesses=["w"], suggestions=["x"])
    return JournalReviewReport(
        methodology=crit,
        contribution=crit,
        novelty=crit,
        writing_quality=crit,
        overall_recommendation="minor_revision",
        summary="Good.",
    )


def _reviewer_session(tmp_path):
    md_file = tmp_path / "working.md"
    md_file.write_text(REVIEWER_DOC, encoding="utf-8")
    return _make_session(str(md_file)), md_file


class TestJournalReviewerMode:
    """Reviewer branch isolation, read-only guarantee, follow-ups and session reset."""

    def test_first_turn_generates_report_without_touching_classifiers(self, tmp_path):
        from gradio_app.handlers.review import review_chat_turn

        session, _ = _reviewer_session(tmp_path)
        with (
            patch(RUN_REVIEW, return_value=_reviewer_report()) as gen,
            patch("gradio_app.handlers.review.run_review_agent") as main_agent,
            patch("gradio_app.handlers.review._resolve_target_hint") as target,
            patch("gradio_app.handlers.review._classify_reference_intent") as ref_intent,
            patch("gradio_app.handlers.review._classify_phrase_reference_intent") as phrase,
            patch("gradio_app.handlers.review._is_image_request") as image,
        ):
            history, state, status, _ = review_chat_turn(
                "revisar a seção 2", [], session, reviewer_mode=True
            )

        gen.assert_called_once()
        for untouched in (main_agent, target, ref_intent, phrase, image):
            untouched.assert_not_called()
        assert state["journal_review_report"]["overall_recommendation"] == "minor_revision"
        assert "/10" in history[-1]["content"] and history[-1]["role"] == "assistant"
        assert "✅" in status

    def test_reviewer_turns_are_read_only(self, tmp_path):
        from gradio_app.handlers.review import review_chat_turn

        session, md_file = _reviewer_session(tmp_path)
        pending = {"before": "a", "after": "b", "start": 0, "end": 1}
        session["pending_edit"] = dict(pending)
        with (
            patch(RUN_REVIEW, return_value=_reviewer_report()),
            patch(
                RUN_AGENT_IN_REVIEWER,
                return_value={
                    "action": "edit_proposal",
                    "reply": "Answer.\nREVISED_TEXT_START\nnew text\nREVISED_TEXT_END",
                    "edit_proposal": {"before": "x", "after": "y"},
                    "trace": [],
                },
            ),
        ):
            review_chat_turn("go", [], session, reviewer_mode=True)
            history, state, _, _ = review_chat_turn(
                "why is novelty 6?", [], session, reviewer_mode=True
            )

        assert md_file.read_text(encoding="utf-8") == REVIEWER_DOC
        assert state["pending_edit"] == pending
        assert "REVISED_TEXT" not in history[-1]["content"]
        assert history[-1]["content"].startswith("Answer.")

    def test_second_turn_is_follow_up_using_stored_report(self, tmp_path):
        from gradio_app.handlers.review import review_chat_turn

        session, _ = _reviewer_session(tmp_path)
        with (
            patch(RUN_REVIEW, return_value=_reviewer_report()) as gen,
            patch(
                RUN_AGENT_IN_REVIEWER,
                return_value={"action": "answer", "reply": "Because.", "trace": []},
            ) as agent,
        ):
            review_chat_turn("start", [], session, reviewer_mode=True)
            history, _, _, _ = review_chat_turn(
                "why is novelty 6?", [], session, reviewer_mode=True
            )

        assert gen.call_count == 1
        kwargs = agent.call_args.kwargs
        assert session["journal_review_markdown"] in kwargs["system_prompt_override"]
        assert "First paragraph." in kwargs["system_prompt_override"]
        assert kwargs["pending_edit"] is None and kwargs["target_hint"] is None
        assert history[-1]["content"] == "Because."

    def test_regenerate_intent_replaces_stored_report(self, tmp_path):
        from gradio_app.handlers.review import review_chat_turn

        session, _ = _reviewer_session(tmp_path)
        second = _reviewer_report().model_copy(update={"overall_recommendation": "reject"})
        with patch(RUN_REVIEW, side_effect=[_reviewer_report(), second]) as gen:
            review_chat_turn("start", [], session, reviewer_mode=True)
            _, state, _, _ = review_chat_turn("gerar novamente", [], session, reviewer_mode=True)

        assert gen.call_count == 2
        assert state["journal_review_report"]["overall_recommendation"] == "reject"

    def test_generation_error_is_reported_and_not_stored(self, tmp_path):
        from gradio_app.handlers.review import review_chat_turn

        session, _ = _reviewer_session(tmp_path)
        with patch(RUN_REVIEW, return_value="⚠️ boom"):
            history, state, status, _ = review_chat_turn("go", [], session, reviewer_mode=True)

        assert "journal_review_report" not in state
        assert history[-1]["content"] == "⚠️ boom" and "❌" in status

    def test_toggle_off_keeps_report_and_restores_normal_flow(self, tmp_path):
        from gradio_app.handlers.review import review_chat_turn

        session, _ = _reviewer_session(tmp_path)
        with patch(RUN_REVIEW, return_value=_reviewer_report()):
            review_chat_turn("go", [], session, reviewer_mode=True)
        fake = {"action": "answer", "reply": "Normal.", "trace": []}
        with patch("gradio_app.handlers.review.run_review_agent", return_value=fake) as agent:
            history, state, _, _ = review_chat_turn("what is in section 2?", [], session)

        agent.assert_called_once()
        assert history[-1]["content"] == "Normal."
        assert "journal_review_report" in state

    def test_new_session_clears_stored_report(self, tmp_path, monkeypatch):
        from gradio_app.handlers.review import start_review_session

        monkeypatch.chdir(tmp_path)
        (tmp_path / "reviews").mkdir()
        (tmp_path / "reviews" / "doc.md").write_text(REVIEWER_DOC, encoding="utf-8")
        stale = {"journal_review_report": {"x": 1}, "journal_review_markdown": "old"}

        _, state, _, _ = start_review_session("reviews/doc.md", [], stale)

        assert "journal_review_report" not in state
        assert "journal_review_markdown" not in state


class TestReviewerHelpers:
    """Regenerate detection and edit-marker stripping."""

    @pytest.mark.parametrize(
        "msg",
        [
            "gerar novamente",
            "Gere o relatório novamente, por favor",
            "refazer a avaliação",
            "regenerate",
            "Please regenerate the review",
            "generate the report again",
            "re-run the review",
            "Regenerate please",
            "please regenerate",
            "regenerate it",
            "run the review again",
            "generate a new report",
            "gera o relatório",
            "faça um novo relatório",
            "nova avaliação",
        ],
    )
    def test_regenerate_requests_detected(self, msg):
        from gradio_app.handlers.review_parts.reviewer import _is_regenerate_request

        assert _is_regenerate_request(msg)

    @pytest.mark.parametrize(
        "msg",
        [
            "Can you regenerate the explanation of the novelty score?",
            "preciso de uma nova avaliação do trecho 2?",
            "why is methodology 5?",
            "revisar a seção 2",
            "não gere o relatório de novo",
            "don't regenerate the report",
            "regenerate the review of section 3",
            "gere a avaliação da novidade",
            "exagere o relatório",
            "como gerar relatório em PDF?",
            "Explain how to redo the evaluation of the methodology section",
        ],
    )
    def test_questions_about_report_are_not_regenerate(self, msg):
        from gradio_app.handlers.review_parts.reviewer import _is_regenerate_request

        assert not _is_regenerate_request(msg)

    def test_strip_edit_markers_removes_blocks_and_action_lines(self):
        from gradio_app.handlers.review_parts.reviewer import _strip_edit_markers

        text = "ACTION: APPLY_EDIT\nKeep.\nREVISED_TEXT_START\nx\nREVISED_TEXT_END\nTail."
        assert _strip_edit_markers(text) == "Keep.\n\nTail."


class TestReviewerFollowUpDetails:
    def _generated_session(self, tmp_path):
        from gradio_app.handlers.review import review_chat_turn

        session, md_file = _reviewer_session(tmp_path)
        with patch(RUN_REVIEW, return_value=_reviewer_report()):
            review_chat_turn("go", [], session, reviewer_mode=True)
        return session, md_file

    def test_follow_up_history_excludes_all_report_bubbles_after_regenerate(self, tmp_path):
        from gradio_app.handlers.review import review_chat_turn

        session, _ = self._generated_session(tmp_path)
        second = _reviewer_report().model_copy(update={"summary": "Second report summary."})
        with patch(RUN_REVIEW, return_value=second):
            review_chat_turn(
                "gerar novamente", session["chat_history"], session, reviewer_mode=True
            )
        fake = {"action": "answer", "reply": "ok", "trace": []}
        with patch(RUN_AGENT_IN_REVIEWER, return_value=fake) as agent:
            review_chat_turn("why?", session["chat_history"], session, reviewer_mode=True)

        sent = agent.call_args.kwargs["chat_history"]
        assert not any(m["content"].startswith("## 🧑‍⚖️") for m in sent)

    def test_follow_up_history_excludes_the_report_bubble(self, tmp_path):
        from gradio_app.handlers.review import review_chat_turn

        session, _ = self._generated_session(tmp_path)
        fake = {"action": "answer", "reply": "ok", "trace": []}
        with patch(RUN_AGENT_IN_REVIEWER, return_value=fake) as agent:
            review_chat_turn("why?", session["chat_history"], session, reviewer_mode=True)

        sent = agent.call_args.kwargs["chat_history"]
        assert all(m["content"] != session["journal_review_markdown"] for m in sent)
        assert any(m["content"] == "go" for m in sent)

    @pytest.mark.parametrize(
        ("web", "expected"), [(True, "Web tools are available"), (False, "NOT available")]
    )
    def test_web_flag_selects_prompt_block(self, tmp_path, web, expected):
        from gradio_app.handlers.review import review_chat_turn

        session, _ = self._generated_session(tmp_path)
        fake = {"action": "answer", "reply": "ok", "trace": []}
        with patch(RUN_AGENT_IN_REVIEWER, return_value=fake) as agent:
            review_chat_turn("why?", [], session, web_enabled=web, reviewer_mode=True)

        assert agent.call_args.kwargs["allow_web"] is web
        assert expected in agent.call_args.kwargs["system_prompt_override"]

    def test_follow_up_agent_exception_becomes_error_reply(self, tmp_path):
        from gradio_app.handlers.review import review_chat_turn

        session, md_file = self._generated_session(tmp_path)
        with patch(RUN_AGENT_IN_REVIEWER, side_effect=RuntimeError("llm down")):
            history, state, status, _ = review_chat_turn("why?", [], session, reviewer_mode=True)

        assert "llm down" in history[-1]["content"] and "❌" in status
        assert state["pending_edit"] == {} and md_file.read_text(encoding="utf-8") == REVIEWER_DOC

    def test_stale_report_warns_when_document_changed(self, tmp_path):
        from gradio_app.handlers.review import review_chat_turn

        session, md_file = self._generated_session(tmp_path)
        md_file.write_text(REVIEWER_DOC + "\n## New\n\nAdded after the report.\n", encoding="utf-8")
        fake = {"action": "answer", "reply": "Answer.", "trace": []}
        with patch(RUN_AGENT_IN_REVIEWER, return_value=fake):
            history, _, _, _ = review_chat_turn("why?", [], session, reviewer_mode=True)

        assert "gerar novamente" in history[-1]["content"]
        assert history[-1]["content"].endswith("Answer.")

    def test_unchanged_document_has_no_stale_warning(self, tmp_path):
        from gradio_app.handlers.review import review_chat_turn

        session, _ = self._generated_session(tmp_path)
        fake = {"action": "answer", "reply": "Answer.", "trace": []}
        with patch(RUN_AGENT_IN_REVIEWER, return_value=fake):
            history, _, _, _ = review_chat_turn("why?", [], session, reviewer_mode=True)

        assert history[-1]["content"] == "Answer."
