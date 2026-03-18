"""
Unit tests for the ReAct review agent: response parsing, routing, and
scenario-based regressions.

These tests mock the LLM and tools so they run fast with zero external deps.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from revisao_agents.agents.review_agent import (
    _extract_edit_proposal,
    _parse_agent_response,
    _structure_summary,
    run_review_agent,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

SAMPLE_MARKDOWN = """\
## 1. Introduction

This is the introduction paragraph about climate models.

### Referências desta seção
[1] Smith et al. (2024) Climate model evaluation.
[2] Jones & Lee (2023) LSTM comparison.

## 2. Methodology

We compared Chronos-2 with LSTM models for rainfall prediction.

Second paragraph of methodology with more details.

### Referências desta seção
[1] Amazon (2024) Chronos-2 technical report.
[2] Brown et al. (2023) Time series benchmarks.
[3] Garcia (2024) Streamflow prediction models.

## 3. Conclusion

The results show Chronos-2 outperforms LSTM in most metrics.
"""


def _make_sections():
    """Parse the sample markdown into sections — reuse the handler parser."""
    # We inline a minimal parser here so the test doesn't depend on handlers.py
    import re
    from typing import Optional

    lines = SAMPLE_MARKDOWN.splitlines(keepends=True)
    line_offsets: list[int] = []
    acc = 0
    for line in lines:
        line_offsets.append(acc)
        acc += len(line)

    headers: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        m = re.match(r"^##\s+(.+?)\s*$", line.strip("\n"))
        if m:
            headers.append((idx, m.group(1).strip()))

    sections: list[dict] = []
    for hi, (start_line, title) in enumerate(headers):
        next_sl = headers[hi + 1][0] if hi + 1 < len(headers) else len(lines)
        sec_start = line_offsets[start_line]
        sec_end = line_offsets[next_sl] if next_sl < len(line_offsets) else len(SAMPLE_MARKDOWN)
        section_text = SAMPLE_MARKDOWN[sec_start:sec_end]

        refs_line: Optional[int] = None
        for i in range(start_line + 1, next_sl):
            if lines[i].strip().lower().startswith("### referências desta seção"):
                refs_line = i
                break

        body_end_line = refs_line if refs_line is not None else next_sl
        body_start = line_offsets[start_line + 1] if start_line + 1 < len(line_offsets) else sec_start
        body_end = line_offsets[body_end_line] if body_end_line < len(line_offsets) else sec_end

        references: list[str] = []
        if refs_line is not None:
            for i in range(refs_line + 1, next_sl):
                ref_l = lines[i].strip()
                if re.match(r"^\[\d+\]", ref_l):
                    references.append(ref_l)

        paragraphs: list[dict] = []
        current_lines: list[str] = []
        current_start: Optional[int] = None
        for i in range(start_line + 1, body_end_line):
            stripped = lines[i].strip()
            if not stripped or stripped.startswith("<!--") or stripped.startswith("### "):
                if current_lines:
                    para_text = "".join(current_lines).strip()
                    if para_text:
                        paragraphs.append({"text": para_text, "start": current_start, "end": line_offsets[i]})
                    current_lines = []
                    current_start = None
                continue
            if current_start is None:
                current_start = line_offsets[i]
            current_lines.append(lines[i])
        if current_lines and current_start is not None:
            para_text = "".join(current_lines).strip()
            if para_text:
                paragraphs.append({"text": para_text, "start": current_start, "end": body_end})

        sections.append({
            "title": title,
            "start": sec_start,
            "end": sec_end,
            "text": section_text,
            "body": SAMPLE_MARKDOWN[body_start:body_end].strip(),
            "paragraphs": paragraphs,
            "references": references,
        })

    return sections


SECTIONS = _make_sections()


def _fake_llm_response(content: str, tool_calls=None):
    """Build a fake AIMessage-like object."""
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])


# ── _structure_summary ────────────────────────────────────────────────────


class TestStructureSummary:
    def test_produces_numbered_lines(self):
        summary = _structure_summary(SECTIONS)
        assert "1. Introduction" in summary or "1." in summary
        assert "2. Methodology" in summary or "2." in summary
        assert "3. Conclusion" in summary or "3." in summary

    def test_empty_sections(self):
        assert _structure_summary([]) == "(empty document)"


# ── _parse_agent_response ────────────────────────────────────────────────


class TestParseAgentResponse:
    def test_plain_answer(self):
        result = _parse_agent_response(
            "Here are the main findings...", SECTIONS, None,
        )
        assert result["action"] == "answer"
        assert result["edit_proposal"] is None
        assert "main findings" in result["reply"]

    def test_apply_edit_action(self):
        result = _parse_agent_response(
            "ACTION: APPLY_EDIT\nDone.", SECTIONS, {"section_title": "x"},
        )
        assert result["action"] == "apply_edit"

    def test_cancel_edit_action(self):
        result = _parse_agent_response(
            "ACTION: CANCEL_EDIT\nCancelled.", SECTIONS, {"section_title": "x"},
        )
        assert result["action"] == "cancel_edit"

    def test_implicit_confirm_with_pending_edit(self):
        result = _parse_agent_response(
            "confirm", SECTIONS, {"section_title": "x"},
        )
        assert result["action"] == "apply_edit"

    def test_implicit_cancel_with_pending_edit(self):
        result = _parse_agent_response(
            "cancel", SECTIONS, {"section_title": "x"},
        )
        assert result["action"] == "cancel_edit"

    def test_confirm_without_pending_is_just_answer(self):
        result = _parse_agent_response("confirm", SECTIONS, None)
        assert result["action"] == "answer"


# ── _extract_edit_proposal ───────────────────────────────────────────────


class TestExtractEditProposal:
    def test_valid_revised_text_block_with_target_hint(self):
        text = (
            "REVISED_TEXT_START\n"
            "This is the revised introduction paragraph.\n"
            "REVISED_TEXT_END\n"
            "Explanation follows."
        )
        target_hint = {
            "section_title": "1. Introduction",
            "paragraph_index": 0,
            "start": SECTIONS[0]["paragraphs"][0]["start"],
            "end": SECTIONS[0]["paragraphs"][0]["end"],
            "before": SECTIONS[0]["paragraphs"][0]["text"],
        }
        proposal = _extract_edit_proposal(text, SECTIONS, target_hint)
        assert proposal is not None
        assert proposal["section_title"] == "1. Introduction"
        assert proposal["paragraph_index"] == 0
        assert "revised introduction" in proposal["after"]
        assert proposal["before"] == SECTIONS[0]["paragraphs"][0]["text"]

    def test_legacy_edit_proposal_block_still_supported(self):
        text = (
            "EDIT_PROPOSAL\n"
            "SECTION_NUMBER: 1\n"
            "PARAGRAPH_NUMBER: 1\n"
            "REVISED_TEXT_START\n"
            "This is the revised introduction paragraph.\n"
            "REVISED_TEXT_END\n"
            "Explanation follows."
        )
        proposal = _extract_edit_proposal(text, SECTIONS)
        assert proposal is not None
        assert proposal["section_title"] == "1. Introduction"

    def test_invalid_section_number(self):
        text = (
            "EDIT_PROPOSAL\n"
            "SECTION_NUMBER: 99\n"
            "PARAGRAPH_NUMBER: 1\n"
            "REVISED_TEXT_START\ntext\nREVISED_TEXT_END"
        )
        assert _extract_edit_proposal(text, SECTIONS) is None

    def test_invalid_paragraph_number(self):
        text = (
            "EDIT_PROPOSAL\n"
            "SECTION_NUMBER: 1\n"
            "PARAGRAPH_NUMBER: 99\n"
            "REVISED_TEXT_START\ntext\nREVISED_TEXT_END"
        )
        assert _extract_edit_proposal(text, SECTIONS) is None

    def test_no_proposal_block(self):
        assert _extract_edit_proposal("Just a normal answer.", SECTIONS) is None


# ── run_review_agent: scenario-driven routing ─────────────────────────────


def _run_agent_with_mock_llm(user_message: str, llm_reply: str, *, allow_web=False, pending_edit=None):
    """Helper: run the review agent with a mocked LLM that returns a fixed reply."""
    fake_response = _fake_llm_response(llm_reply)
    fake_llm = MagicMock()
    fake_llm.bind_tools.return_value = fake_llm
    fake_llm.invoke.return_value = fake_response

    with patch(
        "revisao_agents.agents.review_agent.get_raw_llm",
        return_value=fake_llm,
    ):
        return run_review_agent(
            document_content=SAMPLE_MARKDOWN,
            document_sections=SECTIONS,
            user_message=user_message,
            chat_history=[],
            allow_web=allow_web,
            pending_edit=pending_edit,
        )


class TestAgentScenarios:
    """
    Scenario tests for the 7 target asks plus the critical regression.
    These verify the agent receives the right system prompt and that
    response parsing produces correct actions.
    """

    # ── REGRESSION: "liste todas as referencias usadas" ───────────────
    def test_regression_list_all_references_not_summary(self):
        """
        The exact failure that prompted the ReAct refactor:
        'liste todas as referencias usadas' MUST NOT return a summary.
        With the ReAct agent, the LLM sees the full document and the
        system prompt instructs it to extract references.
        """
        llm_reply = (
            "### Todas as referências\n\n"
            "**Seção 1 – Introduction:**\n"
            "[1] Smith et al. (2024)\n"
            "[2] Jones & Lee (2023)\n\n"
            "**Seção 2 – Methodology:**\n"
            "[1] Amazon (2024)\n"
            "[2] Brown et al. (2023)\n"
            "[3] Garcia (2024)"
        )
        result = _run_agent_with_mock_llm(
            "liste todas as referencias usadas", llm_reply,
        )
        assert result["action"] == "answer"
        # Must contain references, not a findings summary
        assert "Smith" in result["reply"] or "referência" in result["reply"].lower()

    # ── Scenario 1: main findings ─────────────────────────────────────
    def test_scenario_main_findings(self):
        llm_reply = "The main findings are: Chronos-2 outperforms LSTM."
        result = _run_agent_with_mock_llm("what are the main findings?", llm_reply)
        assert result["action"] == "answer"
        assert "Chronos-2" in result["reply"]

    # ── Scenario 2: papers cited in section N ─────────────────────────
    def test_scenario_section_citations(self):
        llm_reply = (
            "References in Section 2:\n"
            "[1] Amazon (2024)\n"
            "[2] Brown et al. (2023)\n"
            "[3] Garcia (2024)"
        )
        result = _run_agent_with_mock_llm(
            "what papers are cited in section 2?", llm_reply,
        )
        assert result["action"] == "answer"
        assert "Amazon" in result["reply"]

    # ── Scenario 3: confirm paragraph ─────────────────────────────────
    def test_scenario_confirm_paragraph(self):
        llm_reply = (
            "The paragraph is supported by:\n"
            "- Smith et al. (2024) confirms the claim about climate models\n"
            "- Jones & Lee (2023) provides corroborating data"
        )
        result = _run_agent_with_mock_llm(
            "which authors confirm the first paragraph of section 1?",
            llm_reply,
        )
        assert result["action"] == "answer"
        assert "Smith" in result["reply"]

    # ── Scenario 4: more docs for phrase (local only) ─────────────────
    def test_scenario_more_docs_local(self):
        llm_reply = "I found additional evidence in the corpus about climate models."
        result = _run_agent_with_mock_llm(
            'find more documents about "climate model evaluation"',
            llm_reply,
            allow_web=False,
        )
        assert result["action"] == "answer"

    # ── Scenario 5: edit with source ──────────────────────────────────
    def test_scenario_edit_proposal(self):
        llm_reply = (
            "REVISED_TEXT_START\n"
            "We compared Chronos-2 with LSTM and Transformer models for rainfall prediction, "
            "following the methodology of Amazon (2024).\n"
            "REVISED_TEXT_END\n"
            "Added reference to Amazon (2024)."
        )
        result = _run_agent_with_mock_llm(
            "edit first paragraph of section 2 to mention Amazon paper",
            llm_reply,
            pending_edit=None,
        )
        # Without target_hint, plain revised-text block is treated as answer.
        assert result["action"] == "answer"

    # ── Scenario 6: fix paragraph with web search ─────────────────────
    def test_scenario_edit_with_web(self):
        llm_reply = (
            "REVISED_TEXT_START\n"
            "The results show Chronos-2 outperforms LSTM in most metrics, "
            "consistent with recent benchmarks (2025).\n"
            "REVISED_TEXT_END"
        )
        result = _run_agent_with_mock_llm(
            "fix the conclusion, search the internet if necessary",
            llm_reply,
            allow_web=True,
        )
        assert result["action"] == "answer"

    # ── Scenario 7: style rewrite ─────────────────────────────────────
    def test_scenario_style_rewrite(self):
        llm_reply = (
            "REVISED_TEXT_START\n"
            "Chronos-2 consistently outperforms LSTM across all evaluated metrics.\n"
            "REVISED_TEXT_END"
        )
        result = _run_agent_with_mock_llm(
            "rewrite the conclusion to be more direct",
            llm_reply,
        )
        assert result["action"] == "answer"

    def test_revised_text_block_with_target_hint_creates_proposal(self):
        fake_response = _fake_llm_response(
            "REVISED_TEXT_START\n"
            "Chronos-2 consistently outperforms LSTM across evaluated metrics.\n"
            "REVISED_TEXT_END"
        )
        fake_llm = MagicMock()
        fake_llm.bind_tools.return_value = fake_llm
        fake_llm.invoke.return_value = fake_response

        target_hint = {
            "section_title": "3. Conclusion",
            "paragraph_index": 0,
            "start": SECTIONS[2]["paragraphs"][0]["start"],
            "end": SECTIONS[2]["paragraphs"][0]["end"],
            "before": SECTIONS[2]["paragraphs"][0]["text"],
        }

        with patch(
            "revisao_agents.agents.review_agent.get_raw_llm",
            return_value=fake_llm,
        ):
            result = run_review_agent(
                document_content=SAMPLE_MARKDOWN,
                document_sections=SECTIONS,
                user_message="rewrite this paragraph",
                chat_history=[],
                target_hint=target_hint,
            )

        assert result["action"] == "edit_proposal"
        assert result["edit_proposal"] is not None
        assert "Chronos-2" in result["edit_proposal"]["after"]


# ── Tool-call trace ──────────────────────────────────────────────────────


class TestToolCallTrace:
    def test_trace_contains_tool_calls(self):
        """When the LLM calls tools, the trace should record them."""
        tc = {"name": "search_evidence", "args": {"query": "LSTM"}, "id": "tc1"}
        first_resp = _fake_llm_response("", tool_calls=[tc])
        final_resp = _fake_llm_response("Evidence found for LSTM.")

        fake_llm = MagicMock()
        fake_llm.bind_tools.return_value = fake_llm
        fake_llm.invoke.side_effect = [first_resp, final_resp]

        with patch(
            "revisao_agents.agents.review_agent.get_raw_llm",
            return_value=fake_llm,
        ), patch(
            "revisao_agents.agents.review_agent.get_review_tools",
            return_value=[],
        ):
            result = run_review_agent(
                document_content=SAMPLE_MARKDOWN,
                document_sections=SECTIONS,
                user_message="verify LSTM claim",
                chat_history=[],
            )

        assert len(result["trace"]) >= 1
        assert result["trace"][0]["tool"] == "search_evidence"

    def test_no_trace_when_no_tool_calls(self):
        result = _run_agent_with_mock_llm("hello", "Hello!")
        assert result["trace"] == []


# ── Web gate ──────────────────────────────────────────────────────────────


class TestWebGate:
    def test_web_tool_NOT_available_without_allow_web(self):
        """When allow_web=False, search_web_sources should not be bound."""
        fake_llm = MagicMock()
        fake_llm.bind_tools.return_value = fake_llm
        fake_llm.invoke.return_value = _fake_llm_response("answer")

        with patch(
            "revisao_agents.agents.review_agent.get_raw_llm",
            return_value=fake_llm,
        ):
            run_review_agent(
                document_content=SAMPLE_MARKDOWN,
                document_sections=SECTIONS,
                user_message="find more sources",
                chat_history=[],
                allow_web=False,
            )

        # Check the tools that were bound
        bound_tools = fake_llm.bind_tools.call_args[0][0]
        tool_names = [t.name for t in bound_tools]
        assert "search_web_sources" not in tool_names
        assert "search_evidence" in tool_names

    def test_web_tool_available_with_allow_web(self):
        fake_llm = MagicMock()
        fake_llm.bind_tools.return_value = fake_llm
        fake_llm.invoke.return_value = _fake_llm_response("answer")

        with patch(
            "revisao_agents.agents.review_agent.get_raw_llm",
            return_value=fake_llm,
        ):
            run_review_agent(
                document_content=SAMPLE_MARKDOWN,
                document_sections=SECTIONS,
                user_message="find more sources on the internet",
                chat_history=[],
                allow_web=True,
            )

        bound_tools = fake_llm.bind_tools.call_args[0][0]
        tool_names = [t.name for t in bound_tools]
        assert "search_web_sources" in tool_names


# ── System prompt content ─────────────────────────────────────────────────


class TestSystemPrompt:
    def test_system_prompt_contains_document_and_rules(self):
        """Verify the system prompt includes document content and key rules."""
        fake_llm = MagicMock()
        fake_llm.bind_tools.return_value = fake_llm
        fake_llm.invoke.return_value = _fake_llm_response("ok")

        with patch(
            "revisao_agents.agents.review_agent.get_raw_llm",
            return_value=fake_llm,
        ):
            run_review_agent(
                document_content=SAMPLE_MARKDOWN,
                document_sections=SECTIONS,
                user_message="hello",
                chat_history=[],
            )

        # First call to invoke, first arg is messages list
        messages = fake_llm.invoke.call_args[0][0]
        system_msg = messages[0].content
        # Document content should be embedded
        assert "Introduction" in system_msg
        assert "Methodology" in system_msg
        # Key rules should be present
        assert "list all references" in system_msg.lower() or "referências" in system_msg.lower()
        assert "REVISED_TEXT_START" in system_msg
        assert "search_evidence" in system_msg
        assert "search_evidence_sources" in system_msg
