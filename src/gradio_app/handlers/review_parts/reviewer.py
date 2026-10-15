"""Journal Reviewer Mode turn handler for the interactive review chat.

Reviewer mode is **read-only**: it never touches the working copy, never sets
``pending_edit`` and never emits edit markers. The first reviewer turn (or an
explicit regenerate request) produces a structured referee report that is kept
in ``session_state``; later turns answer follow-up questions against it.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime

from revisao_agents.agents.journal_reviewer_agent import (
    REPORT_HEADING_PREFIX,
    activate_journal_review_experiment,
    build_review_context,
    report_to_markdown,
    run_journal_review,
)
from revisao_agents.agents.review_agent import run_review_agent
from revisao_agents.core.schemas.journal_review import JournalReviewReport
from revisao_agents.utils.llm_utils.prompt_loader import load_prompt

from ..base import _localized_text

REPORT_KEY = "journal_review_report"
REPORT_MARKDOWN_KEY = "journal_review_markdown"
REPORT_DOC_HASH_KEY = "journal_review_doc_hash"

_REGENERATE_RE = re.compile(
    r"""
    ^\W*(?:(?:please|por\s+favor|pode|can\s+you|could\s+you)\s+)?
    (?:
        (?:gerar|gere|gera|refazer|refa[çc]a|fa[çc]a)\s+(?:o\s+|a\s+|um\s+|uma\s+)?
            (?:novo\s+|nova\s+)?(?:relat[óo]rio|revis[ãa]o|avalia[çc][ãa]o|parecer)
            (?:\s+(?:novamente|de\s+novo))?
      | (?:gerar|gere|gera)\s+(?:isso\s+|tudo\s+)?(?:novamente|de\s+novo)
      | nova\s+(?:avalia[çc][ãa]o|revis[ãa]o)
      | re-?generate(?:\s+(?:the\s+|this\s+)?(?:review|report|assessment|evaluation)|\s+it)?
            (?:\s+again)?
      | (?:redo|re-?run)\s+(?:the\s+|this\s+)?(?:review|report|assessment|evaluation)
            (?:\s+again)?
      | (?:run|generate)\s+(?:the\s+|a\s+)?(?:review\s+|report\s+)?again
      | generate\s+(?:a\s+)?new\s+(?:review|report|assessment)
    )
    (?:\s*,?\s*(?:please|por\s+favor))?\W*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

_EDIT_BLOCK_RE = re.compile(
    r"(?:REVISED_TEXT_START|TEXTO_REVISADO_IN[ÍI]CIO).*?(?:REVISED_TEXT_END|TEXTO_REVISADO_FIM)",
    re.DOTALL,
)
_ACTION_LINE_RE = re.compile(r"^\s*ACTION:.*$", re.MULTILINE | re.IGNORECASE)


def _doc_hash(markdown: str) -> str:
    """Return a stable fingerprint of the document the report was written for."""
    return hashlib.sha256(markdown.encode("utf-8")).hexdigest()


def _is_regenerate_request(user_msg: str) -> bool:
    """Return ``True`` when the message explicitly asks to regenerate the whole report.

    The *whole* message must be the request (an imperative verb plus a report
    noun, or "again", with an optional "please"). Negations ("não gere de
    novo"), scoped requests ("regenerate the review of section 3") and
    questions about the report ("can you regenerate the explanation of the
    novelty score?") are therefore ordinary follow-ups.
    """
    return _REGENERATE_RE.search(user_msg) is not None


def _strip_edit_markers(reply: str) -> str:
    """Remove edit-proposal blocks and ``ACTION:`` lines from a follow-up reply."""
    cleaned = _EDIT_BLOCK_RE.sub("", reply)
    return _ACTION_LINE_RE.sub("", cleaned).strip()


def _finish(
    history: list,
    session_state: dict,
    user_msg: str,
    reply: str,
    status: str,
    markdown: str,
) -> tuple[list, dict, str, str]:
    """Append the exchange to the chat history and build the handler return tuple."""
    history = history + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": reply},
    ]
    session_state["chat_history"] = history
    return history, session_state, status, markdown


def _generate_report(
    user_msg: str,
    history: list,
    session_state: dict,
    markdown: str,
    sections: list[dict],
    language: str,
) -> tuple[list, dict, str, str]:
    """Generate (or regenerate) the referee report and store it in session state."""
    result = run_journal_review(
        markdown,
        language,
        sections=sections,
        review_file=str(session_state.get("original_file_path", "")),
    )
    if not isinstance(result, JournalReviewReport):
        return _finish(
            history,
            session_state,
            user_msg,
            result,
            _localized_text(language, "❌ Erro no revisor", "❌ Reviewer error"),
            markdown,
        )

    rendered = report_to_markdown(result, language)
    session_state[REPORT_KEY] = result.model_dump()
    session_state[REPORT_MARKDOWN_KEY] = rendered
    session_state[REPORT_DOC_HASH_KEY] = _doc_hash(markdown)
    session_state.setdefault("retrieval_trace", []).append(
        {
            "action": "journal_review_generate",
            "at": datetime.now().isoformat(timespec="seconds"),
            "tool_calls": [],
        }
    )
    return _finish(
        history,
        session_state,
        user_msg,
        rendered,
        _localized_text(language, "✅ Relatório gerado", "✅ Report generated"),
        markdown,
    )


def _answer_follow_up(
    user_msg: str,
    history: list,
    session_state: dict,
    markdown: str,
    sections: list[dict],
    language: str,
    allow_web: bool,
) -> tuple[list, dict, str, str]:
    """Answer a follow-up question against the stored report and the document."""
    web_block = (
        "Web tools are available (`search_web_sources`, `extract_web_text_from_url`)."
        if allow_web
        else "Web search is NOT available right now; rely on the document and the corpus."
    )
    try:
        prompt = load_prompt(
            "common/journal_reviewer_followup",
            report=str(session_state.get(REPORT_MARKDOWN_KEY, "")),
            document=build_review_context(markdown, sections).text,
            web_block=web_block,
            language=language,
        )
        activate_journal_review_experiment()  # never raises; places the trace
        result = run_review_agent(
            document_content=markdown,
            document_sections=sections,
            user_message=user_msg,
            chat_history=[
                msg
                for msg in session_state.get("chat_history", [])
                if not (
                    msg.get("role") == "assistant"
                    and str(msg.get("content", "")).startswith(REPORT_HEADING_PREFIX)
                )
            ],
            allow_web=allow_web,
            pending_edit=None,
            target_hint=None,
            system_prompt_override=prompt.text,
        )
    except Exception as exc:
        return _finish(
            history,
            session_state,
            user_msg,
            _localized_text(language, f"⚠️ Erro do agente: {exc}", f"⚠️ Agent error: {exc}"),
            _localized_text(language, "❌ Erro no agente", "❌ Agent error"),
            markdown,
        )

    # Only the text is used: any proposed edit or action is deliberately ignored.
    reply = _strip_edit_markers(str(result.get("reply", ""))) or _localized_text(
        language, "Não consegui responder a essa pergunta.", "I could not answer that question."
    )
    if session_state.get(REPORT_DOC_HASH_KEY) != _doc_hash(markdown):
        reply = (
            _localized_text(
                language,
                "⚠️ _O documento mudou desde que o relatório foi gerado; diga "
                "**gerar novamente** para atualizá-lo._",
                "⚠️ _The document changed since the report was generated; say "
                "**regenerate** to refresh it._",
            )
            + "\n\n"
            + reply
        )
    session_state.setdefault("retrieval_trace", []).append(
        {
            "action": "journal_review_followup",
            "web": allow_web,
            "at": datetime.now().isoformat(timespec="seconds"),
            "tool_calls": result.get("trace", []),
        }
    )
    return _finish(
        history,
        session_state,
        user_msg,
        reply,
        _localized_text(language, "✅ Resposta do revisor", "✅ Reviewer answer"),
        markdown,
    )


def handle_reviewer_turn(
    user_msg: str,
    history: list,
    session_state: dict,
    markdown: str,
    sections: list[dict],
    language: str,
    allow_web: bool = False,
) -> tuple[list, dict, str, str]:
    """Handle one chat turn while Journal Reviewer Mode is on.

    Routing: with no stored report (or an explicit regenerate request) the
    referee report is generated and stored; otherwise the message is a
    follow-up answered against the stored report plus the document. The
    working copy and ``session_state["pending_edit"]`` are never modified.

    Args:
        user_msg: The user's message.
        history: Chat history (list of ``{"role", "content"}`` dicts).
        session_state: Mutable review session state.
        markdown: Current working-copy markdown (read-only here).
        sections: Sections of ``markdown`` from ``_split_sections``.
        language: ``"pt"`` or ``"en"``, the detected user language.
        allow_web: Whether web tools may be used in follow-up answers.

    Returns:
        ``(history, session_state, status, preview_markdown)``, the same
        shape as ``review_chat_turn``. Failures (report generation or agent
        errors) are reported in the chat reply and status text; this function
        does not raise.

    Side effects:
        On success, ``session_state`` gains ``journal_review_report``,
        ``journal_review_markdown`` and ``journal_review_doc_hash`` (report
        generation) and a ``retrieval_trace`` entry; ``chat_history`` is updated.
        Follow-up replies get a stale-report warning prepended when the document
        hash no longer matches the one stored with the report.
    """
    if not session_state.get(REPORT_KEY) or _is_regenerate_request(user_msg):
        return _generate_report(user_msg, history, session_state, markdown, sections, language)
    return _answer_follow_up(
        user_msg, history, session_state, markdown, sections, language, allow_web
    )
