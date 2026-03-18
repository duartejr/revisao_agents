# src/revisao_agents/agents/review_agent.py
"""
ReAct review agent for the interactive review chatbot.

Uses tool-calling LLM to reason about user requests, retrieve evidence,
and produce structured actions (answer, edit proposal, confirm/cancel).
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Optional

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from ..tools.review_tools import get_review_tools
from ..utils.llm_utils.llm_providers import get_llm as get_raw_llm

logger = logging.getLogger(__name__)

MAX_AGENT_ITERATIONS = 6


def _recover_tool_call_from_exception(exc: Exception) -> dict[str, Any] | None:
    """Recover a tool call from provider-side failed_generation payload.

    Some providers (notably Groq on certain models) can reject tool calls with
    ``tool_use_failed`` and return the raw attempted call as text, e.g.:
    ``<function=tool_name({"arg": "value"})</function>``.
    This parser extracts the tool name and JSON args so the agent can continue.
    """
    text = str(exc)
    if "tool_use_failed" not in text and "failed_generation" not in text:
        return None

    match = re.search(
        r"<function=([a-zA-Z_][a-zA-Z0-9_]*)\((\{.*?\})\)</function>",
        text,
        flags=re.DOTALL,
    )
    if not match:
        return None

    tool_name = match.group(1)
    raw_args = match.group(2)

    try:
        tool_args = json.loads(raw_args)
    except Exception:
        return None

    if not isinstance(tool_args, dict):
        return None

    return {"name": tool_name, "args": tool_args}


# ── public API ────────────────────────────────────────────────────────────


def run_review_agent(
    document_content: str,
    document_sections: list[dict],
    user_message: str,
    chat_history: list[dict],
    allow_web: bool = False,
    pending_edit: dict | None = None,
    target_hint: dict | None = None,
    max_iterations: int = MAX_AGENT_ITERATIONS,
) -> dict:
    """Execute one turn of the review agent.

    Args:
        document_content: Full markdown of the working copy.
        document_sections: Pre-parsed sections from ``_split_sections()``.
        user_message: Current user input.
        chat_history: Previous ``[{"role": ..., "content": ...}, ...]`` pairs.
        allow_web: ``True`` when user explicitly asked for web search.
        pending_edit: Current pending edit dict, or ``None``.
        max_iterations: Max tool-call round-trips.

    Returns:
        ``dict`` with keys:
        - ``reply``  (str) – assistant message to show
        - ``edit_proposal`` (dict|None) – proposal dict if one was generated
        - ``action`` (str) – ``"answer"`` | ``"edit_proposal"`` |
          ``"apply_edit"`` | ``"cancel_edit"``
        - ``trace`` (list[dict]) – tool-call trace for debugging
    """
    provider_name = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    is_groq = provider_name == "groq"

    compact_history, history_summary = _compact_chat_history(
        chat_history,
        provider_name=provider_name,
    )

    system_prompt = _build_system_prompt(
        document_content,
        document_sections,
        allow_web,
        pending_edit,
        include_full_document=not is_groq,
    )

    # Build LangChain message list
    messages: list[Any] = [SystemMessage(content=system_prompt)]
    if history_summary:
        messages.append(SystemMessage(content=history_summary))

    for msg in compact_history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=user_message))

    # Prepare LLM + tools
    tools = get_review_tools(allow_web=allow_web)
    llm = get_raw_llm(temperature=0.2)
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    trace: list[dict] = []

    # ── ReAct loop ────────────────────────────────────────────────────
    response: Any = None
    for iteration in range(max_iterations):
        try:
            response = llm_with_tools.invoke(messages)
        except Exception as exc:
            recovered = _recover_tool_call_from_exception(exc)
            if recovered is None:
                raise
            logger.warning(
                "Recovered tool call from provider failed_generation: %s",
                recovered.get("name"),
            )
            response = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": recovered["name"],
                        "args": recovered["args"],
                        "id": f"recovered_tool_call_{iteration}",
                    }
                ],
            )
        messages.append(response)

        if not getattr(response, "tool_calls", None):
            break  # Final answer – no more tools to call

        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_fn = tool_map.get(tool_name)

            if tool_fn is None:
                result_str = f"Error: unknown tool '{tool_name}'"
            else:
                try:
                    result_str = str(tool_fn.invoke(tool_args))
                except Exception as exc:
                    result_str = f"Error executing {tool_name}: {exc}"

            trace.append({
                "iteration": iteration,
                "tool": tool_name,
                "args": tool_args,
                "result_length": len(result_str),
            })
            messages.append(
                ToolMessage(content=result_str, tool_call_id=tc["id"]),
            )

    # ── Gemini empty-stop fallback ────────────────────────────────────
    # Gemini 2.5 Flash with bind_tools sometimes returns content='' and
    # output_tokens=0 when the final answer requires plain text (no tool call).
    # In that case, retry once with a plain (tool-free) LLM so the model is
    # no longer in "tool-call mode" and is forced to generate text.
    if response is not None:
        raw_check = getattr(response, "content", None)
        if isinstance(raw_check, list):
            raw_check = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in raw_check
            )
        if not (raw_check or "").strip() and not getattr(response, "tool_calls", None):
            logger.warning(
                "Provider returned empty content with no tool calls "
                "(likely Gemini bind_tools empty-stop). Retrying without tools."
            )
            plain_llm = get_raw_llm(temperature=0.2)
            response = plain_llm.invoke(messages)

    # ── Parse response ────────────────────────────────────────────────
    reply_text = ""
    if response is not None:
        raw_content = getattr(response, "content", None)
        # Gemini (and some other providers) return content as a list of parts
        # e.g. [{"type": "text", "text": "..."}, ...] — flatten to plain str.
        if isinstance(raw_content, list):
            parts = []
            for part in raw_content:
                if isinstance(part, dict):
                    parts.append(part.get("text", "") or str(part))
                else:
                    parts.append(str(part))
            raw_content = "".join(parts)
        reply_text = raw_content or str(response)

    parsed = _parse_agent_response(reply_text, document_sections, pending_edit, target_hint)
    parsed["trace"] = trace
    return parsed


# ── system prompt builder ─────────────────────────────────────────────────


def _build_system_prompt(
    document_content: str,
    sections: list[dict],
    allow_web: bool,
    pending_edit: dict | None,
    include_full_document: bool = True,
) -> str:
    structure = _structure_summary(sections)
    year = datetime.now().year

    pending_block = ""
    if pending_edit:
        pending_block = (
            "\n\n⚠️ PENDING EDIT:\n"
            f"Section: {pending_edit.get('section_title', '?')}\n"
            f"Paragraph: {int(pending_edit.get('paragraph_index', 0)) + 1}\n"
            "If the user confirms → action=apply_edit. "
            "If the user cancels → action=cancel_edit.\n"
        )

    if allow_web:
        web_block = (
            "- You ALSO have web tools: `search_web_sources`, `search_web_images`, "
            "`extract_web_text_from_url`, `get_bibtex_for_reference`, and "
            "`search_article_online`. The last two are specifically for resolving "
            "references: use `search_article_online(title)` to find a DOI via Tavily "
            "when `fetch_reference_metadata` could not resolve it, then pass the DOI "
            "to `get_bibtex_for_reference`."
        )
    else:
        web_block = (
            "- Web search is **NOT** available right now. "
            "If the user needs web results, tell them to include "
            "'internet' or 'web' in their request."
        )

    # Keep full document context only when provider budget allows it.
    # For Groq we intentionally skip the full document block and rely on
    # structure + tools to avoid 413/token-limit failures.
    doc_block = ""
    if include_full_document:
        doc_text = document_content[:30000]
        if len(document_content) > 30000:
            doc_text += "\n\n[...document truncated for context window...]"
        doc_block = f"─── FULL DOCUMENT ───\n{doc_text}\n"

    return f"""\
You are a review assistant. You help the user analyse, verify, and edit an
academic review document written in Markdown. Today is {year}.

─── DOCUMENT STRUCTURE ───
{structure}
{doc_block}
{pending_block}
─── AVAILABLE TOOLS ───
- `search_evidence(query, k)` – search the MongoDB academic corpus for
  evidence chunks.  Use it to verify claims, find sources, or expand a topic.
- `search_evidence_sources(query, k)` – search corpus evidence with source
    metadata (title, URL, DOI, file path). Use it to suggest new sources and
    check whether source names appear in the current review references.
- `search_near_chunks(query, n)` – retrieve anchor chunk + neighboring chunks
    from the same source when one chunk is not enough context.
- `fetch_reference_metadata(title, doi, url)` – **always available** – resolve
    full bibliographic metadata (DOI, BibTeX) for an article. Call this FIRST
    when the user asks for reference formatting.
{web_block}

─── RULES ───
1. When the user asks to **list all references**, extract every citation from
   ALL "### Referências desta seção" blocks in the document and list them.
   Do NOT summarise findings.
2. When asked about citations in a **specific section**, list only that
   section's references.
3. For **paragraph verification / confirmation**, use `search_evidence` to
   find corpus chunks that support or contradict the paragraph, then report
   your findings with source labels.
4. For **edit requests**, reply ONLY with the revised paragraph using the
    `REVISED_TEXT_START` / `REVISED_TEXT_END` block (see FORMAT below).
    Do NOT include section numbers, paragraph numbers, or EDIT_PROPOSAL header.
    NEVER apply edits silently.
5. If you need more context from the academic corpus, call `search_evidence`.
6. When users ask for "more sources not yet cited", call
    `search_evidence_sources` and compare returned source titles against
    references already present in the document.
7. If user asks for "context around this chunk", use `search_near_chunks`.
8. Use `extract_web_text_from_url` when user provides specific URLs and
    asks for validation/summarization from page text.
9. Respond in the **same language** as the user's message.
10. If the question is ambiguous (missing section/paragraph number, unclear
   target), ask a clarifying question instead of guessing.


─── REFERENCE FORMATTING WORKFLOW ───
When the user asks for ABNT, APA, or any citation format, follow these steps
exactly — do NOT skip to formatting from a filename or URL alone.

⚠️  CRITICAL — Understanding source metadata fields:
    • "Title"     → this is the ARTICLE TITLE — use it for Crossref/web search
    • "File"      → local file path (NOT the article title, NOT a URL to cite)
    • "URL"       → web URL when the source is a web article; may be empty for locals
    • "DOI"       → DOI string when already indexed; may be empty

STEP 1 — Call `fetch_reference_metadata(title=<article title>, doi=<doi or "">, url=<url or "">)`.
    • NEVER pass a file path as `title`. From a filename like
        "A-parallel-attention_2026_Journal-of-Hydro.pdf", reconstruct the title:
        remove extension (".pdf"), year token ("_2026"), journal token ("_Journal-of-Hydro"),
        replace dashes/underscores with spaces → "A parallel attention framework..."
    • If the record already has a DOI, pass it in `doi=`.
    • This step also tries Crossref automatically.

STEP 2 (if web available and Step 1 found no DOI) — Call
    `search_article_online(title=<article title>)` to search Tavily.
    Inspect results for a DOI. Then call `get_bibtex_for_reference(<doi>)`.

STEP 3 (if still no DOI and a web URL is available) — Call
    `extract_web_text_from_url(<url>)` and extract authors, year, journal,
    title, and URL from the first ~1000 characters of the page.

STEP 4 — Format the reference using the collected data. For ABNT:
    SOBRENOME, Nome. Título. Periódico, v. X, n. Y, p. ZZ, ano. DOI/URL. Acesso em: DD mês AAAA.
    Fill every field you found. Only mark [?] for a field if it is truly absent
    after all lookup steps. NEVER use the file path as the reference URL.
─── RESPONSE FORMAT ───
For a normal answer, just write your response text.

For an edit proposal, use EXACTLY this format at the START of your reply:

REVISED_TEXT_START
<complete revised paragraph text>
REVISED_TEXT_END

Then optionally add explanation text after the block.

For confirming a pending edit, start your reply with: ACTION: APPLY_EDIT
For cancelling a pending edit, start your reply with: ACTION: CANCEL_EDIT
"""


def _compact_chat_history(
    chat_history: list[dict],
    provider_name: str,
) -> tuple[list[dict], str]:
    """Reduce chat payload size while preserving recent conversational context.

    Returns:
        (recent_messages, summary_text)
        - recent_messages: trimmed + per-message clipped history
        - summary_text: compact system summary of dropped older messages
    """
    if not chat_history:
        return [], ""

    is_groq = provider_name == "groq"
    max_recent_turns = 6 if is_groq else 10
    per_message_limit = 700 if is_groq else 2000
    older_summary_items = 8 if is_groq else 12

    def _clip(text: str, limit: int) -> str:
        cleaned = (text or "").strip()
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[:limit] + "... [truncated]"

    recent = chat_history[-max_recent_turns:]
    clipped_recent: list[dict] = []
    for msg in recent:
        clipped_recent.append(
            {
                "role": msg.get("role", ""),
                "content": _clip(str(msg.get("content", "")), per_message_limit),
            }
        )

    older = chat_history[:-max_recent_turns]
    if not older:
        return clipped_recent, ""

    # Keep a compact memory of older turns without sending full raw text.
    summary_lines = [
        "Conversation memory (older turns, compact):",
    ]
    for msg in older[-older_summary_items:]:
        role = str(msg.get("role", "assistant")).upper()
        text = _clip(str(msg.get("content", "")), 220 if is_groq else 320)
        summary_lines.append(f"- {role}: {text}")

    summary = "\n".join(summary_lines)
    return clipped_recent, summary


# ── helpers ───────────────────────────────────────────────────────────────


def _structure_summary(sections: list[dict]) -> str:
    """One-line-per-section summary with paragraph counts and reference counts."""
    if not sections:
        return "(empty document)"
    lines = []
    for i, sec in enumerate(sections, 1):
        n_para = len(sec.get("paragraphs", []))
        n_refs = len(sec.get("references", []))
        lines.append(
            f"  {i}. {sec['title']}  "
            f"({n_para} paragraph{'s' if n_para != 1 else ''}, "
            f"{n_refs} ref{'s' if n_refs != 1 else ''})"
        )
    return "\n".join(lines)


def _parse_agent_response(
    text: str,
    sections: list[dict],
    pending_edit: dict | None,
    target_hint: dict | None = None,
) -> dict:
    """Parse the LLM reply text into a structured result dict."""

    # ── Check for apply / cancel actions ──────────────────────────────
    first_line = text.strip().split("\n", 1)[0].strip().upper()

    if "ACTION:" in first_line and "APPLY_EDIT" in first_line:
        return {
            "reply": text,
            "edit_proposal": None,
            "action": "apply_edit",
        }
    if "ACTION:" in first_line and "CANCEL_EDIT" in first_line:
        return {
            "reply": text,
            "edit_proposal": None,
            "action": "cancel_edit",
        }

    # Also detect implicit confirm/cancel in short replies
    stripped = text.strip().lower()
    if pending_edit and stripped in {
        "confirm", "yes", "apply", "apply edit", "confirm edit",
        "confirmar", "sim", "aplicar",
    }:
        return {"reply": text, "edit_proposal": None, "action": "apply_edit"}
    if pending_edit and stripped in {
        "cancel", "no", "discard", "cancel edit",
        "cancelar", "não", "descartar",
    }:
        return {"reply": text, "edit_proposal": None, "action": "cancel_edit"}

    # ── Check for edit proposal block ─────────────────────────────────
    proposal = _extract_edit_proposal(text, sections, target_hint)
    if proposal is not None:
        return {
            "reply": text,
            "edit_proposal": proposal,
            "action": "edit_proposal",
        }

    # ── Default: plain answer ─────────────────────────────────────────
    return {
        "reply": text,
        "edit_proposal": None,
        "action": "answer",
    }


def _extract_edit_proposal(
    text: str,
    sections: list[dict],
    target_hint: dict | None = None,
) -> dict | None:
    """Parse edit proposal from output.

    Preferred format is a plain REVISED_TEXT block. Legacy EDIT_PROPOSAL
    format is still accepted for compatibility.
    """
    normalized = text.replace("\r\n", "\n")
    normalized = re.sub(r"```(?:markdown|md)?\s*", "", normalized, flags=re.IGNORECASE)
    normalized = normalized.replace("```", "")

    revised_block = re.search(
        r"REVISED_TEXT_START\s*\n(.*?)\nREVISED_TEXT_END",
        normalized,
        re.DOTALL | re.IGNORECASE,
    )
    if revised_block is not None and target_hint:
        revised = revised_block.group(1).strip()
        if not revised:
            return None
        return {
            "section_title": str(target_hint.get("section_title", "")),
            "paragraph_index": int(target_hint.get("paragraph_index", 0)),
            "start": int(target_hint.get("start", 0)),
            "end": int(target_hint.get("end", 0)),
            "before": str(target_hint.get("before", "")),
            "after": revised,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    m = re.search(
        r"EDIT_PROPOSAL\s*\n"
        r"(?:SECTION_NUMBER\s*:\s*(\d+)\s*\n)?"
        r"(?:SECTION_TITLE\s*:\s*(.*?)\s*\n)?"
        r"PARAGRAPH_NUMBER\s*:\s*(\d+)\s*\n"
        r"REVISED_TEXT_START\s*\n"
        r"(.*?)\n"
        r"REVISED_TEXT_END",
        normalized,
        re.DOTALL | re.IGNORECASE,
    )
    if m is None:
        return None

    sec_num_raw = m.group(1)
    sec_title_raw = (m.group(2) or "").strip()
    para_num = int(m.group(3))
    revised = m.group(4).strip()

    section = None
    if sec_num_raw:
        sec_idx = int(sec_num_raw) - 1
        if 0 <= sec_idx < len(sections):
            section = sections[sec_idx]

    if section is None and sec_title_raw:
        title_norm = sec_title_raw.lower().strip()
        for sec in sections:
            if sec.get("title", "").lower().strip() == title_norm:
                section = sec
                break
        if section is None:
            for sec in sections:
                if title_norm in sec.get("title", "").lower():
                    section = sec
                    break

    if section is None:
        return None

    para_idx = para_num - 1
    paragraphs = section.get("paragraphs", [])
    if para_idx < 0 or para_idx >= len(paragraphs):
        return None
    paragraph = paragraphs[para_idx]

    return {
        "section_title": section["title"],
        "paragraph_index": para_idx,
        "start": paragraph["start"],
        "end": paragraph["end"],
        "before": paragraph["text"],
        "after": revised,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
