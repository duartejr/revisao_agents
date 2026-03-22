# src/revisao_agents/agents/review_agent.py
"""
ReAct review agent for the interactive review chatbot.

Uses tool-calling LLM to reason about user requests, retrieve evidence,
and produce structured actions (answer, edit proposal, confirm/cancel).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
from datetime import datetime
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from ..tools.review_tools import get_review_tools
from ..utils.llm_utils.llm_providers import get_llm as get_raw_llm
from ..utils.llm_utils.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

MAX_AGENT_ITERATIONS = 6


def _clip_text(text: str, limit: int) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + "... [truncated]"


def _normalize_tool_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Normalize common tool-arg typing issues from provider tool calls.

    Some providers may emit numeric args as strings (e.g. ``{"k": "3"}``),
    which can fail provider-side schema validation on retries.
    """
    normalized = dict(args or {})
    int_like_keys = {"k", "n", "max_results", "top_k", "limit"}

    for key in int_like_keys:
        value = normalized.get(key)
        if isinstance(value, str):
            stripped = value.strip()
            if re.fullmatch(r"[-+]?\d+", stripped):
                with contextlib.suppress(Exception):
                    normalized[key] = int(stripped)

    return normalized


def _recover_tool_call_from_exception(exc: Exception) -> dict[str, Any] | None:
    """Recover a tool call from provider-side failed_generation payload.

    Some providers (notably Groq on certain models) can reject tool calls with
    ``tool_use_failed`` and return the raw attempted call as text, e.g.:
    ``<function=tool_name({"arg": "value"})</function>``.
    This parser extracts the tool name and JSON args so the agent can continue.

    Args:
        exc: The exception raised by the LLM provider, which may contain a failed_generation payload

    Returns:
        A dict with 'name' and 'args' keys if a tool call was successfully recovered, or None otherwise.
    """
    text = str(exc)
    if "tool_use_failed" not in text and "failed_generation" not in text:
        return None

    match = re.search(
        r"<function=([a-zA-Z_][a-zA-Z0-9_]*)\((\{.*?\})\)</function>",
        text,
        flags=re.DOTALL,
    )
    if match:
        tool_name = match.group(1)
        raw_args = match.group(2)
    else:
        match = re.search(
            r"<function=([a-zA-Z_][a-zA-Z0-9_]*)>(\{.*?\})</function>",
            text,
            flags=re.DOTALL,
        )
        if not match:
            match = re.search(
                r'"failed_generation"\s*:\s*"<function=([a-zA-Z_][a-zA-Z0-9_]*)>(\{.*?\})</function>',
                text,
                flags=re.DOTALL,
            )
        if not match:
            return None
        tool_name = match.group(1)
        raw_args = match.group(2)

    raw_args = raw_args.replace('\\"', '"')
    raw_args = raw_args.replace("\\n", " ").strip()

    if not raw_args.startswith("{"):
        return None

    try:
        tool_args = json.loads(raw_args)
    except Exception:
        return None

    if not isinstance(tool_args, dict):
        return None

    tool_args = _normalize_tool_args(tool_name, tool_args)

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
    effective_max_iterations = min(max_iterations, 4) if is_groq else max_iterations
    tool_result_limit = 1200 if is_groq else 5000

    response: Any = None
    for iteration in range(effective_max_iterations):
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
            tool_args = _normalize_tool_args(tool_name, tc["args"])
            tool_fn = tool_map.get(tool_name)

            if tool_fn is None:
                result_str = f"Error: unknown tool '{tool_name}'"
            else:
                try:
                    result_str = str(tool_fn.invoke(tool_args))
                except Exception as exc:
                    result_str = f"Error executing {tool_name}: {exc}"

            result_str = _clip_text(result_str, tool_result_limit)

            trace.append(
                {
                    "iteration": iteration,
                    "tool": tool_name,
                    "args": tool_args,
                    "result_length": len(result_str),
                }
            )
            messages.append(
                ToolMessage(content=result_str, tool_call_id=tc["id"]),
            )

    # ── Empty-content fallback ────────────────────────────────────────
    # Some providers can return empty content after tool execution.
    # Retry once with a plain (tool-free) LLM to force a user-facing text.
    if response is not None:
        raw_check = _extract_message_text(response)
        if not raw_check:
            logger.warning(
                "Provider returned empty content after tool loop. "
                "Retrying once without tools for final text output."
            )
            try:
                plain_llm = get_raw_llm(temperature=0.2)
                response = plain_llm.invoke(messages)
            except Exception as exc:
                logger.warning("Plain-LLM fallback failed: %s", exc)

    # ── Parse response ────────────────────────────────────────────────
    reply_text = _extract_message_text(response)
    if not reply_text:
        reply_text = (
            "I executed the requested analysis steps, but the model returned "
            "an empty final message. Please retry the same instruction once."
        )

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
    """Load and render the system prompt from YAML with contextualized placeholders.

    Args:
        document_content: Full markdown text of the document under review.
        sections: List of section dicts with titles and paragraph counts.
        allow_web: Whether web search tools are available in this turn.
        pending_edit: Dict with pending edit details, or None if no pending edit.
        include_full_document: Whether to include the full document content in the prompt (defaults to True).

    Returns:
        Rendered system prompt string ready to send to the LLM.
    """
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

    # Load the prompt from YAML and render with placeholders
    prompt = load_prompt(
        "common/review_agent_system",
        year=year,
        structure=structure,
        doc_block=doc_block,
        pending_block=pending_block,
        web_block=web_block,
    )
    return prompt.text


def _compact_chat_history(
    chat_history: list[dict],
    provider_name: str,
) -> tuple[list[dict], str]:
    """Reduce chat payload size while preserving recent conversational context.

    Args:
        chat_history: List of message dicts with 'role' and 'content'.
        provider_name: Name of the LLM provider (e.g., "gemini", "groq") to adjust limits.

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
        return _clip_text(text, limit)

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
    """One-line-per-section summary with paragraph counts and reference counts.

    Args:
        sections: List of section dicts, each with 'title', 'paragraphs', and 'references' keys.

    Returns:
        A formatted string summarizing the document structure, or "(empty document)" if no sections.
    """
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


def _extract_message_text(message: Any) -> str:
    """Extract human-readable text from a provider message.

    Avoids falling back to ``str(message)``, which may dump internal
    provider metadata (tool calls, thought signatures, token details).
    """
    if message is None:
        return ""

    raw_content = getattr(message, "content", None)
    if isinstance(raw_content, list):
        parts: list[str] = []
        for part in raw_content:
            if isinstance(part, dict):
                text_part = part.get("text", "")
                if text_part:
                    parts.append(str(text_part))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts).strip()

    if isinstance(raw_content, str):
        return raw_content.strip()

    return ""


def _parse_agent_response(
    text: str,
    sections: list[dict],
    pending_edit: dict | None,
    target_hint: dict | None = None,
) -> dict:
    """Parse the LLM reply text into a structured result dict.

    Args:
        text: The raw text content of the LLM's reply.
        sections: List of document sections for context in parsing edit proposals.
        pending_edit: Current pending edit dict, or None if no pending edit.
        target_hint: Optional dict with hints about the target section/paragraph for edit proposals.

    Returns:
        A dict with keys:
        - 'reply': the original text reply from the LLM
        - 'edit_proposal': a dict with edit proposal details if an edit was proposed, or None
        - 'action': one of "answer", "edit_proposal", "apply_edit", "cancel_edit"
    """

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
        "confirm",
        "yes",
        "apply",
        "apply edit",
        "confirm edit",
        "confirmar",
        "confirmar edição",
        "confirmar edicao",
        "sim",
        "aplicar",
        "aplicar edição",
        "aplicar edicao",
    }:
        return {"reply": text, "edit_proposal": None, "action": "apply_edit"}
    if pending_edit and stripped in {
        "cancel",
        "no",
        "discard",
        "cancel edit",
        "cancelar",
        "cancelar edição",
        "cancelar edicao",
        "não",
        "nao",
        "descartar",
        "descartar edição",
        "descartar edicao",
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

    Args:
        text: The raw text content of the LLM's reply.
        sections: List of document sections for context in parsing edit proposals.
        target_hint: Optional dict with hints about the target section/paragraph for edit proposals.

    Returns:
        A dict with edit proposal details if an edit was proposed, or None if no valid proposal was found.
    """
    normalized = text.replace("\r\n", "\n")
    normalized = re.sub(r"```(?:markdown|md)?\s*", "", normalized, flags=re.IGNORECASE)
    normalized = normalized.replace("```", "")

    revised_block = re.search(
        r"(?:REVISED_TEXT_START|TEXTO_REVISADO_IN[ÍI]CIO)\s*\n(.*?)\n(?:REVISED_TEXT_END|TEXTO_REVISADO_FIM)",
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
        r"(?:EDIT_PROPOSAL|PROPOSTA_DE_EDI[CÇ][AÃ]O)\s*\n"
        r"(?:(?:SECTION_NUMBER|N[ÚU]MERO_DA?_SE[ÇC][AÃ]O)\s*:\s*(\d+)\s*\n)?"
        r"(?:(?:SECTION_TITLE|T[ÍI]TULO_DA?_SE[ÇC][AÃ]O)\s*:\s*(.*?)\s*\n)?"
        r"(?:PARAGRAPH_NUMBER|N[ÚU]MERO_D[EO]_PAR[ÁA]GRAFO)\s*:\s*(\d+)\s*\n"
        r"(?:REVISED_TEXT_START|TEXTO_REVISADO_IN[ÍI]CIO)\s*\n"
        r"(.*?)\n"
        r"(?:REVISED_TEXT_END|TEXTO_REVISADO_FIM)",
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
