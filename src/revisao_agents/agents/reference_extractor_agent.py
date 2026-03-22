# src/revisao_agents/agents/reference_extractor_agent.py
"""
Reference Extractor Agent.

Receives a raw text block of reference entries from a document (from any
section heading that looks like a bibliography), enriches each entry with
complete metadata via tool calls (MongoDB → CrossRef → Tavily), and returns
a structured enriched list ready to be passed to the formatter agent.

Uses LangGraph ``create_react_agent`` for built-in ReAct loop and
first-class LangSmith tracing.
"""

from __future__ import annotations

import logging
from datetime import date

from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import create_agent

from ..tools.reference_tools import get_reference_tools
from ..utils.llm_utils.llm_providers import get_llm, create_agent_easy
from ..utils.llm_utils.prompt_loader import load_prompt

logger = logging.getLogger(__name__)


def _today() -> str:
    return date.today().strftime("%d %b. %Y").lower()


def _count_entries(raw_references: str) -> int:
    count = 0
    for line in raw_references.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            count += 1
    return max(count, 1)


def run_reference_extractor_agent(
    raw_references: str,
    citation_context: dict[int, list[str]] | None = None,
    allow_web: bool = True,
) -> str:
    """Run the reference extractor agent on a raw list of reference entries.

    The agent enriches each entry using MongoDB, CrossRef, ArXiv, local PDF
    reading, and optionally Tavily, then returns a structured pipe-separated
    list ready for the formatter agent.

    Args:
        raw_references: Raw text block with one reference entry per line.
                        May include numbered [N] entries, file paths, partial
                        citations, author-year references, etc.
        citation_context: Optional mapping from reference number to a list of
                          paragraphs (up to 2) from the document body that cite
                          that reference. Used for Type-B in-text citations.
        allow_web: Whether to include Tavily web search tool.

    Returns:
        Enriched structured reference list as a string, or an error message.
    """
    if not raw_references or not raw_references.strip():
        return "No references provided to the extractor agent."

    total_items = _count_entries(raw_references)

    if citation_context:
        ctx_lines: list[str] = []
        for ref_num, paragraphs in sorted(citation_context.items()):
            ctx_lines.append(f"[{ref_num}]:")
            for para in paragraphs:
                ctx_lines.append(f"  - {para.strip()}")
        citation_context_text = "\n".join(ctx_lines)
    else:
        citation_context_text = "(no citation context provided)"

    allow_web_hint = (
        (
            "## WEB SEARCH DISABLED\n"
            "  The `search_web_for_reference` tool is NOT available in this session.\n"
            "  Skip ALL web search steps — do NOT attempt to call `search_web_for_reference`.\n"
        )
        if not allow_web
        else ""
    )

    try:
        prompt = load_prompt(
            "common/reference_extractor",
            today_date=_today(),
            total_items=str(total_items),
            raw_references=raw_references,
            citation_context=citation_context_text,
            allow_web_hint=allow_web_hint,
        )
    except Exception as exc:
        logger.error("Failed to load reference_extractor prompt: %s", exc)
        return f"Extractor prompt load error: {exc}"

    tools = get_reference_tools(allow_web=allow_web)

    agent = create_agent_easy(
        tools=tools,
        system_prompt=prompt.text,
        temperature=prompt.temperature,
        name="reference_extractor",
    )

    user_msg = HumanMessage(
        content=(
            f"Please enrich all {total_items} reference entries listed in the "
            "system prompt and return the structured output."
        )
    )

    try:
        result = agent.invoke({"messages": [user_msg]})
    except Exception as exc:
        logger.error("Extractor agent invocation failed: %s", exc)
        return f"Extractor agent error: {exc}"

    messages = result.get("messages", [])
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        if getattr(msg, "tool_calls", None):
            continue
        content = getattr(msg, "content", "") or ""
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        if content.strip():
            return content.strip()

    return "Extractor agent returned an empty response."
