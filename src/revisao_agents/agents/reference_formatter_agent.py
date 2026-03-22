# src/revisao_agents/agents/reference_formatter_agent.py
"""
Reference Formatter Agent.

Receives a list of references (raw user text, or structured output from the
extractor agent) and formats every entry into ABNT NBR 6023 standard,
calling tools (CrossRef, MongoDB, Tavily) when metadata is still incomplete.

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


def run_reference_formatter_agent(
    references_input: str,
    allow_web: bool = True,
) -> str:
    """Run the reference formatter agent on a list of references.

    Formats every reference into ABNT NBR 6023 standard, using tool calls
    to look up any missing metadata (DOI, authors, year, journal, etc.).

    Args:
        references_input: Text containing the references to format. May be:
            - Raw user text pasted into the chat
            - Structured pipe-separated output from the extractor agent
            - Any mix of partial citations, file paths, DOIs, or URLs
        allow_web: Whether to include Tavily web search tool.

    Returns:
        Formatted ABNT list as markdown string, or an error message.
    """
    if not references_input or not references_input.strip():
        return "No references provided to the formatter agent."

    try:
        allow_web_hint = (
            "## WEB SEARCH DISABLED\n"
            "  The `search_web_for_reference` tool is NOT available in this session.\n"
            "  Skip step 5 — do NOT attempt to call `search_web_for_reference`.\n"
        ) if not allow_web else ""
        prompt = load_prompt(
            "common/reference_formatter",
            today_date=_today(),
            references_input=references_input,
            allow_web_hint=allow_web_hint,
        )
    except Exception as exc:
        logger.error("Failed to load reference_formatter prompt: %s", exc)
        return f"Formatter prompt load error: {exc}"

    tools = get_reference_tools(allow_web=allow_web)

    agent = create_agent_easy(
        tools=tools,
        system_prompt=prompt.text,
        temperature=prompt.temperature,
        name="reference_formatter",
    )

    user_msg = HumanMessage(
        content=(
            "Please format all references listed in the system prompt into "
            "perfect ABNT NBR 6023 citations. Use tools to resolve any missing "
            "metadata before formatting. Return only the final numbered ABNT list."
        )
    )

    try:
        result = agent.invoke({"messages": [user_msg]})
    except Exception as exc:
        logger.error("Formatter agent invocation failed: %s", exc)
        return f"Formatter agent error: {exc}"

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

    return "Formatter agent returned an empty response."
