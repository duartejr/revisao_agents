# src/revisao_agents/agents/image_suggestion_agent.py
"""
Image Suggestion Agent.

Receives a document excerpt and a scope description (all sections, a specific
section, or a specific paragraph) and searches for academic/technical images
to illustrate the content.  Returns Markdown figure blocks ready to insert,
with best-effort source references, using LangGraph create_react_agent.
"""

from __future__ import annotations

import logging
import re
from datetime import date

from langchain_core.messages import AIMessage, HumanMessage

from ..tools.image_tools import get_image_tools
from ..utils.llm_utils.llm_providers import create_agent_easy
from ..utils.llm_utils.prompt_loader import load_prompt

logger = logging.getLogger(__name__)


def _today() -> str:
    return date.today().strftime("%d %b. %Y").lower()


def _detect_language(text: str) -> str:
    """Detect document language (pt or en) from a text sample."""
    pt_words = [
        r"\bcomo\b",
        r"\bpara\b",
        r"\bque\b",
        r"\bnão\b",
        r"\bsão\b",
        r"\buma\b",
        r"\bpor\b",
        r"\bcom\b",
        r"\besta\b",
        r"\bneste\b",
        r"\besses\b",
        r"\bmodelos\b",
        r"\bdados\b",
        r"\bprevisão\b",
        r"\bseção\b",
        r"\bresultados\b",
    ]
    en_words = [
        r"\bthe\b",
        r"\bof\b",
        r"\band\b",
        r"\bin\b",
        r"\bto\b",
        r"\bis\b",
        r"\bare\b",
        r"\bthis\b",
        r"\bthat\b",
        r"\bfor\b",
        r"\bmodels\b",
        r"\bdata\b",
        r"\bforecast\b",
        r"\bsection\b",
        r"\bresults\b",
    ]
    sample = text[:2000].lower()
    pt_score = sum(1 for p in pt_words if re.search(p, sample))
    en_score = sum(1 for p in en_words if re.search(p, sample))
    return "pt" if pt_score >= en_score else "en"


def run_image_suggestion_agent(
    document_excerpt: str,
    user_request: str,
    scope_description: str,
) -> str:
    """Run the image suggestion agent on a document excerpt and scope.

    Args:
        document_excerpt: The relevant part of the document to illustrate
            (e.g., a single section or the whole document condensed to a
            few key paragraphs — keep under ~4 000 chars for latency).
        user_request: The original user message requesting images, used for
            language detection and framing.
        scope_description: Human-readable description of what should be
            illustrated, e.g.  "all sections", "section 2 — Methodology",
            or "paragraph 3 of Introduction".

    Returns:
        Markdown string with image suggestions and best-effort source
        references, or an error message.
    """
    if not document_excerpt and not scope_description:
        return "No document content or scope provided to the image suggestion agent."

    # Use user_request as a secondary signal when the excerpt is empty or
    # language detection is inconclusive (both scores 0).
    doc_language = _detect_language(document_excerpt or "")
    if doc_language == "en" and not (document_excerpt or "").strip():
        doc_language = _detect_language(user_request or "")
    if doc_language == "pt":
        lang_instruction = (
            "Write ALL your output in Brazilian Portuguese (same language as the document). "
            "Use 'Figura' (not 'Figure'), 'ver Figura' (not 'see Figure')."
        )
        figure_label = "Figura"
        see_figure_label = "ver Figura"
        available_at_label = "Dispon\u00edvel em:"
        accessed_label = "Acesso em:"
        unknown_source_label = "FONTE N\u00c3O IDENTIFICADA"
        no_images_msg = "Nenhuma imagem t\u00e9cnica adequada encontrada para esta se\u00e7\u00e3o."
    else:
        lang_instruction = (
            "Write ALL your output in English (same language as the document). "
            "Use 'Figure', 'see Figure'."
        )
        figure_label = "Figure"
        see_figure_label = "see Figure"
        available_at_label = "Available at:"
        accessed_label = "Accessed:"
        unknown_source_label = "SOURCE NOT IDENTIFIED"
        no_images_msg = "No suitable technical image found for this section."

    try:
        prompt = load_prompt(
            "common/image_suggestion",
            today_date=_today(),
            document_excerpt=document_excerpt if document_excerpt else "(not provided)",
            user_request=user_request or "(not provided)",
            scope_description=scope_description or "all sections of the document",
            lang_instruction=lang_instruction,
            figure_label=figure_label,
            see_figure_label=see_figure_label,
            available_at_label=available_at_label,
            accessed_label=accessed_label,
            unknown_source_label=unknown_source_label,
            no_images_msg=no_images_msg,
        )
    except Exception as exc:
        logger.error("Failed to load image_suggestion prompt: %s", exc)
        return f"Image suggestion prompt load error: {exc}"

    tools = get_image_tools()

    agent = create_agent_easy(
        tools=tools,
        system_prompt=prompt.text,
        temperature=prompt.temperature,
        name="image_suggestion",
    )

    user_msg = HumanMessage(
        content=(
            f"Search for images to illustrate: {scope_description}. "
            f"User request: {user_request}. "
            f"IMPORTANT: {lang_instruction} "
            f"Use '{figure_label}' for all figure labels and captions. "
            f"Unavailable source label: '{unknown_source_label}'. "
            f"Reference 'available at' label: '{available_at_label}'. "
            f"Reference 'accessed' label: '{accessed_label}'. "
            "Do not claim full bibliographic metadata verification; include the "
            "source limitation warning once at the top of the response."
        )
    )

    try:
        result = agent.invoke({"messages": [user_msg]})
    except Exception as exc:
        logger.error("Image suggestion agent invocation failed: %s", exc)
        return f"Image suggestion agent error: {exc}"

    messages = result.get("messages", [])
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        if getattr(msg, "tool_calls", None):
            continue
        content = getattr(msg, "content", "") or ""
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
        if content.strip():
            return content.strip()

    return "Image suggestion agent returned an empty response."
