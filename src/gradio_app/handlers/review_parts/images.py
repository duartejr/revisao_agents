"""Image suggestion helpers for the interactive review handler.

Detects image-related requests and builds the scope description and
confirmation prompt sent to the image suggestion agent.
"""

from __future__ import annotations

import re

from ..base import _localized_text
from .document import _extract_quoted_snippet, _resolve_section_index

_IMAGE_REQUEST_KEYWORDS = [
    "imagem",
    "imagens",
    "figur",
    "ilustr",
    "foto",
    "diagrama",
    "inserir imagem",
    "adicionar imagem",
    "sugerir imagem",
    "buscar imagem",
    "encontrar imagem",
    "colocar imagem",
    "incluir imagem",
    "imagem para",
    "imagens para",
    "image",
    "images",
    "figure",
    "illustration",
    "diagram",
    "picture",
    "insert image",
    "add image",
    "suggest image",
    "find image",
    "search image",
    "place image",
]


def _is_image_request(user_text: str) -> bool:
    """Return True when the user is asking for image suggestions.

    Args:
        user_text: The input text from the user.

    Returns:
        True if the message indicates an image request.
    """
    text = user_text.lower()
    return any(kw in text for kw in _IMAGE_REQUEST_KEYWORDS)


def _build_image_scope_description(
    user_text: str, sections: list[dict], language: str = "en"
) -> tuple[str, str]:
    """Derive a human-readable scope and a document excerpt for the image agent.

    Args:
        user_text: The input text from the user indicating the scope.
        sections: Parsed document sections with paragraphs.
        language: Language code for localization.

    Returns:
        A tuple containing the scope description and a markdown excerpt.
    """
    text = user_text.lower()

    def _paragraphs_excerpt(section: dict, max_chars: int = 3500) -> str:
        accumulated = f"## {section['title']}\n\n"
        for i, para in enumerate(section.get("paragraphs", []), 1):
            para_text = para.get("text", "").strip()
            if not para_text:
                continue
            block = f"[PARAGRAPH {i}]\n{para_text}\n\n"
            if len(accumulated) + len(block) > max_chars:
                break
            accumulated += block
        return accumulated

    sec_idx = _resolve_section_index(user_text, sections)
    if sec_idx is not None and 0 <= sec_idx < len(sections):
        section = sections[sec_idx]
        scope = _localized_text(
            language,
            f"seção {sec_idx + 1} — {section['title']}",
            f"section {sec_idx + 1} — {section['title']}",
        )
        excerpt = _paragraphs_excerpt(section)
        return scope, excerpt

    para_match = re.search(r"par[áa]grafo\s*(\d+)|paragraph\s*(\d+)", text)
    if para_match and sections:
        num_str = para_match.group(1) or para_match.group(2)
        para_num = int(num_str) - 1
        section = None
        for candidate in sections:
            title = str(candidate.get("title", "")).strip()
            if title and title.lower() in text:
                section = candidate
                break
        if section is None:
            section = sections[0]
        paragraphs = section.get("paragraphs", [])
        if 0 <= para_num < len(paragraphs):
            para = paragraphs[para_num]
            scope = _localized_text(
                language,
                f"parágrafo {para_num + 1} da seção '{section['title']}'",
                f"paragraph {para_num + 1} of section '{section['title']}'",
            )
            excerpt = (
                f"## {section['title']}\n\n[PARAGRAPH {para_num + 1}]\n{para.get('text', '')}\n"
            )
            return scope, excerpt

    snippet = _extract_quoted_snippet(user_text)
    if snippet and sections:
        for section in sections:
            for p_idx, paragraph in enumerate(section.get("paragraphs", [])):
                if snippet.lower() in paragraph.get("text", "").lower():
                    scope = _localized_text(
                        language,
                        f"parágrafo contendo \"{snippet[:60]}\"... na seção '{section['title']}'",
                        f"paragraph containing \"{snippet[:60]}\"... in section '{section['title']}'",
                    )
                    excerpt = (
                        f"## {section['title']}\n\n"
                        f"[PARAGRAPH {p_idx + 1}]\n{paragraph.get('text', '')}\n"
                    )
                    return scope, excerpt

    scope = _localized_text(
        language, "todas as seções do documento", "all sections of the document"
    )
    parts: list[str] = []
    total = 0
    for sec in sections[:6]:
        paragraphs = sec.get("paragraphs", [])
        if not paragraphs:
            continue
        block = f"## {sec['title']}\n\n"
        for i, para in enumerate(paragraphs[:3], 1):
            para_text = para.get("text", "").strip()
            if para_text:
                block += f"[PARAGRAPH {i}]\n{para_text}\n\n"
        if total + len(block) > 4000:
            break
        parts.append(block)
        total += len(block)
    excerpt = "\n".join(parts)
    return scope, excerpt


def _build_image_confirmation_prompt(scope: str, language: str) -> str:
    """Return a confirmation prompt asking user to confirm image search scope.

    Args:
        scope: The scope description that will be confirmed.
        language: Language code for localization.

    Returns:
        A localized prompt string.
    """
    return _localized_text(
        language,
        f"Vou buscar imagens para ilustrar: **{scope}**.\n\n"
        "Confirme o escopo ou especifique uma seção/parágrafo diferente.\n"
        "Responda **sim** para confirmar ou descreva o escopo desejado.",
        f"I will search for images to illustrate: **{scope}**.\n\n"
        "Confirm the scope or specify a different section/paragraph.\n"
        "Reply **yes** to confirm or describe the desired scope.",
    )
