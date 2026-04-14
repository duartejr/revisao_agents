"""Markdown document parsing utilities for the review handler.

Provides helpers for listing review files, splitting documents into
sections and paragraphs, and resolving user-supplied section/paragraph
targets for edit proposals.
"""

from __future__ import annotations

import glob
import os
import re
from datetime import datetime

from src.revisao_agents.config import REVIEWS_DIR


def list_review_files() -> list[str]:
    """List available review files in the reviews directory.

    Returns:
        Sorted list of review file paths (*.md files in reviews/).
    """
    return sorted(glob.glob(os.path.join(REVIEWS_DIR, "*.md")))


def _working_copy_path(review_file: str) -> str:
    """Generate a unique working copy path for a given review file.

    Args:
        review_file: The original review file path for which to create a working copy.

    Returns:
        A new file path for the working copy, incorporating a timestamp to ensure uniqueness.
    """
    base_dir = os.path.dirname(review_file) or REVIEWS_DIR
    base_name = os.path.basename(review_file)
    name, ext = os.path.splitext(base_name)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    return os.path.join(base_dir, f"{name}__review_edit_{ts}{ext}")


def _split_sections(markdown: str) -> list[dict]:
    """Split a markdown document into sections based on "## " headers.

    Args:
        markdown: The full markdown text of the review document.

    Returns:
        A list of sections, where each section is a dictionary containing the title, text, body,
        paragraphs, and references.
    """
    lines = markdown.splitlines(keepends=True)
    if not lines:
        return []

    line_offsets: list[int] = []
    acc = 0
    for line in lines:
        line_offsets.append(acc)
        acc += len(line)

    headers: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        match = re.match(r"^##\s+(.+?)\s*$", line.strip("\n"))
        if match:
            headers.append((idx, match.group(1).strip()))

    sections: list[dict] = []
    for header_idx, (start_line, title) in enumerate(headers):
        next_start_line = (
            headers[header_idx + 1][0] if header_idx + 1 < len(headers) else len(lines)
        )
        section_start = line_offsets[start_line]
        section_end = (
            line_offsets[next_start_line] if next_start_line < len(line_offsets) else len(markdown)
        )
        section_text = markdown[section_start:section_end]

        references_start_line: int | None = None
        for i in range(start_line + 1, next_start_line):
            if re.match(
                r"^###\s+(?:References\s+for\s+this\s+section|Refer[êe]ncias\s+desta\s+se[çc][ãa]o)\s*$",
                lines[i].strip(),
                re.IGNORECASE,
            ):
                references_start_line = i
                break

        body_end_line = (
            references_start_line if references_start_line is not None else next_start_line
        )
        body_start = (
            line_offsets[start_line + 1] if start_line + 1 < len(line_offsets) else section_start
        )
        body_end = line_offsets[body_end_line] if body_end_line < len(line_offsets) else section_end
        body_text = markdown[body_start:body_end].strip()

        references: list[str] = []
        if references_start_line is not None:
            for i in range(references_start_line + 1, next_start_line):
                ref_line = lines[i].strip()
                if re.match(r"^\[\d+\]", ref_line):
                    references.append(ref_line)

        paragraphs: list[dict] = []
        current_lines: list[str] = []
        current_start: int | None = None
        for i in range(start_line + 1, body_end_line):
            stripped = lines[i].strip()
            if not stripped or stripped.startswith("<!--") or stripped.startswith("### "):
                if current_lines:
                    para_text = "".join(current_lines).strip()
                    if para_text:
                        paragraphs.append(
                            {
                                "text": para_text,
                                "start": current_start,
                                "end": line_offsets[i],
                            }
                        )
                    current_lines = []
                    current_start = None
                continue
            if current_start is None:
                current_start = line_offsets[i]
            current_lines.append(lines[i])

        if current_lines and current_start is not None:
            para_text = "".join(current_lines).strip()
            if para_text:
                paragraphs.append(
                    {
                        "text": para_text,
                        "start": current_start,
                        "end": body_end,
                    }
                )

        sections.append(
            {
                "title": title,
                "start": section_start,
                "end": section_end,
                "text": section_text,
                "body": body_text,
                "paragraphs": paragraphs,
                "references": references,
            }
        )

    return sections


def _resolve_section_index(user_text: str, sections: list[dict]) -> int | None:
    """Resolve the intended section index from user text based on heuristics.

    Args:
        user_text: The input text from the user indicating a section.
        sections: The list of section dictionaries to match against.

    Returns:
        The index of the matched section in the sections list, or None if no match is found.
    """
    text = user_text.lower()

    sec_match = re.search(
        r"(?:section|sec|se[çc][ãa]o|chapter|cap[ií]tulo)\s*[^\w\d]*(\d+)",
        text,
    )
    if sec_match:
        number = sec_match.group(1)
        for idx, section in enumerate(sections):
            if re.match(rf"^{number}[\.)\s]", section["title"], flags=re.IGNORECASE):
                return idx

    md_match = re.search(r"##\s*(\d+)[.\s]", text)
    if md_match:
        number = md_match.group(1)
        for idx, section in enumerate(sections):
            if re.match(rf"^{number}[\.)\s]", section["title"], flags=re.IGNORECASE):
                return idx

    if re.search(r"\b(?:conclusion|conclus[ãa]o)\b", text):
        for idx, section in enumerate(sections):
            t = section["title"].lower()
            if re.search(r"\b(?:conclusion|conclus[ãa]o)\b", t):
                return idx

    stopwords = {
        "a",
        "o",
        "e",
        "de",
        "da",
        "do",
        "em",
        "para",
        "no",
        "na",
        "com",
        "the",
        "of",
        "in",
        "on",
        "at",
        "to",
        "and",
        "for",
        "or",
        "an",
        "quero",
        "contexto",
        "seção",
        "section",
        "apenas",
        "only",
        "sobre",
        "about",
        "this",
        "esse",
        "este",
        "essa",
    }
    user_words = {
        w for w in re.findall(r"[a-záàâãéèêíïóôõöúüçñ]+", text) if w not in stopwords and len(w) > 3
    }
    best_idx: int | None = None
    best_hits = 0
    for idx, section in enumerate(sections):
        title_words = set(re.findall(r"[a-záàâãéèêíïóôõöúüçñ]+", section["title"].lower()))
        hits = len(user_words & title_words)
        if hits > best_hits:
            best_hits = hits
            best_idx = idx
    if best_hits >= 1:
        return best_idx

    return None


def _resolve_paragraph_index(user_text: str, paragraph_count: int) -> int | None:
    """Resolve the intended paragraph index from user text based on heuristics.

    Args:
        user_text: The input text from the user indicating a paragraph.
        paragraph_count: The total number of paragraphs in the section to validate indices against.

    Returns:
        The index of the matched paragraph, or None if no match is found.
    """
    if paragraph_count <= 0:
        return None
    text = user_text.lower()
    if re.search(r"\b(?:last\s+paragraph|[úu]ltimo\s+par[áa]grafo)\b", text):
        return paragraph_count - 1
    para_match = re.search(r"(?:paragraph|par[áa]grafo)\s*(\d+)", text)
    if para_match:
        idx = int(para_match.group(1)) - 1
        return idx if 0 <= idx < paragraph_count else None

    ordinals = {
        "first": 0,
        "second": 1,
        "third": 2,
        "fourth": 3,
        "fifth": 4,
        "primeiro": 0,
        "segundo": 1,
        "terceiro": 2,
        "quarto": 3,
        "quinto": 4,
    }
    for token, idx in ordinals.items():
        if token in text and re.search(r"\b(?:paragraph|par[áa]grafo)\b", text):
            return idx if idx < paragraph_count else None
    return None


def _extract_quoted_snippet(user_text: str) -> str:
    """Extract a quoted snippet from the user text.

    Args:
        user_text: The input text from the user containing a quoted snippet.

    Returns:
        The extracted quoted snippet, or an empty string if no match is found.
    """
    match = re.search(r'"([^"]{12,})"', user_text)
    if match:
        return match.group(1).strip()
    match = re.search(r"'([^']{12,})'", user_text)
    return match.group(1).strip() if match else ""


def _resolve_target_hint(
    user_text: str,
    sections: list[dict],
    last_target: dict | None = None,
) -> dict | None:
    """Resolve target paragraph for edit proposals.

    Priority: quoted snippet match > explicit section/paragraph > last target.

    Args:
        user_text: The input text from the user indicating a target for an edit proposal.
        sections: The list of section dictionaries to match against.
        last_target: The last resolved target, used as a fallback if no explicit target is found.

    Returns:
        A dictionary containing the resolved target information, or None if no valid target is found.
    """
    if not sections:
        return None

    snippet = _extract_quoted_snippet(user_text)
    if snippet:
        for section in sections:
            for p_idx, paragraph in enumerate(section.get("paragraphs", [])):
                if snippet.lower() in paragraph.get("text", "").lower():
                    return {
                        "section_title": section.get("title", ""),
                        "paragraph_index": p_idx,
                        "start": paragraph.get("start", 0),
                        "end": paragraph.get("end", 0),
                        "before": paragraph.get("text", ""),
                    }

    sec_idx = _resolve_section_index(user_text, sections)
    if sec_idx is None and last_target:
        target_section = str(last_target.get("section", ""))
        for idx, section in enumerate(sections):
            if section.get("title", "") == target_section:
                sec_idx = idx
                break

    if sec_idx is None or sec_idx < 0 or sec_idx >= len(sections):
        return None

    section = sections[sec_idx]
    paragraphs = section.get("paragraphs", [])
    if not paragraphs:
        return None

    para_idx = _resolve_paragraph_index(user_text, len(paragraphs))
    if para_idx is None and last_target:
        maybe_idx = int(last_target.get("paragraph_index", -1))
        if 0 <= maybe_idx < len(paragraphs):
            para_idx = maybe_idx
    if para_idx is None:
        para_idx = 0

    paragraph = paragraphs[para_idx]
    return {
        "section_title": section.get("title", ""),
        "paragraph_index": para_idx,
        "start": paragraph.get("start", 0),
        "end": paragraph.get("end", 0),
        "before": paragraph.get("text", ""),
    }
