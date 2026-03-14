"""
text_filters.py — regex patterns and LLM output cleanup helpers.

Contains:
- _ANCHORS_PATTERN          : matches [ANCHOR: "..."] in generated text.
- _strip_justification_blocks : removes LLM reasoning blocks after the paragraph.
- _strip_meta_sentences    : removes meta-organisational opening sentences.
- _strip_figure_table_refs : removes dangling Figure/Table/Equation references.
"""

import re

# ---------------------------------------------------------------------------
# Core anchor pattern (used across multiple modules)
# ---------------------------------------------------------------------------
_ANCHORS_PATTERN = re.compile(r'\[ANCHOR:\s*"((?:[^"\\]|\\.)*)"\]', re.DOTALL)

# ---------------------------------------------------------------------------
# Patterns for LLM-generated justification/meta blocks
# ---------------------------------------------------------------------------
_JUSTIFICATION_BLOCK_RE = re.compile(
    r'(?:^|\n{0,2})(\*{0,2}(?:Justificativa|Correções\s+aplicadas|Correção\s+aplicada|'
    r'Raciocínio|Correção|Justification|Applied\s+corrections|Reasoning)\s*[:\：\*]\*{0,2}'
    r'[\s\S]*)',
    re.IGNORECASE,
)

# Sentence-level meta-organizational patterns
_META_SENTENCE_RE = re.compile(
    r'(?:^|(?<=\n))'
    r'(?:O objetivo d[eo](?: estud[oa]| capítulo| se[çc][ãa]o| revis[ãa]o)?[^.]*?[.!]\s*'
    r'|The objective of(?:this| the)[^.]*?[.!]\s*'
    r'|This (?:section|chapter|review|study) (?:aims|seeks|intends|presents|provides|explores)[^.]*?[.!]\s*'
    r'|Esta (?:se[çc][ãa]o|revis[ãa]o|an[áa]lise|pesquisa) (?:busca|visa|objetiva|apresenta|explora|aborda)[^.]*?[.!]\s*'
    r'|Nesta se[çc][ãa]o[^.]*?(?:apresentamos|discutimos|analisamos|exploraremos|abordaremos)[^.]*?[.!]\s*'
    r')',
    re.IGNORECASE | re.MULTILINE,
)

# Pattern to detect references to figures/tables/equations not present in the essay
_FIGURE_TABLE_RE = re.compile(
    r'(?:^|(?<=\.\s)|(?<=\n))'
    r'[^.]*?'
    r'(?:'
    r'[Ff]igura\s+\d+|[Ff]ig(?:ure)?\.?\s*\d+'
    r'|[Tt]abela\s+\d+|[Tt]able\s+\d+'
    r'|[Ee]qua[çc][ãa]o\s+\d+|[Ee]quation\s+\d+'
    r'|[Qq]uadro\s+\d+|[Gg]r[áa]fico\s+\d+'
    r')'
    r'[^.]*\.\s*',
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def _strip_justification_blocks(text: str) -> str:
    """Remove LLM-generated justification/reasoning blocks after the paragraph text."""
    m = _JUSTIFICATION_BLOCK_RE.search(text)
    if m:
        text = text[:m.start()].rstrip()
    return text


def _strip_meta_sentences(text: str) -> str:
    """Remove meta-organizational opening sentences from a paragraph."""
    return _META_SENTENCE_RE.sub("", text).strip()


def _strip_figure_table_refs(text: str) -> str:
    """Remove sentences that reference figures/tables/equations not present in the essay."""
    cleaned = _FIGURE_TABLE_RE.sub("", text)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()
