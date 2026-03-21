"""
text_filters.py — regex patterns and LLM output cleanup helpers.

Contains:
- _ANCHORS_PATTERN          : matches [ANCHOR: "..."] in generated text.
- _JUSTIFICATION_BLOCK_RE   : removes LLM reasoning blocks after the paragraph.
- _META_SENTENCE_RE         : removes meta-organizational opening sentences.
- _STRIP_FIGURE_TABLE_REFS  : removes dangling Figure/Table/Equation references.
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
    r'(?:^|\n{0,2})(\*{0,2}(?:Justification|Justificativa|Applied\s+corrections|Correções\s+aplicadas|'
    r'Correction\s+applied|Correção\s+aplicada|'
    r'Reasoning|Raciocínio|Correction|Correção|'
    r'Revised\s+text|Texto\s+revisado|Corrected\s+text|Texto\s+corrigido)\s*[:\：\*]\*{0,2}'
    r'[\s\S]*)',
    re.IGNORECASE,
)

# Sentence-level meta-organizational patterns
_META_SENTENCE_RE = re.compile(
    r'(?:^|(?<=\n))'
    r'(?:The (?:objective|purpose) of(?:this| the)[^.]*?[.!]\s*'
    r'|O (?:objetivo|propósito) d[eo](?: estud[oa]| capítulo| se[çc][ãa]o| revis[ãa]o)?[^.]*?[.!]\s*'
    r'|This (?:section|subsection|chapter|review|study|analysis) '
    r'(?:aims|seeks|intends|presents|provides|explores|addresses|discusses)[^.]*?[.!]\s*'
    r'|In this (?:section|subsection|chapter|review|study)[^.]*?'
    r'(?:we|the authors) (?:present|discuss|analyze|explore|address)[^.]*?[.!]\s*'
    r'|Esta (?:se[çc][ãa]o|subse[çc][ãa]o|revis[ãa]o|an[áa]lise|pesquisa|capítulo) '
    r'(?:busca|visa|objetiva|apresenta|explora|aborda|discute)[^.]*?[.!]\s*'
    r'|Nest[ae] (?:se[çc][ãa]o|subse[çc][ãa]o|capítulo)[^.]*?'
    r'(?:apresentamos|discutimos|analisamos|exploraremos|abordaremos)[^.]*?[.!]\s*'
    r')',
    re.IGNORECASE | re.MULTILINE,
)

# Pattern to detect references to figures/tables/equations not present in the essay
_FIGURE_TABLE_RE = re.compile(
    r'(?:^|(?<=\.\s)|(?<=\n))'
    r'[^.]*?'
    r'(?:'
    r'Figures?\s+\d+|Figuras?\s+\d+|Fig(?:ures?)?\.?\s*\d+'
    r'|Tables?\s+\d+|Tabelas?\s+\d+'
    r'|Equations?\s+\d+|Equa(?:ç|c)(?:ão|ao|ões|oes)\s+\d+|Eq\.?\s*\d+'
    r'|Frames?\s+\d+|Quadros?\s+\d+'
    r'|Charts?\s+\d+|Graphs?\s+\d+|Gr[áa]ficos?\s+\d+'
    r')'
    r'[^.]*\.\s*',
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def _strip_justification_blocks(text: str) -> str:
    """Remove LLM-generated justification/reasoning blocks after the paragraph text.
    
    Args:
        text (str): The original text potentially containing justification blocks.
    
    Returns:
        str: Cleaned text with justification blocks removed.
    """
    m = _JUSTIFICATION_BLOCK_RE.search(text)
    if m:
        text = text[:m.start()].rstrip()
    return text


def _strip_meta_sentences(text: str) -> str:
    """Remove meta-organizational opening sentences from a paragraph.
    
    Args:
        text (str): The original text potentially containing meta-organizational sentences.
    
    Returns:
        str: Cleaned text with meta-organizational sentences removed.
    """
    return _META_SENTENCE_RE.sub("", text).strip()


def _strip_figure_table_refs(text: str) -> str:
    """Remove sentences that reference figures/tables/equations not present in the essay.
    
    Args:
        text (str): The original text potentially containing figure/table/equation references.
    
    Returns:
        str: Cleaned text with figure/table/equation references removed.
    """
    cleaned = _FIGURE_TABLE_RE.sub("", text)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()
