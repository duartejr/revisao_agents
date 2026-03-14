"""
Compatibility shim for old import path: from ..utils.helpers import X
Now located at: utils/file_utils/helpers.py
"""

from .file_utils.helpers import (
    fmt_chunks,
    fmt_snippets,
    summarize_hist,
    truncate,
    save_md,
    summarize_section,
    parse_technical_plan,
    parse_academic_plan,
    normalize,
    fuzzy_sim,
    fuzzy_search_in_text,
    extract_anchors,
    is_paragraph_verifiable,
)

__all__ = [
    "fmt_chunks",
    "fmt_snippets",
    "summarize_hist",
    "truncate",
    "save_md",
    "summarize_section",
    "parse_technical_plan",
    "parse_academic_plan",
    "normalize",
    "fuzzy_sim",
    "fuzzy_search_in_text",
    "extract_anchors",
    "is_paragraph_verifiable",
]
