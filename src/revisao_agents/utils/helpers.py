"""
Compatibility shim for old import path: from ..utils.helpers import X
Now located at: utils/file_utils/helpers.py
"""

from .file_utils.helpers import (
    extract_anchors,
    fmt_chunks,
    fmt_snippets,
    fuzzy_search_in_text,
    fuzzy_sim,
    is_paragraph_verifiable,
    normalize,
    parse_academic_plan,
    parse_technical_plan,
    save_md,
    summarize_hist,
    summarize_section,
    truncate,
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
