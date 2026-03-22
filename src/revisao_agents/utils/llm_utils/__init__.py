"""
LLM utilities: prompt loading, LLM providers, and citation handling.
"""

from .fix_citation_remapping import (
    CitationTracker,
    create_remap_map,
    extract_numbered_citations,
    remap_text_with_tracking,
    synchronize_text_with_references,
)
from .llm_providers import LLMProvider, get_llm, llm_call, parse_json_safe
from .prompt_loader import load_prompt

__all__ = [
    "load_prompt",
    "llm_call",
    "parse_json_safe",
    "get_llm",
    "LLMProvider",
    "CitationTracker",
    "create_remap_map",
    "extract_numbered_citations",
    "remap_text_with_tracking",
    "synchronize_text_with_references",
]
