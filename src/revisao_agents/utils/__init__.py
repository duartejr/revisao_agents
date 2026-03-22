"""
Utility modules for the review agent.

Subpackages:
- core: Constants, logging, and common utilities
- llm_utils: Prompt loading, LLM providers, citation handling
- search_utils: Web search, Tavily integration
- vector_utils: MongoDB, vector store, PDF processing
- bib_utils: Bibliography, DOI/BibTeX retrieval
- file_utils: File operations, text helpers

Backwards-compatible imports from subfolders (for existing code):
"""

# Re-export from subfolders for backwards compatibility
from .llm_utils import load_prompt, llm_call, parse_json_safe, get_llm, LLMProvider
from .search_utils import search_web, search_images, extract_urls, score_url
from .vector_utils import CorpusMongoDB, search_chunks, accumulate_chunks
from .bib_utils import get_reference_data_react, bibtex_to_abnt, search_doi_in_text
from .file_utils import (
    summarize_section,
    parse_technical_plan,
    parse_academic_plan,
    fmt_chunks,
    fmt_snippets,
    summarize_hist,
    truncate,
    save_md,
)

# Legacy imports for backwards compatibility
from .search_utils.tavily_client import search_technical_content

__all__ = [
    # LLM
    "load_prompt",
    "llm_call",
    "parse_json_safe",
    "get_llm",
    "LLMProvider",
    # Search
    "search_web",
    "search_images",
    "extract_urls",
    "score_url",
    "search_technical_content",
    # Vector
    "CorpusMongoDB",
    "search_chunks",
    "accumulate_chunks",
    # Bibliography
    "get_reference_data_react",
    "bibtex_to_abnt",
    "search_doi_in_text",
    # File
    "summarize_section",
    "parse_technical_plan",
    "parse_academic_plan",
    "fmt_chunks",
    "fmt_snippets",
    "summarize_hist",
    "truncate",
    "save_md",
]
