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
from .file_utils import resumir_secao, parse_plano_tecnico, parse_plano_academico, fmt_chunks, fmt_snippets, resumir_hist, truncar, salvar_md

# Legacy imports for backwards compatibility
from .search_utils.tavily_client import buscar_conteudo_tecnico

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
    "buscar_conteudo_tecnico",
    # Vector
    "CorpusMongoDB",
    "search_chunks",
    "accumulate_chunks",
    # Bibliography
    "get_reference_data_react",
    "bibtex_to_abnt",
    "search_doi_in_text",
    # File
    "resumir_secao",
    "parse_plano_tecnico",
    "parse_plano_academico",
    "fmt_chunks",
    "fmt_snippets",
    "resumir_hist",
    "truncar",
    "salvar_md",
]