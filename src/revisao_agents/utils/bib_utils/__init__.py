"""
Bibliography utilities: DOI/BibTeX retrieval and ABNT citation formatting.
Uses REACT agent to intelligently fetch bibliographic data from multiple sources.
"""

from .crossref_bibtex import (
    get_reference_data_react,
    get_bibtex_from_doi,
    get_bibtex_from_arxiv,
    extract_doi_from_url,
    extract_arxiv_id,
    search_doi_in_text,
    bibtex_to_abnt,
)

__all__ = [
    "get_reference_data_react",
    "get_bibtex_from_doi",
    "get_bibtex_from_arxiv",
    "extract_doi_from_url",
    "extract_arxiv_id",
    "search_doi_in_text",
    "bibtex_to_abnt",
]
