# src/revisao_agents/tools/reference_tools.py
"""
LangChain tool wrappers for bibliographic reference resolution.

Provides DOI lookup, CrossRef search, ArXiv lookup, MongoDB corpus search,
PDF text extraction, and Tavily web search as @tool functions.
Used by the reference extractor and formatter agents.
"""

from __future__ import annotations

import os
import re

from langchain_core.tools import tool

from ..utils.bib_utils.doi_utils import get_bibtex_from_doi, search_crossref_by_title
from ..utils.bib_utils.arxiv_utils import extract_arxiv_id, get_bibtex_from_arxiv
from ..utils.vector_utils.vector_store import search_chunk_records
from .tavily_web_search import search_tavily_incremental, extract_tavily


@tool
def lookup_doi_bibtex(doi: str) -> str:
    """Look up complete BibTeX metadata for a DOI via CrossRef.

    This is the most authoritative source — always call this first when
    a DOI is available. Returns full author, title, journal, year, pages.

    Args:
        doi: The DOI string (e.g. '10.1162/neco.1997.9.8.1735').

    Returns:
        BibTeX string with full metadata, or an error message.
    """
    if not doi or not doi.strip():
        return "Error: empty DOI provided."
    bibtex = get_bibtex_from_doi(doi.strip(), timeout=15)
    if not bibtex:
        return f"No BibTeX metadata found on CrossRef for DOI: {doi!r}"
    return bibtex


@tool
def crossref_search_by_title(title: str) -> str:
    """Search CrossRef for a paper by title and return its DOI.

    Use this when you have a title but no DOI. The returned DOI can then
    be passed to lookup_doi_bibtex for full metadata.

    Args:
        title: Paper title to search (first 200 characters are used).

    Returns:
        The DOI string (format: 'DOI: 10.xxxx/xxxx') if found, or a
        'not found' message.
    """
    if not title or not title.strip():
        return "Error: empty title provided."
    doi = search_crossref_by_title(title.strip()[:200])
    if not doi:
        return f"No DOI found on CrossRef for title: '{title[:80]}'"
    return f"DOI: {doi}"


@tool
def search_mongodb_corpus(query: str, k: int = 5) -> str:
    """Search the local MongoDB academic corpus for reference metadata.

    Use this to find papers that may be stored locally in the project
    corpus. Returns title, DOI, URL, and file path for each match.

    Especially useful for resolving file-path-style entries like
    'Artigo_16315_PT_PB' — search with the numeric ID to find the
    real paper title and DOI.

    Args:
        query: Search text (title fragment, author name, numeric article ID,
               or any keywords from the reference).
        k: Number of results to return (default 5, max 10).

    Returns:
        Formatted records with title, DOI, URL, file path, and relevance score.
    """
    records = search_chunk_records(query[:500], k=min(k, 10))
    if not records:
        return f"No records found in MongoDB corpus for: '{query[:80]}'"
    parts = []
    for i, r in enumerate(records, 1):
        parts.append(
            f"[{i}] Title: {r.get('source_title', 'N/A')}\n"
            f"    DOI:   {r.get('doi', 'N/A')}\n"
            f"    URL:   {r.get('source_url', 'N/A')}\n"
            f"    File:  {r.get('file_path', 'N/A')}\n"
            f"    Score: {float(r.get('score', 0)):.3f}"
        )
    return "\n\n".join(parts)


@tool
def search_web_for_reference(query: str, max_results: int = 4) -> str:
    """Search the web (Tavily) for academic reference metadata.

    Use as a last resort when CrossRef and MongoDB cannot provide complete
    metadata. Good for finding DOI, journal name, publisher, or year.

    Args:
        query: Search query — combine title + first author for best results,
               or use a DOI/citation string.
        max_results: Number of web results to retrieve (default 4).

    Returns:
        Snippets with title, URL, and brief content from web results.
    """
    web = search_tavily_incremental(
        query=query[:400], previous_urls=[], max_results=max_results
    )
    urls = web.get("new_urls", [])[:3]
    if not urls:
        return f"No web results found for: '{query[:80]}'"
    extracted = extract_tavily.invoke({"urls": urls, "include_images": False})
    results: list[str] = []
    for item in extracted.get("extracted", [])[:3]:
        results.append(
            f"Title:   {item.get('title', 'N/A')}\n"
            f"URL:     {item.get('url', 'N/A')}\n"
            f"Content: {str(item.get('content', ''))[:600]}"
        )
    return "\n\n---\n\n".join(results) if results else "No content retrieved."


@tool
def lookup_arxiv_bibtex(arxiv_id: str) -> str:
    """Look up BibTeX metadata for an ArXiv paper by its ArXiv ID.

    Use this when the reference is an ArXiv preprint (e.g. '2403.07815').
    Can also accept ArXiv URLs or file paths containing the ID.

    Args:
        arxiv_id: ArXiv identifier such as '2403.07815' or '2403.07815v1',
                  or a URL/path containing the ID.

    Returns:
        BibTeX string with author, title, year, eprint fields, or an error.
    """
    if not arxiv_id or not arxiv_id.strip():
        return "Error: empty ArXiv ID provided."
    # Try to extract a clean ID from whatever was passed
    clean_id = extract_arxiv_id(arxiv_id.strip()) or arxiv_id.strip()
    bibtex = get_bibtex_from_arxiv(clean_id, timeout=15)
    if not bibtex:
        return f"No metadata found on ArXiv for ID: {clean_id!r}"
    return bibtex


@tool
def extract_pdf_text_from_disk(file_path: str, max_pages: int = 1) -> str:
    """Extract text from a PDF file on disk.

    Use this when a reference entry is a local file path and MongoDB did
    not return useful metadata. Extracts text from the first N pages so
    the agent can identify title, authors, DOI, journal, and year.

    Use max_pages=5 for theses, dissertations, monographs, or TCC files
    (they require more pages to find the full bibliographic information).

    Args:
        file_path: Absolute path to the PDF file.
        max_pages: Number of pages to extract (default 1; use 5 for
                   academic theses/dissertations/TCC/monographs).

    Returns:
        Extracted text from the first N pages, or an error message.
    """
    if not file_path or not file_path.strip():
        return "Error: empty file path provided."

    path = file_path.strip()
    if not os.path.isfile(path):
        return f"File not found on disk: {path!r}"

    try:
        from pdfminer.high_level import extract_text_to_fp
        from pdfminer.layout import LAParams
        from pdfminer.pdfpage import PDFPage
        import io

        with open(path, "rb") as f:
            pages_text: list[str] = []
            for i, page in enumerate(PDFPage.get_pages(f)):
                if i >= max_pages:
                    break
                out = io.StringIO()
                extract_text_to_fp(
                    open(path, "rb"),  # noqa: WPS515 — pdfminer requires seek
                    out,
                    laparams=LAParams(),
                    page_numbers={i},
                )
                pages_text.append(out.getvalue())

        text = "\n".join(pages_text).strip()
        if not text:
            return f"Could not extract text from: {os.path.basename(path)}"
        # Limit to 3000 chars to avoid flooding the context
        return text[:3000] + ("... [truncated]" if len(text) > 3000 else "")
    except Exception as exc:
        return f"PDF extraction error for {os.path.basename(path)!r}: {exc}"


def get_reference_tools(allow_web: bool = True) -> list:
    """Return the tool list for reference agents.

    Args:
        allow_web: If False, excludes Tavily web search (CrossRef via
                   lookup_doi_bibtex and crossref_search_by_title still work
                   since they call CrossRef API, not general web search).

    Returns:
        List of LangChain tool instances ready for bind_tools().
    """
    tools = [
        lookup_doi_bibtex,
        crossref_search_by_title,
        lookup_arxiv_bibtex,
        search_mongodb_corpus,
        extract_pdf_text_from_disk,
    ]
    if allow_web:
        tools.append(search_web_for_reference)
    return tools
