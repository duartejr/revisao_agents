# src/revisao_agents/tools/review_tools.py
"""
Tools specific to the interactive review agent.

These are kept separate from the global tool registry because they are
bound *only* to the review-agent LLM, not to the general writing workflows.
"""

from __future__ import annotations

import glob
import os
import re

from langchain_core.tools import tool
from typing import List

from ..utils.vector_utils.vector_store import search_chunks, search_chunk_records
from ..tools.tavily_web_search import search_tavily_incremental, extract_tavily, search_tavily_images
from ..utils.bib_utils.doi_utils import (
    extract_doi_from_url,
    get_bibtex_from_doi,
    search_crossref_by_title,
)
from ..utils.bib_utils.doi_utils import search_doi_in_text

@tool
def search_evidence(query: str, k: int = 6) -> str:
    """Search the MongoDB academic corpus for evidence chunks related to a query.

    Use this tool to find supporting evidence, verify claims, or discover
    which sources back a particular statement in the review.

    Args:
        query: The text or topic to search for (e.g. a paragraph excerpt,
               a claim, or keywords).
        k: Maximum number of chunks to return (default 6).

    Returns:
        Formatted string with numbered evidence chunks, or a
        'no results' message.
    """
    chunks: List[str] = search_chunks(query[:600], k=min(k, 10))
    if not chunks:
        return "No relevant evidence found in the academic corpus."
    parts = [f"[Chunk {i+1}]:\n{c}" for i, c in enumerate(chunks)]
    return "\n\n---\n\n".join(parts)


@tool
def search_web_sources(query: str, max_results: int = 5) -> str:
    """Search the web via Tavily for additional academic sources.

    **Only call this tool when the user has explicitly asked for web /
    internet search.**  If the user did not mention 'internet', 'web',
    or 'online', do NOT use this tool.

    Args:
        query: Search query (academic topic or claim to verify).
        max_results: Maximum web results to fetch (default 5).

    Returns:
        Formatted string with title, URL, and snippet for each result,
        or a 'no results' message.
    """
    web = search_tavily_incremental(
        query=query[:400], previous_urls=[], max_results=max_results,
    )
    urls = web.get("new_urls", [])[:3]
    if not urls:
        return "No web results found."
    extracted = extract_tavily.invoke({"urls": urls, "include_images": False})
    results: List[str] = []
    for item in extracted.get("extracted", [])[:3]:
        results.append(
            f"Title: {item.get('title', '')}\n"
            f"URL: {item.get('url', '')}\n"
            f"Content: {str(item.get('content', ''))[:800]}"
        )
    return "\n\n---\n\n".join(results) if results else "No content extracted."


@tool
def search_evidence_sources(query: str, k: int = 6) -> str:
    """Search corpus chunks and return source metadata (title/URL/DOI).

    Use this when the user asks for additional sources, or asks whether
    sources are already cited in the current review text.

    Args:
        query: Topic or paragraph text to search.
        k: Maximum number of results (default 6, clamped to 10).

    Returns:
        Formatted list with source title, URL/DOI, file path, and snippet.
    """
    records = search_chunk_records(query[:600], k=min(k, 10))
    if not records:
        return "No evidence sources found in the academic corpus."

    lines: List[str] = []
    for idx, record in enumerate(records, start=1):
        title = record.get("source_title", "") or "(untitled source)"
        url = record.get("source_url", "")
        doi = record.get("doi", "")
        file_path = record.get("file_path", "")
        snippet = str(record.get("chunk", ""))[:500]
        score = float(record.get("score", 0.0) or 0.0)

        lines.append(
            "\n".join(
                [
                    f"[Source {idx}]",
                    f"Title: {title}",
                    f"URL: {url or '(not available)'}",
                    f"DOI: {doi or '(not available)'}",
                    f"File: {file_path or '(not available)'}",
                    f"Score: {score:.4f}",
                    f"Snippet: {snippet}",
                ]
            )
        )

    return "\n\n---\n\n".join(lines)


@tool
def search_near_chunks(query: str, n: int = 2) -> str:
    """Return neighboring chunks around the top retrieved chunk.

    Useful when one chunk is insufficient and nearby chunks are needed
    to preserve local context from the same source.

    Args:
        query: Search text/topic.
        n: Number of neighbors before and after anchor chunk (default 2).

    Returns:
        Formatted anchor + near chunks from the same cached source.
    """
    records = search_chunk_records(query[:600], k=1)
    if not records:
        return "No anchor chunk found for neighbor retrieval."

    anchor = records[0]
    file_path = str(anchor.get("file_path", ""))
    if not file_path:
        return "Anchor chunk has no file path metadata."

    base_name = os.path.basename(file_path)
    match = re.match(r"^([a-fA-F0-9]+)_(\d+)\.txt$", base_name)
    if not match:
        return (
            "Unable to infer neighbor chunk family from anchor file name.\n\n"
            f"Anchor title: {anchor.get('source_title', '(unknown)')}\n"
            f"Anchor snippet: {str(anchor.get('chunk', ''))[:800]}"
        )

    family_hash = match.group(1)
    anchor_idx = int(match.group(2))
    window = max(0, min(int(n), 6))
    chunk_dir = os.path.dirname(file_path) or os.getcwd()

    family_files = glob.glob(os.path.join(chunk_dir, f"{family_hash}_*.txt"))
    indexed_paths: list[tuple[int, str]] = []
    for path in family_files:
        m = re.match(r"^[a-fA-F0-9]+_(\d+)\.txt$", os.path.basename(path))
        if not m:
            continue
        indexed_paths.append((int(m.group(1)), path))

    if not indexed_paths:
        return "No neighboring cached chunks found for this source."

    indexed_paths.sort(key=lambda item: item[0])
    selected = [
        (idx, path)
        for idx, path in indexed_paths
        if (anchor_idx - window) <= idx <= (anchor_idx + window)
    ]

    lines = [
        f"Anchor source: {anchor.get('source_title', '(unknown)')}",
        f"Anchor index: {anchor_idx}",
        f"Window: ±{window}",
    ]

    for idx, path in selected:
        try:
            with open(path, "r", encoding="utf-8") as file_handle:
                text = file_handle.read().strip()
        except Exception:
            text = ""
        marker = "(ANCHOR)" if idx == anchor_idx else ""
        lines.append(
            "\n".join(
                [
                    f"[Chunk {idx}] {marker}".strip(),
                    text[:1000] if text else "(empty or unreadable chunk)",
                ]
            )
        )

    return "\n\n---\n\n".join(lines)


@tool
def search_web_images(query: str, max_results: int = 8) -> str:
    """Search images via Tavily and return image URLs with context.
    
    Args:
        query: Search query for images (e.g., "stable diffusion architecture diagram").
        max_results: Maximum number of image results to return (default 8).
    
    Returns:
        Formatted string with image URLs, source page, title, and description,
        or a 'no images found' message.
    """
    result = search_tavily_images.invoke({
        "queries": [query[:400]],
        "max_results": max_results,
    })
    images = result.get("images", []) if isinstance(result, dict) else []
    if not images:
        return "No web images found."

    lines: List[str] = []
    for idx, item in enumerate(images[:8], start=1):
        lines.append(
            "\n".join(
                [
                    f"[Image {idx}]",
                    f"Image URL: {item.get('image_url', '')}",
                    f"Source page: {item.get('source_url', '') or '(not available)'}",
                    f"Page title: {item.get('page_title', '') or '(not available)'}",
                    f"Description: {item.get('description', '') or '(not available)'}",
                ]
            )
        )
    return "\n\n---\n\n".join(lines)


@tool
def extract_web_text_from_url(url: str) -> str:
    """Extract main text from a single URL via Tavily Extract API.
    
    Args:
        url: The URL of the web page to extract text from.
    
    Returns:
        Extracted text content from the URL, or an error message if extraction fails.
    """
    cleaned = (url or "").strip()
    if not cleaned:
        return "Empty URL provided."

    result = extract_tavily.invoke({
        "urls": [cleaned],
        "include_images": False,
    })
    extracted = result.get("extracted", []) if isinstance(result, dict) else []
    if not extracted:
        return "No extractable content found for the URL."

    item = extracted[0]
    title = item.get("title", "")
    content = str(item.get("content", ""))
    return (
        f"Title: {title or '(untitled)'}\n"
        f"URL: {item.get('url', cleaned)}\n"
        f"Content:\n{content[:5000]}"
    )


@tool
def get_bibtex_for_reference(query_or_doi: str) -> str:
    """Get BibTeX for a DOI or title-like query using Crossref.

    This tool is intended for reference formatting support.

    Args:
        query_or_doi: A DOI string, a URL containing a DOI, or an article title.
    
    Returns:
        A string containing the resolved DOI and its BibTeX entry, or an error message if not found.
    """
    value = (query_or_doi or "").strip()
    if not value:
        return "Empty query provided for BibTeX search."

    doi = extract_doi_from_url(value)
    if doi is None:
        doi_match = re.search(r"(10\.\d{4,9}/[^\s]+)", value)
        doi = doi_match.group(1) if doi_match else None
    if doi is None:
        doi = search_crossref_by_title(value)
        if not doi:
            return "No DOI found for the provided query/title."

    bibtex = get_bibtex_from_doi(doi)
    if not bibtex:
        return f"DOI found but BibTeX could not be retrieved: {doi}"

    return f"DOI: {doi}\nBibTeX:\n{bibtex}"


@tool
def fetch_reference_metadata(
    title: str,
    doi: str = "",
    url: str = "",
) -> str:
    """Fetch full bibliographic metadata for an article to format a reference.

    ALWAYS use this tool first when the user asks for reference formatting
    (ABNT, APA, etc.).  Do NOT try to format references from filenames or URLs
    alone — call this tool first to recover proper authors, journal, year, DOI.

    IMPORTANT: ``title`` must be the ARTICLE TITLE, not a filename or URL.
    When you only have a local file path like
    ``A-parallel-attention_2026_Journal-of-Hydro.pdf``, extract a human-readable
    title from the filename: remove the extension, year, journal suffix, and
    replace underscores/dashes with spaces (e.g. "A parallel attention based
    framework for multi step mu").

    Workflow executed internally:
      1. If ``doi`` is provided → fetch BibTeX directly.
      2. If ``url`` contains a DOI pattern → extract and fetch BibTeX.
      3. Search Crossref by title to find a DOI → fetch BibTeX.
      4. Returns all metadata found, flagging any fields that are missing.

    Args:
        title: Article title (use for Crossref lookup).
        doi:   DOI string if already known (optional).
        url:   Article URL if available — a DOI may be embedded (optional).

    Returns:
        Structured metadata block including DOI, BibTeX when available, and
        a clear note when web search is needed for local-only sources.
    """
    title = (title or "").strip()
    doi = (doi or "").strip()
    url = (url or "").strip()

    # Step 1 — resolve DOI from args
    resolved_doi: str | None = None
    if doi:
        resolved_doi = extract_doi_from_url(doi) or (doi if re.match(r"^10\.\d", doi) else None)
    if resolved_doi is None and url:
        resolved_doi = extract_doi_from_url(url)
    # Step 2 — Crossref title search as fallback
    if resolved_doi is None and title:
        resolved_doi = search_crossref_by_title(title)

    bibtex: str | None = None
    if resolved_doi:
        bibtex = get_bibtex_from_doi(resolved_doi)

    lines: List[str] = ["=== Reference Metadata ==="]
    lines.append(f"Provided title : {title or '(none)'}")
    lines.append(f"Provided DOI   : {doi or '(none)'}")
    lines.append(f"Provided URL   : {url or '(none)'}")
    lines.append(f"Resolved DOI   : {resolved_doi or '(not found via Crossref)'}")

    if bibtex:
        lines.append(f"\nBibTeX:\n{bibtex}")
        lines.append(
            "\nINSTRUCTION: Use the BibTeX fields above to format the reference "
            "in the requested citation style (ABNT, APA, etc.)."
        )
    else:
        lines.append(
            "\nBibTeX: (not available — Crossref lookup failed or yielded no match)"
        )
        if url and not url.startswith("/"):
            lines.append(
                f"\nSUGGESTION: Call extract_web_text_from_url('{url}') to "
                "fetch the first ~1000 characters of the article page and "
                "extract authors, year, journal manually."
            )
        elif not url or url.startswith("/"):
            lines.append(
                "\nSUGGESTION: This is a local file. "
                "If web access is enabled, call search_article_online with the "
                "article title to find the DOI, then call get_bibtex_for_reference."
            )
    return "\n".join(lines)


@tool
def search_article_online(title: str) -> str:
    """Search Tavily for a scholarly article by title to find its DOI and metadata.

    Use this tool when ``fetch_reference_metadata`` could not find the DOI via
    Crossref and you need to locate the article on the web.

    IMPORTANT: ``title`` must be the ARTICLE TITLE — not a filename or URL.
    Strip file extensions, year tokens, and journal tokens from filenames to
    recover the human-readable title before calling this tool.

    After this tool returns an article URL with an embedded DOI, call
    ``get_bibtex_for_reference`` with that DOI or URL to get full BibTeX.
    If no DOI is found, call ``extract_web_text_from_url`` with the article
    URL to fetch the first ~1000 characters for manual extraction.

    Args:
        title: The article title to search for.

    Returns:
        List of web results (title, URL, snippet) for the article search.
    """
    query = f'"{title[:200]}" scholarly article DOI'
    web = search_tavily_incremental(query=query, previous_urls=[], max_results=5)
    urls = web.get("new_urls", [])[:3]
    if not urls:
        return f"No web results found for title: {title}"

    extracted = extract_tavily.invoke({"urls": urls, "include_images": False})
    results: List[str] = []
    for item in extracted.get("extracted", [])[:3]:
        url_found = item.get("url", "")
        content = str(item.get("content", ""))
        # Try to find a DOI in the page content or URL
        doi_in_content = search_doi_in_text(content[:2000])
        doi_in_url = extract_doi_from_url(url_found)
        doi = doi_in_content or doi_in_url or ""
        results.append(
            "\n".join([
                f"Title: {item.get('title', '')}",
                f"URL: {url_found}",
                f"DOI found: {doi or '(not found in page)'}",
                f"Snippet: {content[:600]}",
            ])
        )
    return "\n\n---\n\n".join(results) if results else "No content extracted."

def get_review_tools(allow_web: bool = False) -> list:
    """Return the list of tools available to the review agent.

    Args:
        allow_web: If True, include the web search tool.
    
    Returns:
        List of tool functions to bind to the review agent.
    """
    tools = [search_evidence, search_evidence_sources, search_near_chunks,
             fetch_reference_metadata]
    if allow_web:
        tools.extend([
            search_web_sources,
            search_web_images,
            extract_web_text_from_url,
            get_bibtex_for_reference,
                    search_article_online,
        ])
    return tools
