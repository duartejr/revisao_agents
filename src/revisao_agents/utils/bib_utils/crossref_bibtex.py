"""
REACT agent for bibliographic data retrieval.
Uses multiple strategies (Crossref, ArXiv, Tavily, MongoDB chunks, PDF metadata)
to find DOI and generate ABNT citations.

Helper implementations live in:
    utils/bib_utils/doi_utils.py    -- DOI extraction + Crossref API
    utils/bib_utils/arxiv_utils.py  -- ArXiv ID extraction + API
    utils/bib_utils/abnt_utils.py   -- BibTeX -> ABNT formatting
"""

import logging
import re
from typing import Optional, Dict, List, Any
from pathlib import Path

from .doi_utils import (
    CROSSREF_API_BASE,
    extract_doi_from_url,
    search_doi_in_text,
    get_bibtex_from_doi,
    search_crossref_by_title,
)
from .arxiv_utils import (
    ARXIV_API_BASE,
    extract_arxiv_id,
    get_bibtex_from_arxiv,
)
from .abnt_utils import (
    bibtex_to_abnt,
    generate_fallback_abnt as _generate_fallback_abnt,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CROSSREF_API_BASE",
    "ARXIV_API_BASE",
    "extract_doi_from_url",
    "search_doi_in_text",
    "get_bibtex_from_doi",
    "search_crossref_by_title",
    "extract_arxiv_id",
    "get_bibtex_from_arxiv",
    "bibtex_to_abnt",
    "search_doi_in_mongo_chunks",
    "search_paper_with_tavily",
    "get_reference_data_react",
]

def search_doi_in_mongo_chunks(file_path: str, mongo_corpus: Any) -> Optional[str]:
    """
    Search for DOI in MongoDB chunks of the document.
    Useful for local PDFs where we have indexed content.
    
    Args:
        file_path: Path to the document
        mongo_corpus: CorpusMongoDB instance
        
    Returns:
        DOI if found in chunks, None otherwise
    """
    if not file_path or not mongo_corpus:
        return None
    
    try:
        file_name = Path(file_path).stem
        query_text = f"DOI digital object identifier publication {file_name}"
        chunks = mongo_corpus.query(query_text, top_k=10)

        for chunk in chunks[:5]:
            if isinstance(chunk, dict):
                text = chunk.get("text") or chunk.get("texto") or ""
            else:
                text = getattr(chunk, "texto", "") or getattr(chunk, "text", "") or ""

            doi = search_doi_in_text(text)
            if doi:
                logger.info(f"Found DOI in MongoDB chunk: {doi}")
                return doi
    except Exception as e:
        logger.debug(f"MongoDB chunk search failed: {e}")
    
    return None


def search_paper_with_tavily(file_path: str, tavily_client: Any = None) -> Optional[Dict[str, str]]:
    """
    Use Tavily web search to find paper metadata and DOI.
    
    Args:
        file_path: URL or path to search for
        tavily_client: Tavily search client (if available)
        
    Returns:
        Dict with 'doi', 'title', 'url' if found, None otherwise
    """
    if not tavily_client:
        # Try to import and use tavily_client if available
        try:
            from ..search_utils.tavily_client import search_web
            tavily_client = search_web
        except ImportError:
            return None
    
    # Extract filename as search query
    file_name = Path(file_path).stem
    file_name_clean = re.sub(r'[\._\-]+', ' ', file_name)
    
    if len(file_name_clean) < 10:
        return None
    
    try:
        # Search for the paper title
        results = tavily_client(f'"{file_name_clean}" DOI', max_results=3)
        
        for result in results:
            url = result.get('url', '')
            content = result.get('content', '')
            
            # Try to extract DOI from URL or content
            doi = extract_doi_from_url(url) or search_doi_in_text(content)
            if doi:
                return {
                    'doi': doi,
                    'title': result.get('title', ''),
                    'url': url
                }
    except Exception as e:
        logger.debug(f"Tavily search failed: {e}")
    
    return None


def get_reference_data_react(
    file_path: str,
    mongo_corpus: Any = None,
    tavily_enabled: bool = False,
    max_iterations: int = 5,
    timeout: int = 10
) -> Dict[str, Any]:
    """
    REACT agent for retrieving bibliographic data.
    Tries multiple strategies intelligently to find DOI and generate citation.
    
    Strategies:
    1. Extract DOI from URL
    2. Extract ArXiv ID from URL
    3. Search DOI in MongoDB chunks (if corpus available)
    4. Use Tavily web search (if enabled)
    5. Search Crossref by filename/title
    
    Args:
        file_path: URL or path to document
        mongo_corpus: CorpusMongoDB instance (optional)
        tavily_enabled: Whether to use Tavily search
        max_iterations: Maximum number of strategies to try
        timeout: HTTP request timeout
        
    Returns:
        Dict with keys: 'doi', 'arxiv_id', 'bibtex', 'abnt', 'source'
    """
    result = {
        'doi': None,
        'arxiv_id': None,
        'bibtex': None,
        'abnt': None,
        'source': 'unknown',
        'url': file_path
    }
    
    logger.info(f"🔍 REACT: Starting bibliography search for: {Path(file_path).name}")
    
    # Strategy 1: Extract DOI from URL
    doi = extract_doi_from_url(file_path)
    if doi:
        logger.info(f"📍 REACT: Found DOI in URL: {doi}")
        bibtex = get_bibtex_from_doi(doi, timeout=timeout)
        if bibtex:
            logger.info(f"✅ Successfully retrieved BibTeX for DOI: {doi}")
            result['doi'] = doi
            result['bibtex'] = bibtex
            result['abnt'] = bibtex_to_abnt(bibtex, url=file_path)
            result['source'] = 'url_doi'
            return result
        else:
            logger.warning(f"⚠️ DOI found but BibTeX fetch failed: {doi}")
            result['doi'] = doi  # Keep DOI even if BibTeX fetch failed
    
    # Strategy 2: Extract ArXiv ID from URL
    arxiv_id = extract_arxiv_id(file_path)
    if arxiv_id:
        logger.info(f"REACT: Found ArXiv ID: {arxiv_id}")
        bibtex = get_bibtex_from_arxiv(arxiv_id, timeout=timeout)
        if bibtex:
            result['arxiv_id'] = arxiv_id
            result['bibtex'] = bibtex
            result['abnt'] = bibtex_to_abnt(bibtex, url=file_path)
            result['source'] = 'arxiv'
            return result
    
    # Strategy 3: Search DOI in MongoDB chunks (for local PDFs)
    if mongo_corpus and Path(file_path).exists():
        logger.info("REACT: Searching DOI in MongoDB chunks...")
        doi = search_doi_in_mongo_chunks(file_path, mongo_corpus)
        if doi:
            logger.info(f"REACT: Found DOI in chunks: {doi}")
            bibtex = get_bibtex_from_doi(doi, timeout=timeout)
            if bibtex:
                result['doi'] = doi
                result['bibtex'] = bibtex
                result['abnt'] = bibtex_to_abnt(bibtex, url=file_path)
                result['source'] = 'mongo_chunks'
                return result
            else:
                result['doi'] = doi
    
    # Strategy 4: Use Tavily search (if enabled)
    if tavily_enabled:
        logger.info("REACT: Trying Tavily web search...")
        tavily_result = search_paper_with_tavily(file_path)
        if tavily_result and tavily_result.get('doi'):
            doi = tavily_result['doi']
            logger.info(f"REACT: Found DOI via Tavily: {doi}")
            bibtex = get_bibtex_from_doi(doi, timeout=timeout)
            if bibtex:
                result['doi'] = doi
                result['bibtex'] = bibtex
                result['abnt'] = bibtex_to_abnt(bibtex, url=file_path)
                result['source'] = 'tavily'
                return result
            else:
                result['doi'] = doi
    
    # Strategy 5: Search Crossref by title (extracted from filename)
    file_name = Path(file_path).stem
    file_name_clean = re.sub(r'[\._\-]+', ' ', file_name)
    if len(file_name_clean) > 10:
        logger.info("REACT: Searching Crossref by title...")
        doi = search_crossref_by_title(file_name_clean, timeout=timeout)
        if doi:
            logger.info(f"REACT: Found DOI via Crossref title search: {doi}")
            bibtex = get_bibtex_from_doi(doi, timeout=timeout)
            if bibtex:
                result['doi'] = doi
                result['bibtex'] = bibtex
                result['abnt'] = bibtex_to_abnt(bibtex, url=file_path)
                result['source'] = 'crossref_title'
                return result
            else:
                result['doi'] = doi
    
    # Fallback: Generate simple ABNT citation from path
    logger.warning(f"📝 REACT: No bibliographic data found for {Path(file_path).name}, using fallback citation")
    result['abnt'] = _generate_fallback_abnt(file_path)
    result['source'] = 'fallback'
    
    return result


