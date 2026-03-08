"""
REACT agent for bibliographic data retrieval.
Uses multiple strategies (Crossref, ArXiv, Tavily, MongoDB chunks, PDF metadata)
to find DOI and generate ABNT citations.
"""

import re
import urllib.request
import urllib.error
import json
from typing import Optional, Dict, List, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

CROSSREF_API_BASE = "https://api.crossref.org/v1/works"
ARXIV_API_BASE = "http://export.arxiv.org/api/query"


def get_bibtex_from_doi(doi: str, timeout: int = 10) -> Optional[str]:
    """
    Retrieve BibTeX citation from Crossref API using DOI.
    
    Args:
        doi: Digital Object Identifier (e.g., "10.1234/example")
        timeout: HTTP request timeout in seconds
        
    Returns:
        BibTeX string if successful, None if DOI not found or API error
    """
    if not doi:
        return None
    
    # Clean DOI: remove "https://doi.org/" prefix if present
    doi_clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    
    # Validate DOI format
    if not re.match(r'^10\.\d+/.+', doi_clean):
        logger.warning(f"⚠️ Invalid DOI format: {doi_clean}")
        return None
    
    # URL encode the DOI to handle special characters
    doi_encoded = urllib.parse.quote(doi_clean, safe='/')
    
    url = f"{CROSSREF_API_BASE}/{doi_encoded}/transform"
    logger.debug(f"🌐 Requesting BibTeX from: {url}")
    
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "ReviewAgent/1.0 (mailto:support@example.com)",
                "Accept": "application/x-bibtex"
            }
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            bibtex = response.read().decode('utf-8').strip()
            if bibtex and bibtex.startswith("@"):
                logger.debug(f"✅ Successfully retrieved BibTeX for DOI {doi_clean}")
                return bibtex
            else:
                logger.warning(f"❌ Invalid BibTeX response for DOI {doi_clean}: {bibtex[:100] if bibtex else 'Empty response'}")
                return None
    except urllib.error.HTTPError as e:
        logger.warning(f"❌ Failed to fetch BibTeX for DOI {doi_clean}: HTTP Error {e.code} - {e.reason}")
        logger.debug(f"   URL attempted: {url}")
        
        # Try alternative approach: use content negotiation directly
        if e.code == 400:
            logger.debug("   Trying alternative Crossref endpoint...")
            alt_url = f"https://dx.doi.org/{doi_encoded}"
            try:
                alt_req = urllib.request.Request(
                    alt_url,
                    headers={
                        "User-Agent": "ReviewAgent/1.0 (mailto:support@example.com)",
                        "Accept": "application/x-bibtex"
                    }
                )
                with urllib.request.urlopen(alt_req, timeout=timeout) as alt_response:
                    bibtex = alt_response.read().decode('utf-8')
                    if bibtex and bibtex.startswith("@"):
                        logger.info(f"✅ Successfully retrieved BibTeX via dx.doi.org for DOI {doi_clean}")
                        return bibtex
                    else:
                        logger.debug(f"   Alternative endpoint returned invalid BibTeX")
            except Exception as alt_e:
                logger.debug(f"   Alternative endpoint also failed: {alt_e}")
        
        return None
    except (urllib.error.URLError, TimeoutError) as e:
        logger.warning(f"❌ Network error fetching BibTeX for DOI {doi}: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Unexpected error fetching BibTeX for DOI {doi}: {e}")
        return None


def search_crossref_by_title(title: str, timeout: int = 10) -> Optional[str]:
    """
    Search Crossref API by title to find DOI.
    
    Args:
        title: Paper title or keywords
        timeout: HTTP request timeout in seconds
        
    Returns:
        DOI string if found, None otherwise
    """
    if not title or len(title) < 5:
        return None
    
    # Encode the query
    encoded_title = urllib.parse.quote(title[:100])
    url = f"{CROSSREF_API_BASE}?query.title={encoded_title}&rows=1&select=DOI"
    
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ReviewAgent/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode('utf-8'))
            if data.get("message", {}).get("items"):
                doi = data["message"]["items"][0].get("DOI")
                return doi
            return None
    except Exception as e:
        logger.debug(f"Crossref search by title failed: {e}")
        return None


def extract_doi_from_url(file_path: str) -> Optional[str]:
    """
    Extract DOI from URL or file path.
    Handles common DOI URL patterns and embedded DOIs.
    
    Args:
        file_path: URL or path that might contain a DOI
        
    Returns:
        DOI string if found, None otherwise
    """
    if not file_path:
        return None
    
    # Pattern 1: doi.org URLs
    doi_url_match = re.search(r'doi\.org/([10]\.\S+)', file_path)
    if doi_url_match:
        return doi_url_match.group(1).rstrip('/')
    
    # Pattern 2: DOI embedded in URL or path (10.XXXX/...)
    doi_match = re.search(r'(10\.\d{4,9}/[^\s\]]+)', file_path)
    if doi_match:
        return doi_match.group(1).rstrip('/')
    
    return None


def extract_arxiv_id(file_path: str) -> Optional[str]:
    """
    Extract ArXiv ID from URL or file path.
    
    Args:
        file_path: URL or path that might contain an ArXiv ID
        
    Returns:
        ArXiv ID if found, None otherwise
    """
    if not file_path:
        return None
    
    # Pattern: arxiv.org/abs/XXXX.XXXXX or arxiv.org/pdf/XXXX.XXXXX
    arxiv_match = re.search(r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})', file_path, re.IGNORECASE)
    if arxiv_match:
        return arxiv_match.group(1)
    
    # Pattern: just the ID in the path
    arxiv_id_match = re.search(r'(\d{4}\.\d{4,5})', file_path)
    if arxiv_id_match:
        return arxiv_id_match.group(1)
    
    return None


def search_doi_in_text(text: str) -> Optional[str]:
    """
    Search for DOI pattern in text (useful for PDF first pages).
    
    Args:
        text: Text content to search
        
    Returns:
        DOI if found, None otherwise
    """
    if not text:
        return None
    
    # Common DOI patterns in academic papers
    # Pattern 1: explicit "DOI: 10.XXXX/..."
    doi_explicit = re.search(r'DOI\s*:?\s*(10\.\d{4,9}/[^\s\]]+)', text, re.IGNORECASE)
    if doi_explicit:
        return doi_explicit.group(1).rstrip('.,;')
    
    # Pattern 2: doi.org URL
    doi_url = re.search(r'doi\.org/(10\.\d{4,9}/[^\s\]]+)', text, re.IGNORECASE)
    if doi_url:
        return doi_url.group(1).rstrip('.,;')
    
    # Pattern 3: standalone DOI
    doi_standalone = re.search(r'\b(10\.\d{4,9}/[^\s\]]+)\b', text)
    if doi_standalone:
        candidate = doi_standalone.group(1).rstrip('.,;')
        # Validate it looks like a real DOI (has at least one / and reasonable length)
        if '/' in candidate and len(candidate) > 8:
            return candidate
    
    return None


def get_bibtex_from_arxiv(arxiv_id: str, timeout: int = 10) -> Optional[str]:
    """
    Retrieve BibTeX from ArXiv API.
    
    Args:
        arxiv_id: ArXiv identifier (e.g., "2301.12345")
        timeout: HTTP request timeout
        
    Returns:
        BibTeX string if successful, None otherwise
    """
    if not arxiv_id:
        return None
    
    url = f"{ARXIV_API_BASE}?id_list={arxiv_id}"
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ReviewAgent/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            xml_data = response.read().decode('utf-8')
            
            # Parse ArXiv XML to extract basic info
            title_match = re.search(r'<title>([^<]+)</title>', xml_data)
            author_match = re.findall(r'<name>([^<]+)</name>', xml_data)
            published_match = re.search(r'<published>(\d{4})-', xml_data)
            
            if title_match and author_match and published_match:
                title = title_match.group(1).strip()
                authors = ' and '.join(author_match[:3])  # First 3 authors
                year = published_match.group(1)
                
                # Generate simple BibTeX
                bibtex = f"""@article{{arxiv{arxiv_id.replace('.', '')},
  author = {{{authors}}},
  title = {{{title}}},
  year = {{{year}}},
  eprint = {{{arxiv_id}}},
  archivePrefix = {{arXiv}}
}}"""
                return bibtex
    except Exception as e:
        logger.debug(f"ArXiv API failed for {arxiv_id}: {e}")
    
    return None


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
        # Query MongoDB for chunks from this document (first 3 chunks likely contain DOI)
        chunks = mongo_corpus.query_similar(
            "DOI digital object identifier publication",
            top_k=10,
            url_filter=file_path
        )
        
        for chunk in chunks[:5]:  # Check first 5 chunks
            text = chunk.get('text', '')
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


def _generate_fallback_abnt(file_path: str) -> str:
    """
    Generate a fallback ABNT citation when no bibliographic data is found.
    
    Args:
        file_path: Path or URL to document
        
    Returns:
        Simple ABNT-like citation string
    """
    file_name = Path(file_path).stem
    file_name_clean = " ".join(file_name.split("_")[:5])  # First 5 words
    
    return f"{file_name_clean}. Disponível em: {file_path}"


def bibtex_to_abnt(bibtex: str, url: Optional[str] = None) -> str:
    """
    Convert BibTeX to ABNT format (simplified).
    ABNT standard: Author(s), Year, Title, Publication, DOI/URL
    
    Args:
        bibtex: BibTeX string
        url: Optional URL to include in reference
        
    Returns:
        ABNT formatted citation string
    """
    # Simple pattern extraction from BibTeX
    # Full ABNT conversion is complex; this is a simplified version
    
    patterns = {
        'author': r'author\s*=\s*["{]([^"}]+)["}]',
        'title': r'title\s*=\s*["{]([^"}]+)["}]',
        'year': r'year\s*=\s*["{]?(\d{4})',
        'journal': r'journal\s*=\s*["{]([^"}]+)["}]',
        'doi': r'doi\s*=\s*["{]([^"}]+)["}]',
    }
    
    extracted = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, bibtex, re.IGNORECASE)
        if match:
            extracted[key] = match.group(1).strip()
    
    # Format as simplified ABNT
    author = extracted.get('author', 'Unknown Author')
    year = extracted.get('year', 'n.d.')
    title = extracted.get('title', 'Unknown Title')
    journal = extracted.get('journal', '')
    doi = extracted.get('doi', '')
    
    # ABNT: AUTHOR(S). Title. Journal (if available), Year. DOI or URL.
    if journal:
        abnt = f"{author}. {title}. {journal}, {year}."
    else:
        abnt = f"{author}. {title}, {year}."
    
    if doi:
        abnt += f" DOI: {doi}"
    elif url:
        abnt += f" Disponível em: {url}"
    
    return abnt


# Import urllib.parse for URL encoding
import urllib.parse
