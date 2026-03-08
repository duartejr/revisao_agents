"""
Crossref API integration for retrieving BibTeX bibliographic data.
Enables conversion of PDF URLs/DOIs to standard citation format.
"""

import re
import urllib.request
import urllib.error
import json
from typing import Optional, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

CROSSREF_API_BASE = "https://api.crossref.org/works"


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
    
    url = f"{CROSSREF_API_BASE}/{doi_clean}/transform?accept=application/x-bibtex"
    
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ReviewAgent/1.0 (mailto:support@example.com)"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            bibtex = response.read().decode('utf-8')
            if bibtex and bibtex.startswith("@"):
                return bibtex
            return None
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        logger.warning(f"Failed to fetch BibTeX for DOI {doi}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching BibTeX for DOI {doi}: {e}")
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
    Attempt to extract or find DOI from a file path.
    Tries common patterns and searches Crossref if needed.
    
    Args:
        file_path: Path to PDF or document URL
        
    Returns:
        DOI string if found, None otherwise
    """
    if not file_path:
        return None
    
    # Pattern 1: DOI already in path (rare but possible)
    doi_match = re.search(r'10\.\d{4,}/[^\s]+', file_path)
    if doi_match:
        return doi_match.group(0)
    
    # Pattern 2: Extract filename and try to search
    file_name = Path(file_path).stem  # filename without extension
    file_name_clean = re.sub(r'[\._\-]+', ' ', file_name)  # Replace separators with spaces
    
    # Search Crossref with the filename (might work for PDF names like "Smith_2023_Title")
    if len(file_name_clean) > 10:
        doi = search_crossref_by_title(file_name_clean)
        if doi:
            return doi
    
    return None


def url_to_bibtex(file_path: str, timeout: int = 10) -> Optional[str]:
    """
    Convert a file path/URL to BibTeX citation.
    Tries to find DOI and retrieve BibTeX from Crossref.
    
    Args:
        file_path: Path to PDF or document
        timeout: HTTP request timeout
        
    Returns:
        BibTeX string if successful, None otherwise
    """
    # Try to extract/find DOI
    doi = extract_doi_from_url(file_path)
    
    if doi:
        bibtex = get_bibtex_from_doi(doi, timeout=timeout)
        if bibtex:
            return bibtex
    
    return None


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


# Add urllib.parse import at the top if not already present
import urllib.parse
