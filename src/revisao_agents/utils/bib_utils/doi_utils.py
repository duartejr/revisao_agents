"""
doi_utils.py — DOI extraction and Crossref API helpers.

Public API
----------
CROSSREF_API_BASE
extract_doi_from_url      : parse DOI from a URL/path.
search_doi_in_text        : find DOI pattern inside raw text.
get_bibtex_from_doi       : fetch BibTeX from Crossref using a DOI.
search_crossref_by_title  : query Crossref by title to find a DOI.
"""

import re
import json
import logging
import urllib.request
import urllib.error
import urllib.parse
from typing import Optional

logger = logging.getLogger(__name__)

CROSSREF_API_BASE = "https://api.crossref.org/v1/works"

_USER_AGENT = "ReviewAgent/1.0 (mailto:support@example.com)"


# ---------------------------------------------------------------------------

def extract_doi_from_url(file_path: str) -> Optional[str]:
    """Extract a DOI from a URL or file path string.

    Handles ``doi.org`` URL patterns and embedded ``10.XXXX/...`` patterns.

    Args:
        file_path: A URL or file path that may contain a DOI.
    
    Returns:
        The extracted DOI as a string, or ``None`` if no DOI is found.
    """
    if not file_path:
        return None

    doi_url_match = re.search(r'doi\.org/([10]\.\S+)', file_path)
    if doi_url_match:
        return doi_url_match.group(1).rstrip('/')

    doi_match = re.search(r'(10\.\d{4,9}/[^\s\]]+)', file_path)
    if doi_match:
        return doi_match.group(1).rstrip('/')

    return None


def search_doi_in_text(text: str) -> Optional[str]:
    """Find a DOI pattern inside raw text (useful for PDF first pages).
    
    Args:
        text: A string containing the text to search for a DOI.
    
    Returns:
        The extracted DOI as a string, or ``None`` if no DOI is found.
    """
    if not text:
        return None

    doi_explicit = re.search(r'DOI\s*:?\s*(10\.\d{4,9}/[^\s\]]+)', text, re.IGNORECASE)
    if doi_explicit:
        return doi_explicit.group(1).rstrip('.,;')

    doi_url = re.search(r'doi\.org/(10\.\d{4,9}/[^\s\]]+)', text, re.IGNORECASE)
    if doi_url:
        return doi_url.group(1).rstrip('.,;')

    doi_standalone = re.search(r'\b(10\.\d{4,9}/[^\s\]]+)\b', text)
    if doi_standalone:
        candidate = doi_standalone.group(1).rstrip('.,;')
        if '/' in candidate and len(candidate) > 8:
            return candidate

    return None


def get_bibtex_from_doi(doi: str, timeout: int = 10) -> Optional[str]:
    """Retrieve BibTeX from the Crossref API for the given DOI.

    Tries the primary ``/transform`` endpoint first; falls back to
    ``dx.doi.org`` on HTTP 400.

    Args:
        doi: The DOI string to query (with or without "https://doi.org/").
        timeout: The timeout for the API request in seconds.
    
    Returns:
        A string containing the BibTeX entry, or ``None`` if retrieval fails.
    """
    if not doi:
        return None

    doi_clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    if not re.match(r'^10\.\d+/.+', doi_clean):
        logger.warning(f"⚠️ Invalid DOI format: {doi_clean}")
        return None

    doi_encoded = urllib.parse.quote(doi_clean, safe='/')
    url = f"{CROSSREF_API_BASE}/{doi_encoded}/transform"
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/x-bibtex"}

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            bibtex = response.read().decode('utf-8').strip()
            if bibtex and bibtex.startswith("@"):
                logger.debug(f"✅ BibTeX retrieved for DOI {doi_clean}")
                return bibtex
            logger.warning(f"❌ Invalid BibTeX response for DOI {doi_clean}")
            return None
    except urllib.error.HTTPError as e:
        logger.warning(f"❌ HTTP {e.code} for DOI {doi_clean}: {e.reason}")
        if e.code == 400:
            alt_url = f"https://dx.doi.org/{doi_encoded}"
            try:
                alt_req = urllib.request.Request(alt_url, headers=headers)
                with urllib.request.urlopen(alt_req, timeout=timeout) as resp:
                    bibtex = resp.read().decode('utf-8')
                    if bibtex and bibtex.startswith("@"):
                        logger.info(f"✅ BibTeX via dx.doi.org for {doi_clean}")
                        return bibtex
            except Exception as alt_e:
                logger.debug(f"   dx.doi.org also failed: {alt_e}")
        return None
    except (urllib.error.URLError, TimeoutError) as e:
        logger.warning(f"❌ Network error for DOI {doi}: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Unexpected error for DOI {doi}: {e}")
        return None


def search_crossref_by_title(title: str, timeout: int = 10) -> Optional[str]:
    """Query Crossref by title to obtain a DOI.

    Args:
        title: The title of the paper to search for.
        timeout: The timeout for the API request in seconds.

    Returns:
        The DOI as a string if found, or ``None`` if not found.
    """
    if not title or len(title) < 5:
        return None

    encoded_title = urllib.parse.quote(title[:100])
    url = f"{CROSSREF_API_BASE}?query.title={encoded_title}&rows=1&select=DOI"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode('utf-8'))
            items = data.get("message", {}).get("items", [])
            if items:
                return items[0].get("DOI")
            return None
    except Exception as e:
        logger.debug(f"Crossref title search failed: {e}")
        return None
