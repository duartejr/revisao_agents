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
import threading
import time
import urllib.request
import urllib.error
import urllib.parse
from typing import Optional

logger = logging.getLogger(__name__)

CROSSREF_API_BASE = "https://api.crossref.org/v1/works"

_USER_AGENT = "ReviewAgent/1.0 (mailto:support@example.com)"

# ---------------------------------------------------------------------------
# Rate limiter — CrossRef allows ~1 req/s; we use 1.2 s minimum interval.
# Cache avoids re-fetching the same DOI/title within the same process run.
# ---------------------------------------------------------------------------
_CROSSREF_MIN_INTERVAL = 1.2          # seconds between requests
_crossref_lock = threading.Lock()
_crossref_last_call: float = 0.0      # timestamp of last request

_doi_cache:   dict[str, Optional[str]] = {}   # doi  -> bibtex or None
_title_cache: dict[str, Optional[str]] = {}   # title -> doi   or None


def _crossref_wait() -> None:
    """Block until at least _CROSSREF_MIN_INTERVAL seconds have passed since
    the last request, then record the current time as the new last call."""
    global _crossref_last_call
    with _crossref_lock:
        elapsed = time.monotonic() - _crossref_last_call
        if elapsed < _CROSSREF_MIN_INTERVAL:
            time.sleep(_CROSSREF_MIN_INTERVAL - elapsed)
        _crossref_last_call = time.monotonic()


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


def _fetch_bibtex_url(url: str, headers: dict, timeout: int, doi_clean: str) -> Optional[str]:
    """Make one HTTP request for BibTeX, with a single 429-backoff retry."""
    for attempt in range(2):
        _crossref_wait()
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                bibtex = response.read().decode('utf-8').strip()
                if bibtex and bibtex.startswith("@"):
                    return bibtex
                logger.warning(f"❌ Invalid BibTeX response for DOI {doi_clean}")
                return None
        except urllib.error.HTTPError as e:
            if e.code == 429:
                retry_after = int(e.headers.get("Retry-After", "10"))
                logger.warning(
                    f"⏳ HTTP 429 for DOI {doi_clean} — waiting {retry_after}s "
                    f"(attempt {attempt + 1}/2)"
                )
                time.sleep(retry_after)
                continue
            raise  # re-raise non-429 HTTP errors
    return None  # both attempts exhausted


def get_bibtex_from_doi(doi: str, timeout: int = 10) -> Optional[str]:
    """Retrieve BibTeX from the Crossref API for the given DOI.

    Tries the primary ``/transform`` endpoint first; falls back to
    ``dx.doi.org`` on HTTP 400.  Results are cached for the process lifetime.

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

    # Return cached result immediately (avoids redundant API calls)
    if doi_clean in _doi_cache:
        return _doi_cache[doi_clean]

    doi_encoded = urllib.parse.quote(doi_clean, safe='/')
    url = f"{CROSSREF_API_BASE}/{doi_encoded}/transform"
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/x-bibtex"}

    bibtex: Optional[str] = None
    try:
        bibtex = _fetch_bibtex_url(url, headers, timeout, doi_clean)
        if bibtex:
            logger.debug(f"✅ BibTeX retrieved for DOI {doi_clean}")
    except urllib.error.HTTPError as e:
        logger.warning(f"❌ HTTP {e.code} for DOI {doi_clean}: {e.reason}")
        if e.code == 400:
            alt_url = f"https://dx.doi.org/{doi_encoded}"
            try:
                bibtex = _fetch_bibtex_url(alt_url, headers, timeout, doi_clean)
                if bibtex:
                    logger.info(f"✅ BibTeX via dx.doi.org for {doi_clean}")
            except Exception as alt_e:
                logger.debug(f"   dx.doi.org also failed: {alt_e}")
    except (urllib.error.URLError, TimeoutError) as e:
        logger.warning(f"❌ Network error for DOI {doi}: {e}")
    except Exception as e:
        logger.error(f"❌ Unexpected error for DOI {doi}: {e}")

    _doi_cache[doi_clean] = bibtex
    return bibtex


def search_crossref_by_title(title: str, timeout: int = 10) -> Optional[str]:
    """Query Crossref by title to obtain a DOI.

    Results are cached for the process lifetime.

    Args:
        title: The title of the paper to search for.
        timeout: The timeout for the API request in seconds.

    Returns:
        The DOI as a string if found, or ``None`` if not found.
    """
    if not title or len(title) < 5:
        return None

    cache_key = title.strip().lower()[:100]
    if cache_key in _title_cache:
        return _title_cache[cache_key]

    encoded_title = urllib.parse.quote(title[:100])
    url = f"{CROSSREF_API_BASE}?query.title={encoded_title}&rows=1&select=DOI"

    doi: Optional[str] = None
    for attempt in range(2):
        _crossref_wait()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = json.loads(response.read().decode('utf-8'))
                items = data.get("message", {}).get("items", [])
                doi = items[0].get("DOI") if items else None
            break
        except urllib.error.HTTPError as e:
            if e.code == 429:
                retry_after = int(e.headers.get("Retry-After", "10"))
                logger.warning(f"⏳ HTTP 429 on title search — waiting {retry_after}s")
                time.sleep(retry_after)
                continue
            logger.debug(f"Crossref title search HTTP error: {e}")
            break
        except Exception as e:
            logger.debug(f"Crossref title search failed: {e}")
            break

    _title_cache[cache_key] = doi
    return doi
