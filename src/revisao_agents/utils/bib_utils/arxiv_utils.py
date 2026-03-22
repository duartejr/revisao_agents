"""
arxiv_utils.py — ArXiv ID extraction and BibTeX retrieval.

Public API
----------
ARXIV_API_BASE
extract_arxiv_id      : parse an ArXiv ID from a URL/path.
get_bibtex_from_arxiv : fetch metadata from the ArXiv API & build minimal BibTeX.
"""

import logging
import re
import threading
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

ARXIV_API_BASE = "http://export.arxiv.org/api/query"

_USER_AGENT = "ReviewAgent/1.0"

# ---------------------------------------------------------------------------
# Rate limiter — ArXiv recommends ≤3 req/s; we use 1.0 s minimum interval.
# Cache avoids re-fetching the same ID within the same process run.
# ---------------------------------------------------------------------------
_ARXIV_MIN_INTERVAL = 1.0
_arxiv_lock = threading.Lock()
_arxiv_last_call: float = 0.0

_arxiv_cache: dict[str, str | None] = {}  # arxiv_id -> bibtex or None


def _arxiv_wait() -> None:
    """Block until at least _ARXIV_MIN_INTERVAL seconds have passed."""
    global _arxiv_last_call
    with _arxiv_lock:
        elapsed = time.monotonic() - _arxiv_last_call
        if elapsed < _ARXIV_MIN_INTERVAL:
            time.sleep(_ARXIV_MIN_INTERVAL - elapsed)
        _arxiv_last_call = time.monotonic()


# ---------------------------------------------------------------------------


def extract_arxiv_id(file_path: str) -> str | None:
    """Return the ArXiv identifier embedded in *file_path*, or ``None``.

    Args:
        file_path: a URL or file path that may contain an ArXiv ID.

    Returns:
        The ArXiv ID as a string, or ``None`` if not found.
    """
    if not file_path:
        return None

    arxiv_match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", file_path, re.IGNORECASE)
    if arxiv_match:
        return arxiv_match.group(1)

    arxiv_id_match = re.search(r"(\d{4}\.\d{4,5})", file_path)
    if arxiv_id_match:
        return arxiv_id_match.group(1)

    return None


def get_bibtex_from_arxiv(arxiv_id: str, timeout: int = 10) -> str | None:
    """Query the ArXiv Atom API and return a minimal BibTeX entry, or ``None``.

    Results are cached for the process lifetime.

    Args:
        arxiv_id: The ArXiv identifier of the paper.
        timeout: The timeout for the API request in seconds.

    Returns:
        A string containing the BibTeX entry, or ``None`` if the request fails.
    """
    if not arxiv_id:
        return None

    clean_id = arxiv_id.strip()
    if clean_id in _arxiv_cache:
        return _arxiv_cache[clean_id]

    url = f"{ARXIV_API_BASE}?id_list={clean_id}"
    bibtex: str | None = None
    for attempt in range(2):
        _arxiv_wait()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                xml_data = response.read().decode("utf-8")

                title_match = re.search(r"<title>([^<]+)</title>", xml_data)
                author_match = re.findall(r"<name>([^<]+)</name>", xml_data)
                published_match = re.search(r"<published>(\d{4})-", xml_data)

                if title_match and author_match and published_match:
                    title = title_match.group(1).strip()
                    authors = " and ".join(author_match[:3])
                    year = published_match.group(1)
                    bibtex = (
                        f"@article{{arxiv{clean_id.replace('.', '')},\n"
                        f"  author = {{{authors}}},\n"
                        f"  title = {{{title}}},\n"
                        f"  year = {{{year}}},\n"
                        f"  eprint = {{{clean_id}}},\n"
                        f"  archivePrefix = {{arXiv}}\n"
                        f"}}"
                    )
            break
        except urllib.error.HTTPError as e:
            if e.code == 429:
                retry_after = int(e.headers.get("Retry-After", "10"))
                logger.warning(
                    f"⏳ HTTP 429 from ArXiv for {clean_id} — waiting {retry_after}s "
                    f"(attempt {attempt + 1}/2)"
                )
                time.sleep(retry_after)
                continue
            logger.debug(f"ArXiv API HTTP error for {clean_id}: {e}")
            break
        except Exception as e:
            logger.debug(f"ArXiv API failed for {clean_id}: {e}")
            break

    _arxiv_cache[clean_id] = bibtex
    return bibtex
