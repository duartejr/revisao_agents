"""
arxiv_utils.py — ArXiv ID extraction and BibTeX retrieval.

Public API
----------
ARXIV_API_BASE
extract_arxiv_id      : parse an ArXiv ID from a URL/path.
get_bibtex_from_arxiv : fetch metadata from the ArXiv API & build minimal BibTeX.
"""

import re
import logging
import urllib.request
import urllib.error
from typing import Optional

logger = logging.getLogger(__name__)

ARXIV_API_BASE = "http://export.arxiv.org/api/query"

_USER_AGENT = "ReviewAgent/1.0"


# ---------------------------------------------------------------------------

def extract_arxiv_id(file_path: str) -> Optional[str]:
    """Return the ArXiv identifier embedded in *file_path*, or ``None``."""
    if not file_path:
        return None

    arxiv_match = re.search(
        r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})', file_path, re.IGNORECASE
    )
    if arxiv_match:
        return arxiv_match.group(1)

    arxiv_id_match = re.search(r'(\d{4}\.\d{4,5})', file_path)
    if arxiv_id_match:
        return arxiv_id_match.group(1)

    return None


def get_bibtex_from_arxiv(arxiv_id: str, timeout: int = 10) -> Optional[str]:
    """Query the ArXiv Atom API and return a minimal BibTeX entry, or ``None``."""
    if not arxiv_id:
        return None

    url = f"{ARXIV_API_BASE}?id_list={arxiv_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            xml_data = response.read().decode('utf-8')

            title_match = re.search(r'<title>([^<]+)</title>', xml_data)
            author_match = re.findall(r'<name>([^<]+)</name>', xml_data)
            published_match = re.search(r'<published>(\d{4})-', xml_data)

            if title_match and author_match and published_match:
                title = title_match.group(1).strip()
                authors = ' and '.join(author_match[:3])
                year = published_match.group(1)
                bibtex = (
                    f"@article{{arxiv{arxiv_id.replace('.', '')},\n"
                    f"  author = {{{authors}}},\n"
                    f"  title = {{{title}}},\n"
                    f"  year = {{{year}}},\n"
                    f"  eprint = {{{arxiv_id}}},\n"
                    f"  archivePrefix = {{arXiv}}\n"
                    f"}}"
                )
                return bibtex
    except Exception as e:
        logger.debug(f"ArXiv API failed for {arxiv_id}: {e}")

    return None
