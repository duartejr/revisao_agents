"""
abnt_utils.py — BibTeX → ABNT citation formatting and fallback generator.

Public API
----------
bibtex_to_abnt         : convert a BibTeX string to a simplified ABNT citation.
generate_fallback_abnt : produce a minimal ABNT-like citation from a file path/URL.
"""

import re
from pathlib import Path
from typing import Optional


_BIBTEX_PATTERNS = {
    'author':  r'author\s*=\s*["{]([^"}]+)["}]',
    'title':   r'title\s*=\s*["{]([^"}]+)["}]',
    'year':    r'year\s*=\s*["{]?(\d{4})',
    'journal': r'journal\s*=\s*["{]([^"}]+)["}]',
    'doi':     r'doi\s*=\s*["{]([^"}]+)["}]',
}


def bibtex_to_abnt(bibtex: str, url: Optional[str] = None) -> str:
    """Convert a BibTeX entry to a simplified ABNT-style citation string.

    Format: ``Author. Title. Journal, Year. DOI: … | Disponível em: …``
    """
    extracted: dict = {}
    for key, pattern in _BIBTEX_PATTERNS.items():
        match = re.search(pattern, bibtex, re.IGNORECASE)
        if match:
            extracted[key] = match.group(1).strip()

    author = extracted.get('author', 'Unknown Author')
    year = extracted.get('year', 'n.d.')
    title = extracted.get('title', 'Unknown Title')
    journal = extracted.get('journal', '')
    doi = extracted.get('doi', '')

    if journal:
        abnt = f"{author}. {title}. {journal}, {year}."
    else:
        abnt = f"{author}. {title}, {year}."

    if doi:
        abnt += f" DOI: {doi}"
    elif url:
        abnt += f" Disponível em: {url}"

    return abnt


def generate_fallback_abnt(file_path: str) -> str:
    """Return a minimal ABNT-like citation when no bibliographic data is available."""
    file_name = Path(file_path).stem
    file_name_clean = " ".join(file_name.split("_")[:5])
    return f"{file_name_clean}. Disponível em: {file_path}"
