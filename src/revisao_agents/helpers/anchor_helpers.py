"""
anchor_helpers.py — utilities for extracting anchors from generated text.

An *anchor* is the explicit text fragment used to tie a claim in the LLM
output back to a specific source chunk:  [ANCHOR: "exact verbatim text"][N]

Public API
----------
_extract_main_anchor                  : longest valid anchor in a block.
_extract_citation_anchor              : citation number [N] attached to an anchor.
_extract_all_anchors_with_citations   : list of (anchor_text, citation_number) pairs.
_clean_anchors                        : remove anchor markup from text.
"""

import re
from typing import List, Optional, Tuple

from ..nodes.writing.text_filters import _ANCHORS_PATTERN


def _extract_main_anchor(block: str) -> Optional[str]:
    """Return the longest (most informative) anchor found in *block*.

    Anchors shorter than 20 characters or that look like LaTeX/special symbols
    are discarded.

    Args:
        block: text block to search for anchors
    
    Returns:
        longest valid anchor text, or None if no valid anchors found
    """
    anchors = _ANCHORS_PATTERN.findall(block)
    valid_anchors = [
        a.strip() for a in anchors  
        if len(a.strip()) >= 20
        and not re.match(r'^[\\\$\{\}\[\]_\^]+', a.strip())
    ]
    if not valid_anchors:
        return None
    return max(valid_anchors, key=len)


def _extract_citation_anchor(text: str, anchor: str) -> Optional[int]:
    """Find the citation number [N] that immediately follows *anchor* in *text*.

    Falls back to scanning the 50 characters after the anchor position.

    Args:
        text: the full text to search within
        anchor: the exact anchor text to find and extract citation for
    
    Returns:
        the citation number N if found, or None if not found
    """
    anchor_escaped = re.escape(anchor)
    pattern = re.compile(
        rf'\[ANCHOR:\s*"{anchor_escaped}"\]\s*\[(\d+)\]',
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        return int(match.group(1))

    # Fallback: citation within 50 chars after anchor text
    anchor_pos = text.find(anchor)
    if anchor_pos >= 0:
        subsequent_text = text[anchor_pos: anchor_pos + 50]
        cit_match = re.compile(r'\[(\d+)\]').search(subsequent_text)
        if cit_match:
            return int(cit_match.group(1))
    return None


def _extract_all_anchors_with_citations(block: str) -> List[Tuple[str, Optional[int]]]:
    """Return a list of *(anchor_text, citation_number)* pairs from *block*

    Args:
        block: text block to search for anchors and citations
    
    Returns:
        list of (anchor_text, citation_number) pairs, where citation_number may be None if
    """
    results: List[Tuple[str, Optional[int]]] = []
    pattern = re.compile(
        r'\[ANCHOR:\s*"((?:[^"\\]|\\.)*)"\]\s*\[(\d+)\]',
        re.DOTALL,
    )
    for match in pattern.finditer(block):
        anchor_text = match.group(1).strip()
        citation = int(match.group(2))
        if len(anchor_text) >= 10:
            results.append((anchor_text, citation))
    return results


def _clean_anchors(text: str) -> str:
    """Remove all anchor tags from *text* and normalize whitespace.
    
    Args:
        text: the text containing anchor tags to clean
    
    Returns:
        the cleaned text with anchor tags removed
    """
    clean_text = _ANCHORS_PATTERN.sub("", text)
    clean_text = re.sub(r"  +", " ", clean_text)
    return clean_text.strip()
