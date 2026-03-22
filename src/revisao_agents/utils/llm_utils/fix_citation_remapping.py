"""
COMPLETE SOLUTION: Citation Tracking with Source → URL Mapping
======================================================================

Problem: Citations [14] in the text do not match URLs [1], [2], [3] in the references
Cause: No tracking of which URL was used in which paragraph

Solution:

1. During writing: Track which URL was used in each section
2. During consolidation: Map citations to actual URLs
3. Re-number everything consistently

This is a patch for write_sections_node and consolidate_node
"""

import re
from typing import Dict, List, Tuple, Set


class CitationTracker:
    """Tracks which source was used in which paragraph."""

    def __init__(self):
        # Map: {citation_number: url}
        self.citation_to_url: Dict[int, str] = {}

        # Counter for sources in this section
        self.source_counter = 1

        # URLs already added (avoids duplicates)
        self.seen_urls: Set[str] = set()

    def add_source(self, url: str) -> int:
        """
        Registers a URL and returns the citation number.

        Example:
            tracker.add_source("https://paper1.pdf")
            → returns 1

            tracker.add_source("https://paper2.pdf")
            → returns 2

            tracker.add_source("https://paper1.pdf")  # duplicate
            → returns 1 (already added)

        Args:
            url: The URL of the source to add.

        Returns:
            The citation number assigned to this URL.
        """
        if url in self.seen_urls:
            # Finds the already assigned number
            for num, u in self.citation_to_url.items():
                if u == url:
                    return num

        # New URL
        self.seen_urls.add(url)
        num = self.source_counter
        self.citation_to_url[num] = url
        self.source_counter += 1
        return num

    def get_ordered_urls(self) -> List[str]:
        """Returns a list of URLs in the order of citations [1], [2], [3]...

        Example:
            If citation_to_url = {1: "https://paper1.pdf", 2: "https://paper2.pdf"}
            → returns ["https://paper1.pdf", "https://paper2.pdf"]
        """
        urls = []
        for i in range(1, self.source_counter):
            if i in self.citation_to_url:
                urls.append(self.citation_to_url[i])
        return urls

    def get_full_map(self) -> Dict[int, str]:
        """Returns the complete dictionary {citation_number: url}

        Example:
            If citation_to_url = {1: "https://paper1.pdf", 2: "https://paper2.pdf"}
            → returns {1: "https://paper1.pdf", 2: "https://paper2.pdf"}
        """
        return self.citation_to_url.copy()


def extract_numbered_citations(text: str) -> List[int]:
    """
    Extracts ALL [N] from the text in the order of appearance.

    Example:
        "conforme [13]... e [14]... e [13] novamente"
        → [13, 14, 13]
    """
    pattern = r"\[(\d+)\]"
    matches = re.findall(pattern, text)
    return [int(m) for m in matches]


def create_remap_map(original_citations: List[int]) -> Dict[int, int]:
    """
    Creates a map old_idx → new_idx in the order of first appearance.

    Example:
        [13, 14, 13, 15]
        → {13: 1, 14: 2, 15: 3}

    Args:
        original_citations: A list of citation numbers extracted from the text.

    Returns:
        A dictionary mapping original citation numbers to new sequential numbers.
    """
    map = {}
    new_idx = 1
    for old_idx in original_citations:
        if old_idx not in map:
            map[old_idx] = new_idx
            new_idx += 1
    return map


def remap_text_with_tracking(
    text: str,
    original_source_map: Dict[int, str],
    remap_map: Dict[int, int],
) -> Tuple[str, Dict[int, str], Dict[int, int]]:
    """
    Re-maps citations AND returns a new source→URL map.

    Args:
        text: "as shown in [13]... and [14]..."
        original_source_map: {13: "url13", 14: "url14", ...}
        remap_map: {13: 1, 14: 2, ...}

    Returns:
        (remapped_text, new_source_map, remap_map)

    Example:
        text = "as shown in [13]... and [14]..."
        original_source_map = {13: "https://paper13.pdf", 14: "https://paper14.pdf"}
        remap_map = {13: 1, 14: 2}

        → remapped_text = "as shown in [1]... and [2]..."
        → new_source_map = {1: "https://paper13.pdf", 2: "https://paper14.pdf"}
        → remap_map = {13: 1, 14: 2}
    """

    def replace_idx(match: re.Match) -> str:
        """Replaces [old_idx] with [new_idx] using remap_map.

        Args:
            match: A regex match object for a citation like [13].

        Returns:
            A string with the citation number remapped, like [1].
        """
        old_idx = int(match.group(1))
        new_idx = remap_map.get(old_idx, old_idx)
        return f"[{new_idx}]"

    remapped_text = re.sub(r"\[(\d+)\]", replace_idx, text)

    # Create new map source→url with new indices
    new_source_map = {}
    for old_idx, new_idx in remap_map.items():
        if old_idx in original_source_map:
            new_source_map[new_idx] = original_source_map[old_idx]

    return remapped_text, new_source_map, remap_map


def synchronize_text_with_references(
    text: str,
    original_source_map: Dict[int, str],
) -> Tuple[str, List[str]]:
    """
    Completely synchronizes text and references.

    Workflow:
    1. Extracts [N] from the text
    2. Creates a remap map
    3. Re-numbers [N]
    4. Re-orders URLs

    Args:
        text: "as shown in [13]... and [14]..."
        original_source_map: {13: "url", 14: "url", ...}

    Returns:
        (text_with_[1][2], ordered_urls_[1][2])
    """

    # Extract original citations
    citations = extract_numbered_citations(text)

    if not citations:
        return text, list(original_source_map.values())

    # Create remap map
    remap_map = create_remap_map(citations)

    # Re-map
    new_text, new_source_map, _ = remap_text_with_tracking(
        text, original_source_map, remap_map
    )

    # Extract URLs in new order
    ordered_urls = []
    for i in range(1, len(new_source_map) + 1):
        if i in new_source_map:
            ordered_urls.append(new_source_map[i])

    return new_text, ordered_urls
