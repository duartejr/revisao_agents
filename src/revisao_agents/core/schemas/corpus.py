"""
core/schemas/corpus.py - Data schemas for document corpus and retrieval.
"""

from typing import NamedTuple


class Chunk(NamedTuple):
    """Represents a text chunk retrieved from the MongoDB corpus."""

    text: str
    url: str
    title: str
    source_idx: int
    file_path: str | None = None  # optional, for compatibility
    chunk_idx: str = ""  # identifies the specific chunk
