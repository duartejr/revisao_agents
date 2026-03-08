"""
core/schemas/corpus.py - Data schemas for document corpus and retrieval.
"""

from typing import Optional, NamedTuple


class Chunk(NamedTuple):
    """Represents a text chunk retrieved from the MongoDB corpus."""

    texto: str
    url: str
    titulo: str
    fonte_idx: int
    file_path: Optional[str] = None  # optional, for compatibility
    chunk_idx: str = ""              # identifies the specific chunk
