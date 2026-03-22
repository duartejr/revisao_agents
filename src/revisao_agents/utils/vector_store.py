"""
Compatibility shim for old import path: from ..utils.vector_store import X
Now located at: utils/vector_utils/vector_store.py
"""

from .vector_utils.vector_store import (
    accumulate_chunks,
    search_chunks,
)

__all__ = [
    "search_chunks",
    "accumulate_chunks",
]
