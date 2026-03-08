"""
Compatibility shim for old import path: from ..utils.vector_store import X
Now located at: utils/vector_utils/vector_store.py
"""

from .vector_utils.vector_store import (
    buscar_chunks,
    acumular_chunks,
)

__all__ = [
    "buscar_chunks",
    "acumular_chunks",
]
