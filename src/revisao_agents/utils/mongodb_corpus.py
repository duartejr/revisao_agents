"""
Compatibility shim for old import path: from ..utils.mongodb_corpus import X
Now located at: utils/vector_utils/mongodb_corpus.py
"""

from .vector_utils.mongodb_corpus import CorpusMongoDB

__all__ = [
    "CorpusMongoDB",
]
