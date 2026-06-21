"""
Compatibility shim for old import path: from ..utils.llm_providers import X
Now located at: utils/llm_utils/llm_providers.py
"""

from .llm_utils.llm_providers import (
    LLMProvider,
    get_llm,
    llm_call,
)

__all__ = [
    "get_llm",
    "LLMProvider",
    "llm_call",
]
