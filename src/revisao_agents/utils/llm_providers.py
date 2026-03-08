"""
Compatibility shim for old import path: from ..utils.llm_providers import X
Now located at: utils/llm_utils/llm_providers.py
"""

from .llm_utils.llm_providers import (
    get_llm,
    LLMProvider,
    llm_call,
    parse_json_safe,
)

__all__ = [
    "get_llm",
    "LLMProvider",
    "llm_call",
    "parse_json_safe",
]
