"""
Compatibility shim for old import path: from ..utils.prompt_loader import X
Now located at: utils/llm_utils/prompt_loader.py
"""

from .llm_utils.prompt_loader import (
    get_prompt_field,
    load_prompt,
)

__all__ = [
    "load_prompt",
    "get_prompt_field",
]
