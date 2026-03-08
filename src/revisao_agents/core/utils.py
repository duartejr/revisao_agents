"""
core/utils.py - Shared utilities for core schemas and data processing.

Pure functions with no side-effects — safe to import anywhere.
"""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json_safe(text: str, default: Any = None) -> Any:
    """
    Attempt to parse *text* as JSON.

    Tries the raw string first; if that fails, looks for the first
    JSON object/array embedded in the text (e.g. after a markdown
    fence or trailing prose).

    Args:
        text:    String that should contain JSON.
        default: Value to return when parsing fails entirely.

    Returns:
        Parsed Python object, or *default*.
    """
    if not text:
        return default

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract first {...} or [...] block
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    return default


def truncate(text: str, max_chars: int = 2000, suffix: str = "…") -> str:
    """Truncate *text* to *max_chars*, appending *suffix* if cut."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + suffix


def slugify(text: str) -> str:
    """Convert *text* to a filesystem-safe slug (lowercase, hyphens)."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text
