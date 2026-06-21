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


_PT_WORDS = [
    "para",
    "como",
    "que",
    "com",
    "mais",
    "dos",
    "das",
    "pela",
    "pelo",
    "são",
    "foi",
    "está",
    "sobre",
    "entre",
    "através",
    "também",
    "ser",
    "por",
    "uma",
    "seus",
    "suas",
    "este",
    "esta",
    "pode",
    "podem",
]

_EN_WORDS = [
    "the",
    "and",
    "for",
    "with",
    "this",
    "from",
    "that",
    "have",
    "was",
    "are",
    "been",
    "their",
    "which",
    "were",
    "when",
    "through",
    "where",
    "using",
    "can",
    "these",
    "those",
    "such",
    "would",
    "should",
]


def detect_language(text: str, tie_break: str = "en") -> str:
    """Detect whether *text* is predominantly Portuguese or English.

    Uses a cheap keyword-frequency heuristic (substring counts plus a bonus
    for Portuguese-specific diacritics) over the full *text*; not a
    substitute for a real language-detection library, but good enough for
    choosing which language to respond/write in.

    Args:
        text:      The input text to analyze.
        tie_break: Value to return when *text* is empty or scores tie.

    Returns:
        "pt" or "en".
    """
    if not text:
        return tie_break

    sample = text.lower()
    count_pt = sum(1 for w in _PT_WORDS if f" {w} " in f" {sample} ")
    count_en = sum(1 for w in _EN_WORDS if f" {w} " in f" {sample} ")

    if "ã" in sample or "ç" in sample or "õ" in sample:
        count_pt += 3

    if count_pt == count_en:
        return tie_break
    return "pt" if count_pt > count_en else "en"


def truncate(text: str, max_chars: int = 2000, suffix: str = "…") -> str:
    """Truncate *text* to *max_chars*, appending *suffix* if cut.
    If *text* is shorter than *max_chars*, returns it unchanged.

    Args:
        text:      Input string to truncate.
        max_chars: Maximum allowed characters before truncation.
        suffix:    String to append when truncation occurs (default: "…").

    Returns:
        Truncated string with suffix if applicable.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + suffix
