# src/revisao_agents/tools/registry.py
"""
Central registry of ALL project tools.
"""

from langchain_core.tools import BaseTool
from typing import List

# === Tools for MongoDB corpus ===
from .academic_corpus_search import search_academic_corpus

# === Tools for Tavily (all 5) ===
from .tavily_web_search import (
    search_tavily,
    search_tavily_incremental,
    search_tavily_technical,
    search_tavily_images,
    extract_tavily,
)

# === Tool to get current date ===
from .get_current_date import get_current_date

TOOLS: List[BaseTool] = [
    get_current_date,
    search_academic_corpus,
    search_tavily,
    search_tavily_incremental,
    search_tavily_technical,
    search_tavily_images,
    extract_tavily,
]

def get_all_tools() -> List[BaseTool]:
    """Returns all tools ready for bind_tools() or agent."""
    return TOOLS