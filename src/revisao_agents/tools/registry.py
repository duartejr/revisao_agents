# src/revisao_agents/tools/registry.py
"""
Registro central de TODAS as tools do projeto.
"""

from langchain_core.tools import BaseTool
from typing import List

# === Tools do corpus MongoDB ===
from .academic_corpus_search import search_academic_corpus

# === Tools do Tavily (todas as 5) ===
from .tavily_web_search import (
    search_tavily,
    search_tavily_incremental,
    search_tavily_tecnico,
    search_tavily_images,
    extract_tavily,
)

TOOLS: List[BaseTool] = [
    search_academic_corpus,
    search_tavily,
    search_tavily_incremental,
    search_tavily_tecnico,
    search_tavily_images,
    extract_tavily,
    # ← novas tools basta adicionar aqui
]

def get_all_tools() -> List[BaseTool]:
    """Retorna todas as tools prontas para bind_tools() ou agent."""
    return TOOLS