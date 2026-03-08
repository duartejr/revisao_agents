# src/revisao_agents/tools/registry.py
"""
Registro central de TODAS as tools do projeto.
Basta adicionar novas tools aqui — o resto do código nunca muda.
"""

from langchain_core.tools import BaseTool
from typing import List

# Importe aqui todas as tools (elas já vêm com @tool)
from .academic_corpus_search import search_academic_corpus

# Lista central (fácil de manter e escalar)
TOOLS: List[BaseTool] = [
    search_academic_corpus,
    # ← adicione novas tools aqui no futuro (ex: web_search, calculator, etc.)
]

def get_all_tools() -> List[BaseTool]:
    """Retorna a lista pronta para bind_tools() ou .bind_tools() no agent."""
    return TOOLS