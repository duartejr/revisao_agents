# src/revisao_agents/__init__.py
"""
Revisão de textos acadêmicos com Agents (baseado em LangGraph).

Pacote principal para o sistema de revisão agentica.
"""

__version__ = "0.1.0"

# Exporta os itens mais importantes para facilitar imports de alto nível
from .config import get_settings
from .state import ReviewState
from .hitl import human_feedback_node
from .cli import app as cli_app

# Opcional: se quiser expor graphs/agents diretamente
# from .graphs.review_graph import build_review_graph