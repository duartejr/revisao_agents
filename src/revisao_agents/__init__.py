"""
Revisão de textos acadêmicos com Agents (baseado em LangGraph).
"""

__version__ = "0.1.0"

from .state import RevisaoState, EscritaTecnicaState, ReviewState

__all__ = [
    "RevisaoState",
    "EscritaTecnicaState",
    "ReviewState",
]