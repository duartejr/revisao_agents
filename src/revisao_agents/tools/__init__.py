# src/revisao_agents/tools/__init__.py
"""
Ferramentas LangChain que os agents podem chamar.
Importe tudo daqui para ter acesso limpo.
"""

from .registry import get_all_tools

__all__ = ["get_all_tools"]