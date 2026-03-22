# src/revisao_agents/tools/__init__.py
"""
LangChain tools that agents can call.
Import everything from here for clean access.
"""

from .registry import get_all_tools

__all__ = ["get_all_tools"]
