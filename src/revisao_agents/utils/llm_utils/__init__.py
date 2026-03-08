"""
LLM utilities: prompt loading, LLM providers, and citation handling.
"""

from .prompt_loader import load_prompt
from .llm_providers import llm_call, parse_json_safe, get_llm, LLMProvider
from .fix_citation_remapping import (
    sincronizar_texto_com_references,
    remapear_texto_com_rastreamento,
    criar_mapa_remapeamento,
    extrair_citacoes_numeradas,
)

__all__ = [
    "load_prompt",
    "llm_call",
    "parse_json_safe",
    "get_llm",
    "LLMProvider",
    "sincronizar_texto_com_references",
    "remapear_texto_com_rastreamento",
    "criar_mapa_remapeamento",
    "extrair_citacoes_numeradas",
]
