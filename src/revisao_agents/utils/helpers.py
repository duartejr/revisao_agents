"""
Compatibility shim for old import path: from ..utils.helpers import X
Now located at: utils/file_utils/helpers.py
"""

from .file_utils.helpers import (
    fmt_chunks,
    fmt_snippets,
    resumir_hist,
    truncar,
    salvar_md,
    resumir_secao,
    parse_plano_tecnico,
    parse_plano_academico,
    normalizar,
    fuzzy_sim,
    fuzzy_search_in_text,
    extrair_anchors,
    eh_paragrafo_verificavel,
)

__all__ = [
    "fmt_chunks",
    "fmt_snippets",
    "resumir_hist",
    "truncar",
    "salvar_md",
    "resumir_secao",
    "parse_plano_tecnico",
    "parse_plano_academico",
    "normalizar",
    "fuzzy_sim",
    "fuzzy_search_in_text",
    "extrair_anchors",
    "eh_paragrafo_verificavel",
]
