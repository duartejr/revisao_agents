"""
anchor_helpers.py — utilities for extracting anchors from generated text.

An *anchor* is the explicit text fragment used to tie a claim in the LLM
output back to a specific source chunk:  [ÂNCORA: "exact verbatim text"][N]

Public API
----------
_extrair_ancora_principal          : longest valid anchor in a block.
_extrair_citacao_ancora            : citation number [N] attached to an anchor.
_extrair_todas_ancoras_com_citacoes: list of (anchor_text, citation_number) pairs.
"""

import re
from typing import List, Optional, Tuple

from .text_filters import _ANCORA_PATTERN


def _extrair_ancora_principal(bloco: str) -> Optional[str]:
    """Return the longest (most informative) anchor found in *bloco*.

    Anchors shorter than 20 characters or that look like LaTeX/special symbols
    are discarded.
    """
    ancoras = _ANCORA_PATTERN.findall(bloco)
    ancoras_validas = [
        a.strip() for a in ancoras
        if len(a.strip()) >= 20
        and not re.match(r'^[\\\$\{\}\[\]_\^]+', a.strip())
    ]
    if not ancoras_validas:
        return None
    return max(ancoras_validas, key=len)


def _extrair_citacao_ancora(texto: str, ancora: str) -> Optional[int]:
    """Find the citation number [N] that immediately follows *ancora* in *texto*.

    Falls back to scanning the 50 characters after the anchor position.
    """
    ancora_escaped = re.escape(ancora)
    pattern = re.compile(
        rf'\[ÂNCORA:\s*"{ancora_escaped}"\]\s*\[(\d+)\]',
        re.IGNORECASE,
    )
    match = pattern.search(texto)
    if match:
        return int(match.group(1))

    # Fallback: citation within 50 chars after anchor text
    ancora_pos = texto.find(ancora)
    if ancora_pos >= 0:
        trecho_posterior = texto[ancora_pos: ancora_pos + 50]
        cit_match = re.compile(r'\[(\d+)\]').search(trecho_posterior)
        if cit_match:
            return int(cit_match.group(1))
    return None


def _extrair_todas_ancoras_com_citacoes(bloco: str) -> List[Tuple[str, Optional[int]]]:
    """Return a list of *(anchor_text, citation_number)* pairs from *bloco*.

    Only includes anchors ≥ 10 characters.  Anchors without a directly
    trailing [N] are included with citation_number = None.
    """
    resultados: List[Tuple[str, Optional[int]]] = []
    pattern = re.compile(
        r'\[ÂNCORA:\s*"((?:[^"\\]|\\.)*)"\]\s*\[(\d+)\]',
        re.DOTALL,
    )
    for match in pattern.finditer(bloco):
        texto_ancora = match.group(1).strip()
        citacao = int(match.group(2))
        if len(texto_ancora) >= 10:
            resultados.append((texto_ancora, citacao))
    return resultados
