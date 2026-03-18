"""
technical_writing.py — LangGraph graph nodes for technical/academic chapter authoring.

All logic has been extracted into the `nodes/writing/` subpackage for maintainability:
    text_filters.py   : regex patterns and LLM output cleanup.
    anchor_helpers.py : anchor extraction utilities.
    phase_runners.py  : phases 1-6 (plan, observe, draft, extract).
    verification.py   : adaptive judge (REACT verification loop).
    node_parsear.py   : parsear_plano_node implementation.
    node_escrever.py  : escrever_secoes_node implementation.
    node_consolidar.py: consolidar_node implementation.
"""

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Re-export helper symbols (kept for backward compatibility)
# ---------------------------------------------------------------------------
from .writing import (
    _ANCHORS_PATTERN,
    _strip_justification_blocks,
    _strip_meta_sentences,
    _strip_figure_table_refs,
    _extract_main_anchor,
    _extract_citation_anchor,
    _extract_all_anchors_with_citations,
    _fase_pensamento,
    _fase_observacao,
    _fase_rascunho,
    _extrair_com_fallback,
    _contar_claims_verificaveis,
    _juiz_paragrafo_melhorado,
    _monitorar_taxa_verificacao,
    _buscar_conteudo_complementar,
    _verificar_e_corrigir_secao_adaptativa,
    _verificar_paragrafo_com_anchor,
    _verificar_e_corrigir_secao_com_anchor,
)

# ---------------------------------------------------------------------------
# Graph nodes (implementations in node_parsear, node_escrever, node_consolidar)
# ---------------------------------------------------------------------------
from .writing.node_parsear import parsear_plano_node
from .writing.node_escrever import escrever_secoes_node
from .writing.node_consolidar import consolidar_node

# Canonical English aliases (Portuguese names kept for backward compatibility)
parse_plan_node = parsear_plano_node
write_sections_node = escrever_secoes_node
consolidate_node = consolidar_node

__all__ = [
    "parse_plan_node",
    "write_sections_node",
    "consolidate_node",
    "parsear_plano_node",
    "escrever_secoes_node",
    "consolidar_node",
]
