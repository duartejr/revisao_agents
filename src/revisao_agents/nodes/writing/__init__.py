"""
nodes.writing — internal helpers and graph-node implementations for section writing.

Submodules
----------
text_filters    : regex patterns and strip helpers for LLM output cleanup.
anchor_helpers  : anchor extraction utilities.
phase_runners   : the individual writing phases (phases 1-6).
verification    : adaptive paragraph verification (judge + REACT loop).
node_parsear    : parsear_plano_node — parses a plan file and extracts sections.
node_escrever   : escrever_secoes_node — writes sections with search and verification.
node_consolidar : consolidar_node — consolidates written sections into a final document.
"""
from .text_filters import (
    _ANCORA_PATTERN,
    _strip_justification_blocks,
    _strip_meta_sentences,
    _strip_figure_table_refs,
)
from .anchor_helpers import (
    _extrair_ancora_principal,
    _extrair_citacao_ancora,
    _extrair_todas_ancoras_com_citacoes,
)
from .phase_runners import (
    _fase_pensamento,
    _fase_observacao,
    _fase_rascunho,
    _extrair_com_fallback,
)
from .verification import (
    _contar_claims_verificaveis,
    _juiz_paragrafo_melhorado,
    _monitorar_taxa_verificacao,
    _buscar_conteudo_complementar,
    _verificar_e_corrigir_secao_adaptativa,
    _verificar_paragrafo_com_ancora,
    _verificar_e_corrigir_secao_com_ancora,
)

__all__ = [
    # text_filters
    "_ANCORA_PATTERN",
    "_strip_justification_blocks",
    "_strip_meta_sentences",
    "_strip_figure_table_refs",
    # anchor_helpers
    "_extrair_ancora_principal",
    "_extrair_citacao_ancora",
    "_extrair_todas_ancoras_com_citacoes",
    # phase_runners
    "_fase_pensamento",
    "_fase_observacao",
    "_fase_rascunho",
    "_extrair_com_fallback",
    # verification
    "_contar_claims_verificaveis",
    "_juiz_paragrafo_melhorado",
    "_monitorar_taxa_verificacao",
    "_buscar_conteudo_complementar",
    "_verificar_e_corrigir_secao_adaptativa",
    "_verificar_paragrafo_com_ancora",
    "_verificar_e_corrigir_secao_com_ancora",
]
