"""
parsear_plano_node — parses a plan file and extracts sections
Part of the nodes/writing subpackage.
"""
import re
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

from ...state import EscritaTecnicaState
from ...config import (
    llm_call, parse_json_safe,
    TECNICO_MAX_RESULTS, MAX_CORPUS_PROMPT, EXTRACT_MIN_CHARS,
    MAX_URLS_EXTRACT, CTX_RESUMO_CHARS, SECAO_MIN_PARAGRAFOS,
    DELAY_ENTRE_SECOES, MAX_REACT_ITERATIONS, TOP_K_OBSERVACAO,
)
from ...core.schemas.techinical_writing import RespostaSecao, Fonte
from ...utils.vector_utils.mongodb_corpus import CorpusMongoDB
from ...utils.file_utils.helpers import resumir_secao, parse_plano_tecnico, parse_plano_academico
from ...core.schemas.writer_config import WriterConfig
from ...utils.search_utils.tavily_client import search_web, search_images, extract_urls, score_url
from ...utils.llm_utils.prompt_loader import load_prompt
from ...utils.bib_utils.crossref_bibtex import get_reference_data_react, bibtex_to_abnt
from .text_filters import _strip_justification_blocks, _strip_meta_sentences, _strip_figure_table_refs
from .anchor_helpers import _ANCORA_PATTERN, _extrair_ancora_principal, _extrair_citacao_ancora, _extrair_todas_ancoras_com_citacoes
from .phase_runners import _fase_pensamento, _fase_observacao, _fase_rascunho, _extrair_com_fallback
from .verification import (
    _contar_claims_verificaveis, _juiz_paragrafo_melhorado,
    _monitorar_taxa_verificacao, _buscar_conteudo_complementar,
    _verificar_e_corrigir_secao_adaptativa,
    _verificar_paragrafo_com_ancora, _verificar_e_corrigir_secao_com_ancora,
)

def parsear_plano_node(state: EscritaTecnicaState) -> dict:
    """Parses a plan file and extracts sections. Supports both technical and academic modes."""
    config = WriterConfig.from_dict(state.get("writer_config", {}))
    caminho = state["caminho_plano"]
    print(f"\n📖 Lendo plano: {caminho} (modo: {config.mode})")
    with open(caminho, "r", encoding="utf-8") as f:
        texto = f.read()
    if config.mode == "academic":
        tema, resumo, secoes = parse_plano_academico(texto)
    else:
        tema, resumo, secoes = parse_plano_tecnico(texto)
    print(f"   ✅ Tema: {tema} | {len(secoes)} seções")
    for s in secoes:
        print(f"      [{s['indice']+1}] {s['titulo']}")
    return {
        "tema": tema,
        "resumo_plano": resumo,
        "secoes": secoes,
        "secoes_escritas": [],
        "refs_urls": [],
        "refs_imagens": [],
        "resumo_acumulado": "",
        "react_log": [],
        "stats_verificacao": [],
        "status": "plano_parseado",
        "caminho_plano": caminho,
        "writer_config": state.get("writer_config", {}),
    }


