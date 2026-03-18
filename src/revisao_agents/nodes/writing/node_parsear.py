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

from ...state import TechnicalWriterState
from ...config import (
    llm_call, parse_json_safe,
    TECHNICAL_MAX_RESULTS, MAX_CORPUS_PROMPT, EXTRACT_MIN_CHARS,
    MAX_URLS_EXTRACT, CTX_ABSTRACT_CHARS, MIN_SECTION_PARAGRAPHS,
    DELAY_BETWEEN_SECTIONS, MAX_REACT_ITERATIONS, TOP_K_OBSERVATION,
)
from ...utils.vector_utils.mongodb_corpus import CorpusMongoDB
from ...utils.file_utils.helpers import parse_technical_plan, parse_academic_plan
from ...core.schemas.writer_config import WriterConfig
from ...utils.search_utils.tavily_client import search_web, search_images, extract_urls, score_url
from ...utils.llm_utils.prompt_loader import load_prompt
from ...utils.bib_utils.crossref_bibtex import get_reference_data_react, bibtex_to_abnt
from .text_filters import _strip_justification_blocks, _strip_meta_sentences, _strip_figure_table_refs
from ...helpers.anchor_helpers import _ANCHORS_PATTERN, _extract_main_anchor, _extract_citation_anchor, _extract_all_anchors_with_citations
from .phase_runners import _fase_pensamento, _fase_observacao, _fase_rascunho, _extrair_com_fallback
from .verification import (
    _contar_claims_verificaveis, _juiz_paragrafo_melhorado,
    _monitorar_taxa_verificacao, _buscar_conteudo_complementar,
    _verificar_e_corrigir_secao_adaptativa,
    _verificar_paragrafo_com_anchor, _verificar_e_corrigir_secao_com_anchor,
)

def parsear_plano_node(state: TechnicalWriterState) -> dict:
    """Parses a plan file and extracts sections. Supports both technical and academic modes."""
    config = WriterConfig.from_dict(state.get("writer_config", {}))
    plan_path = state["plan_path"]
    print(f"\n📖 Lendo plano: {plan_path} (modo: {config.mode})")
    with open(plan_path, "r", encoding="utf-8") as f:
        text = f.read()
    if config.mode == "academic":
        theme, plan_summary, sections = parse_academic_plan(text)
    else:
        theme, plan_summary, sections = parse_technical_plan(text)
    print(f"   ✅ Tema: {theme} | {len(sections)} seções")
    for s in sections:
        print(f"      [{s['index']+1}] {s['title']}")
    return {
        "theme": theme,
        "plan_summary": plan_summary,
        "sections": sections,
        "written_sections": [],
        "refs_urls": [],
        "refs_images": [],
        "cumulative_summary": "",
        "react_log": [],
        "verification_stats": [],
        "status": "plano_parseado",
        "plan_path": plan_path,
        "writer_config": state.get("writer_config", {}),
    }


