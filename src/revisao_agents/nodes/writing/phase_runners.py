"""
phase_runners.py — the individual LLM writing phases (1 – 6).

Phases
------
1  _fase_pensamento         : plan queries and required information.
5  _fase_observacao         : check if the existing corpus is sufficient.
6  _fase_rascunho           : generate the anchored draft via LLM.
   _extrair_com_fallback    : URL extraction with Tavily + fallback retry.
"""

import re
import time
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ...utils.vector_utils.mongodb_corpus import CorpusMongoDB

from ...config import (
    llm_call, parse_json_safe,
    TECHNICAL_MAX_RESULTS, MAX_CORPUS_PROMPT, EXTRACT_MIN_CHARS,
    MAX_URLS_EXTRACT, CTX_ABSTRACT_CHARS, MIN_SECTION_PARAGRAPHS,
    TOP_K_OBSERVATION,
)
from ...core.schemas.techinical_writing import SectionAnswer
from ...utils.llm_utils.prompt_loader import load_prompt
from ...utils.search_utils.tavily_client import search_web, extract_urls, score_url


# ---------------------------------------------------------------------------
# Phase 1: Pensamento (planning)
# ---------------------------------------------------------------------------

def _fase_pensamento(
    tema: str,
    titulo: str,
    cont_esp: str,
    recursos: str,
    prompt_dir: str = "technical_writing",
    language: str = "pt",
) -> dict:
    """Phase 1: generate search queries and list of required information."""
    prompt = load_prompt(
        f"{prompt_dir}/fase_pensamento",
        tema=tema, titulo=titulo, cont_esp=cont_esp, recursos=recursos,
        language=language,
    )
    resp = llm_call(prompt.text, temperature=prompt.temperature)
    resultado = parse_json_safe(resp)
    if resultado:
        return resultado
    return {
        "informacoes_necessarias": [cont_esp[:120]],
        "queries_busca": [f"{tema} {titulo}", f"{titulo} technical details"],
        "queries_imagens": [f"{titulo} diagram architecture"],
    }


# ---------------------------------------------------------------------------
# Phase 5: Observação (corpus-sufficiency check)
# ---------------------------------------------------------------------------

def _fase_observacao(
    informacoes_necessarias: List[str],
    corpus: "CorpusMongoDB",
    prompt_dir: str = "technical_writing",
    language: str = "pt",
) -> dict:
    """Phase 5: decide if the existing corpus is sufficient to write the section."""
    if corpus._n_docs == 0:
        return {
            "suficiente": False,
            "lacunas": informacoes_necessarias,
            "query_complementar": informacoes_necessarias[0] if informacoes_necessarias else "",
            "resumo": "Corpus vazio.",
        }

    query_obs = " ".join(informacoes_necessarias[:3])
    chunks_obs = corpus.query(query_obs, top_k=TOP_K_OBSERVATION)
    amostra_corpus = "\n\n".join(c.text for c in chunks_obs)[:4000]

    informacoes_lista = "\n".join(f"- {i}" for i in informacoes_necessarias)
    prompt_obs = load_prompt(
        f"{prompt_dir}/fase_observacao",
        informacoes_lista=informacoes_lista,
        amostra_corpus=amostra_corpus,
        language=language,
    )
    resp = llm_call(prompt_obs.text, temperature=prompt_obs.temperature)
    resultado = parse_json_safe(resp)
    if resultado:
        return resultado
    return {
        "suficiente": True,
        "lacunas": [],
        "query_complementar": None,
        "resumo": amostra_corpus[:200],
    }


# ---------------------------------------------------------------------------
# Phase 6: Rascunho (anchored draft generation)
# ---------------------------------------------------------------------------

def _fase_rascunho(
    tema: str,
    titulo: str,
    cont_esp: str,
    recursos: str,
    corpus: str,
    urls_secao: List[str],
    resumo_acumulado: str,
    pos: int,
    n_total: int,
    titulos_todos: List[str],
    n_extraidos: int,
    prompt_dir: str = "technical_writing",
    language: str = "pt",
    min_sources: int = 0,
) -> tuple:
    """Phase 6: generate the anchored draft for one section using the LLM."""
    ctx_anteriores = ""
    if resumo_acumulado.strip():
        ctx_anteriores = (
            "══ SEÇÕES JÁ ESCRITAS (não repita estes conceitos) ══\n"
            f"{resumo_acumulado[:CTX_ABSTRACT_CHARS]}\n"
            "══════════════════════════════════════════════════════\n\n"
        )

    todas_txt = "\n".join(
        f"  {'→ ' if i == pos else '  '}{i+1}. {t}"
        for i, t in enumerate(titulos_todos)
    )

    instru = load_prompt(
        f"{prompt_dir}/fase_rascunho",
        secao_min_paragrafos=MIN_SECTION_PARAGRAPHS,
        language=language,
        min_sources=min_sources if min_sources > 0 else 2,
    )
    prompt = (
        f"TEMA: {tema}\n"
        f"SEÇÃO: {pos+1}/{n_total} — {titulo}\n"
        f"OBJETIVOS: {cont_esp}\n"
        f"RECURSOS OBRIGATÓRIOS: {recursos if recursos else 'conforme conteúdo técnico'}\n\n"
        f"ESTRUTURA DO CAPÍTULO:\n{todas_txt}\n\n"
        f"{ctx_anteriores}"
        f"{'━'*60}\n"
        f"CORPUS DE FONTES — {n_extraidos} documentos indexados "
        f"(abaixo: trechos mais relevantes recuperados por similaridade)\n"
        f"{'━'*60}\n"
        f"{corpus}\n\n"
        + instru.text
        + f"\n## {titulo}\n"
    )
    resultado: SectionAnswer = llm_call(
        prompt, temperature=instru.temperature, response_schema=SectionAnswer
    )
    draft = resultado.draft
    used_sources = resultado.used_sources

    return draft, used_sources


# ---------------------------------------------------------------------------
# URL extraction helper (phases 2-4)
# ---------------------------------------------------------------------------

def _extrair_com_fallback(
    resultados: List[dict],
    queries_fallback: List[str],
    urls_tentadas: set,
    corpus: "CorpusMongoDB",
) -> tuple:
    """Extract full text from priority URLs with Tavily; retry with fallback queries.

    Returns (extraidos_validos, resultados_enriquecidos, urls_tentadas).
    URLs already indexed in MongoDB are skipped.
    """
    scored = sorted(
        [
            (
                r.get("url", ""),
                score_url(r.get("url", ""), r.get("snippet", ""), float(r.get("score", 0))),
            )
            for r in resultados if r.get("url")
        ],
        key=lambda x: x[1],
        reverse=True,
    )

    urls_para_extrair = []
    for url, sc in scored:
        if url in urls_tentadas:
            continue
        if corpus.url_exists(url):
            print(f"      ⏭️ URL já indexada, pulando extração: {url[:]}")
            urls_tentadas.add(url)
            continue
        urls_para_extrair.append(url)
        if len(urls_para_extrair) >= MAX_URLS_EXTRACT:
            break

    if not urls_para_extrair:
        return [], resultados, urls_tentadas

    urls_tentadas.update(urls_para_extrair)
    validos = []
    falhos = []

    tavily_enabled = getattr(corpus, "tavily_enabled", True)
    if tavily_enabled:
        raw = extract_urls(urls_para_extrair)
        for item in raw:
            url = item.get("url", "")
            c = item.get("content", "")
            if len(c) >= EXTRACT_MIN_CHARS:
                validos.append(item)
                print(f"      ✅ {url[:]} ({len(c):,} chars)")
            else:
                falhos.append(url)
                print(f"      ✖  {url[:]} (<{EXTRACT_MIN_CHARS} chars)")
    else:
        print("      ⏭️ Tavily search/extract disabled by user.")
        falhos.extend(urls_para_extrair)

    if len(falhos) > len(validos) and queries_fallback and tavily_enabled:
        print(f"      🔄 {len(falhos)} falha(s) → buscando alternativas...")
        for q in queries_fallback[:2]:
            res_alt = search_web(f"{q} filetype:pdf", max_results=6)
            for r in res_alt:
                u = r.get("url", "")
                if u and u not in urls_tentadas and not corpus.url_exists(u):
                    urls_tentadas.add(u)
                    for item in extract_urls([u]):
                        if len(item.get("content", "")) >= EXTRACT_MIN_CHARS:
                            validos.append(item)
                            resultados.extend(res_alt)
                            print(f"      ✅ Fallback: {item.get('url', '')[:72]}")
            if validos:
                break
    elif len(falhos) > len(validos) and queries_fallback and not tavily_enabled:
        print("      ⏭️ Tavily fallback search disabled by user.")

    return validos, resultados, urls_tentadas
