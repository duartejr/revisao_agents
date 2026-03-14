"""
escrever_secoes_node — writes sections using search, extraction and verification
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
from ...utils.file_utils.helpers import summarize_section
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

def escrever_secoes_node(state: TechnicalWriterState) -> dict:
    """Main node for writing sections with search, extraction and verification."""
    config = WriterConfig.from_dict(state.get("writer_config", {}))
    prompt_dir = config.prompt_dir
    theme = state["theme"]
    sections = state["sections"]
    written_sections = []
    all_refs_urls = list(state.get("refs_urls", []))
    all_refs_images = list(state.get("refs_images", []))
    cumulative_summary = state.get("cumulative_summary", "")
    react_log = list(state.get("react_log", []))
    verification_stats = list(state.get("verification_stats", []))
    n_total = len(sections)
    titulos_todos = [s["title"] for s in sections]
    # CorpusMongoDB instance for URL existence checks (no build)
    corpus_check = CorpusMongoDB()

    tavily_enabled = state.get("tavily_enabled", True)
    for pos, secao in enumerate(sections):
        titulo = secao["title"]
        cont_esp = secao.get("expected_content", "")
        recursos = secao.get("resources", "")
        idx_num = secao.get("index", pos)

        print(f"\n{'━'*70}")
        print(f"  [{pos+1}/{n_total}] PROCESSANDO: {titulo}")
        print(f"{'━'*70}")

        log = [
            f"\n{'='*70}",
            f"SEÇÃO [{pos+1}/{n_total}]: {titulo}",
            f"Timestamp: {datetime.now().isoformat()}",
            f"{'='*70}",
        ]

        # FASE 1: Pensamento
        print(f"\n  🧠 FASE 1 — Pensamento...")
        log.append("\n── FASE 1: PENSAMENTO ──")
        plano = _fase_pensamento(theme, titulo, cont_esp, recursos, prompt_dir=prompt_dir, language=config.language)
        queries = plano.get("queries_busca", [f"{theme} {titulo}"])
        queries_img = plano.get("queries_imagens", [f"{titulo} diagram"])
        informacoes = plano.get("informacoes_necessarias", [cont_esp[:120]])
        log.extend([f"Queries: {queries}", f"Informações: {informacoes}"])
        print(f"     Queries: {queries}")

        # FASES 2-4: Busca + Extração
        print(f"\n  🔎 FASE 2-4 — Busca e Extração...")
        log.append("\n── FASE 2-4: BUSCA + EXTRAÇÃO ──")
        extraidos = []
        resultados = []
        urls_vistas = set()

        # Corpus-first strategy: query existing MongoDB before hitting the web
        _corpus_suficiente = False
        if config.is_corpus_first:
            print(f"\n  🔬 FASE 5 — Observação (corpus-first, antes da busca)...")
            log.append("\n── FASE 5: OBSERVAÇÃO (corpus-first) ──")
            obs = _fase_observacao(informacoes, corpus_check, prompt_dir=prompt_dir, language=config.language)
            _corpus_suficiente = obs.get("suficiente", False)
            log.append(
                f"Corpus suficiente: {_corpus_suficiente} | "
                f"{obs.get('resumo', '')[:120]}"
            )
            print(f"     Corpus suficiente: {_corpus_suficiente}")

        if not _corpus_suficiente and tavily_enabled:
            for q in queries[:4]:
                res = search_web(q, TECHNICAL_MAX_RESULTS)
                # Pass tavily_enabled to corpus for fallback
                corpus_check.tavily_enabled = tavily_enabled
                novos, resultados, urls_vistas = _extrair_com_fallback(
                    res,
                    queries_fallback=[q, titulo],
                    urls_tentadas=urls_vistas,
                    corpus=corpus_check,
                )
                extraidos.extend(novos)
                time.sleep(1)
        elif not _corpus_suficiente and not tavily_enabled:
            print("  ⏭️ Tavily web search disabled by user. Skipping web search.")

        log.append(f"Fontes extraídas: {len(extraidos)}")

        # MongoDB Indexing / corpus selection
        print(f"\n  🗄️  Indexando no MongoDB...")
        log.append("\n── INDEXAÇÃO MONGODB ──")
        slug_secao = re.sub(r"[^\w]", "_", titulo[:30]).lower()
        prefix = f"s{pos+1:02d}_{slug_secao}"

        if _corpus_suficiente:
            # Reuse the global check corpus — no new documents to index
            corpus = corpus_check
        else:
            corpus = CorpusMongoDB().build(extraidos, resultados, prefix=prefix)

        query_retrieval = f"{titulo} {cont_esp} {recursos}"
        corpus_prompt, urls_secao, _ = corpus.render_prompt(
            query_retrieval, max_chars=MAX_CORPUS_PROMPT
        )
        log.append(f"MongoDB: {corpus._n_docs} docs | {corpus._total_chunks} chunks")

        if not corpus_prompt.strip() and not _corpus_suficiente and tavily_enabled:
            print("  ⚠️  Corpus vazio! Busca de último recurso...")
            log.append("⚠️  Corpus vazio — busca de emergência")
            q_emerg = f"{titulo} {theme} technical documentation filetype:pdf"
            res_emerg = search_web(q_emerg, 6)
            corpus_check.tavily_enabled = tavily_enabled
            novos_emerg, _, urls_vistas = _extrair_com_fallback(
                res_emerg, queries_fallback=[titulo], urls_tentadas=urls_vistas,
                corpus=corpus_check,
            )
            if novos_emerg:
                extraidos.extend(novos_emerg)
                corpus = CorpusMongoDB().build(extraidos, resultados, prefix=prefix)
                corpus_prompt, urls_secao, _ = corpus.render_prompt(
                    query_retrieval, max_chars=MAX_CORPUS_PROMPT
                )
                log.append(f"Emergência: {len(novos_emerg)} fontes adicionais")
        elif not corpus_prompt.strip() and not _corpus_suficiente and not tavily_enabled:
            print("  ⏭️ Tavily emergency search disabled by user. Skipping.")

        if not corpus_prompt.strip():
            print("  ❌ FALHA CRÍTICA: nenhuma fonte encontrada.")
            log.append("❌ Nenhuma fonte encontrada.")
            corpus_prompt = (
                "AVISO: Nenhuma fonte encontrada. Escreva apenas conceitos "
                "amplamente estabelecidos, sem afirmações específicas com anchors."
            )

        # FASE 5: Observação (skipped in web-first mode)
        if not config.is_corpus_first:
            print(f"\n  🔬 FASE 5 — Observação (pulada)...")
            log.append("\n── FASE 5: OBSERVAÇÃO (pulada) ──")

        # Imagens
        print(f"\n  🖼️  Buscando imagens...")
        imagens = []
        if tavily_enabled:
            imagens = search_images(queries_img, max_results=3)
        else:
            print("  ⏭️ Tavily image search disabled by user. Skipping.")
        img_txt = ""
        for i, img in enumerate(imagens, 1):
            url_img = img.get("url_imagem", "")
            desc = img.get("descricao", "") or "(sem descrição)"
            origem = img.get("titulo_pagina", img.get("url_origem", ""))
            img_txt += f"  [{i}] {url_img}\n       Desc imagem: {desc}\n       Fonte imagem: {origem}\n"

        referencia_completa = corpus_prompt
        if img_txt:
            referencia_completa += f"\n\nIMAGENS DISPONÍVEIS:\n{img_txt}"

        # FASE 6: Anchored Draft
        print(f"\n  ✍️  FASE 6 — Rascunho anchored...")
        log.append("\n── FASE 6: RASCUNHO ──")
        rascunho, urls_usadas_fase6 = _fase_rascunho(
            theme, titulo, cont_esp, recursos, referencia_completa, urls_secao,
            cumulative_summary, pos, n_total, titulos_todos, len(extraidos),
            prompt_dir=prompt_dir,
            language=config.language,
            min_sources=config.min_sources_per_section,
        )
        # Track source map
        fonte_map_secao = {}
        for i, fonte in enumerate(urls_usadas_fase6, 1):
            if hasattr(fonte, 'id') and hasattr(fonte, 'url'):
                fonte_map_secao[fonte.id] = fonte.url
            elif isinstance(fonte, dict):
                fonte_map_secao[fonte.get('id', i)] = fonte.get('url', '')
            else:
                fonte_map_secao[i] = str(fonte)

        # ── Source diversity check ─────────────────────────────────────
        min_src = config.min_sources_per_section
        n_distinct = len(set(fonte_map_secao.values()))
        if min_src > 0 and n_distinct < min_src:
            print(f"     ⚠️  Apenas {n_distinct} fontes distintas (mínimo: {min_src}). Tentando novamente...")
            log.append(f"⚠️  Retry: {n_distinct}/{min_src} fontes distintas")
            diversity_hint = (
                f"\n\n{'━'*60}\n"
                f"INSTRUÇÃO OBRIGATÓRIA: Use pelo menos {min_src} fontes DISTINTAS nesta seção.\n"
                f"Distribua as citações entre diferentes documentos do corpus.\n"
                f"NÃO dependa de apenas 1-2 artigos.\n"
                f"{'━'*60}\n"
            )
            rascunho_retry, urls_retry = _fase_rascunho(
                theme, titulo, cont_esp, recursos,
                referencia_completa + diversity_hint, urls_secao,
                cumulative_summary, pos, n_total, titulos_todos, len(extraidos),
                prompt_dir=prompt_dir,
                language=config.language,
                min_sources=config.min_sources_per_section,
            )
            fonte_map_retry = {}
            for i, fonte in enumerate(urls_retry, 1):
                if hasattr(fonte, 'id') and hasattr(fonte, 'url'):
                    fonte_map_retry[fonte.id] = fonte.url
                elif isinstance(fonte, dict):
                    fonte_map_retry[fonte.get('id', i)] = fonte.get('url', '')
                else:
                    fonte_map_retry[i] = str(fonte)
            n_distinct_retry = len(set(fonte_map_retry.values()))
            if n_distinct_retry > n_distinct:
                print(f"     ✅ Retry melhorou: {n_distinct_retry} fontes distintas")
                rascunho = rascunho_retry
                urls_usadas_fase6 = urls_retry
                fonte_map_secao = fonte_map_retry
                n_distinct = n_distinct_retry
            else:
                print(f"     ℹ️  Retry não melhorou ({n_distinct_retry} fontes). Mantendo original.")
            if n_distinct < min_src:
                log.append(f"<!-- WARNING: apenas {n_distinct} fontes distintas usadas (mín: {min_src}) -->")

        n_anchors = len(_ANCHORS_PATTERN.findall(rascunho))
        log.append(f"Rascunho: {len(rascunho):,} chars | {n_anchors} anchors (hints)")
        print(f"     {len(rascunho):,} chars | {n_anchors} anchors")

        # FASE 7: Adaptive verification with REACT loop
        print(f"\n  🔍 FASE 7 — Verificação adaptativa...")
        log.append("\n── FASE 7: VERIFICAÇÃO ADAPTATIVA (REACT) ──")
        texto_final, relatorio_verif, stats = _verificar_e_corrigir_secao_com_anchor(
            rascunho,
            corpus,
            fonte_map_secao,
            titulo,
            cont_esp,
            prompt_dir=prompt_dir,
            language=config.language,
        )

        log.append(relatorio_verif)
        verification_stats.append({"secao": titulo, **stats})

        if not texto_final.strip().startswith("## "):
            texto_final = f"## {titulo}\n\n{texto_final.strip()}"

        verificaveis = stats.get("verificaveis", 0)
        if verificaveis == 0:
            verificaveis = stats.get("total", 1)

        verificados = stats.get("aprovados", 0) + stats.get("ajustados", 0)
        taxa = (verificados / verificaveis * 100) if verificaveis > 0 else 100

        num_corrigidos = stats.get("corrigidos", 0)
        if stats["total"] > 0 and (taxa < 40 or num_corrigidos > stats["total"] * 0.3):
            aviso = (
                f"> ⚠️ **Confiabilidade: {taxa:.0f}%** "
                f"({verificados}/{verificaveis} verificados). "
                f"Revisão manual pode ser necessária.\n\n"
            )
            texto_final = re.sub(r"(## .+?\n)", r"\1\n" + aviso, texto_final, count=1)
        elif taxa < 60 and stats["total"] > 0:
            aviso = (
                f"> ℹ️ **Verificação**: {taxa:.0f}% dos parágrafos verificados.\n\n"
            )
            texto_final = re.sub(
                r"(## .+?\n)", r"\1\n" + aviso, texto_final, count=1, flags=re.DOTALL
            )

        # Add per-section references
        print(f"\n  📚 Adicionando referências da seção...")

        citacoes_encontradas = set()
        for match in re.finditer(r'\[(\d+)\]', texto_final):
            num = int(match.group(1))
            citacoes_encontradas.add(num)

        todas_urls_corpus = corpus._used_urls if hasattr(corpus, '_used_urls') else urls_secao

        referencias_secao = []
        urls_faltantes = []

        for idx in sorted(citacoes_encontradas):
            # Priority 1: use the fonte_map built from _fase_rascunho (id → url)
            if idx in fonte_map_secao:
                url = fonte_map_secao[idx]
                referencias_secao.append(f"[{idx}] {url}")
            # Priority 2: fall back to ordered URL list from corpus
            elif 1 <= idx <= len(todas_urls_corpus):
                url = todas_urls_corpus[idx - 1]
                referencias_secao.append(f"[{idx}] {url}")
            else:
                urls_faltantes.append(idx)

        if referencias_secao:
            texto_final += "\n\n### Referências desta seção\n\n"
            texto_final += "\n".join(referencias_secao)
            print(f"     ✅ {len(referencias_secao)} referências adicionadas")
            if urls_faltantes:
                print(f"     ⚠️  Citações sem URL correspondente: {urls_faltantes}")
        else:
            print(f"     ⚠️  Nenhuma citação encontrada nesta seção")

        print(f"  ✅ [{pos+1}/{n_total}] Seção concluída ({taxa:.0f}% verificado)")

        written_sections.append({
            "index": idx_num,
            "title": titulo,
            "text": texto_final,
            "urls_usadas": urls_secao,
            "fonte_map": fonte_map_secao,
            "imagens": imagens,
        })

        for u in urls_secao:
            if u not in all_refs_urls:
                all_refs_urls.append(u)

        for img in imagens:
            if img not in all_refs_images:
                all_refs_images.append(img)

        resumo_sec = summarize_section(titulo, texto_final)
        if cumulative_summary:
            cumulative_summary += f"\n\n[Seção {pos+1}: {titulo}] {resumo_sec}"
        else:
            cumulative_summary = f"[Seção {pos+1}: {titulo}] {resumo_sec}"
        if len(cumulative_summary) > CTX_ABSTRACT_CHARS * 3:
            partes = cumulative_summary.split("\n\n[Seção ")
            cumulative_summary = "\n\n[Seção ".join([""] + partes[-4:]).strip()

        react_log.extend(log)

        if pos < n_total - 1:
            print(f"\n  ⏳ Aguardando {DELAY_BETWEEN_SECTIONS}s...")
            time.sleep(DELAY_BETWEEN_SECTIONS)

    return {
        "written_sections": written_sections,
        "refs_urls": all_refs_urls,
        "refs_images": all_refs_images,
        "cumulative_summary": cumulative_summary,
        "react_log": react_log,
        "verification_stats": verification_stats,
        "status": "secoes_escritas",
    }


