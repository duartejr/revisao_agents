"""
Technical writing nodes - LangGraph nodes for technical chapter authoring.

Full implementation:
- Plan parsing and section extraction
- Section authoring with search, extraction and verification
- Document consolidation with introduction and conclusion
"""

import re
import os
import time
import logging
from datetime import datetime
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

from ..state import EscritaTecnicaState
from ..config import (
    llm_call, parse_json_safe,
    TECNICO_MAX_RESULTS, MAX_CORPUS_PROMPT, EXTRACT_MIN_CHARS,
    MAX_URLS_EXTRACT, CTX_RESUMO_CHARS, SECAO_MIN_PARAGRAFOS,
    DELAY_ENTRE_SECOES, MAX_REACT_ITERATIONS, TOP_K_OBSERVACAO,
)
from ..core.schemas.techinical_writing import RespostaSecao, Fonte
from ..utils.mongodb_corpus import CorpusMongoDB
from ..utils.helpers import resumir_secao, parse_plano_tecnico, parse_plano_academico
from ..core.schemas.writer_config import WriterConfig
from ..utils.tavily_client import search_web, search_images, extract_urls, score_url
from ..utils.prompt_loader import load_prompt
from ..utils.crossref_bibtex import get_reference_data_react, bibtex_to_abnt

# Anchor pattern (kept local — not a simple scalar constant)
_ANCORA_PATTERN = re.compile(r'\[ÂNCORA:\s*"((?:[^"\\]|\\.)*)"\]', re.DOTALL)

# Patterns for LLM-generated justification/meta blocks that must be stripped from output
_JUSTIFICATION_BLOCK_RE = re.compile(
    r'(?:^|\n{0,2})(\*{0,2}(?:Justificativa|Correções\s+aplicadas|Correção\s+aplicada|'
    r'Raciocínio|Correção|Justification|Applied\s+corrections|Reasoning)\s*[:\：\*]\*{0,2}'
    r'[\s\S]*)',
    re.IGNORECASE,
)

# Sentence-level meta-organizational patterns to remove from generated paragraphs
_META_SENTENCE_RE = re.compile(
    r'(?:^|(?<=\n))'
    r'(?:O objetivo d[eo](?: estud[oa]| capítulo| se[çc][ãa]o| revis[ãa]o)?[^.]*?[.!]\s*'
    r'|The objective of(?:this| the)[^.]*?[.!]\s*'
    r'|This (?:section|chapter|review|study) (?:aims|seeks|intends|presents|provides|explores)[^.]*?[.!]\s*'
    r'|Esta (?:se[çc][ãa]o|revis[ãa]o|an[áa]lise|pesquisa) (?:busca|visa|objetiva|apresenta|explora|aborda)[^.]*?[.!]\s*'
    r'|Nesta se[çc][ãa]o[^.]*?(?:apresentamos|discutimos|analisamos|exploraremos|abordaremos)[^.]*?[.!]\s*'
    r')',
    re.IGNORECASE | re.MULTILINE,
)


def _strip_justification_blocks(text: str) -> str:
    """Remove LLM-generated justification/reasoning blocks from verified paragraph text."""
    m = _JUSTIFICATION_BLOCK_RE.search(text)
    if m:
        text = text[:m.start()].rstrip()
    return text


def _strip_meta_sentences(text: str) -> str:
    """Remove meta-organizational opening sentences from a paragraph."""
    return _META_SENTENCE_RE.sub("", text).strip()


# Pattern to detect references to figures/tables/equations not present in the essay
_FIGURE_TABLE_RE = re.compile(
    r'(?:^|(?<=\.\s)|(?<=\n))'                       # sentence boundary
    r'[^.]*?'                                         # leading part of sentence
    r'(?:'
    r'[Ff]igura\s+\d+|[Ff]ig(?:ure)?\.?\s*\d+'
    r'|[Tt]abela\s+\d+|[Tt]able\s+\d+'
    r'|[Ee]qua[çc][ãa]o\s+\d+|[Ee]quation\s+\d+'
    r'|[Qq]uadro\s+\d+|[Gg]r[áa]fico\s+\d+'
    r')'
    r'[^.]*\.\s*',                                    # rest of sentence
    re.MULTILINE,
)


def _strip_figure_table_refs(text: str) -> str:
    """Remove sentences referencing figures/tables/equations not present in the essay."""
    cleaned = _FIGURE_TABLE_RE.sub("", text)
    # Normalize extra blank lines left by removals
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


# ============================================================================
# INTERNAL PHASE FUNCTIONS
# ============================================================================

def _fase_pensamento(tema: str, titulo: str, cont_esp: str, recursos: str, prompt_dir: str = "technical_writing", language: str = "pt") -> dict:
    """Phase 1: search planning — uses prompts/{prompt_dir}/fase_pensamento.yaml."""
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


def _fase_observacao(informacoes_necessarias: List[str], corpus: CorpusMongoDB, prompt_dir: str = "technical_writing", language: str = "pt") -> dict:
    """Phase 5: check if corpus is sufficient — uses prompts/{prompt_dir}/fase_observacao.yaml."""
    if corpus._n_docs == 0:
        return {
            "suficiente": False,
            "lacunas": informacoes_necessarias,
            "query_complementar": informacoes_necessarias[0] if informacoes_necessarias else "",
            "resumo": "Corpus vazio.",
        }

    query_obs = " ".join(informacoes_necessarias[:3])
    chunks_obs = corpus.query(query_obs, top_k=TOP_K_OBSERVACAO)
    amostra_corpus = "\n\n".join(c.texto for c in chunks_obs)[:4000]

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


def _fase_rascunho(
    tema: str, titulo: str, cont_esp: str, recursos: str,
    corpus: str, urls_secao: List[str], resumo_acumulado: str,
    pos: int, n_total: int, titulos_todos: List[str], n_extraidos: int,
    prompt_dir: str = "technical_writing",
    language: str = "pt",
    min_sources: int = 0,
) -> tuple:
    """Phase 6: generate anchored draft — uses prompts/{prompt_dir}/fase_rascunho.yaml."""
    ctx_anteriores = ""
    if resumo_acumulado.strip():
        ctx_anteriores = (
            "══ SEÇÕES JÁ ESCRITAS (não repita estes conceitos) ══\n"
            f"{resumo_acumulado[:CTX_RESUMO_CHARS]}\n"
            "══════════════════════════════════════════════════════\n\n"
        )

    todas_txt = "\n".join(
        f"  {'→ ' if i == pos else '  '}{i+1}. {t}"
        for i, t in enumerate(titulos_todos)
    )

    instru = load_prompt(
        f"{prompt_dir}/fase_rascunho",
        secao_min_paragrafos=SECAO_MIN_PARAGRAFOS,
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
    resultado: RespostaSecao = llm_call(prompt, temperature=instru.temperature, response_schema=RespostaSecao)
    rascunho = resultado.rascunho
    fontes_usadas = resultado.fontes_usadas
    return rascunho, fontes_usadas


def _extrair_com_fallback(
    resultados: List[dict],
    queries_fallback: List[str],
    urls_tentadas: set,
    corpus: "CorpusMongoDB",
) -> tuple:
    """
    Extracts full text from priority URLs with fallback.
    URLs already in MongoDB are skipped.
    Returns (extraidos_validos, resultados_enriquecidos, urls_tentadas)
    """
    # Score and sort URLs
    scored = sorted(
        [(r.get("url", ""), score_url(r.get("url", ""), r.get("snippet", ""), float(r.get("score", 0))))
         for r in resultados if r.get("url")],
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
    # Check Tavily flag
    tavily_enabled = getattr(corpus, 'tavily_enabled', True)
    if tavily_enabled:
        raw = extract_urls(urls_para_extrair)
        for item in raw:
            url = item.get("url", "")
            c = item.get("conteudo", "")
            if len(c) >= EXTRACT_MIN_CHARS:
                validos.append(item)
                print(f"      ✅ {url[:]} ({len(c):,} chars)")
            else:
                falhos.append(url)
                print(f"      ✖  {url[:]} (<{EXTRACT_MIN_CHARS} chars)")
    else:
        # If Tavily disabled, skip extraction
        print("      ⏭️ Tavily search/extract disabled by user.")
        falhos.extend(urls_para_extrair)

    # Fallback search on failure
    if len(falhos) > len(validos) and queries_fallback and tavily_enabled:
        print(f"      🔄 {len(falhos)} falha(s) → buscando alternativas...")
        for q in queries_fallback[:2]:
            res_alt = search_web(f"{q} filetype:pdf", max_results=6)
            for r in res_alt:
                u = r.get("url", "")
                if u and u not in urls_tentadas and not corpus.url_exists(u):
                    urls_tentadas.add(u)
                    for item in extract_urls([u]):
                        if len(item.get("conteudo", "")) >= EXTRACT_MIN_CHARS:
                            validos.append(item)
                            resultados.extend(res_alt)
                            print(f"      ✅ Fallback: {item.get('url', '')[:72]}")
            if validos:
                break
    elif len(falhos) > len(validos) and queries_fallback and not tavily_enabled:
        print("      ⏭️ Tavily fallback search disabled by user.")

    return validos, resultados, urls_tentadas


# ============================================================================
# ANCHOR HELPER FUNCTIONS
# ============================================================================

def _extrair_ancora_principal(bloco: str) -> Optional[str]:
    """Extracts the most relevant (longest) anchor from a text block."""
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
    """Finds the citation number [N] closest to a specific anchor."""
    ancora_escaped = re.escape(ancora)
    pattern = re.compile(
        rf'\[ÂNCORA:\s*"{ancora_escaped}"\]\s*\[(\d+)\]',
        re.IGNORECASE
    )
    match = pattern.search(texto)
    if match:
        return int(match.group(1))
    # Fallback: find citation near anchor (up to 50 chars after)
    ancora_pos = texto.find(ancora)
    if ancora_pos >= 0:
        trecho_posterior = texto[ancora_pos:ancora_pos + 50]
        cit_match = re.compile(r'\[(\d+)\]').search(trecho_posterior)
        if cit_match:
            return int(cit_match.group(1))
    return None


def _extrair_todas_ancoras_com_citacoes(bloco: str) -> List[Tuple[str, Optional[int]]]:
    """Extracts all anchors from the block along with their citations."""
    resultados = []
    pattern = re.compile(
        r'\[ÂNCORA:\s*"((?:[^"\\]|\\.)*)"\]\s*\[(\d+)\]',
        re.DOTALL
    )
    for match in pattern.finditer(bloco):
        texto_ancora = match.group(1).strip()
        citacao = int(match.group(2))
        if len(texto_ancora) >= 10:
            resultados.append((texto_ancora, citacao))
    return resultados


# ============================================================================
# ADAPTIVE VERIFICATION SYSTEM
# ============================================================================

def _contar_claims_verificaveis(paragrafo: str) -> int:
    """Counts claims that MUST be verified (not general knowledge)."""
    p = paragrafo.strip()
    if len(p) < 80:
        return 0
    if p.startswith("#"):
        return 0
    if re.match(r"^\s*[-*]\s", p):
        return 0
    if p.startswith("```"):
        return 0
    if p.startswith("$$") or re.match(r"^\s*\$[^$]+\$", p):
        return 0
    if p.startswith("*Figura") or p.startswith("!["):
        return 0

    num_claims = 0
    num_claims += len(re.findall(r'\b\d+[\d.,]*\b', p))
    num_claims += len(re.findall(r'\b[A-Z][a-z]+\s+(?:et\s+al|[A-Z][a-z]+|\(\d{4}\))', p))
    num_claims += len(re.findall(r'\b(19|20)\d{2}\b|\bv\d+\.\d+', p))
    num_claims += len(re.findall(r'[\+\-\*=/<>]', p))

    assertivas = ["foi", "é", "são", "demonstra", "prova", "mostra", "evidencia",
                  "encontrou", "observou", "descobriu", "propôs", "definiu"]
    for ass in assertivas:
        num_claims += len(re.findall(rf'\b{ass}\b', p, re.IGNORECASE))

    return min(num_claims, 5)


def _juiz_paragrafo_melhorado(
    paragrafo_limpo: str,
    fontes: str,
    titulo_secao: str,
    prompt_dir: str = "technical_writing",
    language: str = "pt",
) -> tuple:
    """
    3-level judge + anchor detection.
    Returns: (texto_final, nivel, log_entry, eh_verificavel)
    """
    ancoras = _ANCORA_PATTERN.findall(paragrafo_limpo)
    tem_ancoras = len([a for a in ancoras if len(a.strip()) > 20]) > 0
    num_claims = _contar_claims_verificaveis(paragrafo_limpo)

    if num_claims == 0 and not tem_ancoras:
        log_entry = f"⏭️  ESTRUTURAL  | {paragrafo_limpo[:70].replace(chr(10), ' ')}..."
        return paragrafo_limpo, "ESTRUTURAL", log_entry, False

    if tem_ancoras and len(paragrafo_limpo) < 100:
        log_entry = f"✅ APROVADO  | {paragrafo_limpo[:70].replace(chr(10), ' ')}..."
        return paragrafo_limpo, "APROVADO", log_entry, True

    if num_claims == 0:
        log_entry = f"✅ CONC.GERAL  | {paragrafo_limpo[:70].replace(chr(10), ' ')}..."
        return paragrafo_limpo, "APROVADO", log_entry, True

    p = load_prompt(
        f"{prompt_dir}/writer_judge",
        paragrafo_limpo=paragrafo_limpo,
        titulo_secao=titulo_secao,
        fontes=fontes,
        language=language,
    )
    resp = llm_call(p.text, temperature=0.1)

    texto_final = paragrafo_limpo
    nivel = "APROVADO"  # default

    m_dec = re.search(r"DECIS[ÃA]O\s*:\s*(APROVADO|AJUSTADO|CORRIGIDO)", resp, re.IGNORECASE)
    if m_dec:
        nivel = m_dec.group(1).upper()

    m_txt = re.search(r"TEXTO\s*:\s*([\s\S]+)", resp, re.IGNORECASE)
    if m_txt:
        candidato = m_txt.group(1).strip()
        candidato = re.sub(r"^DECIS[ÃA]O\s*:.*\n?", "", candidato, flags=re.IGNORECASE).strip()
        # Strip any justification/reasoning block the LLM appended after the paragraph
        candidato = _strip_justification_blocks(candidato)
        # Remove meta-organizational sentences
        candidato = _strip_meta_sentences(candidato).strip()
        # Remove references to figures/tables/equations not present in the essay
        candidato = _strip_figure_table_refs(candidato)
        if candidato and len(candidato) > 20:
            texto_final = candidato

    trecho = paragrafo_limpo[:70].replace('\n', ' ')
    if nivel == "APROVADO":
        log_entry = f"✅ APROVADO  | {trecho}..."
    elif nivel == "AJUSTADO":
        corr = texto_final[:70].replace('\n', ' ')
        log_entry = f"🔵 AJUSTADO  | {trecho}...\n     → {corr}..."
    else:
        corr = texto_final[:70].replace('\n', ' ')
        log_entry = f"🔧 CORRIGIDO | {trecho}...\n     → {corr}..."

    return texto_final, nivel, log_entry, True


def _monitorar_taxa_verificacao(stats: dict, titulo_secao: str) -> tuple:
    """
    Monitors if verification rate is acceptable.
    Returns: (precisa_buscar_mais, motivo)
    """
    total = stats.get("total", 0)
    if total == 0:
        return False, "Nenhum parágrafo verificado"
    verificaveis = stats.get("verificaveis", 0)
    if verificaveis == 0:
        return False, "Sem parágrafos verificáveis"
    verificados = stats.get("aprovados", 0) + stats.get("ajustados", 0)
    taxa = (verificados / verificaveis * 100) if verificaveis > 0 else 100
    if taxa < 40:
        return True, f"Taxa crítica {taxa:.0f}%"
    elif taxa < 60:
        return True, f"Taxa baixa {taxa:.0f}%"
    return False, f"Taxa OK {taxa:.0f}%"


def _buscar_conteudo_complementar(
    titulo_secao: str,
    conteudo_esperado: str,
    corpus_atual: CorpusMongoDB,
    urls_tentadas: set,
    prompt_dir: str = "technical_writing",
    language: str = "pt",
) -> tuple:
    """Searches for complementary content when many paragraphs fail."""
    print(f"\n      🔄 BUSCA COMPLEMENTAR — {titulo_secao}")

    queries_complementares = []
    try:
        p = load_prompt(
            f"{prompt_dir}/busca_complementar",
            titulo_secao=titulo_secao,
            conteudo_esperado=conteudo_esperado[:100],
            language=language,
        )
        resp = llm_call(p.text, temperature=p.temperature)
        queries_complementares = [q.strip() for q in resp.split('\n') if q.strip()][:2]
    except Exception as e:
        print(f"      ⚠️  Erro: {e}")
        queries_complementares = [f"{titulo_secao} tutorial", f"{titulo_secao} técnico"]

    num_novos = 0
    extraidos_novos = []

    tavily_enabled = getattr(corpus_atual, 'tavily_enabled', True)
    if not tavily_enabled:
        print("      ⏭️ Tavily complementary search disabled by user.")
        return 0, corpus_atual, "Nenhum novo conteúdo"

    for q in queries_complementares:
        print(f"      • {q[:70]}")
        try:
            res = search_web(q, max_results=8)
            urls_para_extrair = []
            for r in res:
                u = r.get("url", "")
                if u and u not in urls_tentadas and not corpus_atual.url_exists(u):
                    urls_para_extrair.append(u)
                    urls_tentadas.add(u)
                    if len(urls_para_extrair) >= 4:
                        break
            if urls_para_extrair:
                raw = extract_urls(urls_para_extrair)
                for item in raw:
                    if len(item.get("conteudo", "")) >= EXTRACT_MIN_CHARS:
                        extraidos_novos.append(item)
                        num_novos += 1
            time.sleep(1)
        except Exception as e:
            print(f"      ⚠️  Erro '{q[:50]}': {e}")

    if not extraidos_novos:
        return 0, corpus_atual, "Nenhum novo conteúdo"

    corpus_novo = CorpusMongoDB().build(extraidos_novos, [])
    if corpus_novo._n_docs > 0:
        corpus_atual._urls_usadas.extend(corpus_novo._urls_usadas)
        corpus_atual._total_chunks += corpus_novo._total_chunks
        print(f"      ✅ +{num_novos} chunks indexados")

    return num_novos, corpus_atual, f"+{num_novos} chunks"


def _verificar_e_corrigir_secao_adaptativa(
    texto_secao: str,
    corpus: CorpusMongoDB,
    corpus_prompt_completo: str,
    titulo: str,
    fontes_secao: List[Fonte],
    conteudo_esperado: str = "",
    language: str = "pt",
) -> tuple:
    """Verification with adaptive loop."""
    from ..helpers.ancora_helpers import extrair_ancora_principal, limpar_ancoras, extrair_citacao_ancora, extrair_ancoras_com_citacoes

    urls_tentadas = set()
    iteracao = 0

    while iteracao < 3:
        iteracao += 1
        print(f"\n     └─ Verificação iter {iteracao}/3")

        blocos = re.split(r'\n{2,}', texto_secao.strip())
        resultado = []
        log_linhas = [f"\n### Verificação Adaptativa — {titulo} (iter {iteracao})"]

        stats = {
            "total": 0,
            "aprovados": 0,
            "ajustados": 0,
            "corrigidos": 0,
            "estruturais": 0,
            "verificaveis": 0,
            "pulados": 0,
        }

        for i, bloco in enumerate(blocos):
            bloco = bloco.strip()
            if not bloco:
                continue

            bloco_limpo = re.sub(r'\[ÂNCORA:\s*"[^"]*"\]', "", bloco).strip()
            bloco_limpo = re.sub(r'  +', ' ', bloco_limpo)

            if bloco_limpo.startswith("#") or len(bloco_limpo) < 60:
                resultado.append(bloco_limpo)
                stats["pulados"] += 1
                continue

            stats["total"] += 1

            fontes = corpus.render_prompt(bloco_limpo, max_chars=3000)[0]

            if not fontes.strip():
                resultado.append(bloco_limpo)
                stats["aprovados"] += 1
                stats["estruturais"] += 1
                log_linhas.append(f"  par.{i+1}: ⏭️  SEM FONTES")
                continue

            texto_final, nivel, log_entry, eh_verificavel = _juiz_paragrafo_melhorado(
                bloco_limpo, fontes, titulo
            )

            resultado.append(texto_final)
            log_linhas.append(f"  par.{i+1}: {log_entry}")

            if eh_verificavel:
                stats["verificaveis"] += 1

            if "APROVADO" in nivel or "ESTRUTURAL" in nivel:
                stats["aprovados"] += 1
            elif "AJUSTADO" in nivel:
                stats["ajustados"] += 1
            else:
                stats["corrigidos"] += 1

        precisa_mais, motivo = _monitorar_taxa_verificacao(stats, titulo)
        verificados = stats["aprovados"] + stats["ajustados"]
        taxa = (verificados / stats["verificaveis"] * 100) if stats["verificaveis"] > 0 else 100

        log_linhas.append(f"\n**Resultado:** {verificados}/{stats['verificaveis']} ({taxa:.0f}%) — {motivo}")
        print(f"     📊 {taxa:.0f}% | {motivo}")

        if not precisa_mais or iteracao >= 3:
            break

        num_novos, corpus, msg = _buscar_conteudo_complementar(
            titulo, conteudo_esperado, corpus, urls_tentadas, language=language
        )
        log_linhas.append(f"\n**Busca:** {msg}")
        if num_novos == 0:
            break

    texto_corrigido = "\n\n".join(p for p in resultado if p)
    texto_corrigido = re.sub(r'\[ÂNCORA:\s*"[^"]*"\]', "", texto_corrigido)
    texto_corrigido = re.sub(r'\n{3,}', '\n\n', texto_corrigido)

    verificados = stats["aprovados"] + stats["ajustados"]
    taxa_final = (verificados / stats["verificaveis"] * 100) if stats["verificaveis"] > 0 else 100
    print(f"\n     📊 FINAL: {verificados}/{stats['verificaveis']} ({taxa_final:.0f}%)")

    relatorio = "\n".join(log_linhas)
    return texto_corrigido, relatorio, stats


def _verificar_paragrafo_com_ancora(
    bloco: str,
    corpus: CorpusMongoDB,
    fonte_map: dict,
    titulo_secao: str,
    prompt_dir: str = "technical_writing",
    language: str = "pt",
) -> Tuple[str, str, str, bool]:
    """
    Verifies a paragraph using anchors for directed search.
    Returns: (texto_final, nivel, log_entry, eh_verificavel)
    """
    bloco_limpo = re.sub(r'\[ÂNCORA:\s*"[^"]*"\]', "", bloco).strip()
    bloco_limpo = re.sub(r'  +', ' ', bloco_limpo)

    if bloco_limpo.startswith("#") or len(bloco_limpo) < 60:
        return bloco_limpo, "ESTRUTURAL", "⏭️  ESTRUTURAL", False

    ancora_principal = _extrair_ancora_principal(bloco)

    if ancora_principal:
        num_citacao = _extrair_citacao_ancora(bloco, ancora_principal)

        if num_citacao and num_citacao in fonte_map:
            url_citada = fonte_map[num_citacao]
            print(f"     🎯 Âncora encontrada ({len(ancora_principal)} chars) → [{num_citacao}]")
            print(f"        URL: {url_citada[:60]}")

            fontes, urls_usadas, n_chunks = corpus.render_prompt_url(
                texto_ancora=ancora_principal,
                url_citada=url_citada,
                max_chars=3000,
                top_k=5,
                include_neighbors=True,
                neighbor_window=2,
            )

            if fontes:
                print(f"        ✅ {n_chunks} chunks da URL citada")
            else:
                print(f"        ⚠️  Nenhum chunk encontrado, usando busca geral")
                fontes = corpus.render_prompt(bloco_limpo[:300], max_chars=3000)[0]
        else:
            print(f"     ⚠️  Âncora sem citação válida")
            fontes = corpus.render_prompt(bloco_limpo[:300], max_chars=3000)[0]
    else:
        ancoras_com_cit = _extrair_todas_ancoras_com_citacoes(bloco)

        if ancoras_com_cit:
            ancoras_com_urls = []
            for ancora_txt, num_cit in ancoras_com_cit:
                if num_cit in fonte_map:
                    ancoras_com_urls.append((ancora_txt, fonte_map[num_cit]))

            if ancoras_com_urls:
                print(f"     🎯 {len(ancoras_com_urls)} âncoras com URLs")
                fontes, urls_usadas, n_chunks = corpus.render_prompt_ancoras(
                    ancoras_com_urls=ancoras_com_urls,
                    max_chars=3000,
                )
                if fontes:
                    print(f"        ✅ {n_chunks} chunks das URLs citadas")
                else:
                    fontes = corpus.render_prompt(bloco_limpo[:300], max_chars=3000)[0]
            else:
                fontes = corpus.render_prompt(bloco_limpo[:300], max_chars=3000)[0]
        else:
            fontes = corpus.render_prompt(bloco_limpo[:300], max_chars=3000)[0]

    if not fontes.strip():
        return bloco_limpo, "APROVADO", "✅ SEM FONTES", True

    texto_final, nivel, log_entry, eh_verificavel = _juiz_paragrafo_melhorado(
        bloco_limpo, fontes, titulo_secao, prompt_dir=prompt_dir, language=language
    )
    return texto_final, nivel, log_entry, eh_verificavel


def _verificar_e_corrigir_secao_com_ancora(
    texto_secao: str,
    corpus: CorpusMongoDB,
    fonte_map: dict,
    titulo: str,
    conteudo_esperado: str = "",
    prompt_dir: str = "technical_writing",
    language: str = "pt",
) -> Tuple[str, str, dict]:
    """
    Adaptive verification using anchors for directed search.
    Replaces: _verificar_e_corrigir_secao_adaptativa
    """
    urls_tentadas = set()
    iteracao = 0

    while iteracao < 3:
        iteracao += 1
        print(f"\n     └─ Verificação iter {iteracao}/3 (com âncoras)")

        blocos = re.split(r'\n{2,}', texto_secao.strip())
        resultado = []
        log_linhas = [f"\n### Verificação com Âncoras — {titulo} (iter {iteracao})"]

        stats = {
            "total": 0,
            "aprovados": 0,
            "ajustados": 0,
            "corrigidos": 0,
            "estruturais": 0,
            "verificaveis": 0,
            "pulados": 0,
            "ancoras_usadas": 0,
        }

        for i, bloco in enumerate(blocos):
            bloco = bloco.strip()
            if not bloco:
                continue

            tem_ancoras = bool(re.search(r'\[ÂNCORA:', bloco))
            if tem_ancoras:
                stats["ancoras_usadas"] += 1

            texto_final, nivel, log_entry, eh_verificavel = _verificar_paragrafo_com_ancora(
                bloco=bloco,
                corpus=corpus,
                fonte_map=fonte_map,
                titulo_secao=titulo,
                prompt_dir=prompt_dir,
                language=language,
            )

            resultado.append(texto_final)
            log_linhas.append(f"  par.{i+1}: {log_entry}")

            stats["total"] += 1
            if eh_verificavel:
                stats["verificaveis"] += 1

            if "APROVADO" in nivel or "ESTRUTURAL" in nivel:
                stats["aprovados"] += 1
            elif "AJUSTADO" in nivel:
                stats["ajustados"] += 1
            else:
                stats["corrigidos"] += 1

        precisa_mais, motivo = _monitorar_taxa_verificacao(stats, titulo)
        verificados = stats["aprovados"] + stats["ajustados"]
        taxa = (verificados / stats["verificaveis"] * 100) if stats["verificaveis"] > 0 else 100

        log_linhas.append(
            f"\n**Resultado:** {verificados}/{stats['verificaveis']} ({taxa:.0f}%) — {motivo}"
        )
        log_linhas.append(f"**Âncoras usadas:** {stats['ancoras_usadas']} parágrafos")
        print(f"     📊 {taxa:.0f}% | {motivo} | {stats['ancoras_usadas']} âncoras")

        if not precisa_mais or iteracao >= 3:
            break

        num_novos, corpus, msg = _buscar_conteudo_complementar(
            titulo, conteudo_esperado, corpus, urls_tentadas,
            prompt_dir=prompt_dir,
            language=language,
        )
        log_linhas.append(f"\n**Busca:** {msg}")
        if num_novos == 0:
            break

    texto_corrigido = "\n\n".join(p for p in resultado if p)
    texto_corrigido = re.sub(r'\[ÂNCORA:\s*"[^"]*"\]', "", texto_corrigido)
    texto_corrigido = re.sub(r'\n{3,}', '\n\n', texto_corrigido)

    verificados = stats["aprovados"] + stats["ajustados"]
    taxa_final = (verificados / stats["verificaveis"] * 100) if stats["verificaveis"] > 0 else 100

    print(f"\n     📊 FINAL: {verificados}/{stats['verificaveis']} ({taxa_final:.0f}%)")
    print(f"     🎯 Âncoras utilizadas: {stats['ancoras_usadas']} parágrafos")

    relatorio = "\n".join(log_linhas)
    return texto_corrigido, relatorio, stats


# ============================================================================
# GRAPH NODES
# ============================================================================

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


def escrever_secoes_node(state: EscritaTecnicaState) -> dict:
    """Main node for writing sections with search, extraction and verification."""
    config = WriterConfig.from_dict(state.get("writer_config", {}))
    prompt_dir = config.prompt_dir
    tema = state["tema"]
    secoes = state["secoes"]
    secoes_escritas = []
    all_refs_urls = list(state.get("refs_urls", []))
    all_refs_imagens = list(state.get("refs_imagens", []))
    resumo_acumulado = state.get("resumo_acumulado", "")
    react_log = list(state.get("react_log", []))
    stats_verificacao = list(state.get("stats_verificacao", []))
    n_total = len(secoes)
    titulos_todos = [s["titulo"] for s in secoes]
    # CorpusMongoDB instance for URL existence checks (no build)
    corpus_check = CorpusMongoDB()

    tavily_enabled = state.get("tavily_enabled", True)
    for pos, secao in enumerate(secoes):
        titulo = secao["titulo"]
        cont_esp = secao.get("conteudo_esperado", "")
        recursos = secao.get("recursos", "")
        idx_num = secao.get("indice", pos)

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
        plano = _fase_pensamento(tema, titulo, cont_esp, recursos, prompt_dir=prompt_dir, language=config.language)
        queries = plano.get("queries_busca", [f"{tema} {titulo}"])
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
                res = search_web(q, TECNICO_MAX_RESULTS)
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
        prefixo = f"s{pos+1:02d}_{slug_secao}"

        if _corpus_suficiente:
            # Reuse the global check corpus — no new documents to index
            corpus = corpus_check
        else:
            corpus = CorpusMongoDB().build(extraidos, resultados, prefixo=prefixo)

        query_retrieval = f"{titulo} {cont_esp} {recursos}"
        corpus_prompt, urls_secao, _ = corpus.render_prompt(
            query_retrieval, max_chars=MAX_CORPUS_PROMPT
        )
        log.append(f"MongoDB: {corpus._n_docs} docs | {corpus._total_chunks} chunks")

        if not corpus_prompt.strip() and not _corpus_suficiente and tavily_enabled:
            print("  ⚠️  Corpus vazio! Busca de último recurso...")
            log.append("⚠️  Corpus vazio — busca de emergência")
            q_emerg = f"{titulo} {tema} technical documentation filetype:pdf"
            res_emerg = search_web(q_emerg, 6)
            corpus_check.tavily_enabled = tavily_enabled
            novos_emerg, _, urls_vistas = _extrair_com_fallback(
                res_emerg, queries_fallback=[titulo], urls_tentadas=urls_vistas,
                corpus=corpus_check,
            )
            if novos_emerg:
                extraidos.extend(novos_emerg)
                corpus = CorpusMongoDB().build(extraidos, resultados, prefixo=prefixo)
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
                "amplamente estabelecidos, sem afirmações específicas com âncoras."
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
        print(f"\n  ✍️  FASE 6 — Rascunho ancorado...")
        log.append("\n── FASE 6: RASCUNHO ──")
        rascunho, urls_usadas_fase6 = _fase_rascunho(
            tema, titulo, cont_esp, recursos, referencia_completa, urls_secao,
            resumo_acumulado, pos, n_total, titulos_todos, len(extraidos),
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
                tema, titulo, cont_esp, recursos,
                referencia_completa + diversity_hint, urls_secao,
                resumo_acumulado, pos, n_total, titulos_todos, len(extraidos),
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

        n_ancoras = len(_ANCORA_PATTERN.findall(rascunho))
        log.append(f"Rascunho: {len(rascunho):,} chars | {n_ancoras} âncoras (hints)")
        print(f"     {len(rascunho):,} chars | {n_ancoras} âncoras")

        # FASE 7: Adaptive verification with REACT loop
        print(f"\n  🔍 FASE 7 — Verificação adaptativa...")
        log.append("\n── FASE 7: VERIFICAÇÃO ADAPTATIVA (REACT) ──")
        texto_final, relatorio_verif, stats = _verificar_e_corrigir_secao_com_ancora(
            rascunho,
            corpus,
            fonte_map_secao,
            titulo,
            cont_esp,
            prompt_dir=prompt_dir,
            language=config.language,
        )

        log.append(relatorio_verif)
        stats_verificacao.append({"secao": titulo, **stats})

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

        todas_urls_corpus = corpus._urls_usadas if hasattr(corpus, '_urls_usadas') else urls_secao

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

        secoes_escritas.append({
            "indice": idx_num,
            "titulo": titulo,
            "texto": texto_final,
            "urls_usadas": urls_secao,
            "fonte_map": fonte_map_secao,
            "imagens": imagens,
        })

        for u in urls_secao:
            if u not in all_refs_urls:
                all_refs_urls.append(u)

        for img in imagens:
            if img not in all_refs_imagens:
                all_refs_imagens.append(img)

        resumo_sec = resumir_secao(titulo, texto_final)
        if resumo_acumulado:
            resumo_acumulado += f"\n\n[Seção {pos+1}: {titulo}] {resumo_sec}"
        else:
            resumo_acumulado = f"[Seção {pos+1}: {titulo}] {resumo_sec}"
        if len(resumo_acumulado) > CTX_RESUMO_CHARS * 3:
            partes = resumo_acumulado.split("\n\n[Seção ")
            resumo_acumulado = "\n\n[Seção ".join([""] + partes[-4:]).strip()

        react_log.extend(log)

        if pos < n_total - 1:
            print(f"\n  ⏳ Aguardando {DELAY_ENTRE_SECOES}s...")
            time.sleep(DELAY_ENTRE_SECOES)

    return {
        "secoes_escritas": secoes_escritas,
        "refs_urls": all_refs_urls,
        "refs_imagens": all_refs_imagens,
        "resumo_acumulado": resumo_acumulado,
        "react_log": react_log,
        "stats_verificacao": stats_verificacao,
        "status": "secoes_escritas",
    }


def consolidar_node(state: EscritaTecnicaState) -> dict:
    """Consolidates written sections into a final document."""
    config = WriterConfig.from_dict(state.get("writer_config", {}))
    tema = state["tema"]
    secoes = sorted(state["secoes_escritas"], key=lambda s: s["indice"])
    all_urls = list(dict.fromkeys(state.get("refs_urls", [])))
    all_imagens = state.get("refs_imagens", [])
    react_log = state.get("react_log", [])
    stats_global = state.get("stats_verificacao", [])
    resumo_final = state.get("resumo_acumulado", "")[:1000]

    print(f"\n📚 Consolidando {len(secoes)} seções...")

    total_par = sum(s.get("total", 0) for s in stats_global)
    total_aprov = sum(s.get("aprovados", 0) for s in stats_global)
    total_ajust = sum(s.get("ajustados", 0) for s in stats_global)
    total_corr = sum(s.get("corrigidos", 0) for s in stats_global)
    total_verif = total_aprov + total_ajust
    taxa_global = (total_verif / total_par * 100) if total_par > 0 else 100

    print(f"   📊 {total_verif}/{total_par} verificados ({taxa_global:.0f}%) "
          f"— ✅{total_aprov} aprovados  🔵{total_ajust} ajustados  "
          f"🔧{total_corr} corrigidos | {len(all_urls)} fontes")

    titulos = [s["titulo"] for s in secoes]
    p_intro = load_prompt(
        f"{config.prompt_dir}/consolidar_intro",
        tema=tema,
        titulos=", ".join(titulos),
        language=config.language,
    )
    resp_intro = llm_call(p_intro.text, temperature=p_intro.temperature)
    p_concl = load_prompt(
        f"{config.prompt_dir}/consolidar_conclusao",
        tema=tema,
        resumo_final=resumo_final,
        language=config.language,
    )
    resp_concl = llm_call(p_concl.text, temperature=p_concl.temperature)

    partes = [
        f"# {tema}\n",
        f"> **Tipo:** {config.review_type_label}\n",
        f"> **Verificação por parágrafo:** {total_verif}/{total_par} verificados "
        f"({taxa_global:.0f}%) — {total_aprov} aprovados, {total_ajust} ajustados, "
        f"{total_corr} corrigidos | "
        f"**Fontes:** {len(all_urls)} | **Seções:** {len(secoes)}\n",
        "\n---\n", "## Sumário\n", "- Introdução",
    ]
    for s in secoes:
        partes.append(f"- {s['titulo']}")
    partes += ["- Conclusão", "\n\n---\n",
               "## Introdução\n", resp_intro.strip(), "\n\n---\n"]

    for s in secoes:
        stats_s = next(
            (x for x in stats_global if x.get("secao") == s["titulo"]), {}
        )
        t_s = stats_s.get("total", 0)
        a_s = stats_s.get("aprovados", 0) + stats_s.get("ajustados", 0)
        r_s = stats_s.get("corrigidos", 0)
        aj_s = stats_s.get("ajustados", 0)
        tx_s = (a_s / t_s * 100) if t_s > 0 else 100
        partes.append(
            f"<!-- Parágrafos: {a_s}/{t_s} verificados ({tx_s:.0f}%) "
            f"| {stats_s.get('aprovados', 0)} aprovados, {aj_s} ajustados, "
            f"{r_s} corrigidos -->\n"
        )
        partes.append(s["texto"].strip())
        partes.append("\n\n---\n")

    partes += ["## Conclusão\n", resp_concl.strip(), "\n\n"]

    # ══════════════════════════════════════════════════════════════════
    # GLOBAL CITATION SYNCHRONIZATION + PER-SECTION REFERENCE REBUILD
    # ══════════════════════════════════════════════════════════════════
    print(f"\n  🔗 Sincronizando citações globais...")

    # 1. Build consolidated fonte_map: {original_citation_number: url}
    #    Merge all per-section fonte_maps; keep the first URL seen per index.
    #    Keys may be int or str depending on serialization — normalize to int.
    fonte_map_consolidado: dict = {}
    for s in secoes:
        s_map = s.get("fonte_map", {})
        for idx, url in s_map.items():
            idx_int = int(idx)
            if idx_int not in fonte_map_consolidado:
                fonte_map_consolidado[idx_int] = url

    # Also add URLs from corpus that might be cited but not in fonte_maps
    for i, url in enumerate(all_urls, 1):
        if i not in fonte_map_consolidado:
            fonte_map_consolidado[i] = url

    documento_raw = "\n".join(partes)

    # 2. Strip old "### Referências desta seção" blocks before renumbering
    documento_clean = re.sub(
        r'\n*### Referências desta seção\s*\n(?:\[?\d+\]?[^\n]*\n?)*',
        '',
        documento_raw,
    )

    # 3. Strip invalid figure/table/equation references
    documento_clean = _strip_figure_table_refs(documento_clean)

    # 4. Extract all [N] from entire document and create global renumbering
    citacoes_originais = re.findall(r'\[(\d+)\]', documento_clean)
    citacoes_unicas = []
    seen = set()
    for c in citacoes_originais:
        n = int(c)
        if n not in seen:
            seen.add(n)
            citacoes_unicas.append(n)

    # old_idx → new_idx (first-appearance order)
    mapa_global: dict = {}
    for new_idx, old_idx in enumerate(citacoes_unicas, 1):
        mapa_global[old_idx] = new_idx

    # Build synchronized global fonte map: {new_idx: url}
    global_fonte_map_sync: dict = {}
    for old_idx, new_idx in mapa_global.items():
        url = fonte_map_consolidado.get(old_idx, "")
        if url:
            global_fonte_map_sync[new_idx] = url

    # 5. Renumber all [N] in the document
    def _renumber(match):
        old = int(match.group(1))
        new = mapa_global.get(old, old)
        return f"[{new}]"

    documento_sync = re.sub(r'\[(\d+)\]', _renumber, documento_clean)
    # Also handle [N, M] compound citations
    def _renumber_compound(match):
        nums = re.findall(r'\d+', match.group(0))
        new_nums = [str(mapa_global.get(int(n), int(n))) for n in nums]
        return "[" + ", ".join(new_nums) + "]"
    documento_sync = re.sub(r'\[\d+(?:\s*,\s*\d+)+\]', _renumber_compound, documento_sync)

    n_global_sources = len(global_fonte_map_sync)
    print(f"     ✅ {n_global_sources} fontes globais | {len(mapa_global)} citações remapeadas")

    # 6. Rebuild per-section "### Referências desta seção" blocks
    #    First, split out the conclusion so it doesn't contaminate the
    #    last section block (the old code skipped any block containing
    #    '## Conclusão', silently dropping the last section's refs).
    _CONCLUSAO_MARKER = "\n## Conclusão"
    if _CONCLUSAO_MARKER in documento_sync:
        _c_idx = documento_sync.index(_CONCLUSAO_MARKER)
        doc_sections_part = documento_sync[:_c_idx]
        doc_conclusao_part = documento_sync[_c_idx:]
    else:
        doc_sections_part = documento_sync
        doc_conclusao_part = ""

    section_pattern = re.compile(r'(?=\n<!-- Parágrafos:)')
    section_blocks = section_pattern.split(doc_sections_part)

    rebuilt_parts = []
    for block in section_blocks:
        # Only process blocks that contain a numbered section heading
        if not re.search(r'## \d', block):
            rebuilt_parts.append(block)
            continue

        # Extract all [N] referenced in block body
        cits_in_block = set()
        for m in re.finditer(r'\[(\d+)\]', block):
            cits_in_block.add(int(m.group(1)))
        # Also handle [N, M]
        for m in re.finditer(r'\[(\d+(?:\s*,\s*\d+)+)\]', block):
            for n in re.findall(r'\d+', m.group(1)):
                cits_in_block.add(int(n))

        if cits_in_block:
            refs_lines = []
            for idx in sorted(cits_in_block):
                url = global_fonte_map_sync.get(idx, "")
                if url:
                    refs_lines.append(f"[{idx}] {url}")
            if refs_lines:
                # Remove trailing --- if present, we'll re-add it
                block_trimmed = block.rstrip()
                if block_trimmed.endswith("---"):
                    block_trimmed = block_trimmed[:-3].rstrip()
                block = (
                    block_trimmed
                    + "\n\n### Referências desta seção\n\n"
                    + "\n".join(refs_lines)
                    + "\n\n\n---\n"
                )
        rebuilt_parts.append(block)

    documento = "".join(rebuilt_parts) + doc_conclusao_part

    # Update all_urls count for header
    all_urls_final = list(global_fonte_map_sync.values())
    # Update the header line with correct source count
    documento = re.sub(
        r'\*\*Fontes:\*\* \d+',
        f'**Fontes:** {len(all_urls_final)}',
        documento,
        count=1,
    )

    print(f"\n  ℹ️  Referências reconstruídas por seção ({n_global_sources} fontes globais)")

    # ══════════════════════════════════════════════════════════════════
    # BUILD UNIFIED ABNT REFERENCES SECTION using REACT agent
    # ══════════════════════════════════════════════════════════════════
    print(f"\n  📚 Construindo seção de Referências em ABNT...")
    
    # Prepare MongoDB corpus and tavily_enabled flag
    tavily_enabled = state.get("tavily_enabled", False)
    try:
        mongo_corpus = CorpusMongoDB()
        mongo_corpus.connect()
    except Exception as e:
        logger.warning(f"MongoDB connection failed for bibliography: {e}")
        mongo_corpus = None
    
    abnt_references = []
    for idx in sorted(global_fonte_map_sync.keys()):
        url = global_fonte_map_sync[idx]
        
        # Use REACT agent to intelligently find bibliographic data
        ref_data = get_reference_data_react(
            file_path=url,
            mongo_corpus=mongo_corpus,
            tavily_enabled=tavily_enabled,
            max_iterations=5,
            timeout=10
        )
        
        # Format as "[N] ABNT_citation"
        if ref_data.get('abnt'):
            abnt_citation = f"[{idx}] {ref_data['abnt']}"
        else:
            # Final fallback if REACT failed completely
            file_name = url.split('/')[-1]
            abnt_citation = f"[{idx}] {file_name}. Disponível em: {url}"
        
        abnt_references.append(abnt_citation)
        
        # Log the source strategy used
        logger.info(f"Ref [{idx}]: {ref_data.get('source', 'unknown')} for {url[:60]}")
    
    # Close MongoDB connection if opened
    if mongo_corpus:
        try:
            mongo_corpus.close()
        except:
            pass
    
    # Append unified references section to document
    if abnt_references:
        documento += "\n\n---\n\n## Referências\n\n"
        documento += "\n\n".join(abnt_references)
        print(f"     ✅ {len(abnt_references)} referências em ABNT adicionadas")
    else:
        print(f"     ⚠️  Nenhuma referência para construir")

    slug = re.sub(r"[^\w\s-]", "", tema[:40]).strip().replace(" ", "_").lower()
    output_path = f"reviews/{config.output_prefix}_{slug}.md"
    log_path = f"reviews/{config.output_prefix}_{slug}.log"

    try:
        os.makedirs("reviews", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(documento)
        print(f"\n💾 {output_path} ({len(documento):,} chars)")
    except Exception as e:
        print(f"⚠️  Erro ao salvar: {e}")

    try:
        cabecalho = [
            "=" * 70, f"REACT AUDIT LOG — {tema}",
            f"Gerado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Seções: {len(secoes)} | Fontes: {len(all_urls)}",
            f"Verificados: {total_verif}/{total_par} ({taxa_global:.0f}%) "
            f"— {total_aprov} aprovados, {total_ajust} ajustados, {total_corr} corrigidos",
            "=" * 70, "\n── STATS POR SEÇÃO ──",
        ]
        for s in stats_global:
            t = s.get("total", 0)
            a = s.get("aprovados", 0) + s.get("ajustados", 0)
            r = s.get("corrigidos", 0)
            aj = s.get("ajustados", 0)
            tx = (a / t * 100) if t > 0 else 100
            cabecalho.append(
                f"  [{a}/{t} = {tx:.0f}% | {s.get('aprovados', 0)} aprov "
                f"{aj} ajust {r} corrig] {s.get('secao', '?')[:55]}"
            )
        os.makedirs("reviews", exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(cabecalho + [""] + react_log))
        print(f"📋 {log_path}")
    except Exception as e:
        print(f"⚠️  Erro ao salvar log: {e}")

    return {"status": "concluido"}
