"""
verification.py — adaptive paragraph verification (REACT judge loop).

Contains
--------
_contar_claims_verificaveis          : count claims that need fact-checking.
_juiz_paragrafo_melhorado            : 3-level LLM judge (APROVADO/AJUSTADO/CORRIGIDO).
_monitorar_taxa_verificacao          : decide if more context is needed.
_buscar_conteudo_complementar        : complementary web search when rate is low.
_verificar_e_corrigir_secao_adaptativa : full adaptive loop (legacy, no anchors).
_verificar_paragrafo_com_ancora      : anchor-directed single-paragraph check.
_verificar_e_corrigir_secao_com_ancora : full anchor-directed adaptive loop.
"""

import re
import time
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ...utils.vector_utils.mongodb_corpus import CorpusMongoDB

from ...config import llm_call, EXTRACT_MIN_CHARS
from ...utils.llm_utils.prompt_loader import load_prompt
from ...utils.search_utils.tavily_client import search_web, extract_urls
from ..writing.text_filters import (
    _ANCORA_PATTERN,
    _strip_justification_blocks,
    _strip_meta_sentences,
    _strip_figure_table_refs,
)
from ..writing.anchor_helpers import (
    _extrair_ancora_principal,
    _extrair_citacao_ancora,
    _extrair_todas_ancoras_com_citacoes,
)


# ---------------------------------------------------------------------------
# Verifiability heuristic
# ---------------------------------------------------------------------------

def _contar_claims_verificaveis(paragrafo: str) -> int:
    """Estimate the number of specific claims in *paragrafo* that must be verified."""
    p = paragrafo.strip()
    if len(p) < 80:
        return 0
    if p.startswith("#") or re.match(r"^\s*[-*]\s", p):
        return 0
    if p.startswith("```") or p.startswith("$$") or re.match(r"^\s*\$[^$]+\$", p):
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


# ---------------------------------------------------------------------------
# Single-paragraph judge
# ---------------------------------------------------------------------------

def _juiz_paragrafo_melhorado(
    paragrafo_limpo: str,
    fontes: str,
    titulo_secao: str,
    prompt_dir: str = "technical_writing",
    language: str = "pt",
) -> Tuple[str, str, str, bool]:
    """3-level judge:  APROVADO / AJUSTADO / CORRIGIDO.

    Returns (texto_final, nivel, log_entry, eh_verificavel).
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
    nivel = "APROVADO"

    m_dec = re.search(r"DECIS[ÃA]O\s*:\s*(APROVADO|AJUSTADO|CORRIGIDO)", resp, re.IGNORECASE)
    if m_dec:
        nivel = m_dec.group(1).upper()

    m_txt = re.search(r"TEXTO\s*:\s*([\s\S]+)", resp, re.IGNORECASE)
    if m_txt:
        candidato = m_txt.group(1).strip()
        candidato = re.sub(r"^DECIS[ÃA]O\s*:.*\n?", "", candidato, flags=re.IGNORECASE).strip()
        candidato = _strip_justification_blocks(candidato)
        candidato = _strip_meta_sentences(candidato).strip()
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


# ---------------------------------------------------------------------------
# Verification rate monitor
# ---------------------------------------------------------------------------

def _monitorar_taxa_verificacao(stats: dict, titulo_secao: str) -> Tuple[bool, str]:
    """Return (precisa_buscar_mais, motivo) based on current verification stats."""
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


# ---------------------------------------------------------------------------
# Complementary search
# ---------------------------------------------------------------------------

def _buscar_conteudo_complementar(
    titulo_secao: str,
    conteudo_esperado: str,
    corpus_atual: "CorpusMongoDB",
    urls_tentadas: set,
    prompt_dir: str = "technical_writing",
    language: str = "pt",
) -> Tuple[int, "CorpusMongoDB", str]:
    """Search for complementary content when paragraph verification rate is low."""
    # Import here to avoid circular dependency
    from ...utils.vector_utils.mongodb_corpus import CorpusMongoDB

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

    tavily_enabled = getattr(corpus_atual, "tavily_enabled", True)
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


# ---------------------------------------------------------------------------
# Adaptive verification — legacy (no explicit anchors)
# ---------------------------------------------------------------------------

def _verificar_e_corrigir_secao_adaptativa(
    texto_secao: str,
    corpus: "CorpusMongoDB",
    corpus_prompt_completo: str,
    titulo: str,
    fontes_secao,
    conteudo_esperado: str = "",
    language: str = "pt",
) -> Tuple[str, str, dict]:
    """Adaptive verification without explicit anchor routing (legacy fallback)."""
    urls_tentadas: set = set()
    iteracao = 0

    while iteracao < 3:
        iteracao += 1
        print(f"\n     └─ Verificação iter {iteracao}/3")

        blocos = re.split(r'\n{2,}', texto_secao.strip())
        resultado = []
        log_linhas = [f"\n### Verificação Adaptativa — {titulo} (iter {iteracao})"]

        stats = {
            "total": 0, "aprovados": 0, "ajustados": 0, "corrigidos": 0,
            "estruturais": 0, "verificaveis": 0, "pulados": 0,
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


# ---------------------------------------------------------------------------
# Anchor-directed single paragraph
# ---------------------------------------------------------------------------

def _verificar_paragrafo_com_ancora(
    bloco: str,
    corpus: "CorpusMongoDB",
    fonte_map: dict,
    titulo_secao: str,
    prompt_dir: str = "technical_writing",
    language: str = "pt",
) -> Tuple[str, str, str, bool]:
    """Verify one paragraph using its explicit anchors for source retrieval.

    Returns (texto_final, nivel, log_entry, eh_verificavel).
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
            ancoras_com_urls = [
                (at, fonte_map[nc])
                for at, nc in ancoras_com_cit
                if nc in fonte_map
            ]
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

    return _juiz_paragrafo_melhorado(
        bloco_limpo, fontes, titulo_secao, prompt_dir=prompt_dir, language=language
    )


# ---------------------------------------------------------------------------
# Anchor-directed adaptive loop (main)
# ---------------------------------------------------------------------------

def _verificar_e_corrigir_secao_com_ancora(
    texto_secao: str,
    corpus: "CorpusMongoDB",
    fonte_map: dict,
    titulo: str,
    conteudo_esperado: str = "",
    prompt_dir: str = "technical_writing",
    language: str = "pt",
) -> Tuple[str, str, dict]:
    """Full adaptive verification loop using explicit anchors for directed retrieval."""
    urls_tentadas: set = set()
    iteracao = 0

    while iteracao < 3:
        iteracao += 1
        print(f"\n     └─ Verificação iter {iteracao}/3 (com âncoras)")

        blocos = re.split(r'\n{2,}', texto_secao.strip())
        resultado = []
        log_linhas = [f"\n### Verificação com Âncoras — {titulo} (iter {iteracao})"]

        stats = {
            "total": 0, "aprovados": 0, "ajustados": 0, "corrigidos": 0,
            "estruturais": 0, "verificaveis": 0, "pulados": 0,
            "ancoras_usadas": 0,
        }

        for i, bloco in enumerate(blocos):
            bloco = bloco.strip()
            if not bloco:
                continue

            if bool(re.search(r'\[ÂNCORA:', bloco)):
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
