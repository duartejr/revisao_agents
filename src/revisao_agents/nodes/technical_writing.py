"""
Technical writing agents - LangGraph nodes for technical chapter authoring.

Advanced nodes for the technical writing workflow:
- Plan parsing and section extraction
- Section authoring with search and verification
- Document consolidation with introduction and conclusion
"""

import re
import time
from datetime import datetime
from typing import List, Tuple, Optional

# Updated import paths (relative imports)
from ..state import EscritaTecnicaState
from ..utils.llm_providers import get_llm as llm_call  # Note: might be named differently
from ..utils.prompt_loader import load_prompt
from ..core.schemas.techinical_writing import RespostaSecao, Fonte
from ..utils.mongodb_corpus import CorpusMongoDB
from ..utils.helpers import (
    resumir_secao, parse_plano_tecnico
)
from ..utils.tavily_client import search_web, search_images, extract_urls, score_url

# Constants - may need to be moved to config.py
TECNICO_MAX_RESULTS = 5
MAX_URLS_EXTRACT = 10
CTX_RESUMO_CHARS = 1000
SECAO_MIN_PARAGRAFOS = 4
DELAY_ENTRE_SECOES = 2
MAX_REACT_ITERATIONS = 3
EXTRACT_MIN_CHARS = 500
TOP_K_OBSERVACAO = 5
MAX_CORPUS_PROMPT = 3000


# Padrão para âncoras
_ANCORA_PATTERN = re.compile(r'\[ÂNCORA:\s*"((?:[^"\\]|\\.)*)"\]', re.DOTALL)


# --- Fases internas (reescritas para usar CorpusMongoDB) ---

def _fase_pensamento(tema: str, titulo: str, cont_esp: str, recursos: str) -> dict:
    """Fase 1: planejamento da busca."""
    p = load_prompt(
        "technical_writing/fase_pensamento",
        titulo=titulo,
        tema=tema,
        cont_esp=cont_esp,
        recursos=recursos,
    )
    resp = llm_call(p.text, temperature=p.temperature)
    # Note: parse_json_safe function call is replaced with try-except below
    try:
        import json
        resultado = json.loads(resp)
    except:
        resultado = None
    
    if resultado:
        return resultado
    return {
        "informacoes_necessarias": [cont_esp[:120]],
        "queries_busca":           [f"{tema} {titulo}", f"{titulo} technical details"],
        "queries_imagens":         [f"{titulo} diagram architecture"],
    }


def _extrair_ancora_principal(bloco: str) -> Optional[str]:
    """
    Extrai a âncora mais relevante (mais longa) de um bloco de texto.
    """
    ancoras = _ANCORA_PATTERN.findall(bloco)
    
    # Filtra âncoras válidas (mínimo 20 chars)
    ancoras_validas = [
        a.strip() for a in ancoras
        if len(a.strip()) >= 20
        and not re.match(r'^[\\\$\{\}\[\]_\^]+', a.strip())
    ]
    
    if not ancoras_validas:
        return None
    
    # Retorna a mais longa (geralmente a mais específica)
    return max(ancoras_validas, key=len)


def _extrair_citacao_ancora(texto: str, ancora: str) -> Optional[int]:
    """
    Encontra o número da citação [N] mais próxima de uma âncora específica.
    """
    # Procura pela âncora no texto
    ancora_escaped = re.escape(ancora)
    pattern = re.compile(
        rf'\[ÂNCORA:\s*"{ancora_escaped}"\]\s*\[(\d+)\]',
        re.IGNORECASE
    )
    
    match = pattern.search(texto)
    if match:
        return int(match.group(1))
    
    # Fallback: procura citação próxima à âncora (até 50 chars depois)
    ancora_pos = texto.find(ancora)
    if ancora_pos >= 0:
        trecho_posterior = texto[ancora_pos:ancora_pos + 50]
        cit_match = re.compile(r'\[(\d+)\]').search(trecho_posterior)
        if cit_match:
            return int(cit_match.group(1))
    
    return None


def _extrair_todas_ancoras_com_citacoes(bloco: str) -> List[Tuple[str, Optional[int]]]:
    """
    Extrai todas as âncoras do bloco junto com suas citações.
    
    Returns:
        Lista de (texto_ancora, numero_citacao)
    """
    resultados = []
    
    # Procura padrão: [ÂNCORA: "texto"] [N]
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


def _contar_claims_verificaveis(paragrafo: str) -> int:
    """Conta claims que DEVEM ser verificadas (sem contar conhecimento geral)."""
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
) -> tuple:
    """
    Juiz com 3 níveis + detecção de âncoras.
    Retorna: (texto_final, nivel, log_entry, eh_verificavel)
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
        "technical_writing/writer_judge",
        paragrafo_limpo=paragrafo_limpo,
        titulo_secao=titulo_secao,
        fontes=fontes,
    )
    resp = llm_call(p.text, temperature=p.temperature)
    
    texto_final = paragrafo_limpo
    nivel = "APROVADO"  # default
    
    m_dec = re.search(r"DECIS[ÃA]O\s*:\s*(APROVADO|AJUSTADO|CORRIGIDO)", resp, re.IGNORECASE)
    if m_dec:
        nivel = m_dec.group(1).upper()
    
    m_txt = re.search(r"TEXTO\s*:\s*([\s\S]+)", resp, re.IGNORECASE)
    if m_txt:
        candidato = m_txt.group(1).strip()
        candidato = re.sub(r"^DECIS[ÃA]O\s*:.*\n?", "", candidato, flags=re.IGNORECASE).strip()
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
    Monitora se a taxa de verificação é aceitável.
    Retorna: (precisa_buscar_mais, motivo)
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


def _verificar_paragrafo_com_ancora(
    bloco: str,
    corpus: CorpusMongoDB,
    fonte_map: dict,
    titulo_secao: str,
) -> Tuple[str, str, str, bool]:
    """
    Verifica um parágrafo usando âncoras para busca direcionada.
    
    Args:
        bloco: texto do parágrafo (COM âncoras)
        corpus: objeto CorpusMongoDB
        fonte_map: dicionário {numero_citacao: url}
        titulo_secao: título da seção (para contexto)
    
    Returns:
        (texto_final, nivel, log_entry, eh_verificavel)
    """
    # Remove âncoras para o texto limpo
    bloco_limpo = re.sub(r'\[ÂNCORA:\s*"[^"]*"\]', "", bloco).strip()
    bloco_limpo = re.sub(r'  +', ' ', bloco_limpo)
    
    # Verifica se é parágrafo verificável
    if bloco_limpo.startswith("#") or len(bloco_limpo) < 60:
        return bloco_limpo, "ESTRUTURAL", f"⏭️  ESTRUTURAL", False
    
    # === ESTRATÉGIA 1: Busca por âncora principal + URL ===
    ancora_principal = _extrair_ancora_principal(bloco)
    
    if ancora_principal:
        # Encontra citação da âncora
        num_citacao = _extrair_citacao_ancora(bloco, ancora_principal)
        
        if num_citacao and num_citacao in fonte_map:
            url_citada = fonte_map[num_citacao]
            
            print(f"     🎯 Âncora encontrada ({len(ancora_principal)} chars) → [{num_citacao}]")
            print(f"        URL: {url_citada[:60]}")
            
            # Busca DIRECIONADA: âncora + URL específica
            # NOTE: This assumes render_prompt_url exists. May need adjustment.
            try:
                fontes, urls_usadas, n_chunks = corpus.render_prompt_url(
                    texto_ancora=ancora_principal,
                    url_citada=url_citada,
                    max_chars=3000,
                    top_k=5,
                )
            except:
                fontes = corpus.render_prompt(bloco_limpo[:300], max_chars=3000)[0]
            
            if fontes:
                print(f"        ✅ {n_chunks} chunks da URL citada")
            else:
                print(f"        ⚠️  Nenhum chunk encontrado, usando busca geral")
                fontes = corpus.render_prompt(bloco_limpo[:300], max_chars=3000)[0]
        else:
            # Âncora sem citação válida → busca geral
            print(f"     ⚠️  Âncora sem citação válida")
            fontes = corpus.render_prompt(bloco_limpo[:300], max_chars=3000)[0]
    else:
        # === ESTRATÉGIA 2: Busca por múltiplas âncoras (se houver) ===
        ancoras_com_cit = _extrair_todas_ancoras_com_citacoes(bloco)
        
        if ancoras_com_cit:
            print(f"     🎯 {len(ancoras_com_cit)} âncoras com URLs")
            fontes = corpus.render_prompt(bloco_limpo[:300], max_chars=3000)[0]
        else:
            # Sem âncoras → busca geral
            fontes = corpus.render_prompt(bloco_limpo[:300], max_chars=3000)[0]
    
    # === VERIFICAÇÃO COM O JUIZ ===
    if not fontes.strip():
        return bloco_limpo, "APROVADO", f"✅ SEM FONTES", True
    
    # Chama o juiz para verificar
    texto_final, nivel, log_entry, eh_verificavel = _juiz_paragrafo_melhorado(
        bloco_limpo, fontes, titulo_secao
    )
    
    return texto_final, nivel, log_entry, eh_verificavel


def _verificar_e_corrigir_secao_com_ancora(
    texto_secao: str,
    corpus: CorpusMongoDB,
    fonte_map: dict,
    titulo: str,
    conteudo_esperado: str = "",
) -> Tuple[str, str, dict]:
    """
    Verificação adaptativa usando âncoras para busca direcionada.
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
            
            # Verifica se bloco tem âncoras
            tem_ancoras = bool(re.search(r'\[ÂNCORA:', bloco))
            if tem_ancoras:
                stats["ancoras_usadas"] += 1
            
            # === USA VERIFICAÇÃO COM ÂNCORA ===
            texto_final, nivel, log_entry, eh_verificavel = _verificar_paragrafo_com_ancora(
                bloco=bloco,
                corpus=corpus,
                fonte_map=fonte_map,
                titulo_secao=titulo,
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
        
        # Monitora taxa de verificação
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
    
    # Monta texto final
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
# Nós do grafo (ainda a ser implementados)
# ============================================================================

def parsear_plano_node(state: EscritaTecnicaState) -> dict:
    """Parses a technical plan and extracts sections."""
    caminho = state["caminho_plano"]
    print(f"\n📖 Lendo plano: {caminho}")
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            texto = f.read()
    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {caminho}")
        return {"status": "erro_arquivo", "error": "Plano não encontrado"}
    
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
    }


def escrever_secoes_node(state: EscritaTecnicaState) -> dict:
    """
    Main node for writing sections with search, extraction, and verification.
    NOTE: This is a simplified stub - the full implementation is very large.
    """
    tema = state["tema"]
    secoes = state.get("secoes", [])
    
    print(f"\n{'━'*70}")
    print(f"  ESCREVENDO {len(secoes)} SEÇÕES")
    print(f"{'━'*70}")
    
    # Simplified: just return empty state for now
    # Full implementation would be here
    
    return {
        "secoes_escritas": [],
        "refs_urls": [],
        "refs_imagens": [],
        "resumo_acumulado": "",
        "react_log": [],
        "stats_verificacao": [],
        "status": "secoes_escritas",
    }


def consolidar_node(state: EscritaTecnicaState) -> dict:
    """
    Consolidates written sections into a final document.
    NOTE: Simplified stub - full implementation needed.
    """
    tema = state["tema"]
    secoes = state.get("secoes_escritas", [])
    
    print(f"\n📚 Consolidando {len(secoes)} seções...")
    
    # Simplified consolidation
    documento = f"# {tema}\n\n"
    for s in secoes:
        documento += f"## {s.get('titulo', 'Seção')}\n\n"
        documento += s.get('texto', '') + "\n\n"
    
    slug = re.sub(r"[^\w\s-]", "", tema[:40]).strip().replace(" ", "_").lower()
    output_path = f"reviews/revisao_tecnica_{slug}.md"
    
    try:
        import os as _os
        _os.makedirs("reviews", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(documento)
        print(f"\n💾 {output_path} ({len(documento):,} chars)")
    except Exception as e:
        print(f"⚠️  Erro ao salvar: {e}")
    
    return {"status": "concluido"}
