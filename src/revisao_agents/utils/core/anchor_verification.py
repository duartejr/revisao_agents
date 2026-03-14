import re
from typing import List, Tuple
from ..vector_utils.vector_store import search_chunks
from ..file_utils.helpers import extract_anchors, is_paragraph_verifiable
from ...config import JUIZ_MAX_CORPUS_CHARS, JUIZ_TOP_K, get_llm

def search_chunks_for_paragraph(
    paragrafo: str,
    corpus_prompt_completo: str,
    tema_secao: str,
) -> str:
    """
    Monta o bloco de fontes relevantes para verificar o parágrafo.
    Usa as anchors declaradas como queries de busca no MongoDB.
    """
    # Limpa anchors e LaTeX pesado do texto para a query
    texto_sem_anchors = re.sub(r'\[ANCHOR:\s*"[^"]*"\]', "", paragrafo)
    texto_sem_anchors = re.sub(r'\$\$[^$]+\$\$', "", texto_sem_anchors)
    texto_sem_anchors = re.sub(r'\$[^$]+\$', "", texto_sem_anchors)
    texto_sem_anchors = re.sub(r'\\\([^)]+\\\)', "", texto_sem_anchors).strip()

    # Anchors declaradas são os melhores hints de busca
    anchors = extract_anchors(paragrafo)
    # Filtra anchors que são só LaTeX ou muito curtas
    anchors_validas = [
        a for a in anchors
        if len(a.strip()) >= 20
        and not re.match(r'^[\\\$\{\}\[\]_\^]+', a.strip())
    ]

    queries = anchors_validas[:3] + ([texto_sem_anchors[:200]] if texto_sem_anchors else [])

    if not queries:
        return corpus_prompt_completo[:JUIZ_MAX_CORPUS_CHARS]

    # Uses search_chunks from vector_store (MongoDB)
    # Note: search_chunks returns only text; source metadata is not included.
    # Vamos enriquecer com metadados? Por enquanto só texto.
    partes: List[str] = []
    chars = 0
    chunks_vistos = set()

    for q in queries:
        q = q.strip()
        if not q:
            continue
        # search_chunks returns a list of chunk text strings
        resultados = search_chunks(q, k=JUIZ_TOP_K)
        for texto_chunk in resultados:
            # deduplicação simples
            chave = texto_chunk[:100]
            if chave in chunks_vistos:
                continue
            chunks_vistos.add(chave)
            bloco = f"{texto_chunk}\n\n"
            if chars + len(bloco) > JUIZ_MAX_CORPUS_CHARS:
                break
            partes.append(bloco)
            chars += len(bloco)

    if partes:
        return "".join(partes)

    # fallback
    return corpus_prompt_completo[:JUIZ_MAX_CORPUS_CHARS]

def juiz_paragrafo(paragrafo_limpo: str, fontes: str) -> Tuple[str, str, str]:
    """
    Juiz LLM com 3 níveis.
    Retorna (texto_final, nivel, log_entry)
    nivel ∈ {"APROVADO", "AJUSTADO", "CORRIGIDO"}
    """
    prompt = f"""Você é um verificador de fatos técnicos. Analise o parágrafo abaixo contra as fontes.

PARÁGRAFO:
{paragrafo_limpo}

FONTES DISPONÍVEIS:
{fontes}

TAREFA: Classifique o parágrafo em um de três níveis e retorne o texto adequado.

NÍVEL 1 — APROVADO
  Use quando: todas as afirmações têm suporte nas fontes OU são conhecimento técnico universal estabelecido.
  Ação: retorne o parágrafo sem nenhuma modificação.
  DECISÃO: APROVADO
  TEXTO: [parágrafo sem modificação]

NÍVEL 2 — AJUSTADO
  Use quando: o conteúdo essencial está correto mas há imprecisão de escopo (ex: "bacias hidrográficas"
  quando a fonte diz "séries temporais") ou escolha de palavra inexata — não é erro factual grave.
  Ação: corrija apenas a imprecisão específica, mantenha o restante idêntico.
  DECISÃO: AJUSTADO
  TEXTO: [parágrafo com mínima correção]

NÍVEL 3 — CORRIGIDO
  Use SOMENTE quando: há afirmação factualmente errada em relação às fontes, OU há afirmação específica
  sem nenhum suporte nas fontes disponíveis (não é conhecimento universal).
  Ação:
    - Afirmação errada → corrija para o que a fonte diz
    - Afirmação sem suporte → REMOVA a frase inteira
    - NÃO adicione informações que não estejam nas fontes
  DECISÃO: CORRIGIDO
  TEXTO: [parágrafo corrigido]

IMPORTANTE:
  - Se as fontes forem insuficientes para verificar o parágrafo → use APROVADO (benefício da dúvida)
  - NÃO use CORRIGIDO por diferença de estilo ou vocabulário
  - NÃO use CORRIGIDO porque a fonte não confirma explicitamente algo que é conhecimento técnico geral
  - RESPONDA APENAS com DECISÃO e TEXTO. Sem explicações adicionais."""

    resp = get_llm(temperature=0.0).invoke(prompt)
    resp_text = resp.content if hasattr(resp, "content") else str(resp)

    nivel = "APROVADO"
    texto_final = paragrafo_limpo

    m_dec = re.search(
        r"DECIS[ÃA]O\s*:\s*(APROVADO|AJUSTADO|CORRIGIDO)",
        resp_text, re.IGNORECASE
    )
    if m_dec:
        nivel = m_dec.group(1).upper()

    m_txt = re.search(r"TEXTO\s*:\s*([\s\S]+)", resp_text, re.IGNORECASE)
    if m_txt:
        candidato = m_txt.group(1).strip()
        candidato = re.sub(
            r"^DECIS[ÃA]O\s*:.*\n?", "", candidato, flags=re.IGNORECASE
        ).strip()
        if candidato:
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

    return texto_final, nivel, log_entry