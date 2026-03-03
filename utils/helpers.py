import re
import json
import difflib
from typing import List, Optional
from config import HIST_MAX_TURNS, PLANO_MAX_CHARS


def fmt_chunks(chunks: List[str], max_chars: int = 1200) -> str:
    bloco = ""
    for i, c in enumerate(chunks, 1):
        linha = f"[{i}] {c}\n"
        if len(bloco) + len(linha) > max_chars:
            break
        bloco += linha
    return bloco.strip()

def fmt_snippets(resultados: List[dict], max_chars: int = 1200) -> str:
    bloco = ""
    for i, r in enumerate(resultados, 1):
        titulo  = r.get("title",   "")[:60]
        snippet = r.get("snippet", "")[:250]
        url     = r.get("url",     "")[:80]
        linha   = f"[{i}] {titulo}\n    {snippet}\n    {url}\n\n"
        if len(bloco) + len(linha) > max_chars:
            break
        bloco += linha
    return bloco.strip()

def resumir_hist(historico: List[tuple], max_turns: int = HIST_MAX_TURNS) -> str:
    if not historico:
        return "(sem historico)"
    recentes = historico[-(max_turns * 2):]
    linhas = []
    for role, c in recentes:
        label  = "Agente" if role == "assistant" else "Usuario"
        resumo = c[:300] + "..." if len(c) > 300 else c
        linhas.append(f"{label}: {resumo}")
    return "\n".join(linhas)

def truncar(s: str, n: int = PLANO_MAX_CHARS) -> str:
    return s if len(s) <= n else s[:n] + "\n...[truncado]"

def salvar_md(conteudo: str, prefixo: str, tema: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", tema[:40]).strip().replace(" ", "_").lower()
    path = f"{prefixo}_{slug}.md"
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(conteudo)
        print("\nSalvo em:", path)
    except Exception as e:
        print("Nao foi possivel salvar:", str(e))
    return path


def normalizar(texto: str) -> str:
    """Lowercase, sem pontuação, espaços simples."""
    t = texto.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def fuzzy_sim(a: str, b: str) -> float:
    """SequenceMatcher ratio entre dois textos normalizados."""
    return difflib.SequenceMatcher(None, a, b).ratio()

def fuzzy_search_in_text(ancora_norm: str, corpus_norm: str) -> tuple:
    """
    Desliza uma janela pelo corpus tentando encontrar a âncora por fuzzy.
    Returns: (melhor_score, trecho_original[:120])
    """
    palavras_ancora = ancora_norm.split()
    palavras_corpus = corpus_norm.split()
    n = len(palavras_ancora)
    if n == 0:
        return 0.0, ""

    melhor = 0.0
    melhor_trecho = ""
    passo = max(1, n // 4)

    for i in range(0, max(1, len(palavras_corpus) - n), passo):
        janela = " ".join(palavras_corpus[i: i + n + n // 3])
        score  = fuzzy_sim(ancora_norm, janela)
        if score > melhor:
            melhor = score
            melhor_trecho = janela[:120]

    return melhor, melhor_trecho

def resumir_secao(titulo: str, texto: str) -> str:
    """Gera um resumo curto de uma seção usando LLM."""
    from config import llm_call
    resp = llm_call(
        f"Resuma em 3-4 frases CONCISAS os conceitos técnicos centrais de "
        f"'{titulo}'. Destaque fundamentos, fórmulas-chave e conclusões.\n\n"
        f"SEÇÃO:\n{texto[:2500]}",
        temperature=0.1,
    )
    return resp[:400]

def parse_plano_tecnico(texto: str) -> tuple:
    """
    Extrai tema, resumo e lista de seções do plano em Markdown.
    Retorna (tema, resumo, secoes)
    """
    tema = "Revisao Tecnica"
    m = re.search(r"\*\*Tema:\*\*\s*(.+)", texto)
    if m:
        tema = m.group(1).replace("*", "").strip()
    resumo = texto[:1200].strip()
    secoes = []
    pattern = r"\|\s*([0-9\.]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]*)\s*\|"
    for nivel, titulo, cont_esp, recursos in re.findall(pattern, texto):
        nivel_clean = nivel.strip()
        if not nivel_clean or "Nível" in nivel_clean or "---" in nivel_clean:
            continue
        secoes.append({
            "indice":            len(secoes),
            "titulo":            f"{nivel_clean} {titulo.strip().replace('**', '')}",
            "conteudo_esperado": cont_esp.strip(),
            "recursos":          recursos.strip(),
        })
    if not secoes:
        for i, t in enumerate(re.findall(r"^##\s+([0-9]+\..+)$", texto, re.MULTILINE)):
            secoes.append({"indice": i, "titulo": t,
                           "conteudo_esperado": t, "recursos": ""})
    if not secoes:
        raise ValueError("❌ Nenhuma seção encontrada no plano.")
    return tema, resumo, secoes