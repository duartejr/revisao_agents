import os
import re
import json
import difflib
from typing import List, Optional
from ...config import HIST_MAX_TURNS, PLAN_MAX_CHARS


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

def truncar(s: str, n: int = PLAN_MAX_CHARS) -> str:
    return s if len(s) <= n else s[:n] + "\n...[truncado]"

def salvar_md(conteudo: str, prefixo: str, tema: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", tema[:40]).strip().replace(" ", "_").lower()
    path = f"{prefixo}_{slug}.md"
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
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
    from ...config import llm_call
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

def parse_plano_academico(texto: str) -> tuple:
    """
    Extract theme, summary, and section list from an academic plan Markdown file.

    Academic plans use a 3-column table:
      | N. Título | Objetivo | Tópicos |

    Returns (tema, resumo, secoes) in the same shape as parse_plano_tecnico so the
    entire downstream writer pipeline is unaffected.
    """
    tema = "Revisão Acadêmica"
    m = re.search(r"\*\*Tema:\*\*\s*(.+)", texto)
    if m:
        tema = m.group(1).replace("*", "").strip()

    # Strip a fenced code block wrapper added by the planner (``` markdown ... ```)
    inner = re.search(r"```(?:markdown)?\n([\s\S]+?)\n```", texto)
    conteudo = inner.group(1) if inner else texto

    resumo = conteudo[:1200].strip()
    secoes = []

    # Primary: 3-column table  | N. Title | Objetivo | Tópicos |
    pattern = r"\|\s*\*?\*?(\d[\d\.]*\.?\s+[^|*]+?)\*?\*?\s*\|\s*([^|]+)\s*\|\s*([^|]*)\s*\|"
    for titulo_raw, objetivo, topicos in re.findall(pattern, conteudo):
        titulo_clean = titulo_raw.strip().replace("**", "")
        if not titulo_clean or "Título" in titulo_clean or "---" in titulo_clean:
            continue
        secoes.append({
            "indice": len(secoes),
            "titulo": titulo_clean,
            "conteudo_esperado": objetivo.strip(),
            "recursos": topicos.strip(),
        })

    # Fallback: H2 / H3 numbered headings  (## 1. Title)
    if not secoes:
        for i, t in enumerate(re.findall(r"^#{2,3}\s+(\d[\d\.]*\s+.+)$", conteudo, re.MULTILINE)):
            secoes.append({
                "indice": i,
                "titulo": t.strip(),
                "conteudo_esperado": t.strip(),
                "recursos": "",
            })

    if not secoes:
        raise ValueError("❌ Nenhuma seção encontrada no plano acadêmico.")

    return tema, resumo, secoes


def extrair_ancoras(texto: str) -> list:
    """Extracts anchor texts [ÂNCORA: "..."] from a text block."""
    pattern = re.compile(r'\[ÂNCORA:\s*"((?:[^"\\]|\\.)*)"\]', re.DOTALL)
    return [m.strip() for m in pattern.findall(texto)]


def eh_paragrafo_verificavel(paragrafo: str) -> bool:
    """
    Returns True if the paragraph contains verifiable claims
    that require anchor/source support.
    """
    p = paragrafo.strip()
    if len(p) < 60:
        return False
    if p.startswith("#"):
        return False
    if re.match(r"^\s*[-*]\s", p):
        return False
    if p.startswith("```"):
        return False
    if p.startswith("$$") or re.match(r"^\s*\$[^$]+\$", p):
        return False
    if p.startswith("*Figura") or p.startswith("!["):
        return False
    # Has numbers, citations, or strong verbs → likely verifiable
    has_numbers = bool(re.search(r'\b\d+[\d.,]*\b', p))
    has_citations = bool(re.search(r'\[\d+\]', p))
    has_anchors = bool(re.search(r'\[ÂNCORA:', p))
    has_verbs = bool(re.search(
        r'\b(foi|é|são|demonstra|prova|mostra|evidencia|encontrou|'
        r'observou|descobriu|propôs|definiu)\b', p, re.IGNORECASE
    ))
    return has_numbers or has_citations or has_anchors or has_verbs
