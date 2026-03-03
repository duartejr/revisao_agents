import re
import time
from datetime import datetime
from typing import List, Dict, Any

from state import EscritaTecnicaState  # precisamos definir este estado
from config import (
    llm_call, parse_json_safe,
    TECNICO_MAX_RESULTS, MAX_URLS_EXTRACT, CTX_PLANO_CHARS, CTX_RESUMO_CHARS,
    SECAO_MIN_PARAGRAFOS, MAX_IMAGENS_SECAO, DELAY_ENTRE_SECOES,
    MAX_REACT_ITERATIONS, EXTRACT_MIN_CHARS, TOP_K_OBSERVACAO,
    MAX_CORPUS_PROMPT, JUIZ_TOP_K, ANCORA_MIN_SIM_FAISS
)
from utils.tavily_client import search_web, search_images, extract_urls, score_url
from utils.mongodb_corpus import CorpusMongoDB
from utils.helpers import (
    normalizar, fuzzy_sim, fuzzy_search_in_text,
    resumir_secao, parse_plano_tecnico
)

# Padrão para âncoras
_ANCORA_PATTERN = re.compile(r'\[ÂNCORA:\s*"((?:[^"\\]|\\.)*)"\]', re.DOTALL)

# --- Fases internas (reescritas para usar CorpusMongoDB) ---

def _fase_pensamento(tema: str, titulo: str, cont_esp: str, recursos: str) -> dict:
    """Fase 1: planejamento da busca."""
    schema = (
        'Responda EXCLUSIVAMENTE em JSON válido, sem markdown:\n'
        '{\n'
        '  "informacoes_necessarias": ["lista do que precisa ser encontrado"],\n'
        '  "queries_busca": ["query 1", "query 2", "query 3", "query 4"],\n'
        '  "queries_imagens": ["query imagem 1", "query imagem 2"]\n'
        '}'
    )
    resp = llm_call(
        f"Você planeja a pesquisa para escrever a seção '{titulo}' "
        f"do capítulo técnico sobre '{tema}'.\n\n"
        f"Conteúdo esperado: {cont_esp}\n"
        f"Recursos requeridos: {recursos}\n\n"
        "Liste:\n"
        "1. Quais informações, fatos, fórmulas, algoritmos, conceitos "
        "e dados de qualquer tipo precisam ser encontrados em fontes primárias?\n"
        "2. Queries de busca que maximizem a chance de encontrar fontes "
        "primárias completas (artigos, teses, documentações técnicas).\n"
        "3. Queries para busca de imagens (diagramas, arquiteturas, gráficos).\n"
        "4. Todas as queries geradas em inglê e português. Sempre gera uma versão da mesma query nos dois idiomas\n"
        + schema,
        temperature=0.1,
    )
    resultado = parse_json_safe(resp)
    if resultado:
        return resultado
    return {
        "informacoes_necessarias": [cont_esp[:120]],
        "queries_busca":           [f"{tema} {titulo}", f"{titulo} technical details"],
        "queries_imagens":         [f"{titulo} diagram architecture"],
    }

def _fase_observacao(informacoes_necessarias: List[str], corpus: CorpusMongoDB) -> dict:
    """Fase 5: verifica se corpus é suficiente."""
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

    schema = (
        'Responda EXCLUSIVAMENTE em JSON válido, sem markdown:\n'
        '{\n'
        '  "suficiente": true,\n'
        '  "lacunas": ["informações ausentes no corpus"],\n'
        '  "query_complementar": null,\n'
        '  "resumo": "resumo em 2 frases do que o corpus contém"\n'
        '}'
    )
    resp = llm_call(
        f"Analise se o corpus abaixo contém informações suficientes para escrever "
        f"a seção sem precisar inventar NADA.\n\n"
        f"Informações necessárias:\n"
        + "\n".join(f"- {i}" for i in informacoes_necessarias) +
        f"\n\nCORPUS — trechos mais relevantes recuperados:\n{amostra_corpus}\n\n"
        "'suficiente' é true se o corpus permite escrever a seção ancorando "
        "todas as afirmações relevantes em trechos do corpus.\n"
        "Se false, 'query_complementar' deve ser uma busca diferente das já feitas, sempre query com versão português e inglês.\n\n"
        + schema,
        temperature=0.1,
    )
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
    corpus: CorpusMongoDB, imagens_txt: str, resumo_acumulado: str,
    pos: int, n_total: int, titulos_todos: List[str], n_extraidos: int
) -> tuple:
    """Fase 6: gera rascunho ancorado."""
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

    query_retrieval = f"{titulo} {cont_esp} {recursos}"
    corpus_prompt, urls_usadas, _ = corpus.render_prompt(
        query_retrieval, max_chars=MAX_CORPUS_PROMPT
    )

    _PROMPT_ANCORA_INSTRUCOES = """
══════════════════════════════════════════════════════════════════
SISTEMA DE ESCRITA ANCORADA — LEIA COM ATENÇÃO ABSOLUTA
══════════════════════════════════════════════════════════════════

REGRA ÚNICA E ABSOLUTA:
  Toda afirmação factual DEVE ser imediatamente seguida de uma âncora:
  [ÂNCORA: "trecho exato copiado do corpus"]

  A âncora deve ser um fragmento copiado literalmente do corpus acima.
  NÃO parafraseie a âncora. NÃO invente. Copie o trecho como está.

COBERTURA OBRIGATÓRIA DA ÂNCORA — para TUDO que for escrito:
  ✅ Números, estatísticas, percentuais, métricas
  ✅ Fórmulas matemáticas e notação simbólica
  ✅ Passos de algoritmos e pseudocódigo
  ✅ Datas, períodos, versões de software/hardware
  ✅ Nomes de autores, instituições, estudos específicos
  ✅ Conceitos técnicos definidos ou caracterizados
  ✅ Comparações, resultados, conclusões de estudos
  ✅ Qualquer afirmação que não seja de conhecimento universal básico

EXEMPLOS CORRETOS:
  "O modelo convergiu após 100 épocas [ÂNCORA: "convergiu após 100 épocas de treinamento"] [2]."
  "A função de perda utilizada é o MSE [ÂNCORA: "mean squared error (MSE) is used as loss function"] [1]."
  "LSTM supera modelos clássicos em séries não-estacionárias [ÂNCORA: "LSTM outperforms classical models on non-stationary time series"] [3]."

EXEMPLOS PROIBIDOS (âncoras inválidas):
  ❌ [ÂNCORA: "conforme esperado teoricamente"]      — invenção
  ❌ [ÂNCORA: "como é amplamente conhecido"]         — evasão
  ❌ [ÂNCORA: "resultados similares à literatura"]   — vaga
  ❌ Afirmação factual SEM âncora nenhuma             — proibido

SE NÃO HÁ ÂNCORA DISPONÍVEL NO CORPUS:
  Escreva explicitamente: "Informação não disponível nas fontes consultadas."
  Ou descreva de forma puramente qualitativa sem afirmar fatos específicos.
  NUNCA invente uma âncora que não existe no corpus.

AFIRMAÇÕES QUE NÃO PRECISAM DE ÂNCORA (conhecimento universal básico):
  - Definições matemáticas elementares (ex: "a derivada é o limite...")
  - Conceitos de nível enciclopédico amplamente estabelecidos SEM dados específicos
  - Transições narrativas e conectivos

══════════════════════════════════════════════════════════════════
"""

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
        f"{corpus_prompt}\n\n"
        f"IMAGENS DISPONÍVEIS:\n{imagens_txt if imagens_txt else '  (Nenhuma)'}\n\n"
        + _PROMPT_ANCORA_INSTRUCOES +
        f"\nREGRAS ADICIONAIS DE ESCRITA:\n"
        f"1. NÃO comece com meta-texto ('Esta seção aborda...').\n"
        f"2. Mínimo {SECAO_MIN_PARAGRAFOS} parágrafos densos (8-12 linhas).\n"
        f"3. LaTeX: inline $...$, bloco $$...$$ para equações.\n"
        f"4. Algoritmos: blocos ```algorithm ... ```.\n"
        f"5. Citações: [N] após cada âncora (N = índice da FONTE no corpus).\n"
        f"6. Imagens: insira inline ![desc](URL) com *Figura N: legenda*.\n"
        f"7. NÃO repita conceitos das seções anteriores.\n\n"
        f"## {titulo}\n"
    )
    return llm_call(prompt, temperature=0.25), urls_usadas

def _extrair_com_fallback(
    resultados: List[dict],
    queries_fallback: List[str],
    urls_tentadas: set,
    corpus: "CorpusMongoDB", 
) -> tuple:
    """
    Extrai texto completo das URLs prioritárias com fallback.
    URLs já existentes no MongoDB são ignoradas (não extraídas novamente).
    Retorna (extraidos_validos, resultados_enriquecidos, urls_tentadas)
    """
    from utils.tavily_client import extract_urls, search_web
    from utils.mongodb_corpus import CorpusMongoDB

    # Seleciona URLs com score
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
            print(f"      ⏭️ URL já indexada, pulando extração: {url[:60]}")
            urls_tentadas.add(url)  # marca como tentada para não reprocessar
            continue
        urls_para_extrair.append(url)
        if len(urls_para_extrair) >= MAX_URLS_EXTRACT:
            break

    if not urls_para_extrair:
        return [], resultados, urls_tentadas

    urls_tentadas.update(urls_para_extrair)
    raw = extract_urls(urls_para_extrair)
    validos = []
    falhos = []

    for item in raw:
        url = item.get("url", "")
        c = item.get("conteudo", "")
        if len(c) >= EXTRACT_MIN_CHARS:
            validos.append(item)
            print(f"      ✅ {url[:72]} ({len(c):,} chars)")
        else:
            falhos.append(url)
            print(f"      ✖  {url[:72]} (<{EXTRACT_MIN_CHARS} chars)")

    # Fallback
    if len(falhos) > len(validos) and queries_fallback:
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
                            print(f"      ✅ Fallback: {item.get('url','')[:72]}")
            if validos:
                break

    return validos, resultados, urls_tentadas

def _juiz_paragrafo(paragrafo_limpo: str, fontes: str) -> tuple:
    """Juiz de 3 níveis para parágrafo."""
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

    resp = llm_call(prompt, temperature=0.0).strip()

    nivel = "APROVADO"
    texto_final = paragrafo_limpo

    m_dec = re.search(r"DECIS[ÃA]O\s*:\s*(APROVADO|AJUSTADO|CORRIGIDO)", resp, re.IGNORECASE)
    if m_dec:
        nivel = m_dec.group(1).upper()

    m_txt = re.search(r"TEXTO\s*:\s*([\s\S]+)", resp, re.IGNORECASE)
    if m_txt:
        candidato = m_txt.group(1).strip()
        candidato = re.sub(r"^DECIS[ÃA]O\s*:.*\n?", "", candidato, flags=re.IGNORECASE).strip()
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

def _eh_paragrafo_verificavel(paragrafo: str) -> bool:
    """Retorna False para blocos que não precisam de verificação factual."""
    p = paragrafo.strip()
    if len(p) < 60:
        return False
    if p.startswith("#"): return False
    if p.startswith("```"): return False
    if p.startswith("$$"): return False
    if p.startswith("---"): return False
    if p.startswith("==="): return False
    if p.startswith("*Figura"): return False
    if p.startswith("!["): return False
    if p.startswith(">"): return False
    if p.startswith("```mermaid"): return False
    if re.match(r"^\s*[-*]\s", p): return False
    return True

def _buscar_chunks_para_paragrafo(
    paragrafo: str,
    corpus: CorpusMongoDB,
    corpus_prompt_completo: str,
) -> str:
    """Recupera chunks relevantes para verificar o parágrafo."""
    texto_sem_ancoras = re.sub(r'\[ÂNCORA:\s*"[^"]*"\]', "", paragrafo)
    texto_sem_ancoras = re.sub(r'\$\$[^$]+\$\$', "", texto_sem_ancoras)
    texto_sem_ancoras = re.sub(r'\$[^$]+\$', "", texto_sem_ancoras)
    texto_sem_ancoras = re.sub(r'\\\([^)]+\\\)', "", texto_sem_ancoras).strip()

    ancoras = _ANCORA_PATTERN.findall(paragrafo)
    ancoras_validas = [
        a for a in ancoras
        if len(a.strip()) >= 20
        and not re.match(r'^[\\\$\{\}\[\]_\^]+', a.strip())
    ]

    queries = ancoras_validas[:3] + ([texto_sem_ancoras[:200]] if texto_sem_ancoras else [])

    if (corpus._collection is not None) or (not queries):
        return corpus_prompt_completo[:3000]

    chunks_vistos = set()
    partes = []
    chars = 0
    JUIZ_MAX_CORPUS_CHARS = 3000

    for q in queries:
        q = q.strip()
        if not q:
            continue
        for chunk in corpus.query(q, top_k=5):
            chave = chunk.texto[:100]
            if chave in chunks_vistos:
                continue
            chunks_vistos.add(chave)

            bloco = (
                f"[FONTE {chunk.fonte_idx} | {chunk.url[:70]}]\n"
                f"{chunk.texto}\n\n"
            )
            if chars + len(bloco) > JUIZ_MAX_CORPUS_CHARS:
                break
            partes.append(bloco)
            chars += len(bloco)

    if partes:
        return "".join(partes)
    return corpus_prompt_completo[:JUIZ_MAX_CORPUS_CHARS]

def _verificar_e_corrigir_secao(
    texto_secao: str,
    corpus: CorpusMongoDB,
    corpus_prompt_completo: str,
    titulo: str,
) -> tuple:
    """Verifica parágrafo a parágrafo e retorna texto corrigido, log e stats."""
    blocos = re.split(r'\n{2,}', texto_secao.strip())

    resultado = []
    log_linhas = [f"### Verificação por Parágrafo — {titulo}\n"]
    stats = {"total": 0, "aprovados": 0, "ajustados": 0, "corrigidos": 0, "pulados": 0}

    for i, bloco in enumerate(blocos):
        bloco = bloco.strip()
        if not bloco:
            continue

        bloco_limpo = re.sub(r'\[ÂNCORA:\s*"[^"]*"\]', "", bloco).strip()
        bloco_limpo = re.sub(r'  +', ' ', bloco_limpo)

        if not _eh_paragrafo_verificavel(bloco_limpo):
            resultado.append(bloco_limpo)
            stats["pulados"] += 1
            continue

        stats["total"] += 1

        fontes = _buscar_chunks_para_paragrafo(bloco, corpus, corpus_prompt_completo)

        if not fontes.strip():
            resultado.append(bloco_limpo)
            stats["aprovados"] += 1
            log_linhas.append(f"  ⏭️  par.{i+1} [SEM FONTES → APROVADO]: {bloco_limpo[:60]}...")
            continue

        texto_final, nivel, log_entry = _juiz_paragrafo(bloco_limpo, fontes)
        resultado.append(texto_final)
        log_linhas.append(f"  par.{i+1}: {log_entry}")

        if nivel == "APROVADO":
            stats["aprovados"] += 1
        elif nivel == "AJUSTADO":
            stats["ajustados"] += 1
        else:
            stats["corrigidos"] += 1

    verificados = stats["aprovados"] + stats["ajustados"]
    taxa = (verificados / stats["total"] * 100) if stats["total"] > 0 else 100.0

    log_linhas.append(
        f"\n**Resultado:** {verificados}/{stats['total']} verificados ({taxa:.0f}%) "
        f"— {stats['aprovados']} aprovados, {stats['ajustados']} ajustados, "
        f"{stats['corrigidos']} corrigidos | {stats['pulados']} pulados"
    )

    texto_corrigido = "\n\n".join(p for p in resultado if p)
    texto_corrigido = re.sub(r'\[ÂNCORA:\s*"[^"]*"\]', "", texto_corrigido)
    texto_corrigido = re.sub(r'\n{3,}', '\n\n', texto_corrigido)

    print(f"     📊 {verificados}/{stats['total']} verificados ({taxa:.0f}%) "
          f"— ✅{stats['aprovados']} aprovados  🔵{stats['ajustados']} ajustados  "
          f"🔧{stats['corrigidos']} corrigidos  ⏭️{stats['pulados']} pulados")

    return texto_corrigido, "\n".join(log_linhas), stats

# --- Nós do grafo ---

def parsear_plano_node(state: EscritaTecnicaState) -> dict:
    caminho = state["caminho_plano"]
    print(f"\n📖 Lendo plano: {caminho}")
    with open(caminho, "r", encoding="utf-8") as f:
        texto = f.read()
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
    # Criar um objeto CorpusMongoDB apenas para verificação de URLs (sem build)
    corpus_check = CorpusMongoDB() 

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
        plano = _fase_pensamento(tema, titulo, cont_esp, recursos)
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

        for q in queries[:4]:
            res = search_web(q, TECNICO_MAX_RESULTS)
            novos, resultados, urls_vistas = _extrair_com_fallback(
                res,
                queries_fallback=[q, titulo],
                urls_tentadas=urls_vistas,
                corpus=corpus_check,
            )
            extraidos.extend(novos)
            time.sleep(1)

        log.append(f"Fontes extraídas: {len(extraidos)}")

        # Indexação MongoDB
        print(f"\n  🗄️  Indexando no MongoDB...")
        log.append("\n── INDEXAÇÃO MONGODB ──")
        slug_secao = re.sub(r"[^\w]", "_", titulo[:30]).lower()
        prefixo = f"s{pos+1:02d}_{slug_secao}"
        corpus = CorpusMongoDB().build(extraidos, resultados, prefixo=prefixo)

        query_retrieval = f"{titulo} {cont_esp} {recursos}"
        corpus_prompt, urls_secao, _ = corpus.render_prompt(
            query_retrieval, max_chars=MAX_CORPUS_PROMPT
        )
        log.append(f"MongoDB: {corpus._n_docs} docs | {corpus._total_chunks} chunks")

        if not corpus_prompt.strip():
            print("  ⚠️  Corpus vazio! Busca de último recurso...")
            log.append("⚠️  Corpus vazio — busca de emergência")
            q_emerg = f"{titulo} {tema} technical documentation filetype:pdf"
            res_emerg = search_web(q_emerg, 6)
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

        if not corpus_prompt.strip():
            print("  ❌ FALHA CRÍTICA: nenhuma fonte encontrada.")
            log.append("❌ Nenhuma fonte encontrada.")
            corpus_prompt = (
                "AVISO: Nenhuma fonte encontrada. Escreva apenas conceitos "
                "amplamente estabelecidos, sem afirmações específicas com âncoras."
            )

        # FASE 5: Observação
        print(f"\n  🔬 FASE 5 — Observação...")
        log.append("\n── FASE 5: OBSERVAÇÃO ──")
        obs = _fase_observacao(informacoes, corpus)
        log.extend([f"Suficiente: {obs.get('suficiente')}",
                    f"Lacunas: {obs.get('lacunas', [])}"])

        for iter_n in range(1, MAX_REACT_ITERATIONS + 1):
            if obs.get("suficiente", True):
                break
            q_comp = obs.get("query_complementar") or ""
            if not q_comp:
                break
            print(f"\n  🔄 ITERAÇÃO {iter_n}: '{q_comp}'")
            log.append(f"\n── ITERAÇÃO {iter_n}: '{q_comp}' ──")
            res_comp = search_web(q_comp, max_results=8)
            for r in res_comp:
                u = r.get("url", "")
                if u and u not in urls_vistas:
                    urls_vistas.add(u)
                    resultados.append(r)
            novos, resultados, urls_vistas = _extrair_com_fallback(
                res_comp, 
                queries_fallback=[q_comp], 
                urls_tentadas=urls_vistas,
                corpus=corpus_check, 
            )
            extraidos.extend(novos)
            corpus = CorpusMongoDB().build(extraidos, resultados, prefixo=prefixo)
            corpus_prompt, urls_secao, _ = corpus.render_prompt(
                query_retrieval, max_chars=MAX_CORPUS_PROMPT
            )
            obs = _fase_observacao(informacoes, corpus)
            log.append(f"Suficiente após iter: {obs.get('suficiente')}")
            time.sleep(2)

        # Imagens
        print(f"\n  🖼️  Buscando imagens...")
        imagens = search_images(queries_img)
        img_txt = ""
        for i, img in enumerate(imagens, 1):
            url_img = img.get("url_imagem", "")
            desc = img.get("descricao", "") or "(sem descrição)"
            origem = img.get("titulo_pagina", img.get("url_origem", ""))
            img_txt += f"  [{i}] {url_img}\n       Desc: {desc}\n       Fonte: {origem}\n"

        referencia_completa = corpus_prompt
        if img_txt:
            referencia_completa += f"\n\nIMAGENS DISPONÍVEIS:\n{img_txt}"

        # FASE 6: Rascunho ancorado
        print(f"\n  ✍️  FASE 6 — Rascunho ancorado...")
        log.append("\n── FASE 6: RASCUNHO ──")
        rascunho, _ = _fase_rascunho(
            tema, titulo, cont_esp, recursos, corpus, img_txt,
            resumo_acumulado, pos, n_total, titulos_todos, len(extraidos)
        )
        n_ancoras = len(_ANCORA_PATTERN.findall(rascunho))
        log.append(f"Rascunho: {len(rascunho):,} chars | {n_ancoras} âncoras (hints)")
        print(f"     {len(rascunho):,} chars | {n_ancoras} âncoras")

        # FASE 7: Verificação por parágrafo
        print(f"\n  🔍 FASE 7 — Verificação por parágrafo...")
        log.append("\n── FASE 7: VERIFICAÇÃO POR PARÁGRAFO ──")
        texto_final, relatorio_verif, stats = _verificar_e_corrigir_secao(
            rascunho, corpus, referencia_completa, titulo
        )
        log.append(relatorio_verif)
        stats_verificacao.append({"secao": titulo, **stats})

        if not texto_final.strip().startswith("## "):
            texto_final = f"## {titulo}\n\n{texto_final.strip()}"

        verificados = stats.get("aprovados", 0) + stats.get("ajustados", 0)
        taxa = (verificados / stats["total"] * 100) if stats["total"] > 0 else 100
        if stats["total"] > 0 and taxa < 60:
            aviso = (
                f"> ⚠️ **Confiabilidade: {taxa:.0f}%** — apenas "
                f"{verificados}/{stats['total']} parágrafos verificados "
                f"({stats.get('corrigidos',0)} com erros factuais). "
                "Revisão manual recomendada.\n\n"
            )
            texto_final = re.sub(
                r"(## .+?\n)", r"\1\n" + aviso, texto_final, count=1
            )

        print(f"  ✅ [{pos+1}/{n_total}] Seção concluída ({taxa:.0f}% verificado)")

        secoes_escritas.append({
            "indice": idx_num,
            "titulo": titulo,
            "texto": texto_final,
            "urls_usadas": urls_secao,
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
    resp_intro = llm_call(
        f"Escreva a INTRODUÇÃO de um capítulo técnico sobre: '{tema}'.\n"
        f"Cobre: {', '.join(titulos)}.\n\n"
        "4 parágrafos: (1) motivação técnica, (2) pré-requisitos, "
        "(3) organização, (4) o que o leitor aprenderá. "
        "Máximo 600 palavras. Sem meta-texto.",
        temperature=0.3,
    )
    resp_concl = llm_call(
        f"Escreva a CONCLUSÃO de um capítulo técnico sobre: '{tema}'.\n\n"
        f"Síntese: {resumo_final}\n\n"
        "3 parágrafos: (1) síntese, (2) implicações e limitações, "
        "(3) direções futuras. Máximo 400 palavras.",
        temperature=0.3,
    )

    partes = [
        f"# {tema}\n",
        "> **Tipo:** Revisão Técnica\n",
        f"> **Verificação por parágrafo:** {total_verif}/{total_par} verificados "
        f"({taxa_global:.0f}%) — {total_aprov} aprovados, {total_ajust} ajustados, "
        f"{total_corr} corrigidos | "
        f"**Fontes:** {len(all_urls)} | **Seções:** {len(secoes)}\n",
        "\n---\n", "## Sumário\n", "- Introdução",
    ]
    for s in secoes:
        partes.append(f"- {s['titulo']}")
    partes += ["- Conclusão", "- Referências\n\n---\n",
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
            f"| {stats_s.get('aprovados',0)} aprovados, {aj_s} ajustados, "
            f"{r_s} corrigidos -->\n"
        )
        partes.append(s["texto"].strip())
        partes.append("\n\n---\n")

    partes += ["## Conclusão\n", resp_concl.strip(), "\n\n---\n", "## Referências\n"]
    if all_urls:
        for i, url in enumerate(all_urls[:80], 1):
            partes.append(f"[{i}] {url}")
    else:
        partes.append("*Nenhuma fonte utilizada.*")

    all_img_urls = [img.get("url_imagem", "") for img in all_imagens if img.get("url_imagem")]
    urls_inline = {img.get("url_imagem", "") for s in secoes for img in s.get("imagens", [])}
    orfas = [u for u in all_img_urls if u and u not in urls_inline]
    if orfas:
        partes.append("\n\n## Apêndice — Recursos Visuais\n")
        for i, img_url in enumerate(orfas, 1):
            partes.append(f"![Figura extra {i}]({img_url})\n*Figura extra {i}*\n")

    documento = "\n".join(partes)
    slug = re.sub(r"[^\w\s-]", "", tema[:40]).strip().replace(" ", "_").lower()
    output_path = f"revisao_tecnica_{slug}.md"
    log_path = f"revisao_tecnica_{slug}.log"

    try:
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
                f"  [{a}/{t} = {tx:.0f}% | {s.get('aprovados',0)} aprov "
                f"{aj} ajust {r} corrig] {s.get('secao','?')[:55]}"
            )
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(cabecalho + [""] + react_log))
        print(f"📋 {log_path}")
    except Exception as e:
        print(f"⚠️  Erro ao salvar log: {e}")

    return {"status": "concluido"}