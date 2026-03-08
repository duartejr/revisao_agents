# src/revisao_agents/tools/tavily_web_search.py
"""
Tavily Web Search Tools — versão migrada para o novo pacote.
Mantém todas as funcionalidades originais (rastreabilidade, idioma, filtros acadêmicos).
"""

from langchain_core.tools import tool   # ← atualizado (melhor prática 2026)
from typing import List, Optional, Dict
import os
import re
import json
from datetime import datetime

# Imports relativos ao novo pacote (serão ajustados quando migrarmos utils/)
from ..utils.commons import get_clean_key   # ← ajuste automático na próxima etapa
from tavily import TavilyClient


# ============================================================================
# PASTA DE RASTREABILIDADE
# ============================================================================

_SEARCH_LOG_DIR = "tavily_searchs"


def _garantir_pasta_log():
    """Cria a pasta tavily_searchs se não existir."""
    os.makedirs(_SEARCH_LOG_DIR, exist_ok=True)


def _slug(texto: str, max_chars: int = 50) -> str:
    """Gera slug seguro para usar como nome de arquivo."""
    s = re.sub(r"[^\w\s-]", "", texto[:max_chars]).strip()
    s = re.sub(r"[\s]+", "_", s).lower()
    return s or "busca"


def _salvar_pesquisa_md(
    tipo: str,
    query: str,
    resultados: List[dict],
    extra: Optional[dict] = None,
) -> str:
    """
    Salva os resultados de uma busca Tavily em arquivo Markdown.

    Args:
        tipo       : tipo da busca (academica, tecnica, imagens, extract)
        query      : query ou URL pesquisada
        resultados : lista de resultados (cada item é um dict)
        extra      : informações adicionais opcionais (ex: urls_encontrados)

    Returns:
        Caminho do arquivo salvo.
    """
    _garantir_pasta_log()

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ms precision
    slug_q   = _slug(query)
    filename = f"{_SEARCH_LOG_DIR}/{ts}_{tipo}_{slug_q}.md"

    linhas = [
        f"# Pesquisa Tavily — {tipo.upper()}",
        f"",
        f"- **Data/Hora:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Tipo:** {tipo}",
        f"- **Query:** `{query}`",
        f"- **Total de resultados:** {len(resultados)}",
    ]

    if extra:
        for k, v in extra.items():
            linhas.append(f"- **{k}:** {v}")

    linhas += ["", "---", ""]

    for i, r in enumerate(resultados, 1):
        if isinstance(r, dict):
            url      = r.get("url", r.get("source", ""))
            titulo   = r.get("title", r.get("titulo", ""))
            snippet  = r.get("snippet", r.get("content", r.get("conteudo", "")))
            score    = r.get("score", "")
            idioma   = r.get("language", r.get("idioma", ""))
            imagens  = r.get("imagens", r.get("images", []))
            descr    = r.get("image_descriptions", {})

            linhas.append(f"## [{i}] {titulo or url}")
            linhas.append(f"")
            if url:
                linhas.append(f"**URL:** {url}")
            if score:
                linhas.append(f"**Score:** {score:.4f}" if isinstance(score, float) else f"**Score:** {score}")
            if idioma:
                linhas.append(f"**Idioma:** {idioma}")
            if snippet:
                linhas.append(f"")
                linhas.append(f"**Conteúdo:**")
                linhas.append(f"")
                linhas.append(snippet[:2000])
            if imagens:
                linhas.append(f"")
                linhas.append(f"**Imagens encontradas ({len(imagens)}):**")
                for img in imagens:
                    desc = descr.get(img, "") if isinstance(descr, dict) else ""
                    if desc:
                        linhas.append(f"- `{img}` — {desc}")
                    else:
                        linhas.append(f"- `{img}`")
            linhas.append(f"")
            linhas.append(f"---")
            linhas.append(f"")
        else:
            # fallback para strings (ex: lista de URLs)
            linhas.append(f"- {r}")

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(linhas))
        print(f"   📝 Log salvo: {filename}")
    except Exception as e:
        print(f"   ⚠️  Não foi possível salvar log: {e}")

    return filename


# ============================================================================
# DETECÇÃO DE IDIOMA
# ============================================================================

def _detectar_idioma(texto: str) -> str:
    """
    Detecta se o texto é predominantemente em inglês ou português.
    Retorna 'en' ou 'pt'.
    """
    if not texto:
        return 'en'
    
    texto_lower = texto.lower()
    
    # Palavras comuns em português
    palavras_pt = ['para', 'como', 'que', 'com', 'mais', 'dos', 'das', 'pela', 'pelo', 
                   'são', 'foi', 'está', 'sobre', 'entre', 'através', 'também', 'ser',
                   'por', 'uma', 'seus', 'suas', 'este', 'esta', 'pode', 'podem']
    
    # Palavras comuns em inglês
    palavras_en = ['the', 'and', 'for', 'with', 'this', 'from', 'that', 'have',
                   'was', 'are', 'been', 'their', 'which', 'were', 'when', 'through',
                   'where', 'using', 'can', 'these', 'those', 'such', 'would', 'should']
    
    # Conta ocorrências
    count_pt = sum(1 for p in palavras_pt if f' {p} ' in f' {texto_lower} ')
    count_en = sum(1 for p in palavras_en if f' {p} ' in f' {texto_lower} ')
    
    # Também verifica caracteres especiais do português
    if 'ã' in texto_lower or 'ç' in texto_lower or 'õ' in texto_lower:
        count_pt += 3
    
    return 'en' if count_en >= count_pt else 'pt'


def _priorizar_por_idioma(resultados: List[dict], boost_en: float = 0.3) -> List[dict]:
    """
    Reordena resultados priorizando inglês.
    Adiciona boost ao score de resultados em inglês.
    
    Args:
        resultados: lista de resultados com 'score', 'title', 'snippet'
        boost_en: boost adicional para resultados em inglês (0.0 a 1.0)
    
    Returns:
        Lista de resultados reordenada e enriquecida com campo 'language'
    """
    for r in resultados:
        # Detecta idioma baseado no título + snippet
        texto_detectar = f"{r.get('title', '')} {r.get('snippet', r.get('content', ''))}"
        idioma = _detectar_idioma(texto_detectar)
        r['language'] = idioma
        
        # Adiciona boost para inglês
        if idioma == 'en':
            r['score'] = min(1.0, r.get('score', 0) + boost_en)
    
    # Reordena: inglês primeiro, depois por score
    resultados_ordenados = sorted(
        resultados,
        key=lambda x: (x.get('language', 'en') != 'en', -x.get('score', 0))
    )
    
    return resultados_ordenados


# ============================================================================
# DOMÍNIOS BLOQUEADOS
# ============================================================================

BLOCKED_DOMAINS = [
           "wikipedia.org",                    "wikipedia.com",                       "scribd.com",        "lonepatient.top",          
            "linkedin.com",                     "facebook.com",                      "twitter.com",          "instagram.com",
             "youtube.com",                       "reddit.com",                        "quora.com",      "stackexchange.com",
       "stackoverflow.com",                         "ebay.com",                   "aliexpress.com",               "etsy.com",
          "arxivdaily.com",            "answers.microsoft.com",              "merriam-webster.com",         "dictionary.com",
           "thesaurus.com",             "news.ycombinator.com",            "collinsdictionary.com", "oxforddictionaries.com", 
   "thefreedictionary.com",         "dictionary.cambridge.org", "education.nationalgeographic.com",         "britannica.com", 
       "worldometers.info",                     "statista.com",               "ourworldindata.org",           "chrono24.com", 
         "rankinggods.com",                        "theoi.com",                       "tiktok.com",          "pinterest.com", 
              "zantia.com",              "analisemacro.com.br",                     "ibram.org.br", "beacademy.substack.com", 
                  "gov.br",           "blog.dsacademy.com.br/",                   "mariofilho.com",      "pt.hyee-ct-cv.com", 
           "chatpaper.com",                "flusshidro.com.br",                         "otca.org",       "ler.letras.up.pt",
             "oreilly.com",                       "neurips.cc",          "conference.ifas.ufl.edu", "atrium.lib.uoguelph.ca",
           "datadoghq.com",                          "kumo.ai",                      "hydroai.net",         "geoawesome.com",
            "blogs.egu.eu",
]


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def _get_client() -> TavilyClient:
    return TavilyClient(api_key=get_clean_key("TAVILY_API_KEY"))


def filtrar_urls_academicas(urls: List[str]) -> List[str]:
    filtradas = [
        url for url in urls
        if not any(b.lower() in url.lower() for b in BLOCKED_DOMAINS)
    ]
    removidos = len(urls) - len(filtradas)
    if removidos:
        print(f"   🚫 Removidas {removidos} URLs de fontes não acadêmicas")
    return filtradas


def filtrar_urls_tecnicas(urls: List[str]) -> List[str]:
    return [
        url for url in urls
        if not any(b.lower() in url.lower() for b in BLOCKED_DOMAINS)
    ]


# ============================================================================
# BUSCA ACADÊMICA COM PRIORIZAÇÃO DE INGLÊS
# ============================================================================

@tool
def search_tavily(queries: List[str], max_results: int = 5) -> dict:
    """
    Busca artigos acadêmicos no Tavily.
    PRIORIZA conteúdo em INGLÊS, mas permite português.
    Filtra domínios não científicos automaticamente.
    Salva cada busca em ./tavily_searchs/ para rastreabilidade.

    Args:
        queries: lista de queries de busca
        max_results: resultados por query (padrão 5)

    Returns:
        {"urls_encontrados": [...], "resultados": [...]}
    """
    client = _get_client()
    all_urls: List[str] = []
    all_results: List[dict] = []

    for q in queries:
        print(f"🔎 Buscando (acadêmico, EN priorizado): {q}")
        
        # ESTRATÉGIA: Busca dupla - primeiro inglês, depois complementa
        resultados_batch = []
        
        try:
            # FASE 1: Busca prioritária em inglês
            res_en = client.search(
                query=q,
                search_depth="advanced",
                max_results=max_results,
                exclude_domains=BLOCKED_DOMAINS,
            )
            
            for r in res_en.get("results", []):
                if r.get("score", 0) < 0.7:
                    continue
                
                item = {
                    "url":     r["url"],
                    "title":   r.get("title", ""),
                    "snippet": r.get("content", "")[:300],
                    "score":   r.get("score", 0),
                }
                resultados_batch.append(item)
            
            # Prioriza por idioma (detecta e dá boost para inglês)
            resultados_batch = _priorizar_por_idioma(resultados_batch, boost_en=0.3)
            
            # Adiciona aos resultados globais
            for item in resultados_batch:
                all_urls.append(item["url"])
                all_results.append(item)
            
            # Log estatísticas de idioma
            n_en = sum(1 for r in resultados_batch if r.get('language') == 'en')
            n_pt = sum(1 for r in resultados_batch if r.get('language') == 'pt')
            print(f"   📊 Idiomas: {n_en} inglês, {n_pt} português")
            
            # ── Salva log desta query ────────────────────────────────────────
            _salvar_pesquisa_md("academica", q, resultados_batch, 
                               extra={"idioma_en": n_en, "idioma_pt": n_pt})

        except Exception as e:
            print(f"   ⚠️  Erro na query '{q[:50]}': {e}")

    urls_unicos = list(dict.fromkeys(all_urls))
    
    # Estatísticas finais
    total_en = sum(1 for r in all_results if r.get('language') == 'en')
    total_pt = sum(1 for r in all_results if r.get('language') == 'pt')
    print(f"   📊 TOTAL: {total_en} inglês ({total_en/len(all_results)*100:.0f}%), "
          f"{total_pt} português ({total_pt/len(all_results)*100:.0f}%)")
    
    return {"urls_encontrados": urls_unicos, "resultados": all_results}


# ============================================================================
# BUSCA ACADÊMICA INCREMENTAL
# ============================================================================

def search_tavily_incremental(
    query: str,
    urls_anteriores: List[str],
    max_results: int = 5,
) -> dict:
    """
    Busca acadêmica incremental — acumula URLs sem duplicatas.
    PRIORIZA conteúdo em INGLÊS.
    Salva log de cada busca em ./tavily_searchs/.

    Returns:
        {"urls_novos": [...], "total_acumulado": [...]}
    """
    try:
        client = _get_client()
        print(f"\n🔎 Busca Incremental (EN priorizado): '{query}'")

        res = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            exclude_domains=BLOCKED_DOMAINS,
        )
        
        # Prepara resultados
        batch_results = [
            {
                "url":     r["url"],
                "title":   r.get("title", ""),
                "snippet": r.get("content", "")[:2000],
                "score":   r.get("score", 0),
            }
            for r in res.get("results", []) if r.get("score", 0) >= 0.7
        ]
        
        # Prioriza por idioma
        batch_results = _priorizar_por_idioma(batch_results, boost_en=0.3)
        
        urls_encontrados = [r["url"] for r in batch_results]
        urls_encontrados = filtrar_urls_academicas(urls_encontrados)

        urls_novos      = [u for u in urls_encontrados if u not in urls_anteriores]
        total_acumulado = list(dict.fromkeys(urls_anteriores + urls_encontrados))

        # Estatísticas
        n_en = sum(1 for r in batch_results if r.get('language') == 'en')
        n_pt = sum(1 for r in batch_results if r.get('language') == 'pt')
        
        print(f"   ✔ Encontrados : {len(urls_encontrados)} URLs")
        print(f"   ✔ Novos       : {len(urls_novos)} URLs")
        print(f"   ✔ Total acum. : {len(total_acumulado)} URLs")
        print(f"   📊 Idiomas    : {n_en} inglês, {n_pt} português")

        # ── Log ──────────────────────────────────────────────────────────────
        _salvar_pesquisa_md(
            "academica_incremental",
            query,
            batch_results,
            extra={
                "urls_novos": len(urls_novos), 
                "total_acumulado": len(total_acumulado),
                "idioma_en": n_en,
                "idioma_pt": n_pt,
            },
        )

        return {"urls_novos": urls_novos, "total_acumulado": total_acumulado}

    except Exception as e:
        print(f"   ⚠️  Erro na busca Tavily: {e}")
        return {"urls_novos": [], "total_acumulado": urls_anteriores}


# ============================================================================
# BUSCA TÉCNICA COM PRIORIZAÇÃO DE INGLÊS
# ============================================================================

@tool
def search_tavily_tecnico(queries: List[str], max_results: int = 5) -> dict:
    """
    Busca técnica no Tavily — permite documentações, tutoriais,
    Wikipedia em inglês, livros online, páginas de referência, etc.
    PRIORIZA conteúdo em INGLÊS.
    Salva cada busca em ./tavily_searchs/ para rastreabilidade.

    Args:
        queries: lista de queries de busca
        max_results: resultados por query (padrão 5)

    Returns:
        {"urls_encontrados": [...], "resultados": [...]}
    """
    client = _get_client()
    all_urls: List[str] = []
    all_results: List[dict] = []

    for q in queries:
        print(f"🔎 Buscando (técnico, EN priorizado): {q}")
        
        try:
            res = client.search(
                query=q[:400],
                search_depth="advanced",
                max_results=max_results,
                exclude_domains=BLOCKED_DOMAINS,
            )
            
            batch_results = []
            for r in res.get("results", []):
                if r.get("score", 0) < 0.7:
                    continue
                
                item = {
                    "url":     r["url"],
                    "title":   r.get("title", ""),
                    "snippet": r.get("content", "")[:500],
                    "score":   r.get("score", 0),
                }
                batch_results.append(item)
            
            # Prioriza por idioma
            batch_results = _priorizar_por_idioma(batch_results, boost_en=0.3)
            
            for item in batch_results:
                all_urls.append(item["url"])
                all_results.append(item)
            
            # Estatísticas
            n_en = sum(1 for r in batch_results if r.get('language') == 'en')
            n_pt = sum(1 for r in batch_results if r.get('language') == 'pt')
            print(f"   📊 Idiomas: {n_en} inglês, {n_pt} português")

            # ── Log ──────────────────────────────────────────────────────────
            _salvar_pesquisa_md("tecnica", q, batch_results,
                               extra={"idioma_en": n_en, "idioma_pt": n_pt})

        except Exception as e:
            print(f"   ⚠️  Erro na query '{q[:50]}': {e}")

    urls_unicos = list(dict.fromkeys(all_urls))
    urls_filtrados = filtrar_urls_tecnicas(urls_unicos)
    
    # Estatísticas finais
    total_en = sum(1 for r in all_results if r.get('language') == 'en')
    total_pt = sum(1 for r in all_results if r.get('language') == 'pt')
    if all_results:
        print(f"   📊 TOTAL: {total_en} inglês ({total_en/len(all_results)*100:.0f}%), "
              f"{total_pt} português ({total_pt/len(all_results)*100:.0f}%)")
    
    return {"urls_encontrados": urls_filtrados, "resultados": all_results}


# ============================================================================
# BUSCA COM IMAGENS — tool dedicada
# ============================================================================

@tool
def search_tavily_images(
    queries: List[str],
    max_results: int = 8,
) -> dict:
    """
    Busca imagens relacionadas a um tema via Tavily.
    Retorna URLs de imagens com suas descrições quando disponíveis.
    Salva log completo em ./tavily_searchs/.

    Ideal para:
      - Figuras de algoritmos, arquiteturas, diagramas de fluxo
      - Gráficos comparativos de métricas
      - Visualizações de séries temporais / hidrologia
      - Ilustrações técnicas ou científicas

    Args:
        queries    : lista de queries de busca orientadas a imagens
        max_results: resultados por query (padrão 8)

    Returns:
        {
          "imagens": [
            {
              "url_imagem"  : str,   # URL direta da imagem
              "descricao"   : str,   # descrição gerada pelo Tavily (se disponível)
              "url_origem"  : str,   # página onde a imagem foi encontrada
              "titulo_pagina": str,
            }, ...
          ],
          "total": int,
        }
    """
    client = _get_client()
    todas_imagens: List[dict] = []
    vistas: set = set()

    for q in queries:
        print(f"🖼️  Buscando imagens: {q}")
        try:
            res = client.search(
                query=q[:400],
                search_depth="advanced",
                max_results=max_results,
                include_images=True,
                include_image_descriptions=True,
                exclude_domains=BLOCKED_DOMAINS,
            )

            # ── Imagens diretas retornadas pelo Tavily ───────────────────────
            imagens_raw   = res.get("images", [])
            descricoes_raw = res.get("image_descriptions", {})

            # Normaliza: pode ser lista de strings ou lista de dicts
            for item in imagens_raw:
                if isinstance(item, dict):
                    url_img = item.get("url", "")
                    desc    = item.get("description", "")
                else:
                    url_img = str(item)
                    desc    = descricoes_raw.get(url_img, "") if isinstance(descricoes_raw, dict) else ""

                if not url_img or url_img in vistas:
                    continue
                if not any(url_img.lower().endswith(ext)
                           for ext in (".jpg", ".jpeg", ".png", ".svg", ".gif", ".webp")):
                    # Aceita URLs sem extensão explícita também (ex: CDN)
                    if "image" not in url_img.lower() and not re.search(r"\.(jpg|jpeg|png|svg|gif|webp)", url_img, re.I):
                        continue

                vistas.add(url_img)
                todas_imagens.append({
                    "url_imagem":    url_img,
                    "descricao":     desc,
                    "url_origem":    "",
                    "titulo_pagina": "",
                })

            # ── Imagens dos snippets de resultado ────────────────────────────
            for r in res.get("results", []):
                url_pag   = r.get("url", "")
                titulo_pag = r.get("title", "")
                for img_url in r.get("images", []):
                    if img_url and img_url not in vistas:
                        vistas.add(img_url)
                        todas_imagens.append({
                            "url_imagem":    img_url,
                            "descricao":     "",
                            "url_origem":    url_pag,
                            "titulo_pagina": titulo_pag,
                        })

            # ── Log ──────────────────────────────────────────────────────────
            _salvar_pesquisa_md(
                "imagens",
                q,
                [
                    {
                        "url":     img["url_imagem"],
                        "title":   img["titulo_pagina"],
                        "snippet": img["descricao"],
                        "imagens": [img["url_imagem"]],
                    }
                    for img in todas_imagens
                ],
                extra={"total_imagens": len(todas_imagens)},
            )

        except Exception as e:
            print(f"   ⚠️  Erro na query de imagens '{q[:50]}': {e}")

    print(f"   🖼️  Total de imagens encontradas: {len(todas_imagens)}")
    return {"imagens": todas_imagens, "total": len(todas_imagens)}


# ============================================================================
# EXTRACT — extrai conteúdo completo de URLs
# ============================================================================

@tool
def extract_tavily(urls: List[str], incluir_imagens: bool = True) -> dict:
    """
    Extrai conteúdo completo de páginas web via Tavily Extract API.
    Salva log completo em ./tavily_searchs/.

    Args:
        urls           : lista de URLs para extrair (máx. 20 por chamada)
        incluir_imagens: se True, inclui URLs de imagens encontradas

    Returns:
        {
          "extraidos": [
            {
              "url"     : str,
              "titulo"  : str,
              "conteudo": str,
              "imagens" : [str],
            }, ...
          ],
          "falhos": [str],
        }
    """
    client = _get_client()
    extraidos: List[dict] = []
    falhos:    List[str]  = []

    lotes = [urls[i:i+20] for i in range(0, len(urls), 20)]

    for lote in lotes:
        print(f"📥 Extraindo {len(lote)} URL(s)...")
        try:
            res = client.extract(
                urls=lote,
                extract_depth="advanced",
                include_images=incluir_imagens,
            )

            for item in res.get("results", []):
                url      = item.get("url", "")
                conteudo = item.get("raw_content", item.get("content", ""))
                imagens  = item.get("images", []) if incluir_imagens else []

                extraidos.append({
                    "url":      url,
                    "titulo":   item.get("title", ""),
                    "conteudo": conteudo,
                    "imagens":  imagens,
                })
                print(f"   ✔ {url[:60]} — {len(conteudo):,} chars"
                      f"{f', {len(imagens)} img(s)' if imagens else ''}")

            for item in res.get("failed_results", []):
                falhos.append(item.get("url", ""))
                print(f"   ✖ Falhou: {item.get('url','')[:60]}")

        except Exception as e:
            print(f"   ⚠️  Erro no lote: {e}")
            falhos.extend(lote)

    # ── Log ──────────────────────────────────────────────────────────────────
    query_repr = urls[0] if urls else "extract"
    _salvar_pesquisa_md(
        "extract",
        query_repr,
        [
            {
                "url":     e["url"],
                "title":   e["titulo"],
                "snippet": e["conteudo"],#[:3000],
                "imagens": e["imagens"],
            }
            for e in extraidos
        ],
        extra={"urls_solicitadas": len(urls), "extraidas": len(extraidos), "falhas": len(falhos)},
    )

    return {"extraidos": extraidos, "falhos": falhos}


# ============================================================================
# BUSCA TÉCNICA INCREMENTAL (uso direto pelos nós do grafo)
# ============================================================================

def search_tavily_tecnico_incremental(
    query: str,
    urls_anteriores: List[str],
    max_results: int = 8,
) -> dict:
    """
    Busca técnica incremental — acumula URLs sem duplicatas.
    PRIORIZA conteúdo em INGLÊS.
    Salva log em ./tavily_searchs/.

    Returns:
        {"urls_novos": [...], "total_acumulado": [...], "resultados": [...]}
    """
    try:
        client = _get_client()
        print(f"\n🔎 Busca Técnica Incremental (EN priorizado): '{query}'")

        res = client.search(
            query=query[:400],
            search_depth="advanced",
            max_results=max_results,
            exclude_domains=BLOCKED_DOMAINS,
        )
        
        resultados = [
            {
                "url":     r["url"],
                "title":   r.get("title", ""),
                "snippet": r.get("content", "")[:2000],
                "score":   r.get("score", 0),
            }
            for r in res.get("results", []) if r.get("score", 0) >= 0.7
        ]
        
        # Prioriza por idioma
        resultados = _priorizar_por_idioma(resultados, boost_en=0.3)
        
        todos = [r["url"] for r in resultados]
        todos = filtrar_urls_tecnicas(todos)

        urls_novos      = [u for u in todos if u not in urls_anteriores]
        total_acumulado = list(dict.fromkeys(urls_anteriores + todos))
        
        # Estatísticas
        n_en = sum(1 for r in resultados if r.get('language') == 'en')
        n_pt = sum(1 for r in resultados if r.get('language') == 'pt')

        print(f"   ✔ Encontrados : {len(todos)} URLs")
        print(f"   ✔ Novos       : {len(urls_novos)} URLs")
        print(f"   ✔ Total acum. : {len(total_acumulado)} URLs")
        print(f"   📊 Idiomas    : {n_en} inglês, {n_pt} português")

        # ── Log ──────────────────────────────────────────────────────────────
        _salvar_pesquisa_md(
            "tecnica_incremental",
            query,
            resultados,
            extra={
                "urls_novos": len(urls_novos), 
                "total_acumulado": len(total_acumulado),
                "idioma_en": n_en,
                "idioma_pt": n_pt,
            },
        )

        return {
            "urls_novos":      urls_novos,
            "total_acumulado": total_acumulado,
            "resultados":      resultados,
        }

    except Exception as e:
        print(f"   ⚠️  Erro na busca técnica: {e}")
        return {
            "urls_novos":      [],
            "total_acumulado": urls_anteriores,
            "resultados":      [],
        }