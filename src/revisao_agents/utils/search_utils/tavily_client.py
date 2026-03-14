import time
from typing import List, Dict
from ...config import TECHNICAL_MAX_RESULTS, PRIORITY_DOMAINS, BLOCKED_DOMAINS_EXTRACT

def buscar_conteudo_tecnico(query: str, urls_anteriores: List[str]) -> Dict:
    """
    Realiza busca técnica via Tavily (incremental).
    Retorna dicionário com 'urls_novos', 'total_acumulado', 'resultados'.
    """
    try:
        from ...tools.tavily_web_search import search_tavily_tecnico_incremental
        return search_tavily_tecnico_incremental(
            query, urls_anteriores, max_results=TECHNICAL_MAX_RESULTS
        )
    except Exception as e:
        print("   Busca tecnica falhou: " + str(e))
        return {"urls_novos": [], "total_acumulado": urls_anteriores, "resultados": []}

def score_url(url: str, snippet: str = "", score_tavily: float = 0.0) -> float:
    """
    Score de prioridade para extração. Critérios gerais de qualidade de fonte.
    """
    ul = url.lower()
    pts = score_tavily * 2.0

    for d in PRIORITY_DOMAINS:
        if d in ul:
            pts += 3.0
            break

    if ul.endswith(".pdf"):
        pts += 4.0
    if "doi.org" in ul:
        pts += 3.0
    if any(d in ul for d in BLOCKED_DOMAINS_EXTRACT):
        pts -= 10.0

    if len(snippet) > 400:
        pts += 1.0

    return pts

def search_web(query: str, max_results: int = TECHNICAL_MAX_RESULTS) -> List[dict]:
    """Busca técnica no Tavily e retorna lista de resultados."""
    try:
        from ...tools.tavily_web_search import search_tavily_tecnico_incremental
        res = search_tavily_tecnico_incremental(query, [], max_results=max_results)
        return res.get("resultados", [])
    except Exception as e:
        print(f"   ⚠️  search_web('{query[:50]}'): {e}")
        return []

def search_images(queries: List[str], max_results: int = 8) -> List[dict]:
    """Busca imagens via tool dedicada."""
    try:
        from ...tools.tavily_web_search import search_tavily_images
        res = search_tavily_images.invoke({"queries": queries, "max_results": max_results})
        return res.get("imagens", [])[:max_results]  # MAX_IMAGES_SECTION deve estar em config
    except Exception as e:
        print(f"   ⚠️  search_images: {e}")
        return []

def extract_urls(urls: List[str]) -> List[dict]:
    """Extract full page text from URLs and normalize the payload shape."""
    if not urls:
        return []
    try:
        from ...tools.tavily_web_search import extract_tavily
        res = extract_tavily.invoke({"urls": urls, "incluir_imagens": True})
        normalized = []
        for item in res.get("extraidos", []):
            normalized.append({
                "url": item.get("url", ""),
                "title": item.get("title", item.get("titulo", "")),
                "content": item.get("content", item.get("conteudo", "")),
            })
        return normalized
    except Exception as e:
        print(f"   ⚠️  extract: {e}")
        return []