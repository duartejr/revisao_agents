import time
from typing import List, Dict
from ..config import MAX_IMAGENS_SECAO

def extract(urls: List[str]) -> List[dict]:
    """Extrai texto completo de URLs via Tavily Extract."""
    if not urls:
        return []
    try:
        from tools.tavily_tool import extract_tavily
        res = extract_tavily.invoke({"urls": urls, "incluir_imagens": True})
        return res.get("extraidos", [])
    except Exception as e:
        print(f"   ⚠️  extract: {e}")
        return []

def search_images(queries: List[str]) -> List[dict]:
    """Busca imagens via tool dedicada."""
    try:
        from tools.tavily_tool import search_tavily_images
        res = search_tavily_images.invoke({"queries": queries, "max_results": 8})
        return res.get("imagens", [])[:MAX_IMAGENS_SECAO]
    except Exception as e:
        print(f"   ⚠️  search_images: {e}")
        return []