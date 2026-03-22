from typing import List
from ...config import MAX_IMAGES_SECTION


def extract(urls: List[str]) -> List[dict]:
    """Extract full text from URLs using Tavily Extract.

    Args:
        urls: A list of URLs to extract content from.

    Returns:
        A list of dictionaries, each containing 'url', 'title', and 'content' keys.
    """
    if not urls:
        return []
    try:
        from ...tools.tavily_web_search import extract_tavily

        res = extract_tavily.invoke({"urls": urls, "include_images": True})
        return res.get("extracted", [])
    except Exception as e:
        print(f"   ⚠️  extract: {e}")
        return []


def search_images(queries: List[str]) -> List[dict]:
    """Search for images using a dedicated tool.

    Args:
        queries: A list of query strings to search for images.

    Returns:
        A list of dictionaries, each containing 'url', 'title', and 'snippet' keys.
    """
    try:
        from ...tools.tavily_web_search import search_tavily_images

        res = search_tavily_images.invoke({"queries": queries, "max_results": 8})
        return res.get("images", [])[:MAX_IMAGES_SECTION]
    except Exception as e:
        print(f"   ⚠️  search_images: {e}")
        return []
