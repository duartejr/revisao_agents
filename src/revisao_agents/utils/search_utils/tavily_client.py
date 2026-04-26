from ...config import BLOCKED_DOMAINS_EXTRACT, PRIORITY_DOMAINS, TECHNICAL_MAX_RESULTS


def search_technical_content(query: str, previous_urls: list[str]) -> dict:
    """
    Performs technical search via Tavily (incremental).

    Args:
        query: The technical query string.
        previous_urls: List of URLs already retrieved in previous iterations (to avoid duplicates).

    Returns:
        A dictionary with keys 'new_urls', 'total_accumulated', 'results', 'usage', 'urls_found'.
    """
    try:
        from ...tools.tavily_web_search import search_tavily_incremental_technician

        result = search_tavily_incremental_technician(
            query, previous_urls, max_results=TECHNICAL_MAX_RESULTS
        )
        return result
    except Exception as e:
        print("   Technical search failed: " + str(e))
        return {
            "new_urls": [],
            "total_accumulated": previous_urls,
            "results": [],
            "usage": {},
            "urls_found": [],
        }


def score_url(url: str, snippet: str = "", score_tavily: float = 0.0) -> float:
    """
    Priority score for extraction. General source quality criteria.

    Args:
        url: The URL to score.
        snippet: The text snippet extracted from the URL (if any).
        score_tavily: The relevance score returned by Tavily for this URL.

    Returns:
        A float score where higher means more likely to be a good source.
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


def search_web(query: str, max_results: int = TECHNICAL_MAX_RESULTS) -> list[dict]:
    """Technical search on Tavily and returns a list of results.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.

    Returns:
        A list of dictionaries, each containing 'url', 'title', and 'snippet' keys.
    """
    try:
        from ...tools.tavily_web_search import search_tavily_incremental_technician

        res = search_tavily_incremental_technician(query, [], max_results=max_results)
        return res.get("results", [])
    except Exception as e:
        print(f"   ⚠️  search_web('{query[:50]}'): {e}")
        return []


def search_images(queries: list[str], max_results: int = 8) -> list[dict]:
    """Image search via dedicated tool.

    Args:
        queries: A list of query strings to search for images.
        max_results: Maximum number of image results to return.

    Returns:
        A list of dictionaries, each containing 'url', 'title', and 'snippet' keys.
    """
    try:
        from ...tools.tavily_web_search import search_tavily_images

        res = search_tavily_images.invoke({"queries": queries, "max_results": max_results})
        return res.get("images", [])[:max_results]
    except Exception as e:
        print(f"   ⚠️  search_images: {e}")
        return []


def extract_urls(urls: list[str]) -> list[dict]:
    """Extract full page text from URLs and normalize the payload shape.

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
        normalized = []
        for item in res.get("extracted", []):
            normalized.append(
                {
                    "url": item.get("url", ""),
                    "title": item.get("title", item.get("title", "")),
                    "content": item.get("content", item.get("content", "")),
                }
            )
        return normalized
    except Exception as e:
        print(f"   ⚠️  extract: {e}")
        return []
