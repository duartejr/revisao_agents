# src/revisao_agents/tools/tavily_web_search.py
"""
Tavily Web Search Tools — version migrated to the new package.
Retains all original functionalities (crawlability, language, academic filters).
"""

import os
import re
from datetime import datetime

from langchain_core.tools import tool
from tavily import TavilyClient

from ..utils.core.commons import get_clean_key

# ============================================================================
# PASTA DE RASTREABILIDADE
# ============================================================================

_SEARCH_LOG_DIR = "tavily_searchs"


def _guarantee_log_folder():
    """Create the tavily_searches folder if it doesn't already exist.

    Args:
        None (no input parameters)

    Returns:
        None (creates directory if needed)
    """
    os.makedirs(_SEARCH_LOG_DIR, exist_ok=True)


def _slug(text: str, max_chars: int = 50) -> str:
    """Generate a safe slug for use as a filename.

    Args:
        texto: the input string to slugify (e.g. a search query)
        max_chars: the maximum number of characters to include in the slug

    Returns:
        A slugified version of the input string.
    """
    s = re.sub(r"[^\w\s-]", "", text[:max_chars]).strip()
    s = re.sub(r"[\s]+", "_", s).lower()
    return s or "search"


def _save_search_md(
    type: str,
    query: str,
    results: list[dict],
    extra: dict | None = None,
) -> str:
    """
    Saves the results of a Tavily search to a Markdown file.

    Args:
        type       : type of the search (academic, technical, images, extract)
        query      : the query or URL searched
        results    : list of results (each item is a dict)
        extra      : optional additional information (e.g., found URLs) to include in the log

    Returns:
        The path to the saved file.
    """
    _guarantee_log_folder()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ms precision
    slug_q = _slug(query)
    filename = f"{_SEARCH_LOG_DIR}/{ts}_{type}_{slug_q}.md"

    lines = [
        f"# Tavily Search — {type.upper()}",
        "",
        f"- **Date/Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Type:** {type}",
        f"- **Query:** `{query}`",
        f"- **Total Results:** {len(results)}",
    ]

    if extra:
        for k, v in extra.items():
            lines.append(f"- **{k}:** {v}")

    lines += ["", "---", ""]

    for i, r in enumerate(results, 1):
        if isinstance(r, dict):
            url = r.get("url", r.get("source", ""))
            title = r.get("title", r.get("titulo", ""))
            snippet = r.get("snippet", r.get("content", r.get("conteudo", "")))
            score = r.get("score", "")
            language = r.get("language", r.get("idioma", ""))
            images = r.get("imagens", r.get("images", []))
            descr = r.get("image_descriptions", {})

            lines.append(f"## [{i}] {title or url}")
            lines.append("")
            if url:
                lines.append(f"**URL:** {url}")
            if score:
                lines.append(
                    f"**Score:** {score:.4f}" if isinstance(score, float) else f"**Score:** {score}"
                )
            if language:
                lines.append(f"**Language:** {language}")
            if snippet:
                lines.append("")
                lines.append("**Content:**")
                lines.append("")
                lines.append(snippet[:2000])
            if images:
                lines.append("")
                lines.append(f"**Images Found ({len(images)}):**")
                for img in images:
                    desc = descr.get(img, "") if isinstance(descr, dict) else ""
                    if desc:
                        lines.append(f"- `{img}` — {desc}")
                    else:
                        lines.append(f"- `{img}`")
            lines.append("")
            lines.append("---")
            lines.append("")
        else:
            # fallback for strings (e.g., list of URLs)
            lines.append(f"- {r}")

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"   📝 Log saved: {filename}")
    except Exception as e:
        print(f"   ⚠️  Could not save log: {e}")

    return filename


# ============================================================================
# LANGUAGE DETECTION AND PRIORITIZATION
# ============================================================================


def _detect_language(text: str) -> str:
    """
    Detects whether the text is predominantly in English or Portuguese.

    Args:
        text: the input text to analyze

    Returns:
        'en' if English is predominant, 'pt' if Portuguese is predominant
    """
    if not text:
        return "en"

    text_lower = text.lower()

    # Common words in Portuguese
    pt_words = [
        "para",
        "como",
        "que",
        "com",
        "mais",
        "dos",
        "das",
        "pela",
        "pelo",
        "são",
        "foi",
        "está",
        "sobre",
        "entre",
        "através",
        "também",
        "ser",
        "por",
        "uma",
        "seus",
        "suas",
        "este",
        "esta",
        "pode",
        "podem",
    ]

    # Common words in English
    en_words = [
        "the",
        "and",
        "for",
        "with",
        "this",
        "from",
        "that",
        "have",
        "was",
        "are",
        "been",
        "their",
        "which",
        "were",
        "when",
        "through",
        "where",
        "using",
        "can",
        "these",
        "those",
        "such",
        "would",
        "should",
    ]

    # Count occurrences of common words for each language
    count_pt = sum(1 for p in pt_words if f" {p} " in f" {text_lower} ")
    count_en = sum(1 for p in en_words if f" {p} " in f" {text_lower} ")

    # Also checks for special Portuguese characters
    if "ã" in text_lower or "ç" in text_lower or "õ" in text_lower:
        count_pt += 3

    return "en" if count_en >= count_pt else "pt"


def _prioritize_by_language(results: list[dict], boost_en: float = 0.3) -> list[dict]:
    """
    Reorders results prioritizing English.
    Adds a boost to the score of English results.

    Args:
        results: list of results with 'score', 'title', 'snippet'
        boost_en: additional boost for English results (0.0 to 1.0)

    Returns:
        List of reordered results enriched with 'language' field
    """
    for r in results:
        # Detects language based on title + snippet
        text_to_detect = f"{r.get('title', '')} {r.get('snippet', r.get('content', ''))}"
        language = _detect_language(text_to_detect)
        r["language"] = language

        # Adds boost for English results
        if language == "en":
            r["score"] = min(1.0, r.get("score", 0) + boost_en)

    # Reorders: English first, then by score
    sorted_results = sorted(
        results, key=lambda x: (x.get("language", "en") != "en", -x.get("score", 0))
    )

    return sorted_results


def _print_language_totals(results: list[dict]) -> None:
    """Print aggregate language stats safely for a result list.

    Args:
        results: list of result dicts, each with a 'language' key

    Returns:
        None (prints totals to console)
    """
    total = len(results)
    if total == 0:
        print("   📊 TOTAL: 0 results")
        return

    total_en = sum(1 for r in results if r.get("language") == "en")
    total_pt = sum(1 for r in results if r.get("language") == "pt")
    print(
        f"   📊 TOTAL: {total_en} English ({total_en/total*100:.0f}%), "
        f"{total_pt} Portuguese ({total_pt/total*100:.0f}%)"
    )


# ============================================================================
# BLOCKED DOMAINS
# ============================================================================

BLOCKED_DOMAINS = [
    "wikipedia.org",
    "wikipedia.com",
    "scribd.com",
    "lonepatient.top",
    "linkedin.com",
    "facebook.com",
    "twitter.com",
    "instagram.com",
    "youtube.com",
    "reddit.com",
    "quora.com",
    "stackexchange.com",
    "stackoverflow.com",
    "ebay.com",
    "aliexpress.com",
    "etsy.com",
    "arxivdaily.com",
    "answers.microsoft.com",
    "merriam-webster.com",
    "dictionary.com",
    "thesaurus.com",
    "news.ycombinator.com",
    "collinsdictionary.com",
    "oxforddictionaries.com",
    "thefreedictionary.com",
    "dictionary.cambridge.org",
    "education.nationalgeographic.com",
    "britannica.com",
    "worldometers.info",
    "statista.com",
    "ourworldindata.org",
    "chrono24.com",
    "rankinggods.com",
    "theoi.com",
    "tiktok.com",
    "pinterest.com",
    "zantia.com",
    "analisemacro.com.br",
    "ibram.org.br",
    "beacademy.substack.com",
    "gov.br",
    "blog.dsacademy.com.br/",
    "mariofilho.com",
    "pt.hyee-ct-cv.com",
    "chatpaper.com",
    "flusshidro.com.br",
    "otca.org",
    "ler.letras.up.pt",
    "oreilly.com",
    "neurips.cc",
    "conference.ifas.ufl.edu",
    "atrium.lib.uoguelph.ca",
    "datadoghq.com",
    "kumo.ai",
    "hydroai.net",
    "geoawesome.com",
    "blogs.egu.eu",
    "i.imgur.com",
    "g1.globo.com",
    "uol.com.br",
    "globo.com",
    "terra.com.br",
    "ig.com.br",
    "folha.uol.com.br",
    "istoe.com.br",
    "veja.abril.com.br",
    "exame.com",
    "revistapegn.globo.com",
    "pixabay.com",
    "pexels.com",
    "unsplash.com",
    "stock.adobe.com",
    "shutterstock.com",
    "gettyimages.com",
    "depositphotos.com",
    "istockphoto.com",
    "canva.com",
    "freepik.com",
    "vecteezy.com",
    "pixlr.com",
    "flickr.com",
    "500px.com",
    "smugmug.com",
    "photobucket.com",
    "imgur.com",
    "tinypic.com",
    "postimages.org",
    "imgbb.com",
    "imagebb.com",
    "muralsonoro.squarespace.com",
    "brapci.inf.br",
    "ouranos.ca",
    "theconversation.com",
    "aitimeline.world",
    "seas.gwu.edu",
    "en.wikipedia.org",
    "en.wikipedia.com",
    "pt.wikipedia.org",
    "pt.wikipedia.com",
    "av.tib.eu",
]


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================


def _get_client() -> TavilyClient:
    """Initialize and return a TavilyClient instance using the API key from environment variables."""
    return TavilyClient(api_key=get_clean_key("TAVILY_API_KEY"))


def filter_academic_urls(urls: list[str]) -> list[str]:
    """Filter out URLs from non-academic domains.

    Args:
        urls: List of URLs to filter

    Returns:
        List of URLs that do not belong to blocked domains.
    """
    filtered = [url for url in urls if not any(b.lower() in url.lower() for b in BLOCKED_DOMAINS)]
    removed = len(urls) - len(filtered)
    if removed:
        print(f"   🚫 Removed {removed} URLs from non-academic sources")
    return filtered


def filter_technical_urls(urls: list[str]) -> list[str]:
    """Filter out URLs from non-technical domains.

    Args:
        urls: List of URLs to filter

    Returns:
        List of URLs that do not belong to blocked domains.
    """
    return [url for url in urls if not any(b.lower() in url.lower() for b in BLOCKED_DOMAINS)]


# ============================================================================
# ACADEMIC SEARCH WITH ENGLISH PRIORITIZATION
# ============================================================================


@tool
def search_tavily(queries: list[str], max_results: int = 5) -> dict:
    """Search for academic articles on Tavily, prioritizing English content.

    Search for academic articles on Tavily.
    Prioritizes content in ENGLISH, but allows Portuguese.
    Automatically filters out non-scientific domains.
    Save each search in ./tavily_searches/ for traceability.

    Args:
        queries: list of search queries
        max_results: results per query (default 5)

    Returns:
        {"urls_found": [...], "results": [...]}
    """
    client = _get_client()
    all_urls: list[str] = []
    all_results: list[dict] = []

    for q in queries:
        print(f"🔎 Searching (academic, EN prioritized): {q}")

        # STRATEGY: Dual search - first English, then complement with Portuguese if needed
        batch_results = []

        try:
            # PHASE 1: English-priority search
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
                    "url": r["url"],
                    "title": r.get("title", ""),
                    "snippet": r.get("content", "")[:300],
                    "score": r.get("score", 0),
                }
                batch_results.append(item)

            # Prioritizes by language (detects and boosts English)
            batch_results = _prioritize_by_language(batch_results, boost_en=0.3)

            # Adds to global results
            for item in batch_results:
                all_urls.append(item["url"])
                all_results.append(item)

            # Log language statistics
            n_en = sum(1 for r in batch_results if r.get("language") == "en")
            n_pt = sum(1 for r in batch_results if r.get("language") == "pt")
            print(f"   📊 Languages: {n_en} English, {n_pt} Portuguese")

            # ── Save log for this query ────────────────────────────────────────
            _save_search_md(
                "academic",
                q,
                batch_results,
                extra={"idioma_en": n_en, "idioma_pt": n_pt},
            )

        except Exception as e:
            print(f"   ⚠️  Error in query '{q[:50]}': {e}")

    unique_urls = list(dict.fromkeys(all_urls))

    # Final statistics
    _print_language_totals(all_results)

    return {"urls_found": unique_urls, "results": all_results}


# ============================================================================
# INCREMENTAL ACADEMIC SEARCH
# ============================================================================


def search_tavily_incremental(
    query: str,
    previous_urls: list[str],
    max_results: int = 5,
) -> dict:
    """
    Incremental academic search — accumulates URLs without duplicates.
    PRIORITIZES English content.
    Saves log of each search in ./tavily_searches/.

    Args:
        query: the search query
        previous_urls: list of previously found URLs to avoid duplicates
        max_results: number of results to return for this incremental search (default 5)

    Returns:
        {"new_urls": [...], "total_accumulated": [...]}
    """
    try:
        client = _get_client()
        print(f"\n🔎 Incremental Search (EN prioritized): '{query}'")

        ans = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            exclude_domains=BLOCKED_DOMAINS,
        )

        # Prepare results with language prioritization
        batch_results = [
            {
                "url": r["url"],
                "title": r.get("title", ""),
                "snippet": r.get("content", "")[:2000],
                "score": r.get("score", 0),
            }
            for r in ans.get("results", [])
            if r.get("score", 0) >= 0.7
        ]

        # Prioritizes by language
        batch_results = _prioritize_by_language(batch_results, boost_en=0.3)

        urls_found = [r["url"] for r in batch_results]
        urls_found = filter_academic_urls(urls_found)

        urls_new = [u for u in urls_found if u not in previous_urls]
        total_accumulated = list(dict.fromkeys(previous_urls + urls_found))

        # Statistics
        n_en = sum(1 for r in batch_results if r.get("language") == "en")
        n_pt = sum(1 for r in batch_results if r.get("language") == "pt")

        print(f"   ✔ Found      : {len(urls_found)} URLs")
        print(f"   ✔ New        : {len(urls_new)} URLs")
        print(f"   ✔ Total acum. : {len(total_accumulated)} URLs")
        print(f"   📊 Languages : {n_en} English, {n_pt} Portuguese")

        # ── Log ──────────────────────────────────────────────────────────────
        _save_search_md(
            "academic_incremental",
            query,
            batch_results,
            extra={
                "new_urls": len(urls_new),
                "total_accumulated": len(total_accumulated),
                "idioma_en": n_en,
                "idioma_pt": n_pt,
            },
        )

        return {
            "new_urls": urls_new,
            "total_accumulated": total_accumulated,
            "results": batch_results,
        }

    except Exception as e:
        print(f"   ⚠️  Error in Tavily search: {e}")
        return {
            "new_urls": [],
            "total_accumulated": previous_urls,
            "results": [],
        }


# ============================================================================
# Technical search with emphasis on English
# ============================================================================


@tool
def search_tavily_technical(queries: list[str], max_results: int = 5) -> dict:
    """
    Technical search on Tavily — allows documentation, tutorials,
    English Wikipedia, online books, reference pages, etc.
    PRIORITIZES English content.
    Saves each search in ./tavily_searches/ for traceability.

    Args:
        queries: list of search queries
        max_results: results per query (default 5)

    Returns:
        {"found_urls": [...], "results": [...]}
    """
    client = _get_client()
    all_urls: list[str] = []
    all_results: list[dict] = []

    for q in queries:
        print(f"🔎 Searching (technical, EN prioritized): {q}")

        try:
            ans = client.search(
                query=q[:400],
                search_depth="advanced",
                max_results=max_results,
                exclude_domains=BLOCKED_DOMAINS,
            )

            batch_results = []
            for r in ans.get("results", []):
                if r.get("score", 0) < 0.7:
                    continue

                item = {
                    "url": r["url"],
                    "title": r.get("title", ""),
                    "snippet": r.get("content", "")[:500],
                    "score": r.get("score", 0),
                }
                batch_results.append(item)

            # Prioritizes by language (detects and boosts English)
            batch_results = _prioritize_by_language(batch_results, boost_en=0.3)

            for item in batch_results:
                all_urls.append(item["url"])
                all_results.append(item)

            # Statistics
            n_en = sum(1 for r in batch_results if r.get("language") == "en")
            n_pt = sum(1 for r in batch_results if r.get("language") == "pt")
            print(f"   📊 Languages: {n_en} English, {n_pt} Portuguese")

            # ── Log ──────────────────────────────────────────────────────────
            _save_search_md(
                "technical",
                q,
                batch_results,
                extra={"idioma_en": n_en, "idioma_pt": n_pt},
            )

        except Exception as e:
            print(f"   ⚠️  Error in query '{q[:50]}': {e}")

    unique_urls = list(dict.fromkeys(all_urls))
    filtered_academic_urls = filter_technical_urls(unique_urls)

    # Final statistics
    _print_language_totals(all_results)

    return {"found_urls": filtered_academic_urls, "results": all_results}


# ============================================================================
# IMAGE SEARCH — dedicated tool for finding technical/academic images
# ============================================================================


@tool
def search_tavily_images(
    queries: list[str],
    max_results: int = 8,
) -> dict:
    """
    Search for images related to a topic via Tavily.
    Returns image URLs with their descriptions when available.
    Saves complete log in ./tavily_searches/.

    Ideal for:
      - Algorithm figures, architectures, flow diagrams
      - Comparative metric charts
      - Time series / hydrology visualizations
      - Technical or scientific illustrations

    Args:
        queries    : list of image-oriented search queries
        max_results: results per query (default 8)

    Returns:
        {
          "images": [
            {
              "image_url"  : str,   # direct image URL
              "description": str,   # description generated by Tavily (if available)
              "source_url" : str,   # page where the image was found
              "page_title" : str,
            }, ...
          ],
          "total": int,
        }
    """
    client = _get_client()
    all_images: list[dict] = []
    viewed: set = set()

    for q in queries:
        print(f"🖼️  Searching images: {q}")
        try:
            ans = client.search(
                query=q[:400],
                search_depth="advanced",
                max_results=max_results,
                include_images=True,
                include_image_descriptions=True,
                exclude_domains=BLOCKED_DOMAINS,
            )

            # ── Direct images returned by Tavily ───────────────────────────
            raw_images = ans.get("images", [])
            raw_descriptions = ans.get("image_descriptions", {})

            # Normalize: this can be a list of strings or a list of dicts.
            for item in raw_images:
                if isinstance(item, dict):
                    url_img = item.get("url", "")
                    desc = item.get("description", "")
                else:
                    url_img = str(item)
                    desc = (
                        raw_descriptions.get(url_img, "")
                        if isinstance(raw_descriptions, dict)
                        else ""
                    )

                if not url_img or url_img in viewed:
                    continue

                has_valid_ext = any(
                    url_img.lower().endswith(ext)
                    for ext in (".jpg", ".jpeg", ".png", ".svg", ".gif", ".webp")
                )

                if (
                    not has_valid_ext
                    and "image" not in url_img.lower()
                    and not re.search(r"\.(jpg|jpeg|png|svg|gif|webp)", url_img, re.I)
                ):
                    continue

                viewed.add(url_img)
                all_images.append(
                    {
                        "image_url": url_img,
                        "description": desc,
                        "source_url": "",
                        "page_title": "",
                    }
                )

            # ── Images from result snippets ────────────────────────────────
            for r in ans.get("results", []):
                url_pag = r.get("url", "")
                title_pag = r.get("title", "")
                for img_url in r.get("images", []):
                    if img_url and img_url not in viewed:
                        viewed.add(img_url)
                        all_images.append(
                            {
                                "image_url": img_url,
                                "description": "",
                                "source_url": url_pag,
                                "page_title": title_pag,
                            }
                        )

            # ── Log ──────────────────────────────────────────────────────────
            _save_search_md(
                "images",
                q,
                [
                    {
                        "url": img["image_url"],
                        "title": img["page_title"],
                        "snippet": img["description"],
                        "images": [img["image_url"]],
                    }
                    for img in all_images
                ],
                extra={"total_images": len(all_images)},
            )

        except Exception as e:
            print(f"   ⚠️  Error in images query '{q[:50]}': {e}")

    print(f"   🖼️  Total of founded images: {len(all_images)}")
    return {"images": all_images, "total": len(all_images)}


# ============================================================================
# EXTRACT — extracts the complete content from URLs.
# ============================================================================


@tool
def extract_tavily(urls: list[str], include_images: bool = True) -> dict:
    """
    Extracts full content from web pages via the Tavily Extract API.
    Save the complete log to ./tavily_searches/.

    Args:
        urls           : list of URLs to extract (max. 20 per call)
        include_images : if True, includes URLs of found images in the extracted content

    Returns:
        {
          "extracted": [
            {
              "url"     : str,
              "title"   : str,
              "content" : str,
              "images" : [str],
            }, ...
          ],
          "failed": [str],
        }
    """
    client = _get_client()
    extracted: list[dict] = []
    flawed: list[str] = []

    lots = [urls[i : i + 20] for i in range(0, len(urls), 20)]

    for lot in lots:
        print(f"📥 Extracting {len(lot)} URL(s)...")
        try:
            res = client.extract(
                urls=lot,
                extract_depth="advanced",
                include_images=include_images,
            )

            for item in res.get("results", []):
                url = item.get("url", "")
                content = item.get("raw_content", item.get("content", ""))
                images = item.get("images", []) if include_images else []

                extracted.append(
                    {
                        "url": url,
                        "title": item.get("title", ""),
                        "content": content,
                        "images": images,
                    }
                )
                print(
                    f"   ✔ {url[:60]} — {len(content):,} chars"
                    f"{f', {len(images)} img(s)' if images else ''}"
                )

            for item in res.get("failed_results", []):
                flawed.append(item.get("url", ""))
                print(f"   ✖ Failed: {item.get('url','')[:60]}")

        except Exception as e:
            print(f"   ⚠️  Error in lot: {e}")
            flawed.extend(lot)

    # ── Log ──────────────────────────────────────────────────────────────────
    query_repr = urls[0] if urls else "extract"
    _save_search_md(
        "extract",
        query_repr,
        [
            {
                "url": e["url"],
                "title": e["title"],
                "snippet": e["content"],
                "images": e["images"],
            }
            for e in extracted
        ],
        extra={
            "requested_urls": len(urls),
            "extracted": len(extracted),
            "failed": len(flawed),
        },
    )

    return {"extracted": extracted, "failed": flawed}


# ============================================================================
# Incremental technical search (direct use by graph nodes)
# ============================================================================


def search_tavily_incremental_technician(
    query: str,
    previous_urls: list[str],
    max_results: int = 8,
) -> dict:
    """
    Incremental technical search — accumulates URLs without duplicates.
    Prioritizes content in ENGLISH.
    Save log in ./tavily_searches/.

    Args:
        query: the search query
        previous_urls: list of previously found URLs to avoid duplicates
        max_results: number of results to return for this incremental search (default 8)

    Returns:
        {"new_urls": [...], "total_accumulated": [...], "results": [...]}
    """
    try:
        client = _get_client()
        print(f"\n🔎 Incremental Technical Search (EN prioritized): '{query}'")

        ans = client.search(
            query=query[:400],
            search_depth="advanced",
            max_results=max_results,
            exclude_domains=BLOCKED_DOMAINS,
        )

        results = [
            {
                "url": r["url"],
                "title": r.get("title", ""),
                "snippet": r.get("content", "")[:2000],
                "score": r.get("score", 0),
            }
            for r in ans.get("results", [])
            if r.get("score", 0) >= 0.7
        ]

        # Prioritiza por idioma
        results = _prioritize_by_language(results, boost_en=0.3)

        all_urls = [r["url"] for r in results]
        all_urls = filter_technical_urls(all_urls)

        new_urls = [u for u in all_urls if u not in previous_urls]
        total_accumulated = list(dict.fromkeys(previous_urls + all_urls))

        # Estatístics
        n_en = sum(1 for r in results if r.get("language") == "en")
        n_pt = sum(1 for r in results if r.get("language") == "pt")

        print(f"   ✔ Founded : {len(all_urls)} URLs")
        print(f"   ✔ News       : {len(new_urls)} URLs")
        print(f"   ✔ Total acum. : {len(total_accumulated)} URLs")
        print(f"   📊 Languages    : {n_en} english, {n_pt} portuguese")

        # ── Log ──────────────────────────────────────────────────────────────
        _save_search_md(
            "incremental_technical",
            query,
            results,
            extra={
                "new_urls": len(new_urls),
                "total_accumulated": len(total_accumulated),
                "language_en": n_en,
                "language_pt": n_pt,
            },
        )

        return {
            "new_urls": new_urls,
            "total_accumulated": total_accumulated,
            "results": results,
        }

    except Exception as e:
        print(f"   ⚠️  Error in techinical search: {e}")
        return {
            "new_urls": [],
            "total_accumulated": previous_urls,
            "results": [],
        }
