# src/revisao_agents/tools/tavily_web_search.py
"""
Tavily Web Search Tools — version migrated to the new package.
Retains all original functionalities (crawlability, language, academic filters).
"""

import os
import re
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

import mlflow
import requests
from langchain_core.tools import tool
from tavily import TavilyClient
from tavily.errors import (
    BadRequestError,
    ForbiddenError,
    InvalidAPIKeyError,
    UsageLimitExceededError,
)
from tavily.errors import TimeoutError as TavilyTimeoutError

from ..config import SEARCH_LOGS_DIR, TAVILY_CONFIG
from ..core.utils import detect_language
from ..utils.core.commons import get_clean_key

# Exceptions actually raised by `TavilyClient.search()` on network/API failure
# (see tavily.tavily.TavilyClient.search): Tavily's own error classes plus
# `requests.exceptions.RequestException` for connection-level/HTTP failures.
# Deliberately narrower than `Exception` so unrelated errors (e.g. MLflow
# failures inside the same `with` block) are not silently swallowed.
TAVILY_SEARCH_ERRORS = (
    requests.exceptions.RequestException,
    UsageLimitExceededError,
    ForbiddenError,
    InvalidAPIKeyError,
    BadRequestError,
    TavilyTimeoutError,
)


def _slug(text: str, max_chars: int = 50) -> str:
    """Generate a safe slug for use as a filename.

    Args:
        text: The input string to slugify (e.g. a search query).
        max_chars: Maximum number of characters to include in the slug.

    Returns:
        A slugified version of the input string.
    """
    s = re.sub(r"[^\w\s-]", "", text[:max_chars]).strip()
    s = re.sub(r"[\s]+", "_", s).lower()
    return s or "search"


@dataclass
class _TavilySearchSpan:
    """Handle returned by :func:`_tavily_search_span` for logging extra metrics.

    Attributes:
        log_metrics: Callback that forwards keyword arguments to
            ``mlflow.log_metric`` for each entry, allowing callers to log
            domain-specific metrics (e.g. ``credits_used``, ``urls_found``,
            ``valid_academic_urls_found``) without the helper needing to know
            their names in advance.
    """

    log_metrics: Callable[..., None]


@contextmanager
def _tavily_search_span(query: str, search_depth: str) -> Generator[_TavilySearchSpan, None, None]:
    """Open a nested MLflow run instrumenting a single Tavily search call.

    Starts a nested MLflow run named ``search:<query[:30]>``, logs the
    ``search_depth`` parameter on entry, times the duration of the ``with``
    block, and logs it as the ``latency`` metric on exit. Callers can use the
    yielded span's ``log_metrics`` callback to record additional metrics
    (e.g. ``credits_used``, ``urls_found``) without this helper needing to
    know their names ahead of time. MLflow errors are not caught and
    propagate to the caller.

    Args:
        query: The search query string; used to name the nested run.
        search_depth: The Tavily search depth used for this call (e.g.
            ``TAVILY_CONFIG.depth``); logged as the ``search_depth`` param.

    Yields:
        _TavilySearchSpan: An object exposing ``log_metrics(**kwargs)`` to log
        additional metrics under the same nested run.
    """

    def _log_metrics(**kwargs: float) -> None:
        mlflow.log_metrics(kwargs)

    with mlflow.start_run(run_name=f"search:{query[:30]}", nested=True):
        mlflow.log_param("search_depth", search_depth)
        start = time.perf_counter()
        try:
            yield _TavilySearchSpan(log_metrics=_log_metrics)
        finally:
            latency = time.perf_counter() - start
            mlflow.log_metric("latency", latency)


def _save_search_md(
    type: str,
    query: str,
    results: list[dict],
    extra: dict | None = None,
    usage: dict | None = None,
) -> str:
    """Save the results of a Tavily search to a Markdown file.

    Args:
        type: Type of the search (``academic``, ``technical``, ``images``, or ``extract``).
        query: The query string or URL that was searched.
        results: List of result dicts returned by Tavily.
        extra: Optional mapping of extra metadata (e.g. found URLs) to append to the log header.
        usage: Optional dict of API usage statistics to include in the log.

    Returns:
        Path to the saved Markdown file.
    """
    os.makedirs(SEARCH_LOGS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ms precision
    slug_q = _slug(query)
    filename = f"{SEARCH_LOGS_DIR}/{ts}_{type}_{slug_q}.md"

    lines = [
        f"# Tavily Search — {type.upper()}",
        "",
        f"- **Date/Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Type:** {type}",
        f"- **Query:** `{query}`",
        f"- **Total Results:** {len(results)}",
    ]

    if usage:
        lines.append(f"- **Credits Used:** {usage.get('credits', 'N/A')}")
        lines.append(f"- **Request ID:** {usage.get('id', 'N/A')}")

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
        language = detect_language(text_to_detect)
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
        f"   📊 TOTAL: {total_en} English ({total_en / total * 100:.0f}%), "
        f"{total_pt} Portuguese ({total_pt / total * 100:.0f}%)"
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
def search_tavily(queries: list[str], max_results: int = TAVILY_CONFIG.num_results) -> dict:
    """Search for academic articles on Tavily, prioritizing English content.

    Runs one Tavily search per query, allows Portuguese results but boosts
    English ones, and automatically filters out non-scientific domains. Each
    query's results are logged under ``./tavily_searches/`` for traceability.

    Args:
        queries: List of search query strings.
        max_results: Maximum number of results to request per query. Defaults
            to ``TAVILY_CONFIG.num_results``.

    Returns:
        dict: A dictionary with the following keys:
            - ``"urls_found"``: list[str], deduplicated URLs across all queries.
            - ``"results"``: list[dict], result objects with ``url``, ``title``,
              ``snippet``, and ``score`` keys.
            - ``"usage"``: list[dict], per-query Tavily usage info.
    """
    client = _get_client()
    all_urls: list[str] = []
    all_results: list[dict] = []
    all_usage: list[dict] = []

    for q in queries:
        print(f"🔎 Searching (academic, EN prioritized): {q}")

        # STRATEGY: Dual search - first English, then complement with Portuguese if needed
        batch_results = []

        try:
            # PHASE 1: English-priority search
            res_en = client.search(
                query=q,
                search_depth=TAVILY_CONFIG.depth,
                max_results=max_results,
                include_answer=TAVILY_CONFIG.include_answer,
                include_usage=TAVILY_CONFIG.include_usage,
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
            usage_data = {**res_en.get("usage", {}), "id": res_en.get("request_id", "N/A")}
            _save_search_md(
                "academic",
                q,
                batch_results,
                extra={"idioma_en": n_en, "idioma_pt": n_pt},
                usage=usage_data,
            )
            all_usage.append(usage_data)

        except Exception as e:
            print(f"   ⚠️  Error in query '{q[:50]}': {e}")

    unique_urls = list(dict.fromkeys(all_urls))

    # Final statistics
    _print_language_totals(all_results)

    return {"urls_found": unique_urls, "results": all_results, "usage": all_usage}


# ============================================================================
# INCREMENTAL ACADEMIC SEARCH
# ============================================================================


@mlflow.trace(name="search_tavily_incremental", span_type="TOOL")
def search_tavily_incremental(
    query: str,
    previous_urls: list[str],
    max_results: int = TAVILY_CONFIG.num_results,
) -> dict:
    """Incremental academic search that accumulates URLs without duplicates.

    Prioritizes English content. Saves a log of each search under
    ``./tavily_searches/`` for traceability.

    Args:
        query: The search query string.
        previous_urls: URLs already retrieved in previous calls; used to compute
            the ``new_urls`` set and avoid downstream duplicates.
        max_results: Maximum number of results to request from Tavily. Defaults
            to ``TAVILY_CONFIG.num_results``.

    Returns:
        dict: A dictionary with the following keys:
            - ``"new_urls"``: list[str], URLs not present in ``previous_urls``.
            - ``"total_accumulated"``: list[str], union of ``previous_urls`` and
              newly found URLs (order-preserving, no duplicates).
            - ``"results"``: list[dict], result objects with ``url``, ``title``,
              ``snippet``, and ``score`` keys.
            - ``"usage"``: dict, Tavily usage info, including a ``"credits"`` key.

    Raises:
        Exception: Errors raised by Tavily/requests during ``client.search()``
            (``TAVILY_SEARCH_ERRORS``) are caught and result in the empty-result
            dict above. Any other exception — e.g. from result post-processing,
            MLflow span/instrumentation failures, or ``_save_search_md`` — is not
            caught and propagates to the caller.
    """
    print(f"\n🔎 Incremental Search (EN prioritized): '{query}'")

    client = _get_client()

    with _tavily_search_span(query, TAVILY_CONFIG.depth) as span:
        try:
            ans = client.search(
                query=query,
                search_depth=TAVILY_CONFIG.depth,
                max_results=max_results,
                include_answer=TAVILY_CONFIG.include_answer,
                include_usage=TAVILY_CONFIG.include_usage,
                exclude_domains=BLOCKED_DOMAINS,
            )
        except TAVILY_SEARCH_ERRORS as e:
            print(f"   ⚠️  Error in Tavily search: {e}")
            return {
                "new_urls": [],
                "total_accumulated": previous_urls,
                "results": [],
                "usage": {},
            }

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
        n_urls_found = len(urls_found)
        urls_found = filter_academic_urls(urls_found)

        span.log_metrics(
            credits_used=ans.get("usage", {}).get("credits", 0),
            urls_found=n_urls_found,
            valid_academic_urls_found=len(urls_found),
        )

    urls_new = [u for u in urls_found if u not in previous_urls]
    total_accumulated = list(dict.fromkeys(previous_urls + urls_found))

    # Log all 4 quality metrics to the parent workflow run (if one is active).
    # result_reuse_percent and credit_efficiency_aggregated are computed as
    # per-call approximations since session state is not available here.
    if mlflow.active_run():
        from revisao_agents.observability.search_metrics import SearchQualityMetrics

        credits = ans.get("usage", {}).get("credits", 0.0)
        reused = [u for u in urls_found if u in previous_urls]
        result_reuse_pct = round(100 * len(reused) / max(len(urls_found), 1), 2)
        # credit_efficiency_aggregated requires session-level totals, which
        # are unavailable at the tool level; it is computed by the node callers.
        SearchQualityMetrics.log_all_metrics_to_mlflow(
            {
                "search_coverage": SearchQualityMetrics.calculate_search_coverage(urls_new),
                "result_reuse_percent": result_reuse_pct,
                "credit_efficiency_individual": SearchQualityMetrics.calculate_credit_efficiency_individual(
                    credits
                ),
            }
        )
    n_en = sum(1 for r in batch_results if r.get("language") == "en")
    n_pt = sum(1 for r in batch_results if r.get("language") == "pt")

    print(f"   ✔ Found      : {len(urls_found)} URLs")
    print(f"   ✔ New        : {len(urls_new)} URLs")
    print(f"   ✔ Total acum. : {len(total_accumulated)} URLs")
    print(f"   📊 Languages : {n_en} English, {n_pt} Portuguese")

    # ── Log ──────────────────────────────────────────────────────────────────
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
        usage={**ans.get("usage", {}), "id": ans.get("request_id", "N/A")},
    )

    return {
        "new_urls": urls_new,
        "total_accumulated": total_accumulated,
        "results": batch_results,
        "usage": ans.get("usage", {}),
    }


@tool
def search_tavily_incremental_tool(
    query: str,
    previous_urls: list[str],
    max_results: int = TAVILY_CONFIG.num_results,
) -> dict:
    """Incremental academic search that accumulates URLs without duplicates.

    Agent-bindable wrapper around `search_tavily_incremental` — the underlying
    function is also called directly elsewhere in this codebase as a plain
    utility, so it intentionally has no `@tool` decorator of its own.

    Args:
        query: The search query string.
        previous_urls: URLs already retrieved in previous calls; used to compute
            the ``new_urls`` set and avoid downstream duplicates.
        max_results: Maximum number of results to request from Tavily. Defaults
            to ``TAVILY_CONFIG.num_results``.

    Returns:
        dict: Same shape as `search_tavily_incremental`.
    """
    return search_tavily_incremental(query, previous_urls, max_results)


# ============================================================================
# Technical search with emphasis on English
# ============================================================================


@tool
def search_tavily_technical(
    queries: list[str], max_results: int = TAVILY_CONFIG.num_results
) -> dict:
    """Search for technical content on Tavily, prioritizing English content.

    Allows documentation, tutorials, English Wikipedia, online books,
    reference pages, etc., and boosts English results. Each query's results
    are logged under ``./tavily_searches/`` for traceability.

    Args:
        queries: List of search query strings.
        max_results: Maximum number of results to request per query. Defaults
            to ``TAVILY_CONFIG.num_results``.

    Returns:
        dict: A dictionary with the following keys:
            - ``"found_urls"``: list[str], deduplicated, technical-domain-filtered
              URLs across all queries (see :func:`filter_technical_urls`).
            - ``"results"``: list[dict], result objects with ``url``, ``title``,
              ``snippet``, and ``score`` keys.

    Raises:
        Exception: Errors raised by Tavily/requests during ``client.search()``
            (``TAVILY_SEARCH_ERRORS``) are caught per-query and that query is
            skipped. Any other exception — e.g. from result post-processing,
            MLflow span/instrumentation failures, or ``_save_search_md`` — is not
            caught and propagates to the caller.
    """
    client = _get_client()
    all_urls: list[str] = []
    all_results: list[dict] = []

    for q in queries:
        print(f"🔎 Searching (technical, EN prioritized): {q}")

        with _tavily_search_span(q, TAVILY_CONFIG.depth) as span:
            try:
                ans = client.search(
                    query=q[:400],
                    search_depth=TAVILY_CONFIG.depth,
                    max_results=max_results,
                    include_answer=TAVILY_CONFIG.include_answer,
                    include_usage=TAVILY_CONFIG.include_usage,
                    exclude_domains=BLOCKED_DOMAINS,
                )
            except TAVILY_SEARCH_ERRORS as e:
                print(f"   ⚠️  Error in query '{q[:50]}': {e}")
                continue

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

            span.log_metrics(
                credits_used=ans.get("usage", {}).get("credits", 0),
                urls_found=len(batch_results),
            )

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
                usage={**ans.get("usage", {}), "id": ans.get("request_id", "N/A")},
            )

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
    max_results: int = TAVILY_CONFIG.num_results,
) -> dict:
    """Search for images related to a topic via Tavily Search.

    Uses only documented Search image fields (``include_images`` and
    ``include_image_descriptions``) and returns the best available metadata.
    Each query's results are logged under ``./tavily_searches/``.

    Ideal for algorithm figures, architecture/flow diagrams, comparative
    metric charts, time series / hydrology visualizations, and technical or
    scientific illustrations.

    Args:
        queries: List of image-oriented search query strings.
        max_results: Maximum number of images to return per query, capped at
            4 regardless of the value passed in.

    Returns:
        dict: A dictionary with the following keys:
            - ``"images"``: list[dict], each with ``image_url`` (direct image
              URL), ``description`` (Tavily-generated, if available),
              ``source_url`` (best-effort page URL, may be empty), and
              ``page_title`` (best-effort page title, may be empty).
            - ``"total"``: int, total number of images found.
    """
    max_results = min(max_results, 4)
    client = _get_client()
    all_images: list[dict] = []
    viewed: set[str] = set()

    VALID_EXT = (".jpg", ".jpeg", ".png", ".svg", ".gif", ".webp")

    for q in queries:
        print(f"🔍 Searching images for: {q}")
        try:
            search_ans = client.search(
                query=q[:400],
                search_depth=TAVILY_CONFIG.depth,
                max_results=max_results,
                include_images=True,
                include_image_descriptions=True,
                include_usage=TAVILY_CONFIG.include_usage,
                exclude_domains=BLOCKED_DOMAINS,
            )

            # Build best-effort map image_url -> (source_url, page_title) from search results.
            image_source_map: dict[str, tuple[str, str]] = {}
            for result in search_ans.get("results", []):
                source_url = result.get("url", "") or ""
                page_title = result.get("title", "") or ""
                for img in result.get("images", []):
                    img_url = img.get("url", "") if isinstance(img, dict) else str(img)
                    if img_url and img_url not in image_source_map:
                        image_source_map[img_url] = (source_url, page_title)

            batch_images: list[dict] = []
            for img in search_ans.get("images", []):
                img_url = img.get("url", "") if isinstance(img, dict) else str(img)
                img_desc = img.get("description", "") if isinstance(img, dict) else ""

                if not img_url or img_url in viewed:
                    continue

                clean_url = img_url.split("?")[0].lower()
                if (
                    not any(clean_url.endswith(ext) for ext in VALID_EXT)
                    and "image" not in clean_url
                ):
                    continue

                source_url, page_title = image_source_map.get(img_url, ("", ""))
                viewed.add(img_url)
                item = {
                    "image_url": img_url,
                    "description": img_desc,
                    "source_url": source_url,
                    "page_title": page_title,
                }
                batch_images.append(item)
                all_images.append(item)

                if len(batch_images) >= max_results:
                    break

            # ── Log ──────────────────────────────────────────────────────────
            _save_search_md(
                "images_search",
                q,
                [
                    {"url": i["image_url"], "title": i["page_title"], "snippet": i["description"]}
                    for i in batch_images
                ],
                usage={**search_ans.get("usage", {}), "id": search_ans.get("request_id", "N/A")},
            )

        except Exception as e:
            print(f"   ⚠️  Error in image search for '{q[:50]}': {e}")

    print(f"   🖼️  Total images found: {len(all_images)}")
    return {"images": all_images, "total": len(all_images)}


# ============================================================================
# EXTRACT — extracts the complete content from URLs.
# ============================================================================


@tool
def extract_tavily(urls: list[str], include_images: bool = True) -> dict:
    """Extract full content from web pages via the Tavily Extract API.

    URLs are batched in groups of 20 (the Tavily Extract API's per-call
    limit) and extracted with ``extract_depth="advanced"``. The complete log
    is saved under ``./tavily_searches/``.

    Args:
        urls: List of URLs to extract. Internally split into batches of 20.
        include_images: If True, includes URLs of images found on each page
            in the extracted content.

    Returns:
        dict: A dictionary with the following keys:
            - ``"extracted"``: list[dict], each with ``url``, ``title``,
              ``content`` (raw or rendered page content), and ``images``
              (list[str], empty if ``include_images`` is False).
            - ``"failed"``: list[str], URLs that could not be extracted.
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
                print(f"   ✖ Failed: {item.get('url', '')[:60]}")

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
        usage={**res.get("usage", {}), "id": res.get("request_id", "N/A")} if lots else {},
    )

    return {"extracted": extracted, "failed": flawed}


# ============================================================================
# Incremental technical search (direct use by graph nodes)
# ============================================================================


def search_tavily_incremental_technician(
    query: str,
    previous_urls: list[str],
    max_results: int = TAVILY_CONFIG.num_results,
) -> dict:
    """Incremental technical search that accumulates URLs without duplicates.

    Applies technical-domain URL filtering (see :func:`filter_technical_urls`)
    and prioritizes English content. Saves a log under ``./tavily_searches/``.

    Args:
        query: The search query string.
        previous_urls: URLs already retrieved in previous calls; used to compute
            the ``new_urls`` set and avoid downstream duplicates.
        max_results: Maximum number of results to request from Tavily. Defaults
            to ``TAVILY_CONFIG.num_results``.

    Returns:
        dict: A dictionary with the following keys:
            - ``"new_urls"``: list[str], URLs not present in ``previous_urls``.
            - ``"total_accumulated"``: list[str], union of ``previous_urls`` and
              newly found URLs (order-preserving, no duplicates).
            - ``"results"``: list[dict], result objects with ``url``, ``title``,
              ``snippet``, and ``score`` keys.
            - ``"usage"``: dict, Tavily usage info, including a ``"credits"`` key.
            - ``"urls_found"``: list[str], all URLs returned by this search
              (before deduplication against ``previous_urls``).

    Raises:
        Exception: Errors raised by Tavily/requests during ``client.search()``
            (``TAVILY_SEARCH_ERRORS``) are caught and result in the empty-result
            dict above. Any other exception — e.g. from result post-processing,
            MLflow span/instrumentation failures, or ``_save_search_md`` — is not
            caught and propagates to the caller.
    """
    print(f"\n🔎 Incremental Technical Search (EN prioritized): '{query}'")

    client = _get_client()

    with _tavily_search_span(query, TAVILY_CONFIG.depth) as span:
        try:
            ans = client.search(
                query=query[:400],
                search_depth=TAVILY_CONFIG.depth,
                max_results=max_results,
                include_answer=TAVILY_CONFIG.include_answer,
                include_usage=TAVILY_CONFIG.include_usage,
                exclude_domains=BLOCKED_DOMAINS,
            )
        except TAVILY_SEARCH_ERRORS as e:
            print(f"   ⚠️  Error in techinical search: {e}")
            return {
                "new_urls": [],
                "total_accumulated": previous_urls,
                "results": [],
                "usage": {},
                "urls_found": [],
            }

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

        # Prioritize by language
        results = _prioritize_by_language(results, boost_en=0.3)

        all_urls = [r["url"] for r in results]
        n_urls_found = len(all_urls)
        all_urls = filter_technical_urls(all_urls)

        span.log_metrics(
            credits_used=ans.get("usage", {}).get("credits", 0),
            urls_found=n_urls_found,
            valid_technical_urls_found=len(all_urls),
        )

    new_urls = [u for u in all_urls if u not in previous_urls]
    total_accumulated = list(dict.fromkeys(previous_urls + all_urls))

    # Language statistics
    n_en = sum(1 for r in results if r.get("language") == "en")
    n_pt = sum(1 for r in results if r.get("language") == "pt")

    print(f"   ✔ Found      : {len(all_urls)} URLs")
    print(f"   ✔ New        : {len(new_urls)} URLs")
    print(f"   ✔ Total acum.: {len(total_accumulated)} URLs")
    print(f"   📊 Languages : {n_en} English, {n_pt} Portuguese")

    # ── Log ──────────────────────────────────────────────────────────────────
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
        usage={**ans.get("usage", {}), "id": ans.get("request_id", "N/A")},
    )

    return {
        "new_urls": new_urls,
        "total_accumulated": total_accumulated,
        "results": results,
        "usage": ans.get("usage", {}),
        "urls_found": all_urls,
    }
