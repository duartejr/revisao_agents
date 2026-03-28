# src/revisao_agents/tools/image_tools.py
"""
LangChain tool wrappers for discovering academic/technical images.

Used by the image suggestion agent to search for images suitable for
illustrating document sections, with best-effort source attribution.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile

from langchain_core.tools import tool

from .tavily_web_search import search_tavily_images

# ============================================================================
# Simple file-based cache for image search results
# ============================================================================

_CACHE_DIR = os.path.join(tempfile.gettempdir(), "image_search_cache")


def _cache_key(queries: list[str]) -> str:
    payload = json.dumps(sorted(queries), sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()


def _load_cache(key: str) -> list[dict] | None:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = os.path.join(_CACHE_DIR, f"{key}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(key: str, images: list[dict]) -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = os.path.join(_CACHE_DIR, f"{key}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(images, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ============================================================================
# Image tools
# ============================================================================


@tool
def search_images_with_queries(
    queries: list[str],
    max_results: int = 4,
) -> list[dict]:
    """Search for academic/technical images using explicit pre-crafted queries.

    The agent MUST write specific, targeted queries derived from the actual
    paragraph text (e.g. 'Chronos zero-shot transformer architecture diagram'),
    NOT generic ones like 'Discussion technical diagram'.

    Results are cached per query set.

    Args:
        queries: List of 2–4 search queries crafted by the agent.  Each query
            should name the specific model, method, dataset, or visual type
            mentioned in the paragraph.
        max_results: Upper limit of total image candidates to return (default 4,
            matching the upstream Tavily image cap per query).

    Returns:
        List of dicts, each containing:
        - image_url: Direct URL to the image file.
        - description: Text description (if available).
        - source_url: Web page where the image was found (may be empty).
        - page_title: Title of the source page (may be empty).
        - source_note: Limitation message explaining metadata constraints.
    """
    if not queries:
        return [{"error": "No queries provided"}]

    cache_key = _cache_key(queries)
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached[:max_results]

    try:
        result = search_tavily_images.invoke({"queries": queries, "max_results": max_results})
        raw_images: list[dict] = result.get("images", []) if isinstance(result, dict) else []
    except Exception as exc:
        return [{"error": f"Image search failed: {exc}"}]

    enriched: list[dict] = []
    source_note = (
        "Source paper metadata may be incomplete. Verify the original publication manually."
    )

    # Cache the full result set so later calls with larger max_results can be served from cache.
    for img in raw_images:
        image_url = img.get("image_url", "") or img.get("url", "")
        if not image_url:
            continue
        enriched.append(
            {
                "image_url": image_url,
                "description": img.get("description", "") or "",
                "source_url": img.get("source_url", "") or "",
                "page_title": img.get("page_title", "") or "",
                "source_note": img.get("source_note", "") or source_note,
            }
        )

    _save_cache(cache_key, enriched)
    return enriched[:max_results]


@tool
def lookup_page_metadata(
    image_url: str,
    source_url: str = "",
) -> dict:
    """Fetch the web page that hosts an image and extract full citation metadata.

    Call this BEFORE writing the ## REFERENCE block for every image.
    Returns raw structured data — the agent assembles the final reference.

    If `source_url` is provided it is fetched directly.  Otherwise the tool
    derives the article page URL by stripping the image filename from
    `image_url` (e.g. ".../2022/img.png" → ".../2022/").

    Args:
        image_url: Direct URL to the image file.
        source_url: URL of the page where the image was found (preferred).

    Returns:
        Dict with keys (empty string when not found):
        - page_url: URL that was actually fetched.
        - domain: Domain in ALLCAPS (e.g. "HESS.COPERNICUS.ORG").
        - title: Real page/article title.
        - authors: Author string as found on the page (e.g. "Smith, J.; Lee, K.").
        - journal: Journal or venue name.
        - year: 4-digit publication year.
        - volume: Volume number.
        - issue: Issue number.
        - pages: Page range (e.g. "1673–1693").
        - doi: DOI string without prefix (e.g. "10.5194/hess-26-1673-2022").
        - content_excerpt: First 2000 chars of raw page content for the agent.
        - error: Non-empty if the fetch failed.
    """
    # ── Resolve the page URL ──────────────────────────────────────────────
    if source_url:
        page_url = source_url
    elif image_url:
        m = re.match(
            r"(https?://[^/]+(?:/[^/]+)*)/[^/]+\.[a-zA-Z0-9]{2,5}(?:\?.*)?$",
            image_url,
        )
        page_url = (m.group(1) + "/") if m else image_url
    else:
        return {
            "page_url": "",
            "domain": "",
            "title": "",
            "authors": "",
            "journal": "",
            "year": "",
            "volume": "",
            "issue": "",
            "pages": "",
            "doi": "",
            "content_excerpt": "",
            "error": "No URL provided",
        }

    # ── Extract domain ────────────────────────────────────────────────────
    domain_m = re.search(r"https?://(?:www\.)?([^/]+)", page_url)
    domain = domain_m.group(1).upper() if domain_m else ""

    # ── Fetch page via Tavily Extract ─────────────────────────────────────
    title = ""
    authors = ""
    journal = ""
    year = ""
    volume = ""
    issue = ""
    pages = ""
    doi = ""
    content_excerpt = ""
    error_msg = ""

    try:
        from .tavily_web_search import _get_client

        client = _get_client()
        res = client.extract(
            urls=[page_url],
            extract_depth="advanced",
            include_images=False,
        )
        results = res.get("results", [])
        if results:
            r0 = results[0]
            title = (r0.get("title", "") or "").strip()
            raw = (r0.get("raw_content", r0.get("content", "")) or "").strip()
            content_excerpt = raw[:2000]
            haystack = page_url + " " + raw[:3000]

            # DOI
            doi_m = re.search(r"\b(10\.\d{4,}/[^\s\"<>\]\)]+)", haystack)
            if doi_m:
                doi = doi_m.group(1).rstrip(".,)")

            # Year
            yr_m = re.search(r"\b(20\d{2})\b", haystack[:1500])
            if yr_m:
                year = yr_m.group(1)

            # Volume
            vol_m = re.search(r"[Vv]ol(?:\.|ume)?\s*\.?\s*(\d+)", raw[:2000])
            if vol_m:
                volume = vol_m.group(1)

            # Issue / number
            iss_m = re.search(r"[Nn](?:o|um)\.?\s*(\d+)", raw[:2000])
            if iss_m:
                issue = iss_m.group(1)

            # Pages
            pg_m = re.search(r"[Pp](?:p|ages?)\.?\s*([\d]+[-\u2013]\d+)", raw[:2000])
            if pg_m:
                pages = pg_m.group(1)

            # Authors — try common academic page patterns
            for pat in [
                r"(?:Authors?|By)[:\s]+([A-Z][^\n]{5,120})",
                r"^([A-Z][a-z]+,\s+[A-Z]\.?[^\n]*(?:;\s*[A-Z][a-z]+[^\n]*)*)",
                r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,\s+[A-Z]\.(?:\s+[A-Z]\.)?(?:,\s+[A-Z][a-z]+[^\n]{0,60})?)",
            ]:
                auth_m = re.search(pat, raw[:1000], re.MULTILINE)
                if auth_m:
                    candidate = auth_m.group(1).strip()
                    if len(candidate) > 3:
                        authors = candidate
                        break

            # Journal — common patterns
            for pat in [
                r"(?:journal|published in|venue)[:\s]+([A-Z][^.\n]{5,80})",
                r"(?:Hydrology and|Geoscientific|Water Resources|IEEE|Nature|Science)\s[A-Za-z &]{3,60}",
            ]:
                j_m = re.search(pat, raw[:2000], re.IGNORECASE)
                if j_m:
                    journal = j_m.group(0 if "Hydrology" in pat or "IEEE" in pat else 1).strip()
                    break
        else:
            error_msg = "Page extraction returned no results"
    except Exception as exc:
        error_msg = str(exc)

    return {
        "page_url": page_url,
        "domain": domain,
        "title": title,
        "authors": authors,
        "journal": journal,
        "year": year,
        "volume": volume,
        "issue": issue,
        "pages": pages,
        "doi": doi,
        "content_excerpt": content_excerpt,
        "error": error_msg,
    }


@tool
def format_image_markdown(
    image_url: str,
    caption: str,
    abnt_attribution: str,
) -> str:
    """Format an image as a Markdown block ready to insert into a document.

    Produces a figure block with the image, caption, and ABNT attribution
    formatted according to academic document standards.

    Args:
        image_url: Direct URL to the image file.
        caption: Figure caption (e.g., "Figura 1: Architecture of the model.").
        abnt_attribution: ABNT-style attribution string for the image source.

    Returns:
        Markdown-formatted string for inserting into the document.
    """
    safe_caption = (caption or "Figure").replace('"', "'")
    lines = [
        f"![{safe_caption}]({image_url})",
        f"*{caption}*",
        f"*{abnt_attribution}*",
    ]
    return "\n\n".join(line for line in lines if line.strip())


@tool
def search_paper_reference(
    title: str,
    year: str = "",
    author_hint: str = "",
) -> dict:
    """Search Google Scholar / Crossref / DOI registries for a paper's full citation.

    Use as fallback after `lookup_page_metadata` when `authors` or `journal`
    fields are empty.  Queries are targeted at citation databases, not general
    web search.

    Args:
        title: Paper title (partial is fine, ≥ 4 words recommended).
        year: Publication year if known (helps narrow results).
        author_hint: First-author surname or partial name if known.

    Returns:
        Dict with keys: title, authors, journal, volume, issue, pages, year,
        doi, source_url, error.  Empty string for each field not found.
    """
    from .tavily_web_search import _get_client

    client = _get_client()

    base = f'"{title[:120]}"' if title else "academic paper"
    queries = [
        f"{base} {year} doi crossref".strip(),
        f"{base} {author_hint} journal citation".strip(),
    ]

    found: dict = {
        "title": title,
        "authors": "",
        "journal": "",
        "volume": "",
        "issue": "",
        "pages": "",
        "year": year,
        "doi": "",
        "source_url": "",
        "error": "",
    }

    for q in queries:
        try:
            ans = client.search(
                query=q[:400],
                search_depth="advanced",
                max_results=5,
                include_raw_content=True,
            )
        except Exception as exc:
            found["error"] = str(exc)
            continue

        for r in ans.get("results", []):
            raw = (r.get("raw_content") or r.get("content") or "")[:3000]
            url = r.get("url", "")

            if not found["doi"]:
                doi_m = re.search(r"\b(10\.\d{4,}/[^\s\"<>\]\)]+)", raw)
                if doi_m:
                    found["doi"] = doi_m.group(1).rstrip(".,)")
                    found["source_url"] = f"https://doi.org/{found['doi']}"

            if not found["year"]:
                yr_m = re.search(r"\b(20\d{2})\b", raw[:500])
                if yr_m:
                    found["year"] = yr_m.group(1)

            if not found["volume"]:
                vol_m = re.search(r"[Vv]ol(?:\.|ume)?\s*\.?\s*(\d+)", raw[:1500])
                if vol_m:
                    found["volume"] = vol_m.group(1)

            if not found["issue"]:
                iss_m = re.search(r"[Nn](?:o|um)\.?\s*(\d+)", raw[:1500])
                if iss_m:
                    found["issue"] = iss_m.group(1)

            if not found["pages"]:
                pg_m = re.search(r"[Pp](?:p|ages?)\.?\s*([\d]+[-\u2013]\d+)", raw[:1500])
                if pg_m:
                    found["pages"] = pg_m.group(1)

            if not found["authors"]:
                for pat in [
                    r"(?:Authors?|By)[:\s]+([A-Z][^\n]{5,150})",
                    r"^([A-Z][a-z]+,\s+[A-Z]\.?[^\n]*(?:;\s*[A-Z][a-z]+[^\n]*)*)",
                ]:
                    am = re.search(pat, raw[:1000], re.MULTILINE)
                    if am:
                        candidate = am.group(1).strip()
                        if len(candidate) > 3:
                            found["authors"] = candidate
                            break

            if not found["journal"]:
                for pat in [
                    r"(?:journal|published in|venue)[:\s]+([A-Z][^.\n]{5,80})",
                    r"(?:Hydrology|Geoscientific|Water Resources|IEEE|Nature|Science|Remote Sensing)\s[A-Za-z &]{3,60}",
                ]:
                    jm = re.search(pat, raw[:2000], re.IGNORECASE)
                    if jm:
                        found["journal"] = jm.group(
                            0
                            if any(
                                kw in pat
                                for kw in ["Hydrology", "IEEE", "Nature", "Science", "Remote"]
                            )
                            else 1
                        ).strip()
                        break

            if not found["source_url"]:
                found["source_url"] = url

            if found["doi"] and found["authors"] and found["journal"]:
                break

        if found["doi"] and found["authors"] and found["journal"]:
            break

    return found


def get_image_tools() -> list:
    """Return all image search and formatting tools."""
    return [
        search_images_with_queries,
        lookup_page_metadata,
        search_paper_reference,
        format_image_markdown,
    ]
