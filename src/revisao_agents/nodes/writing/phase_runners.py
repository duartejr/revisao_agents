"""
phase_runners.py — the individual LLM writing phases (1 – 6).

Phases
------
1  _thought_phase         : plan queries and required information.
5  _observation_phase     : check if the existing corpus is sufficient.
6  _draft_phase           : generate the anchored draft via LLM.
   _extract_with_fallback : URL extraction with Tavily + fallback retry.
"""

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ...utils.vector_utils.mongodb_corpus import CorpusMongoDB

from ...config import (
    llm_call, parse_json_safe,
    EXTRACT_MIN_CHARS, MAX_URLS_EXTRACT, CTX_ABSTRACT_CHARS, 
    MIN_SECTION_PARAGRAPHS, TOP_K_OBSERVATION,
)
from ...core.schemas.techinical_writing import SectionAnswer
from ...utils.llm_utils.prompt_loader import load_prompt
from ...utils.search_utils.tavily_client import search_web, extract_urls, score_url


# ---------------------------------------------------------------------------
# Phase 1: Thought (planning)
# ---------------------------------------------------------------------------

def _thought_phase(
    theme: str,
    title: str,
    objective: str,
    resources: str,
    prompt_dir: str = "technical_writing",
    language: str = "pt",
) -> dict:
    """Phase 1: generate search queries and list of required information.
    
    Args:
        theme: The overall theme of the chapter or document.
        title: The specific section title being planned.
        objective: The specific content objectives for this section.
        resources: Any mandatory resources or references to include.
        prompt_dir: Directory where the LLM prompt templates are stored.
    
    Returns:
        A dict containing:
        - necessary_information: List of key information points needed for the section.
        - search_queries: List of search query strings to find relevant sources.
        - image_queries: List of search query strings to find relevant images/diagrams.
    """
    prompt = load_prompt(
        f"{prompt_dir}/thought_phase",
        theme=theme, title=title, objective=objective, resources=resources,
        language=language,
    )
    ans = llm_call(prompt.text, temperature=prompt.temperature)
    result = parse_json_safe(ans)
    if result:
        return result
    return {
        "necessary_information": [objective[:120]],
        "search_queries": [f"{theme} {title}", f"{title} technical details"],
        "image_queries": [f"{title} diagram architecture"],
    }


# ---------------------------------------------------------------------------
# Phase 5: Observation (corpus-sufficiency check)
# ---------------------------------------------------------------------------

def _observation_phase(
    necessary_information: List[str],
    corpus: "CorpusMongoDB",
    prompt_dir: str = "technical_writing",
    language: str = "pt",
) -> dict:
    """Phase 5: decide if the existing corpus is sufficient to write the section.
    
    Args:
        necessary_information: List of key information points needed for the section (from thought phase).
        corpus: The CorpusMongoDB instance to query for relevant sources.
        prompt_dir: Directory where the LLM prompt templates are stored.
        language: The language to use for the prompts.
    
    Returns:
        A dict containing:
        - sufficient: bool indicating if the corpus is sufficient.
        - gaps: List of information points that are missing from the corpus.
        - complementary_query: A search query to find missing information (if not sufficient).
        - summary: A brief summary of the relevant corpus content (if sufficient).
    """
    if corpus._n_docs == 0:
        return {
            "sufficient": False,
            "gaps": necessary_information,
            "complementary_query": necessary_information[0] if necessary_information else "",
            "summary": "Corpus vazio.",
        }

    query_obs = " ".join(necessary_information[:3])
    chunks_obs = corpus.query(query_obs, top_k=TOP_K_OBSERVATION)
    sample_corpus = "\n\n".join(c.text for c in chunks_obs)[:4000]

    information_list = "\n".join(f"- {i}" for i in necessary_information)
    prompt_obs = load_prompt(
        f"{prompt_dir}/observation_phase",
        information_list=information_list,
        sample_corpus=sample_corpus,
        language=language,
    )
    ans = llm_call(prompt_obs.text, temperature=prompt_obs.temperature)
    result = parse_json_safe(ans)
    if result:
        return result
    return {
        "sufficient": True,
        "gaps": [],
        "complementary_query": None,
        "summary": sample_corpus[:200],
    }


# ---------------------------------------------------------------------------
# Phase 6: Draft (anchored draft generation)
# ---------------------------------------------------------------------------

def _draft_phase(
    theme: str,
    title: str,
    objective: str,
    resources: str,
    corpus: str,
    section_urls: List[str],
    cumulative_summary: str,
    pos: int,
    n_total: int,
    all_titles: List[str],
    n_extracted: int,
    prompt_dir: str = "technical_writing",
    language: str = "pt",
    min_sources: int = 0,
) -> tuple:
    """Phase 6: generate the anchored draft for one section using the LLM.
    
    Args:
        theme: The overall theme of the chapter or document.
        title: The specific section title being drafted.
        objective: The specific content objectives for this section.
        resources: Any mandatory resources or references to include.
        corpus: The assembled corpus text to use as context for writing.
        section_urls: List of URLs of the sources included in the corpus.
        cumulative_summary: A summary of the content already written in previous sections.
        pos: The position of this section in the overall structure (0-indexed).
        n_total: The total number of sections in the chapter/document.
        all_titles: List of all section titles in the chapter/document.
        n_extracted: The number of sources extracted for the corpus.
        prompt_dir: Directory where the LLM prompt templates are stored.
        language: The language to use for the prompts.
        min_sources: The minimum number of sources to include in the draft.
    
    Returns:
        A tuple containing:
        - draft: The generated draft text for the section.
        - used_sources: List of URLs of the sources that were used in the draft.
    """
    previous_ctx = ""
    if cumulative_summary.strip():
        previous_ctx = (
            "══ SECTIONS ALREADY WRITTEN (do not repeat these concepts) ══\n"
            f"{cumulative_summary[:CTX_ABSTRACT_CHARS]}\n"
            "══════════════════════════════════════════════════════\n\n"
        )

    all_txt = "\n".join(
        f"  {'→ ' if i == pos else '  '}{i+1}. {t}"
        for i, t in enumerate(all_titles)
    )

    instructions = load_prompt(
        f"{prompt_dir}/draft_phase",
        section_min_paragraphs=MIN_SECTION_PARAGRAPHS,
        language=language,
        min_sources=min_sources if min_sources > 0 else 2,
    )
    prompt = (
        f"THEMe: {theme}\n"
        f"SECTION: {pos+1}/{n_total} — {title}\n"
        f"OBJECTIVES: {objective}\n"
        f"MANDATORY RESOURCES: {resources if resources else 'as per technical content'}\n\n"
        f"CHAPTER STRUCTURE:\n{all_txt}\n\n"
        f"{previous_ctx}"
        f"{'━'*60}\n"
        f"SOURCE CORPUS — {n_extracted} indexed documents "
        f"(below: most relevant excerpts retrieved by similarity)\n"
        f"{'━'*60}\n"
        f"{corpus}\n\n"
        + instructions.text
        + f"\n## {title}\n"
    )
    result: SectionAnswer = llm_call(
        prompt, temperature=instructions.temperature, response_schema=SectionAnswer
    )
    draft = result.draft
    used_sources = result.used_sources

    return draft, used_sources


# ---------------------------------------------------------------------------
# URL extraction helper (phases 2-4)
# ---------------------------------------------------------------------------

def _extract_with_fallback(
    results: List[dict],
    queries_fallback: List[str],
    urls_attempted: set,
    corpus: "CorpusMongoDB",
) -> tuple:
    """Extract full text from priority URLs with Tavily; retry with fallback queries.

    Returns (valid_extracted, enriched_results, urls_attempted).
    URLs already indexed in MongoDB are skipped.

    Args:
        - results: List of search result dicts containing at least 'url', 'snippet',
            and 'score' keys.
        - queries_fallback: List of alternative search queries to try if initial extraction fails.
        - urls_attempted: Set of URLs that have already been attempted for extraction (to avoid duplicates).
        - corpus: The CorpusMongoDB instance to check for existing URLs.
    
    Returns:
        - valid_extracted: List of dicts with 'url' and 'content' for successfully extracted sources.
        - enriched_results: The original results list, potentially enriched with fallback results.
        - urls_attempted: Updated set of URLs that have been attempted.
    """
    scored = sorted(
        [
            (
                r.get("url", ""),
                score_url(r.get("url", ""), r.get("snippet", ""), float(r.get("score", 0))),
            )
            for r in results if r.get("url")
        ],
        key=lambda x: x[1],
        reverse=True,
    )

    urls_to_extract = []
    for url, sc in scored:
        if url in urls_attempted:
            continue
        if corpus.url_exists(url):
            print(f"      ⏭️ URL already indexed, skipping extraction: {url[:144]}")
            urls_attempted.add(url)
            continue
        urls_to_extract.append(url)
        if len(urls_to_extract) >= MAX_URLS_EXTRACT:
            break

    if not urls_to_extract:
        return [], results, urls_attempted

    urls_attempted.update(urls_to_extract)
    valid = []
    flawed = []

    tavily_enabled = getattr(corpus, "tavily_enabled", True)
    if tavily_enabled:
        raw = extract_urls(urls_to_extract)
        for item in raw:
            url = item.get("url", "")
            c = item.get("content", "")
            if len(c) >= EXTRACT_MIN_CHARS:
                valid.append(item)
                print(f"      ✅ {url[:144]} ({len(c):,} chars)")
            else:
                flawed.append(url)
                print(f"      ✖  {url[:144]} (<{EXTRACT_MIN_CHARS} chars)")
    else:
        print("      ⏭️ Tavily search/extract disabled by user.")
        flawed.extend(urls_to_extract)

    if len(flawed) > len(valid) and queries_fallback and tavily_enabled:
        print(f"      🔄 {len(flawed)} failure(s) → seeking alternatives...")
        for q in queries_fallback[:2]:
            res_alt = search_web(f"{q} filetype:pdf", max_results=6)
            for r in res_alt:
                u = r.get("url", "")
                if u and u not in urls_attempted and not corpus.url_exists(u):
                    urls_attempted.add(u)
                    for item in extract_urls([u]):
                        if len(item.get("content", "")) >= EXTRACT_MIN_CHARS:
                            valid.append(item)
                            results.extend(res_alt)
                            print(f"      ✅ Fallback: {item.get('url', '')[:144]}")
            if valid:
                break
    elif len(flawed) > len(valid) and queries_fallback and not tavily_enabled:
        print("      ⏭️ Tavily fallback search disabled by user.")

    return valid, results, urls_attempted
