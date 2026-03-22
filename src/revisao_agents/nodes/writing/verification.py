"""
verification.py — adaptive paragraph verification (REACT judge loop).

Contains
--------
_count_verifiable_claims                : count claims that need fact-checking.
_judge_paragraph_improved               : 3-level LLM judge (APPROVED/ADJUSTED/CORRECTED).
_monitor_verification_rate              : decide if more context is needed.
_search_for_additional_content          : complementary web search when rate is low.
_verify_and_correct_adaptative_section  : full adaptive loop (legacy, no anchors).
_verify_paragraph_with_anchor           : anchor-directed single-paragraph check.
_verify_and_correct_section_with_anchor : full anchor-directed adaptive loop.
"""

import re
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...utils.vector_utils.mongodb_corpus import CorpusMongoDB

from ...config import EXTRACT_MIN_CHARS, llm_call
from ...helpers.anchor_helpers import (
    _extract_all_anchors_with_citations,
    _extract_citation_anchor,
    _extract_main_anchor,
)
from ...utils.llm_utils.prompt_loader import load_prompt
from ...utils.search_utils.tavily_client import extract_urls, search_web
from ..writing.text_filters import (
    _ANCHORS_PATTERN,
    _strip_figure_table_refs,
    _strip_justification_blocks,
    _strip_meta_sentences,
)

# ---------------------------------------------------------------------------
# Verifiability heuristic
# ---------------------------------------------------------------------------


def _count_verifiable_claims(paragraph: str) -> int:
    """Estimate the number of specific claims in *paragraph* that must be verified.

    Args:
        paragraph: text of the paragraph to analyze

    Returns:
        An integer count of verifiable claims, capped at 5.
    """
    p = paragraph.strip()
    if len(p) < 80:
        return 0
    if p.startswith("#") or re.match(r"^\s*[-*]\s", p):
        return 0
    if p.startswith("```") or p.startswith("$$") or re.match(r"^\s*\$[^$]+\$", p):
        return 0
    if p.startswith("*Figura") or p.startswith("![") or p.startswith("*Figure"):
        return 0

    num_claims = 0
    num_claims += len(re.findall(r"\b\d+[\d.,]*\b", p))
    num_claims += len(re.findall(r"\b[A-Z][a-z]+\s+(?:et\s+al|[A-Z][a-z]+|\(\d{4}\))", p))
    num_claims += len(re.findall(r"\b(19|20)\d{2}\b|\bv\d+\.\d+", p))
    num_claims += len(re.findall(r"[\+\-\*=/<>]", p))

    assertive = [
        "foi",
        "é",
        "são",
        "demonstra",
        "prova",
        "mostra",
        "evidencia",
        "encontrou",
        "observou",
        "descobriu",
        "propôs",
        "definiu",
        "was",
        "is",
        "are",
        "demonstrates",
        "proves",
        "shows",
        "evidences",
        "found",
        "observed",
        "discovered",
        "proposed",
        "defined",
    ]
    for ass in assertive:
        num_claims += len(re.findall(rf"\b{ass}\b", p, re.IGNORECASE))

    return min(num_claims, 5)


# ---------------------------------------------------------------------------
# Single-paragraph judge
# ---------------------------------------------------------------------------


def _judge_paragraph_improved(
    clean_paragraph: str,
    sources: str,
    section_title: str,
    prompt_dir: str = "technical_writing",
    language: str = "pt",
) -> tuple[str, str, str, bool]:
    """3-level judge:  APPROVED / ADJUSTED / CORRECTED.

    Args:
        clean_paragraph: The text of the paragraph to analyze
        sources: The sources to use for verification
        section_title: The title of the section containing the paragraph
        prompt_dir: The directory containing the prompt templates
        language: The language of the paragraph

    Returns:
        (final_text, level, log_entry, is_verifiable).
    """
    anchors = _ANCHORS_PATTERN.findall(clean_paragraph)
    tem_anchors = len([a for a in anchors if len(a.strip()) > 20]) > 0
    num_claims = _count_verifiable_claims(clean_paragraph)

    if num_claims == 0 and not tem_anchors:
        log_entry = f"⏭️  STRUCTURAL  | {clean_paragraph[:70].replace(chr(10), ' ')}..."
        return clean_paragraph, "STRUCTURAL", log_entry, False

    if tem_anchors and len(clean_paragraph) < 100:
        log_entry = f"✅ APPROVED  | {clean_paragraph[:70].replace(chr(10), ' ')}..."
        return clean_paragraph, "APPROVED", log_entry, True

    if num_claims == 0:
        log_entry = f"✅ GENERAL.CONC  | {clean_paragraph[:70].replace(chr(10), ' ')}..."
        return clean_paragraph, "APPROVED", log_entry, True

    p = load_prompt(
        f"{prompt_dir}/writer_judge",
        clean_paragraph=clean_paragraph,
        section_title=section_title,
        sources=sources,
        language=language,
    )
    ans = llm_call(p.text, temperature=0.1)

    final_text = clean_paragraph
    level = "APPROVED"

    m_dec = re.search(
        r"(?:DECISION|DECIS(?:[ÃA]O|AO))\s*:\s*(APPROVED|ADJUSTED|CORRECTED|APROVADO|AJUSTADO|CORRIGIDO)",
        ans,
        re.IGNORECASE,
    )
    if m_dec:
        level = {
            "APROVADO": "APPROVED",
            "AJUSTADO": "ADJUSTED",
            "CORRIGIDO": "CORRECTED",
        }.get(m_dec.group(1).upper(), m_dec.group(1).upper())

    m_txt = re.search(r"(?:TEXT|TEXTO)\s*:\s*([\s\S]+)", ans, re.IGNORECASE)
    if m_txt:
        candidate = m_txt.group(1).strip()
        candidate = re.sub(
            r"^(?:DECISION|DECIS(?:[ÃA]O|AO))\s*:.*\n?",
            "",
            candidate,
            flags=re.IGNORECASE,
        ).strip()
        candidate = _strip_justification_blocks(candidate)
        candidate = _strip_meta_sentences(candidate).strip()
        candidate = _strip_figure_table_refs(candidate)
        if candidate and len(candidate) > 20:
            final_text = candidate

    trecho = clean_paragraph[:70].replace("\n", " ")
    if level == "APPROVED":
        log_entry = f"✅ APPROVED  | {trecho}..."
    elif level == "ADJUSTED":
        corr = final_text[:70].replace("\n", " ")
        log_entry = f"🔵 ADJUSTED  | {trecho}...\n     → {corr}..."
    else:
        corr = final_text[:70].replace("\n", " ")
        log_entry = f"🔧 CORRECTED | {trecho}...\n     → {corr}..."

    return final_text, level, log_entry, True


# ---------------------------------------------------------------------------
# Verification rate monitor
# ---------------------------------------------------------------------------


def _monitor_verification_rate(stats: dict) -> tuple[bool, str]:
    """Return (needs_more_search, reason) based on current verification stats.

    Args:
        stats: a dictionary with keys 'total', 'verifiable', 'approved', 'adjusted', etc.

    Returns:
        A tuple where the first element is a boolean indicating if more search is needed,
        and the second element is a string explaining the reason.
    """
    total = stats.get("total", 0)
    if total == 0:
        return False, "No paragraphs verified"
    verifiable = stats.get("verifiable", 0)
    if verifiable == 0:
        return False, "No verifiable paragraphs"
    verified = stats.get("approved", 0) + stats.get("adjusted", 0)
    rate = (verified / verifiable * 100) if verifiable > 0 else 100
    if rate < 40:
        return True, f"Critical rate {rate:.0f}%"
    elif rate < 60:
        return True, f"Low rate {rate:.0f}%"
    return False, f"Rate OK {rate:.0f}%"


# ---------------------------------------------------------------------------
# Complementary search
# ---------------------------------------------------------------------------


def _search_for_additional_content(
    section_title: str,
    expected_content: str,
    current_corpus: "CorpusMongoDB",
    attempted_urls: set,
    prompt_dir: str = "technical_writing",
    language: str = "pt",
) -> tuple[int, "CorpusMongoDB", str]:
    """Search for complementary content when paragraph verification rate is low.

    Args:
        - section_title: The title of the section needing more content.
        - expected_content: A brief description of the expected content (used in prompt).
        - current_corpus: The current corpus object to check for existing URLs and to update.
        - attempted_urls: A set of URLs that have already been attempted for extraction.
        - prompt_dir: Directory for prompt templates.
        - language: Language for the prompts.

    Returns:
        A tuple containing:
        - The number of new chunks added to the corpus.
        - The updated corpus object.
        - A message summarizing the search outcome.
    """
    # Import here to avoid circular dependency
    from ...utils.vector_utils.mongodb_corpus import CorpusMongoDB

    print(f"\n      🔄 COMPLEMENTARY SEARCH — {section_title}")

    complementary_queries = []
    try:
        p = load_prompt(
            f"{prompt_dir}/expected_content",
            section_title=section_title,
            expected_content=expected_content[:100],
            language=language,
        )
        ans = llm_call(p.text, temperature=p.temperature)
        complementary_queries = [q.strip() for q in ans.split("\n") if q.strip()][:2]
    except Exception as e:
        print(f"      ⚠️  Erro: {e}")
        complementary_queries = [
            f"{section_title} tutorial",
            f"{section_title} technical",
        ]

    new_numbers = 0
    extracted_new = []

    tavily_enabled = getattr(current_corpus, "tavily_enabled", True)
    if not tavily_enabled:
        print("      ⏭️ Tavily complementary search disabled by user.")
        return 0, current_corpus, "No new content"

    for q in complementary_queries:
        print(f"      • {q[:70]}")
        try:
            ans = search_web(q, max_results=8)
            urls_to_extract = []
            for r in ans:
                u = r.get("url", "")
                if u and u not in attempted_urls and not current_corpus.url_exists(u):
                    urls_to_extract.append(u)
                    attempted_urls.add(u)
                    if len(urls_to_extract) >= 4:
                        break
            if urls_to_extract:
                raw = extract_urls(urls_to_extract)
                for item in raw:
                    if len(item.get("content", "")) >= EXTRACT_MIN_CHARS:
                        extracted_new.append(item)
                        new_numbers += 1
            time.sleep(1)
        except Exception as e:
            print(f"      ⚠️  Error '{q[:50]}': {e}")

    if not extracted_new:
        return 0, current_corpus, "No new content"

    new_corpus = CorpusMongoDB().build(extracted_new, [])
    if new_corpus._n_docs > 0:
        current_corpus._used_urls.extend(new_corpus._used_urls)
        current_corpus._total_chunks += new_corpus._total_chunks
        print(f"      ✅ +{new_numbers} chunks indexed")

    return new_numbers, current_corpus, f"+{new_numbers} chunks"


# ---------------------------------------------------------------------------
# Adaptive verification — legacy (no explicit anchors)
# ---------------------------------------------------------------------------


def _verify_and_correct_adaptative_section(
    text_section: str,
    corpus: "CorpusMongoDB",
    full_corpus_prompt: str,
    title: str,
    section_sources: str,
    expected_content: str = "",
    language: str = "pt",
) -> tuple[str, str, dict]:
    """Adaptive verification without explicit anchor routing (legacy fallback).

    Args:
        text_section: The section text to verify and correct.
        corpus: The corpus object to query for relevant sources.
        full_corpus_prompt: The full corpus context to use as a fallback.
        title: The title of the section (for logging).
        section_sources: The sources relevant to this section (for logging).
        expected_content: A brief description of expected content (used for complementary search).
        language: The language of the section (default is "pt").

    Returns:
        A tuple containing:
            - The corrected section text after adaptive verification.
            - A log report summarizing the verification process.
            - A dictionary of verification statistics.
    """
    attempted_urls: set = set()
    iteration = 0

    while iteration < 3:
        iteration += 1
        print(f"\n     └─ Verification iter {iteration}/3")

        blocks = re.split(r"\n{2,}", text_section.strip())
        result = []
        log_lines = [f"\n### Adaptive Verification — {title} (iter {iteration})"]

        stats = {
            "total": 0,
            "approved": 0,
            "adjusted": 0,
            "corrected": 0,
            "structural": 0,
            "verifiable": 0,
            "skipped": 0,
        }

        for i, block in enumerate(blocks):
            block = block.strip()
            if not block:
                continue

            clean_block = re.sub(r'\[ANCHOR:\s*"[^"]*"\]', "", block).strip()
            clean_block = re.sub(r"  +", " ", clean_block)

            if clean_block.startswith("#") or len(clean_block) < 60:
                result.append(clean_block)
                stats["skipped"] += 1
                continue

            stats["total"] += 1

            sources = corpus.render_prompt(clean_block, max_chars=3000)[0]

            if not sources.strip():
                result.append(clean_block)
                stats["approved"] += 1
                stats["structural"] += 1
                log_lines.append(f"  par.{i+1}: ⏭️  NO SOURCES")
                continue

            final_text, level, log_entry, is_verifiable = _judge_paragraph_improved(
                clean_block, sources, title
            )

            result.append(final_text)
            log_lines.append(f"  par.{i+1}: {log_entry}")

            if is_verifiable:
                stats["verifiable"] += 1
            if "APPROVED" in level or "STRUCTURAL" in level:
                stats["approved"] += 1
            elif "ADJUSTED" in level:
                stats["adjusted"] += 1
            else:
                stats["corrected"] += 1

        needs_more, reason = _monitor_verification_rate(stats)
        verified = stats["approved"] + stats["adjusted"]
        rate = (verified / stats["verifiable"] * 100) if stats["verifiable"] > 0 else 100
        log_lines.append(f"\n**Result:** {verified}/{stats['verifiable']} ({rate:.0f}%) — {reason}")
        print(f"     📊 {rate:.0f}% | {reason}")

        if not needs_more or iteration >= 3:
            break

        num_novos, corpus, msg = _search_for_additional_content(
            title, expected_content, corpus, attempted_urls, language=language
        )
        log_lines.append(f"\n**Search:** {msg}")
        if num_novos == 0:
            break

    corrected_text = "\n\n".join(p for p in result if p)
    corrected_text = re.sub(r'\[ANCHOR:\s*"[^"]*"\]', "", corrected_text)
    corrected_text = re.sub(r"\n{3,}", "\n\n", corrected_text)

    verified = stats["approved"] + stats["adjusted"]
    final_rate = (verified / stats["verifiable"] * 100) if stats["verifiable"] > 0 else 100
    print(f"\n     📊 FINAL: {verified}/{stats['verifiable']} ({final_rate:.0f}%)")

    report = "\n".join(log_lines)
    return corrected_text, report, stats


# ---------------------------------------------------------------------------
# Anchor-directed single paragraph
# ---------------------------------------------------------------------------


def _verify_paragraph_with_anchor(
    block: str,
    corpus: "CorpusMongoDB",
    source_map: dict,
    section_title: str,
    prompt_dir: str = "technical_writing",
    language: str = "pt",
) -> tuple[str, str, str, bool]:
    """Verify one paragraph using its explicit anchors for source retrieval.

    Args:
        block: The paragraph text block to verify (may contain anchors).
        corpus: The corpus object to query for relevant sources.
        source_map: A mapping of citation numbers to URLs for anchor resolution.
        section_title: The title of the section containing the paragraph (for logging).
        prompt_dir: Directory for prompt templates.
        language: The language of the text block.

    Returns:
        A tuple containing:
            - The final paragraph text after verification and possible correction.
            - The verification level (APPROVED, ADJUSTED, CORRECTED, STRUCTURAL).
            - A log entry summarizing the verification outcome.
            - A boolean indicating if the paragraph was considered verifiable.
    """
    clean_block = re.sub(r'\[ANCHOR:\s*"[^"]*"\]', "", block).strip()
    clean_block = re.sub(r"  +", " ", clean_block)

    if clean_block.startswith("#") or len(clean_block) < 60:
        return clean_block, "STRUCTURAL", "⏭️  STRUCTURAL", False

    main_anchor = _extract_main_anchor(block)

    if main_anchor:
        citation_num = _extract_citation_anchor(block, main_anchor)
        if citation_num and citation_num in source_map:
            cited_url = source_map[citation_num]
            print(f"     🎯 Anchor found ({len(main_anchor)} chars) → [{citation_num}]")
            print(f"        URL: {cited_url[:60]}")

            sources, used_urls, n_chunks = corpus.render_prompt_url(
                anchor_text=main_anchor,
                cited_urls=cited_url,
                max_chars=3000,
                top_k=5,
                include_neighbors=True,
                neighbor_window=2,
            )
            if sources:
                print(f"        ✅ {n_chunks} chunks from cited URL")
            else:
                print("        ⚠️  No chunks found, using general search")
                sources = corpus.render_prompt(clean_block[:300], max_chars=3000)[0]
        else:
            print("     ⚠️  Anchor without valid citation")
            sources = corpus.render_prompt(clean_block[:300], max_chars=3000)[0]
    else:
        anchors_with_cit = _extract_all_anchors_with_citations(block)
        if anchors_with_cit:
            anchors_with_urls = [
                (at, source_map[nc]) for at, nc in anchors_with_cit if nc in source_map
            ]
            if anchors_with_urls:
                print(f"     🎯 {len(anchors_with_urls)} anchors with URLs")
                sources, used_urls, n_chunks = corpus.render_prompt_anchors(
                    anchors_com_urls=anchors_with_urls,
                    max_chars=3000,
                )
                if sources:
                    print(f"        ✅ {n_chunks} chunks from cited URLs")
                else:
                    sources = corpus.render_prompt(clean_block[:300], max_chars=3000)[0]
            else:
                sources = corpus.render_prompt(clean_block[:300], max_chars=3000)[0]
        else:
            sources = corpus.render_prompt(clean_block[:300], max_chars=3000)[0]

    if not sources.strip():
        return clean_block, "APPROVED", "✅ NO SOURCES", True

    return _judge_paragraph_improved(
        clean_block, sources, section_title, prompt_dir=prompt_dir, language=language
    )


# ---------------------------------------------------------------------------
# Anchor-directed adaptive loop (main)
# ---------------------------------------------------------------------------


def _verify_and_correct_section_with_anchor(
    section_text: str,
    corpus: "CorpusMongoDB",
    source_map: dict,
    title: str,
    expected_content: str = "",
    prompt_dir: str = "technical_writing",
    language: str = "pt",
) -> tuple[str, str, dict]:
    """Full adaptive verification loop using explicit anchors for directed retrieval.

    Args:
        section_text: The full text of the section to verify and correct (may contain anchors).
        corpus: The corpus object to query for relevant sources.
        source_map: A mapping of citation numbers to URLs for anchor resolution.
        title: The title of the section (for logging).
        expected_content: A brief description of expected content (used for complementary search).
        prompt_dir: Directory for prompt templates.
        language: The language of the section text.

    Returns:
        A tuple containing:
            - The corrected section text.
            - A log report of the verification process.
            - A dictionary of verification statistics.
    """
    attempted_urls: set = set()
    iteration = 0

    while iteration < 3:
        iteration += 1
        print(f"\n     └─ Verification iter {iteration}/3 (with anchors)")

        blocks = re.split(r"\n{2,}", section_text.strip())
        result = []
        log_lines = [f"\n### Verification with Anchors — {title} (iter {iteration})"]

        stats = {
            "total": 0,
            "approved": 0,
            "adjusted": 0,
            "corrected": 0,
            "structural": 0,
            "verifiable": 0,
            "skipped": 0,
            "anchors_used": 0,
        }

        for i, block in enumerate(blocks):
            block = block.strip()
            if not block:
                continue

            if bool(re.search(r"\[ANCHOR:", block)):
                stats["anchors_used"] += 1

            final_text, level, log_entry, is_verifiable = _verify_paragraph_with_anchor(
                block=block,
                corpus=corpus,
                source_map=source_map,
                section_title=title,
                prompt_dir=prompt_dir,
                language=language,
            )

            result.append(final_text)
            log_lines.append(f"  par.{i+1}: {log_entry}")
            stats["total"] += 1
            if is_verifiable:
                stats["verifiable"] += 1
            if "APPROVED" in level or "STRUCTURAL" in level:
                stats["approved"] += 1
            elif "ADJUSTED" in level:
                stats["adjusted"] += 1
            else:
                stats["corrected"] += 1

        needs_more, reason = _monitor_verification_rate(stats)
        verified = stats["approved"] + stats["adjusted"]
        rate = (verified / stats["verifiable"] * 100) if stats["verifiable"] > 0 else 100
        log_lines.append(f"\n**Result:** {verified}/{stats['verifiable']} ({rate:.0f}%) — {reason}")
        log_lines.append(f"**Anchors used:** {stats['anchors_used']} paragraphs")
        print(f"     📊 {rate:.0f}% | {reason} | {stats['anchors_used']} anchors")

        if not needs_more or iteration >= 3:
            break

        num_new, corpus, msg = _search_for_additional_content(
            title,
            expected_content,
            corpus,
            attempted_urls,
            prompt_dir=prompt_dir,
            language=language,
        )
        log_lines.append(f"\n**Search:** {msg}")
        if num_new == 0:
            break

    corrected_text = "\n\n".join(p for p in result if p)
    corrected_text = re.sub(r'\[ANCHOR:\s*"[^"]*"\]', "", corrected_text)
    corrected_text = re.sub(r"\n{3,}", "\n\n", corrected_text)

    verified = stats["approved"] + stats["adjusted"]
    final_rate = (verified / stats["verifiable"] * 100) if stats["verifiable"] > 0 else 100
    print(f"\n     📊 FINAL: {verified}/{stats['verifiable']} ({final_rate:.0f}%)")
    print(f"     🎯 Anchors used: {stats['anchors_used']} paragraphs")

    report = "\n".join(log_lines)
    return corrected_text, report, stats
