"""
write_sections_node — writes sections using search, extraction and verification
Part of the nodes/writing subpackage.
"""

# 1. Standard Library Imports
import logging
import re
import time
from datetime import datetime

# 2. Local/Existing Codebase Imports (MOVIDOS PARA CIMA)
from ...config import (
    CTX_ABSTRACT_CHARS,
    DELAY_BETWEEN_SECTIONS,
    MAX_CORPUS_PROMPT,
    TECHNICAL_MAX_RESULTS,
)
from ...core.schemas.writer_config import WriterConfig
from ...helpers.anchor_helpers import _ANCHORS_PATTERN
from ...state import TechnicalWriterState
from ...utils.file_utils.helpers import summarize_section
from ...utils.search_utils.tavily_client import search_images, search_web
from ...utils.vector_utils.mongodb_corpus import CorpusMongoDB
from .phase_runners import (
    _draft_phase,
    _extract_with_fallback,
    _observation_phase,
    _thought_phase,
)
from .verification import _verify_and_correct_section_with_anchor

# 3. Global Variables and Logger (ABAIXO DE TODOS OS IMPORTS)
logger = logging.getLogger(__name__)


def write_sections_node(state: TechnicalWriterState) -> dict:
    """Main node for writing sections with search, extraction and verification.

    Args:
        state (TechnicalWriterState): The current state of the technical writer, expected to contain:
            - "theme": str, the overall theme of the document.
            - "sections": list of dicts, each with "title", "expected_content", and "resources".
            - "refs_urls": list of URLs collected so far (optional).
            - "refs_images": list of images collected so far (optional).
            - "cumulative_summary": str, summary of written content so far (optional).
            - "react_log": list of log entries for debugging (optional).
            - "verification_stats": list of dicts with verification statistics per section (optional).
            - "writer_config": dict, optional configuration for writing (e.g., prompt_dir, language, min_sources_per_section).

    Returns:
        dict: Updated state with written sections, collected references, cumulative summary, logs, and verification stats.
    """
    config = WriterConfig.from_dict(state.get("writer_config", {}))
    is_pt = config.language == "pt"
    labels = {
        "images_heading": "IMAGENS DISPONÍVEIS" if is_pt else "AVAILABLE IMAGES",
        "image_desc": "Desc imagem" if is_pt else "Image desc",
        "image_source": "Fonte imagem" if is_pt else "Image source",
        "reliability": "Confiabilidade" if is_pt else "Reliability",
        "manual_review": (
            "Revisão manual pode ser necessária." if is_pt else "Manual review may be necessary."
        ),
        "verification": "Verificação" if is_pt else "Verification",
        "verified_paragraphs": (
            "dos parágrafos verificados" if is_pt else "of paragraphs verified"
        ),
        "references_section": (
            "Referências desta seção" if is_pt else "References for this section"
        ),
        "section_marker": "Seção" if is_pt else "Section",
    }
    prompt_dir = config.prompt_dir
    theme = state["theme"]
    sections = state["sections"]
    written_sections = []
    all_refs_urls = list(state.get("refs_urls", []))
    all_refs_images = list(state.get("refs_images", []))
    cumulative_summary = state.get("cumulative_summary", "")
    react_log = list(state.get("react_log", []))
    verification_stats = list(state.get("verification_stats", []))
    n_total = len(sections)
    all_titles = [s["title"] for s in sections]
    # CorpusMongoDB instance for URL existence checks (no build)
    corpus_check = CorpusMongoDB()

    tavily_enabled = state.get("tavily_enabled", True)
    for pos, section in enumerate(sections):
        title = section["title"]
        expected_content = section.get("expected_content", "")
        resources = section.get("resources", "")
        index_num = section.get("index", pos)

        print(f"\n{'━' * 70}")
        print(f"  [{pos + 1}/{n_total}] PROCESSING: {title}")
        print(f"{'━' * 70}")

        log = [
            f"\n{'=' * 70}",
            f"SECTION [{pos + 1}/{n_total}]: {title}",
            f"Timestamp: {datetime.now().isoformat()}",
            f"{'=' * 70}",
        ]

        # PHASE 1: Thought
        print("\n  🧠 PHASE 1 — Thought...")
        log.append("\n── PHASE 1: THOUGHT ──")
        plan = _thought_phase(
            theme,
            title,
            expected_content,
            resources,
            prompt_dir=prompt_dir,
            language=config.language,
        )
        search_queries = plan.get("search_queries", [f"{theme} {title}"])
        image_queries = plan.get("image_queries", [f"{title} diagram"])
        necessary_information = plan.get("necessary_information", [expected_content[:120]])
        log.extend([f"Queries: {search_queries}", f"Information: {necessary_information}"])
        print(f"     Queries: {search_queries}")

        # PHASES 2-4: Search + Extraction
        print("\n  🔎 PHASE 2-4 — Search and Extraction...")
        log.append("\n── PHASE 2-4: SEARCH + EXTRACTION ──")
        extracted = []
        results = []
        urls_seen = set()

        # Corpus-first strategy: query existing MongoDB before hitting the web
        _corpus_sufficient = False
        if config.is_corpus_first:
            print("\n  🔬 PHASE 5 — Observation (corpus-first, before search)...")
            log.append("\n── PHASE 5: OBSERVATION (corpus-first) ──")
            obs = _observation_phase(
                necessary_information,
                corpus_check,
                prompt_dir=prompt_dir,
                language=config.language,
            )
            _corpus_sufficient = obs.get("sufficient", False)
            log.append(f"Corpus sufficient: {_corpus_sufficient} | {obs.get('summary', '')[:120]}")
            print(f"     Corpus sufficient: {_corpus_sufficient}")

        if not _corpus_sufficient and tavily_enabled:
            for q in search_queries[:4]:
                res = search_web(q, TECHNICAL_MAX_RESULTS)
                # Pass tavily_enabled to corpus for fallback
                corpus_check.tavily_enabled = tavily_enabled
                new_extracted, results, urls_seen = _extract_with_fallback(
                    res,
                    queries_fallback=[q, title],
                    urls_tried=urls_seen,
                    corpus=corpus_check,
                )
                extracted.extend(new_extracted)
                time.sleep(1)
        elif not _corpus_sufficient and not tavily_enabled:
            print("  ⏭️ Tavily web search disabled by user. Skipping web search.")

        log.append(f"Sources extracted: {len(extracted)}")

        # MongoDB Indexing / corpus selection
        print("\n  🗄️  Indexing in MongoDB...")
        log.append("\n── MONGODB INDEXING ──")
        slug_section = re.sub(r"[^\w]", "_", title[:30]).lower()
        prefix = f"s{pos + 1:02d}_{slug_section}"

        if _corpus_sufficient:
            # Reuse the global check corpus — no new documents to index
            corpus = corpus_check
        else:
            corpus = CorpusMongoDB().build(extracted, results, prefix=prefix)

        query_retrieval = f"{title} {expected_content} {resources}"
        corpus_prompt, section_urls, _ = corpus.render_prompt(
            query_retrieval, max_chars=MAX_CORPUS_PROMPT
        )
        log.append(f"MongoDB: {corpus._n_docs} docs | {corpus._total_chunks} chunks")

        if not corpus_prompt.strip() and not _corpus_sufficient and tavily_enabled:
            print("  ⚠️  Empty corpus! Emergency search...")
            log.append("⚠️  Empty corpus — emergency search")
            q_emergency = f"{title} {theme} technical documentation filetype:pdf"
            res_emergency = search_web(q_emergency, 6)
            corpus_check.tavily_enabled = tavily_enabled
            new_emergency, _, urls_seen = _extract_with_fallback(
                res_emergency,
                queries_fallback=[title],
                urls_tried=urls_seen,
                corpus=corpus_check,
            )
            if new_emergency:
                extracted.extend(new_emergency)
                corpus = CorpusMongoDB().build(extracted, results, prefix=prefix)
                corpus_prompt, section_urls, _ = corpus.render_prompt(
                    query_retrieval, max_chars=MAX_CORPUS_PROMPT
                )
                log.append(f"Emergency: {len(new_emergency)} additional sources")
        elif not corpus_prompt.strip() and not _corpus_sufficient and not tavily_enabled:
            print("  ⏭️ Tavily emergency search disabled by user. Skipping.")

        if not corpus_prompt.strip():
            print("  ❌ CRITICAL FAILURE: no sources found.")
            log.append("❌ No sources found.")
            corpus_prompt = (
                "WARNING: No sources found. Write only widely established concepts, "
                "without specific claims with anchors."
            )

        # PHASE 5: Observation (skipped in web-first mode)
        if not config.is_corpus_first:
            print("\n  🔬 PHASE 5 — Observation (skipped)...")
            log.append("\n── PHASE 5: OBSERVATION (skipped) ──")

        # Images
        print("\n  🖼️  Searching for images...")
        images = []
        if tavily_enabled:
            images = search_images(image_queries, max_results=3)
        else:
            print("  ⏭️ Tavily image search disabled by user. Skipping.")
        images_text = ""
        for i, img in enumerate(images, 1):
            url_image = img.get("url_imagem", "")
            description = img.get("descricao", "") or "(no description)"
            source = img.get("titulo_pagina", img.get("url_origem", ""))
            images_text += (
                f"  [{i}] {url_image}\n"
                f"       {labels['image_desc']}: {description}\n"
                f"       {labels['image_source']}: {source}\n"
            )

        complete_reference = corpus_prompt
        if images_text:
            complete_reference += f"\n\n{labels['images_heading']}:\n{images_text}"

        # PHASE 6: Anchored Draft
        print("\n  ✍️  PHASE 6 — Anchored draft...")
        log.append("\n── PHASE 6: DRAFT ──")
        draft, urls_used_phase6 = _draft_phase(
            theme,
            title,
            expected_content,
            resources,
            complete_reference,
            section_urls,
            cumulative_summary,
            pos,
            n_total,
            all_titles,
            len(extracted),
            prompt_dir=prompt_dir,
            language=config.language,
            min_sources=config.min_sources_per_section,
        )
        # Track source map
        source_map_section = {}
        for i, source in enumerate(urls_used_phase6, 1):
            if hasattr(source, "id") and hasattr(source, "url"):
                source_map_section[source.id] = source.url
            elif isinstance(source, dict):
                source_map_section[source.get("id", i)] = source.get("url", "")
            else:
                source_map_section[i] = str(source)

        # ── Source diversity check ─────────────────────────────────────
        min_src = config.min_sources_per_section
        n_distinct = len(set(source_map_section.values()))
        if min_src > 0 and n_distinct < min_src:
            print(f"     ⚠️  Only {n_distinct} distinct sources (minimum: {min_src}). Retrying...")
            log.append(f"⚠️  Retry: {n_distinct}/{min_src} distinct sources")
            diversity_hint = (
                f"\n\n{'━' * 60}\n"
                f"MANDATORY INSTRUCTION: Use at least {min_src} DISTINCT sources in this section.\n"
                f"Distribute citations among different corpus documents.\n"
                f"DO NOT rely on only 1-2 papers.\n"
                f"{'━' * 60}\n"
            )
            draft_retry, urls_retry = _draft_phase(
                theme,
                title,
                expected_content,
                resources,
                complete_reference + diversity_hint,
                section_urls,
                cumulative_summary,
                pos,
                n_total,
                all_titles,
                len(extracted),
                prompt_dir=prompt_dir,
                language=config.language,
                min_sources=config.min_sources_per_section,
            )
            source_map_retry = {}
            for i, source in enumerate(urls_retry, 1):
                if hasattr(source, "id") and hasattr(source, "url"):
                    source_map_retry[source.id] = source.url
                elif isinstance(source, dict):
                    source_map_retry[source.get("id", i)] = source.get("url", "")
                else:
                    source_map_retry[i] = str(source)
            n_distinct_retry = len(set(source_map_retry.values()))
            if n_distinct_retry > n_distinct:
                print(f"     ✅ Retry improved: {n_distinct_retry} distinct sources")
                draft = draft_retry
                urls_used_phase6 = urls_retry
                source_map_section = source_map_retry
                n_distinct = n_distinct_retry
            else:
                print(
                    f"     ℹ️  Retry did not improve ({n_distinct_retry} sources). Keeping original."
                )
            if n_distinct < min_src:
                log.append(
                    f"<!-- WARNING: only {n_distinct} distinct sources used (min: {min_src}) -->"
                )

        n_anchors = len(_ANCHORS_PATTERN.findall(draft))
        log.append(f"Draft: {len(draft):,} chars | {n_anchors} anchors (hints)")
        print(f"     {len(draft):,} chars | {n_anchors} anchors")

        # PHASE 7: Adaptive verification with REACT loop
        print("\n  🔍 PHASE 7 — Adaptive verification...")
        log.append("\n── PHASE 7: ADAPTIVE VERIFICATION (REACT) ──")
        final_text, verification_report, stats = _verify_and_correct_section_with_anchor(
            draft,
            corpus,
            source_map_section,
            title,
            expected_content,
            prompt_dir=prompt_dir,
            language=config.language,
        )

        log.append(verification_report)
        verification_stats.append({"section": title, **stats})

        if not final_text.strip().startswith("## "):
            final_text = f"## {title}\n\n{final_text.strip()}"

        verifiable = stats.get("verifiable", 0)
        if verifiable == 0:
            verifiable = stats.get("total", 1)

        verified = stats.get("approved", 0) + stats.get("adjusted", 0)
        rate = (verified / verifiable * 100) if verifiable > 0 else 100

        corrected_count = stats.get("corrected", 0)
        if stats["total"] > 0 and (rate < 40 or corrected_count > stats["total"] * 0.3):
            warning = (
                f"> ⚠️ **{labels['reliability']}: {rate:.0f}%** "
                f"({verified}/{verifiable} verified). "
                f"{labels['manual_review']}\n\n"
            )
            final_text = re.sub(r"(## .+?\n)", r"\1\n" + warning, final_text, count=1)
        elif rate < 60 and stats["total"] > 0:
            warning = f"> ℹ️ **{labels['verification']}**: {rate:.0f}% {labels['verified_paragraphs']}.\n\n"
            final_text = re.sub(
                r"(## .+?\n)", r"\1\n" + warning, final_text, count=1, flags=re.DOTALL
            )

        # Add per-section references
        print("\n  📚 Adding section references...")

        found_citations = set()
        for match in re.finditer(r"\[(\d+)\]", final_text):
            num = int(match.group(1))
            found_citations.add(num)

        all_corpus_urls = corpus._used_urls if hasattr(corpus, "_used_urls") else section_urls

        section_references = []
        missing_urls = []

        for idx in sorted(found_citations):
            # Priority 1: use the source_map built from _draft_phase (id → url)
            if idx in source_map_section:
                url = source_map_section[idx]
                section_references.append(f"[{idx}] {url}")
            # Priority 2: fall back to ordered URL list from corpus
            elif 1 <= idx <= len(all_corpus_urls):
                url = all_corpus_urls[idx - 1]
                section_references.append(f"[{idx}] {url}")
            else:
                missing_urls.append(idx)

        if section_references:
            final_text += f"\n\n### {labels['references_section']}\n\n"
            final_text += "\n".join(section_references)
            print(f"     ✅ {len(section_references)} references added")
            if missing_urls:
                print(f"     ⚠️  Citations without URL: {missing_urls}")
        else:
            print("     ⚠️  No citations found in this section")

        print(f"  ✅ [{pos + 1}/{n_total}] Section completed ({rate:.0f}% verified)")

        written_sections.append(
            {
                "index": index_num,
                "title": title,
                "text": final_text,
                "urls_used": section_urls,
                "source_map": source_map_section,
                "images": images,
            }
        )

        for u in section_urls:
            if u not in all_refs_urls:
                all_refs_urls.append(u)

        for img in images:
            if img not in all_refs_images:
                all_refs_images.append(img)

        section_summary = summarize_section(title, final_text)
        if cumulative_summary:
            cumulative_summary += (
                f"\n\n[{labels['section_marker']} {pos + 1}: {title}] {section_summary}"
            )
        else:
            cumulative_summary = (
                f"[{labels['section_marker']} {pos + 1}: {title}] {section_summary}"
            )
        if len(cumulative_summary) > CTX_ABSTRACT_CHARS * 3:
            split_marker = f"\n\n[{labels['section_marker']} "
            parts = cumulative_summary.split(split_marker)
            cumulative_summary = split_marker.join([""] + parts[-4:]).strip()

        react_log.extend(log)

        if pos < n_total - 1:
            print(f"\n  ⏳ Waiting {DELAY_BETWEEN_SECTIONS}s...")
            time.sleep(DELAY_BETWEEN_SECTIONS)

    return {
        "written_sections": written_sections,
        "refs_urls": all_refs_urls,
        "refs_images": all_refs_images,
        "cumulative_summary": cumulative_summary,
        "react_log": react_log,
        "verification_stats": verification_stats,
        "status": "sections_written",
    }
