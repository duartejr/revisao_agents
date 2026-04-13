"""
consolidate_node — consolidates all written sections into a final document
Part of the nodes/writing subpackage.
"""

import logging
import os
import re
from datetime import datetime

from ...config import REVIEWS_DIR, llm_call
from ...core.schemas.writer_config import WriterConfig
from ...state import TechnicalWriterState
from ...utils.llm_utils.prompt_loader import load_prompt
from .text_filters import _strip_figure_table_refs

logger = logging.getLogger(__name__)


def consolidate_node(state: TechnicalWriterState) -> dict:
    """Consolidates written sections into a final document.

    Args:
        state: The current state of the technical writer, containing all sections, stats, and references.

    Returns:
        dict: Updated state with consolidation status.
    """
    config = WriterConfig.from_dict(state.get("writer_config", {}))
    is_pt = config.language == "pt"
    labels = {
        "type": "Tipo" if is_pt else "Type",
        "paragraph_verification": (
            "Verificação de parágrafos" if is_pt else "Paragraph verification"
        ),
        "verified": "verificados" if is_pt else "verified",
        "approved": "aprovados" if is_pt else "approved",
        "adjusted": "ajustados" if is_pt else "adjusted",
        "corrected": "corrigidos" if is_pt else "corrected",
        "sources": "Fontes" if is_pt else "Sources",
        "sections": "Seções" if is_pt else "Sections",
        "summary": "RESUMO" if is_pt else "SUMMARY",
        "introduction": "Introdução" if is_pt else "Introduction",
        "conclusion": "Conclusão" if is_pt else "Conclusion",
        "references_section": (
            "Referências desta seção" if is_pt else "References for this section"
        ),
        "paragraphs": "Parágrafos" if is_pt else "Paragraphs",
        "generated": "Gerado" if is_pt else "Generated",
    }
    theme = state["theme"]
    sections = sorted(state["written_sections"], key=lambda s: s["index"])
    all_urls = list(dict.fromkeys(state.get("refs_urls", [])))
    react_log = state.get("react_log", [])
    stats_global = state.get("verification_stats", [])
    final_summary = state.get("cumulative_summary", "")[:1000]

    print(f"\n📚 Consolidating {len(sections)} sections...")

    total_par = sum(s.get("total", 0) for s in stats_global)
    total_aprov = sum(s.get("approved", 0) for s in stats_global)
    total_ajust = sum(s.get("adjusted", 0) for s in stats_global)
    total_corr = sum(s.get("corrected", 0) for s in stats_global)
    total_verif = total_aprov + total_ajust
    taxa_global = (total_verif / total_par * 100) if total_par > 0 else 100

    print(
        f"   📊 {total_verif}/{total_par} verified ({taxa_global:.0f}%) "
        f"— ✅{total_aprov} approved  🔵{total_ajust} adjusted  "
        f"🔧{total_corr} corrected | {len(all_urls)} sources"
    )

    titles = [s["title"] for s in sections]
    p_intro = load_prompt(
        f"{config.prompt_dir}/consolidate_intros",
        theme=theme,
        titles=", ".join(titles),
        language=config.language,
    )
    ans_intro = llm_call(p_intro.text, temperature=p_intro.temperature)
    p_concl = load_prompt(
        f"{config.prompt_dir}/consolidate_conclusion",
        theme=theme,
        final_summary=final_summary,
        language=config.language,
    )
    ans_concl = llm_call(p_concl.text, temperature=p_concl.temperature)

    parts = [
        f"# {theme}\n",
        f"> **{labels['type']}:** {config.review_type_label}\n",
        f"> **{labels['paragraph_verification']}:** {total_verif}/{total_par} {labels['verified']} "
        f"({taxa_global:.0f}%) — {total_aprov} {labels['approved']}, {total_ajust} {labels['adjusted']}, "
        f"{total_corr} {labels['corrected']} | "
        f"**{labels['sources']}:** {len(all_urls)} | **{labels['sections']}:** {len(sections)}\n",
        "\n---\n",
        f"## {labels['summary']}\n",
        f"- {labels['introduction'].upper()}",
    ]
    for s in sections:
        parts.append(f"- {s['title']}")
    parts += [
        f"- {labels['conclusion']}",
        "\n\n---\n",
        f"## {labels['introduction']}\n",
        ans_intro.strip(),
        "\n\n---\n",
    ]

    for s in sections:
        stats_s = next((x for x in stats_global if x.get("section") == s["title"]), {})
        t_s = stats_s.get("total", 0)
        a_s = stats_s.get("approved", 0) + stats_s.get("adjusted", 0)
        r_s = stats_s.get("corrected", 0)
        aj_s = stats_s.get("adjusted", 0)
        tx_s = (a_s / t_s * 100) if t_s > 0 else 100
        parts.append(
            f"<!-- {labels['paragraphs']}: {a_s}/{t_s} {labels['verified']} ({tx_s:.0f}%) "
            f"| {stats_s.get('approved', 0)} {labels['approved']}, {aj_s} {labels['adjusted']}, "
            f"{r_s} {labels['corrected']} -->\n"
        )
        parts.append(s["text"].strip())
        parts.append("\n\n---\n")

    parts += [f"## {labels['conclusion']}\n", ans_concl.strip(), "\n\n"]

    # ══════════════════════════════════════════════════════════════════
    # GLOBAL CITATION SYNCHRONIZATION + PER-SECTION REFERENCE REBUILD
    # ══════════════════════════════════════════════════════════════════
    print("\n  🔗 Synchronizing global citations...")

    # 1. Build consolidated source_map: {original_citation_number: url}
    #    Merge all per-section source_maps; keep the first URL seen per index.
    #    Keys may be int or str depending on serialization — normalize to int.
    consolidated_source_map: dict = {}
    for s in sections:
        s_map = s.get("source_map", {})
        for idx, url in s_map.items():
            idx_int = int(idx)
            if idx_int not in consolidated_source_map:
                consolidated_source_map[idx_int] = url

    # Also add URLs from corpus that might be cited but not in source maps.
    for i, url in enumerate(all_urls, 1):
        if i not in consolidated_source_map:
            consolidated_source_map[i] = url

    raw_document = "\n".join(parts)

    # 2. Strip old "### References for this section" blocks before renumbering
    cleaned_document = re.sub(
        r"\n*###\s+(?:References\s+for\s+this\s+section|Refer[êe]ncias\s+desta\s+se[çc][ãa]o)\s*\n(?:\[?\d+\]?[^\n]*\n?)*",
        "",
        raw_document,
        flags=re.IGNORECASE,
    )

    # 3. Strip invalid figure/table/equation references
    cleaned_document = _strip_figure_table_refs(cleaned_document)

    # 4. Extract all [N] from entire document and create global renumbering
    original_citations = re.findall(r"\[(\d+)\]", cleaned_document)
    unique_citations = []
    seen = set()
    for c in original_citations:
        n = int(c)
        if n not in seen:
            seen.add(n)
            unique_citations.append(n)

    # old_idx → new_idx (first-appearance order)
    global_map: dict = {}
    for new_idx, old_idx in enumerate(unique_citations, 1):
        global_map[old_idx] = new_idx

    # Build synchronized global source map: {new_idx: url}
    global_source_map_sync: dict = {}
    for old_idx, new_idx in global_map.items():
        url = consolidated_source_map.get(old_idx, "")
        if url:
            global_source_map_sync[new_idx] = url

    # 5. Renumber all [N] in the document
    def _renumber(match):
        """Renumber a single citation.

        Args:
            match: regex match object for a single [N] citation

        Returns:
            the renumbered citation string [new_N]
        """
        old = int(match.group(1))
        new = global_map.get(old, old)
        return f"[{new}]"

    document_sync = re.sub(r"\[(\d+)\]", _renumber, cleaned_document)

    # Also handle [N, M] compound citations
    def _renumber_compound(match):
        """Renumber a compound citation.

        Args:
            match: regex match object for a compound [N, M] citation

        Returns:
            the renumbered compound citation string [new_N, new_M]
        """
        nums = re.findall(r"\d+", match.group(0))
        new_nums = [str(global_map.get(int(n), int(n))) for n in nums]
        return "[" + ", ".join(new_nums) + "]"

    document_sync = re.sub(r"\[\d+(?:\s*,\s*\d+)+\]", _renumber_compound, document_sync)

    n_global_sources = len(global_source_map_sync)
    print(f"     ✅ {n_global_sources} global sources | {len(global_map)} citations remapped")

    # 6. Rebuild per-section "### References for this section" blocks
    #    First, split out the conclusion so it doesn't contaminate the
    #    last section block (the old code skipped any block containing
    #    '## Conclusion', silently dropping the last section's refs).
    conclusion_match = re.search(
        r"\n##\s+(?:Conclusion|Conclus[ãa]o)\b", document_sync, re.IGNORECASE
    )
    if conclusion_match:
        conclusion_idx = conclusion_match.start()
        doc_sections_part = document_sync[:conclusion_idx]
        doc_conclusion_part = document_sync[conclusion_idx:]
    else:
        doc_sections_part = document_sync
        doc_conclusion_part = ""

    section_pattern = re.compile(r"(?=\n<!--\s*(?:Paragraphs|Par[áa]grafos):)", re.IGNORECASE)
    section_blocks = section_pattern.split(doc_sections_part)

    rebuilt_parts = []
    for block in section_blocks:
        # Only process blocks that contain a numbered section heading
        if not re.search(r"## \d", block):
            rebuilt_parts.append(block)
            continue

        # Extract all [N] referenced in block body
        cits_in_block = set()
        for m in re.finditer(r"\[(\d+)\]", block):
            cits_in_block.add(int(m.group(1)))
        # Also handle [N, M]
        for m in re.finditer(r"\[(\d+(?:\s*,\s*\d+)+)\]", block):
            for n in re.findall(r"\d+", m.group(1)):
                cits_in_block.add(int(n))

        if cits_in_block:
            refs_lines = []
            for idx in sorted(cits_in_block):
                url = global_source_map_sync.get(idx, "")
                if url:
                    refs_lines.append(f"[{idx}] {url}")
            if refs_lines:
                # Remove trailing --- if present, we'll re-add it
                block_trimmed = block.rstrip()
                if block_trimmed.endswith("---"):
                    block_trimmed = block_trimmed[:-3].rstrip()
                block = (
                    block_trimmed
                    + f"\n\n### {labels['references_section']}\n\n"
                    + "\n".join(refs_lines)
                    + "\n\n\n---\n"
                )
        rebuilt_parts.append(block)

    document = "".join(rebuilt_parts) + doc_conclusion_part

    # Update all_urls count for header
    all_urls_final = list(global_source_map_sync.values())
    # Update the header line with correct source count
    document = re.sub(
        r"\*\*(?:Sources|Fontes):\*\* \d+",
        f"**{labels['sources']}:** {len(all_urls_final)}",
        document,
        count=1,
        flags=re.IGNORECASE,
    )

    print(f"\n  ℹ️  References rebuilt per section ({n_global_sources} global sources)")
    print(
        "\n  ℹ️  The final '## References' section is no longer generated automatically.\n"
        "      Use option [5] from the main menu to format your references\n"
        "      in your desired standard (ABNT, APA, IEEE, etc.) from a\n"
        "      YAML/JSON file. See references/README.md for details."
    )

    slug = re.sub(r"[^\w\s-]", "", theme[:40]).strip().replace(" ", "_").lower()
    output_path = f"{REVIEWS_DIR}/{config.output_prefix}_{slug}.md"
    log_path = f"{REVIEWS_DIR}/{config.output_prefix}_{slug}.log"

    try:
        os.makedirs(REVIEWS_DIR, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(document)
        print(f"\n💾 {output_path} ({len(document):,} chars)")
    except Exception as e:
        print(f"⚠️  Error saving: {e}")

    try:
        header = [
            "=" * 70,
            f"REACT AUDIT LOG — {theme}",
            f"{labels['generated']}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"{labels['sections']}: {len(sections)} | {labels['sources']}: {len(all_urls)}",
            f"{labels['verified'].capitalize()}: {total_verif}/{total_par} ({taxa_global:.0f}%) "
            f"— {total_aprov} {labels['approved']}, {total_ajust} {labels['adjusted']}, {total_corr} {labels['corrected']}",
            "=" * 70,
            "\n── STATS PER SECTION ──",
        ]
        for s in stats_global:
            t = s.get("total", 0)
            a = s.get("approved", 0) + s.get("adjusted", 0)
            r = s.get("corrected", 0)
            aj = s.get("adjusted", 0)
            tx = (a / t * 100) if t > 0 else 100
            header.append(
                f"  [{a}/{t} = {tx:.0f}% | {s.get('approved', 0)} appr "
                f"{aj} adj {r} corr] {s.get('section', '?')[:55]}"
            )
        os.makedirs(REVIEWS_DIR, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(header + [""] + react_log))
        print(f"📋 {log_path}")
    except Exception as e:
        print(f"⚠️  Error saving log: {e}")

    return {"status": "completed"}
