import re
from typing import List, Tuple
from ..vector_utils.vector_store import search_chunks
from ..file_utils.helpers import extract_anchors
from ..llm_utils.prompt_loader import load_prompt
from ...config import JUDGE_MAX_CORPUS_CHARS, JUDGE_TOP_K, get_llm


def search_chunks_for_paragraph(
    paragraph: str,
    full_corpus_prompt: str,
) -> str:
    """
    Assemble the block of relevant sources to check the paragraph.
    Use the declared anchors as search queries in MongoDB.

    Args:
        paragraph: The cleaned paragraph text to verify.
        full_corpus_prompt: The full text of the document (for fallback).

    Returns:
        A string containing the concatenated relevant chunks from the corpus.
    """
    # Clean anchors and heavy LaTeX from the text for the query.
    text_without_anchors = re.sub(r'\[ANCHOR:\s*"[^"]*"\]', "", paragraph)
    text_without_anchors = re.sub(r"\$\$[^$]+\$\$", "", text_without_anchors)
    text_without_anchors = re.sub(r"\$[^$]+\$", "", text_without_anchors)
    text_without_anchors = re.sub(r"\\\([^)]+\\\)", "", text_without_anchors).strip()

    # Declared anchors are the best search hints, but if they are too LaTeX-heavy or short, we can also use the cleaned paragraph text as a fallback query.
    anchors = extract_anchors(paragraph)
    # Filter anchors that are only LaTeX or too short
    valid_anchors = [
        a
        for a in anchors
        if len(a.strip()) >= 20 and not re.match(r"^[\\\$\{\}\[\]_\^]+", a.strip())
    ]

    queries = valid_anchors[:3] + (
        [text_without_anchors[:200]] if text_without_anchors else []
    )

    if not queries:
        return full_corpus_prompt[:JUDGE_MAX_CORPUS_CHARS]

    # Uses search_chunks from vector_store (MongoDB)
    parts: List[str] = []
    chars = 0
    chunks_seen = set()

    for q in queries:
        q = q.strip()
        if not q:
            continue
        # search_chunks returns a list of chunk text strings
        results = search_chunks(q, k=JUDGE_TOP_K)
        for chunk_text in results:
            # simple deduplication
            key = chunk_text[:100]
            if key in chunks_seen:
                continue
            chunks_seen.add(key)
            block = f"{chunk_text}\n\n"
            if chars + len(block) > JUDGE_MAX_CORPUS_CHARS:
                break
            parts.append(block)
            chars += len(block)

    if parts:
        return "".join(parts)

    # fallback
    return full_corpus_prompt[:JUDGE_MAX_CORPUS_CHARS]


def judge_paragraph(clean_paragraph: str, sources: str) -> Tuple[str, str, str]:
    """
    LLM Judge with 3 levels.
    Returns (final_text, level, log_entry)
    level ∈ {"APPROVED", "ADJUSTED", "CORRECTED"}

    Args:
        clean_paragraph: The cleaned paragraph text to verify.
        sources: The assembled relevant sources to check against.
    Returns:
        A tuple containing the final paragraph text after judgment, the decision level, and a log entry summarizing the judgment.

    """
    prompt = load_prompt(
        "common/anchor_verification_judge",
        clean_paragraph=clean_paragraph,
        sources=sources,
    )

    resp = get_llm(temperature=0.0).invoke(prompt.text)
    resp_text = resp.content if hasattr(resp, "content") else str(resp)

    level = "APPROVED"
    final_text = clean_paragraph

    m_dec = re.search(
        r"(?:DECISION|DECIS(?:[ÃA]O|AO))\s*:\s*(APPROVED|ADJUSTED|CORRECTED|APROVADO|AJUSTADO|CORRIGIDO)",
        resp_text,
        re.IGNORECASE,
    )
    if m_dec:
        level = {
            "APROVADO": "APPROVED",
            "AJUSTADO": "ADJUSTED",
            "CORRIGIDO": "CORRECTED",
        }.get(m_dec.group(1).upper(), m_dec.group(1).upper())

    m_txt = re.search(r"(?:TEXT|TEXTO)\s*:\s*([\s\S]+)", resp_text, re.IGNORECASE)
    if m_txt:
        candidate = m_txt.group(1).strip()
        candidate = re.sub(
            r"^(?:DECISION|DECIS(?:[ÃA]O|AO))\s*:.*\n?",
            "",
            candidate,
            flags=re.IGNORECASE,
        ).strip()
        if candidate:
            final_text = candidate

    patch = clean_paragraph[:70].replace("\n", " ")
    if level == "APPROVED":
        log_entry = f"✅ APPROVED  | {patch}..."
    elif level == "ADJUSTED":
        corr = final_text[:70].replace("\n", " ")
        log_entry = f"🔵 ADJUSTED  | {patch}...\n     → {corr}..."
    else:
        corr = final_text[:70].replace("\n", " ")
        log_entry = f"🔧 CORRECTED | {patch}...\n     → {corr}..."

    return final_text, level, log_entry
