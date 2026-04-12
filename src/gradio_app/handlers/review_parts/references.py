"""Reference lookup and ABNT formatting helpers for the review handler.

Covers all reference-related user actions: resolving numbered citations,
listing all document references, formatting user-provided entries, and
enriching metadata via MongoDB, DOI resolution, CrossRef, and Tavily.
"""

from __future__ import annotations

import os
import re
from datetime import datetime

from revisao_agents.agents.reference_extractor_agent import run_reference_extractor_agent
from revisao_agents.agents.reference_formatter_agent import run_reference_formatter_agent
from revisao_agents.config import llm_call
from revisao_agents.tools.tavily_web_search import extract_tavily, search_tavily_incremental
from revisao_agents.utils.bib_utils.doi_utils import (
    extract_doi_from_url,
    get_bibtex_from_doi,
    search_crossref_by_title,
    search_doi_in_text,
)
from revisao_agents.utils.llm_utils.prompt_loader import load_prompt
from revisao_agents.utils.vector_utils.vector_store import search_chunk_records, search_chunks

from ..base import _detect_user_language, _localized_text
from .document import (
    _extract_quoted_snippet,
    _resolve_paragraph_index,
    _resolve_section_index,
    _split_sections,
)
from .intent import (
    _build_phrase_reference_query_seed,
    _extract_citation_number,
    _extract_provided_reference_items,
    _extract_requested_citation_numbers,
)

_BIBTEX_FIELD_RE = re.compile(r'(\w+)\s*=\s*["{]([^"}]+)["}]', re.IGNORECASE)


def _search_reference_in_mongo_by_phrase(
    user_text: str, missing_numbers: list[int]
) -> tuple[str, dict]:
    """Search the MongoDB vector store for source metadata matching a phrase citation.

    Args:
        user_text: The user's original message, used to extract a query seed
            phrase via :func:`_build_phrase_reference_query_seed`.
        missing_numbers: Citation numbers that were absent from the reference
            list and triggered this lookup.

    Returns:
        A 2-tuple of:

        - A localized markdown reply string with the candidate reference
          details (title, DOI, URL, and file path when present).
        - A stats dict with keys ``found`` (bool), ``mongo_queries`` (int),
          and ``mongo_hits`` (int).
    """
    language = _detect_user_language(user_text)
    query_seed = _build_phrase_reference_query_seed(user_text)
    if not query_seed:
        return (
            _localized_text(
                language,
                "Não consegui extrair um trecho para busca no MongoDB.",
                "I couldn't extract a phrase to search in MongoDB.",
            ),
            {"found": False, "mongo_queries": 0, "mongo_hits": 0},
        )

    records = search_chunk_records(query_seed[:500], k=5)
    if not records:
        return (
            _localized_text(
                language,
                "Não encontrei candidato no MongoDB para essa frase.",
                "I couldn't find a MongoDB candidate for that phrase.",
            ),
            {"found": False, "mongo_queries": 1, "mongo_hits": 0},
        )

    best = records[0]
    title = str(best.get("source_title") or "").strip()
    doi = str(best.get("doi") or "").strip()
    url = str(best.get("source_url") or "").strip()
    file_path = str(best.get("file_path") or "").strip()

    lines = [
        _localized_text(
            language,
            f"### Referência candidata para {', '.join(f'[{n}]' for n in missing_numbers)} (MongoDB)",
            f"### Candidate reference for {', '.join(f'[{n}]' for n in missing_numbers)} (MongoDB)",
        ),
        "",
        f"- {_localized_text(language, 'Título', 'Title')}: {title or _localized_text(language, '(não identificado)', '(not identified)')}",
    ]
    if doi:
        lines.append(f"- DOI: {doi}")
    if url:
        lines.append(f"- URL: {url}")
    if file_path:
        lines.append(f"- {_localized_text(language, 'Arquivo', 'File')}: {file_path}")

    return "\n".join(lines), {"found": True, "mongo_queries": 1, "mongo_hits": 1}


def _search_reference_on_web_by_phrase(
    user_text: str, missing_numbers: list[int]
) -> tuple[str, dict]:
    """Search the internet via Tavily for source metadata matching a phrase citation.

    Args:
        user_text: The user's original message, used to extract a query seed
            phrase via :func:`_build_phrase_reference_query_seed`.
        missing_numbers: Citation numbers that were absent from the reference
            list and triggered this lookup.

    Returns:
        A 2-tuple of:

        - A localized markdown reply string with the first extracted page
          title and URL.
        - A stats dict with keys ``found`` (bool), ``web_queries`` (int),
          and ``web_hits`` (int).
    """
    language = _detect_user_language(user_text)
    query_seed = _build_phrase_reference_query_seed(user_text)
    if not query_seed:
        return (
            _localized_text(
                language,
                "Não consegui extrair um trecho para busca na internet.",
                "I couldn't extract a phrase to search on the internet.",
            ),
            {"found": False, "web_queries": 0, "web_hits": 0},
        )

    web = search_tavily_incremental(query=query_seed[:400], previous_urls=[], max_results=3)
    urls = web.get("new_urls", [])[:3]
    if not urls:
        return (
            _localized_text(
                language,
                "Não encontrei resultados web para essa frase.",
                "I couldn't find web results for that phrase.",
            ),
            {"found": False, "web_queries": 1, "web_hits": 0},
        )

    extracted = extract_tavily.invoke({"urls": urls, "include_images": False})
    items = extracted.get("extracted", []) if isinstance(extracted, dict) else []
    if not items:
        return (
            _localized_text(
                language,
                "Encontrei URLs, mas não consegui extrair metadados suficientes.",
                "I found URLs, but I couldn't extract enough metadata.",
            ),
            {"found": False, "web_queries": 1, "web_hits": 0},
        )

    first = items[0]
    title = str(first.get("title") or "").strip()
    url = str(first.get("url") or "").strip()

    lines = [
        _localized_text(
            language,
            f"### Referência candidata para {', '.join(f'[{n}]' for n in missing_numbers)} (Internet)",
            f"### Candidate reference for {', '.join(f'[{n}]' for n in missing_numbers)} (Internet)",
        ),
        "",
        f"- {_localized_text(language, 'Título', 'Title')}: {title or _localized_text(language, '(não identificado)', '(not identified)')}",
    ]
    if url:
        lines.append(f"- URL: {url}")

    return "\n".join(lines), {"found": True, "web_queries": 1, "web_hits": 1}


def _build_reference_confirmation_prompt(
    intent: str, user_text: str, allow_web: bool = True
) -> tuple[str, dict]:
    """Build a user-facing confirmation prompt for a pending reference action.

    Args:
        intent: Reference action label; one of ``"list_all"`` or
            ``"format_provided"``.
        user_text: The original user message, used for language detection
            and for extracting provided reference items when intent is
            ``"format_provided"``.
        allow_web: Whether external web search is enabled.  When ``False``
            and incomplete items exist, the prompt warns the user to enable
            it before confirming.

    Returns:
        A 2-tuple of:

        - A localized prompt string asking the user to confirm or cancel.
        - A pending-action payload dict with keys ``intent``,
          ``original_message``, ``requires_web``, and ``incomplete_items``.
    """
    language = _detect_user_language(user_text)
    pending: dict = {"intent": intent, "original_message": user_text}

    if intent == "list_all":
        prompt = _localized_text(
            language,
            "Posso listar todas as referências do documento e formatar em ABNT. Deseja continuar? Responda **sim** ou **não**.",
            "I can list all references from the document and format them in ABNT. Do you want to continue? Reply **yes** or **no**.",
        )
        if not allow_web:
            prompt += "\n\n" + _localized_text(
                language,
                "Observação: a busca na web está desativada, então algumas referências podem ficar incompletas.",
                "Note: web search is disabled, so some references may remain incomplete.",
            )
        pending.update({"requires_web": False, "incomplete_items": []})
        return prompt, pending

    if intent == "format_provided":
        items = _extract_provided_reference_items(user_text)
        incomplete_items: list[int] = []
        for idx, item in enumerate(items, start=1):
            metadata = _metadata_from_raw_reference(idx, item)
            if not _is_metadata_complete(metadata):
                incomplete_items.append(idx)

        requires_web = bool(incomplete_items)
        pending.update({"requires_web": requires_web, "incomplete_items": incomplete_items})

        base_prompt = _localized_text(
            language,
            f"Posso formatar {len(items)} referência(s) fornecida(s) em ABNT. Deseja continuar? Responda **sim** ou **não**.",
            f"I can format the {len(items)} provided reference(s) in ABNT. Do you want to continue? Reply **yes** or **no**.",
        )

        if requires_web and not allow_web:
            warning = _localized_text(
                language,
                "Algumas referências parecem incompletas e podem exigir busca na web para completar.",
                "Some references look incomplete and may require web search to complete.",
            )
            item_list = ", ".join(f"[{idx}]" for idx in incomplete_items)
            detail = _localized_text(
                language,
                f"Itens incompletos: {item_list}. Ative **Allow web search** e confirme novamente com **sim**.",
                f"Incomplete items: {item_list}. Enable **Allow web search** and confirm again with **yes**.",
            )
            return f"{base_prompt}\n\n{warning}\n{detail}", pending

        if requires_web:
            note = _localized_text(
                language,
                "Algumas referências parecem incompletas; vou usar a web para complementar os dados.",
                "Some references look incomplete; I'll use the web to enrich the data.",
            )
            return f"{base_prompt}\n\n{note}", pending

        return base_prompt, pending

    fallback = _localized_text(
        language,
        "Não consegui identificar a ação de referências. Refaça o pedido.",
        "I couldn't identify the reference action. Please resend the request.",
    )
    pending.update({"requires_web": False, "incomplete_items": []})
    return fallback, pending


def _parse_bibtex_fields(bibtex: str) -> dict[str, str]:
    """Parse BibTeX entry fields into a dictionary.

    Args:
        bibtex: A raw BibTeX entry string, e.g. the output of a DOI lookup.

    Returns:
        Dict mapping lowercase field names (e.g. ``"title"``, ``"doi"``) to
        their string values.  Returns an empty dict when *bibtex* is falsy.
    """
    if not bibtex:
        return {}
    return {m.group(1).lower(): m.group(2).strip() for m in _BIBTEX_FIELD_RE.finditer(bibtex)}


def _metadata_from_bibtex(number: int | None, bibtex: str) -> dict:
    """Extract structured reference metadata from a BibTeX entry string.

    Args:
        number: Display position in the reference list, stored under the
            ``"number"`` key.  Pass ``None`` when the position is unknown.
        bibtex: Raw BibTeX entry string to parse.

    Returns:
        A metadata dict with keys: ``number``, ``raw``, ``title``, ``year``,
        ``url``, ``doi``, ``file_path``, ``derived_from_path``.
    """
    fields = _parse_bibtex_fields(bibtex)
    title = fields.get("title", "")
    year = fields.get("year", "")
    url = (fields.get("url", "") or "").strip().rstrip(".,;")
    doi = fields.get("doi", "")
    doi_match = re.search(r"(10\.\d{4,9}/[^\s,;]+)", doi, flags=re.IGNORECASE)
    doi_clean = doi_match.group(1).rstrip(".)],;") if doi_match else ""
    return {
        "number": number,
        "raw": bibtex.strip(),
        "title": title,
        "year": year,
        "url": url,
        "doi": doi_clean,
        "file_path": "",
        "derived_from_path": False,
    }


def _normalize_reference_key(raw: str) -> str:
    """Normalize a raw reference string for deduplication comparisons.

    Strips leading ``[n]`` number prefixes, DOI fragments, bare URLs, and
    punctuation, then lowercases and collapses whitespace.

    Args:
        raw: Raw reference text, optionally prefixed with a numbered tag
            such as ``"[1] Author et al."``.

    Returns:
        A lowercase, punctuation-stripped string suitable for set-based
        deduplication.
    """
    text = re.sub(r"^\[\d+\]\s*", "", raw or "")
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"doi:\s*10\.[^\s,;]+", "", text)
    text = re.sub(r"https?://\S+", "", text)
    return re.sub(r"[^\w\s]", "", text).strip()


def _title_from_file_path(path: str) -> str:
    """Derive a human-readable title from a local PDF file path.

    Removes the ``.pdf`` extension, replaces underscores and hyphens with
    spaces, and collapses consecutive whitespace.

    Args:
        path: Absolute or relative path to a PDF file.

    Returns:
        A title string derived from the filename.  Returns an empty string
        when *path* is empty.
    """
    base = os.path.basename(path or "")
    base = re.sub(r"\.pdf$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"[_+\-]", " ", base)
    base = re.sub(r"\s+", " ", base).strip()
    return base


def _metadata_from_raw_reference(number: int | None, raw_reference: str) -> dict:
    """Extract structured metadata from a free-form reference string.

    Attempts to parse DOI, URL, PDF path, and publication year from the raw
    text.  When a PDF path is found, the title is derived from the filename;
    otherwise the title is approximated from the remaining text after
    stripping identifiers and noise tokens.

    Args:
        number: Display position in the reference list.  Pass ``None`` when
            the position is unknown.
        raw_reference: Free-form reference text, optionally prefixed with a
            numbered tag such as ``"[1]"``.

    Returns:
        A metadata dict with keys: ``number``, ``raw``, ``title``, ``doi``,
        ``url``, ``year``, ``file_path``, ``derived_from_path``.
    """
    raw = (raw_reference or "").strip()
    body = re.sub(r"^\[\d+\]\s*", "", raw).strip()

    doi_match = re.search(r"(10\.\d{4,9}/[^\s,;]+)", body, flags=re.IGNORECASE)
    url_match = re.search(r"(https?://\S+)", body, flags=re.IGNORECASE)
    path_match = re.search(r"(/[^\n]*?\.pdf)", body, flags=re.IGNORECASE)
    year_match = re.search(r"\b(19|20)\d{2}\b", body)

    file_path = path_match.group(1).strip() if path_match else ""
    title_guess = ""

    if file_path:
        title_guess = _title_from_file_path(file_path)
    else:
        text_no_url = re.sub(r"https?://\S+", "", body)
        text_no_doi = re.sub(r"10\.\d{4,9}/[^\s,;]+", "", text_no_url, flags=re.IGNORECASE)
        text_no_path = re.sub(r"/[^\n]*?\.pdf", "", text_no_doi, flags=re.IGNORECASE)
        text_no_labels = re.sub(
            r"\b(?:dispon[ií]vel em|arquivo local|citado em)\b:?.*",
            "",
            text_no_path,
            flags=re.IGNORECASE,
        )
        title_guess = re.sub(r"\s+", " ", text_no_labels).strip(" .;,")

    title_guess = re.sub(
        r"\bDOI\b\s*:?\s*10\.\d{4,9}/[^\s,;]+", "", title_guess, flags=re.IGNORECASE
    )
    title_guess = re.sub(r"https?://\S+", "", title_guess)
    title_guess = re.sub(r"\s+", " ", title_guess).strip(" .;,")

    return {
        "number": number,
        "raw": raw,
        "title": title_guess,
        "doi": doi_match.group(1).rstrip(".)],;") if doi_match else "",
        "url": (url_match.group(1).rstrip(".)],;")) if url_match else "",
        "year": year_match.group(0) if year_match else "",
        "file_path": file_path,
        "derived_from_path": bool(file_path),
    }


def _is_metadata_complete(metadata: dict) -> bool:
    """Determine whether a metadata dict has enough information for ABNT output.

    A DOI alone is considered sufficient.  Otherwise both a non-path-derived
    title **and** at least one of year or URL are required.

    Args:
        metadata: Metadata dict as returned by :func:`_metadata_from_raw_reference`
            or similar helpers.

    Returns:
        ``True`` when the entry is considered complete for ABNT formatting,
        ``False`` otherwise.
    """
    title = (metadata.get("title") or "").strip()
    year = (metadata.get("year") or "").strip()
    doi = (metadata.get("doi") or "").strip()
    url = (metadata.get("url") or "").strip()
    derived_from_path = bool(metadata.get("derived_from_path"))

    if doi:
        return True
    return bool(title and not derived_from_path and (year or url))


def _format_abnt_entry(metadata: dict) -> str:
    """Format a reference metadata dict as an ABNT-style entry string.

    Cleans and normalises title, year, DOI, and URL before assembling them
    into a single-line entry.  Applies post-processing regex rules to remove
    duplicate ``DOI:`` labels, repeated ``[s.d.]`` tokens, and stray dots.

    Args:
        metadata: Metadata dict with optional keys ``number``, ``title``,
            ``year``, ``doi``, ``url``, ``file_path``, and ``raw``.

    Returns:
        A formatted ABNT reference string.  When ``number`` is an integer
        the entry is prefixed with ``[n] ``.
    """
    number = metadata.get("number")
    title = (metadata.get("title") or "").strip()
    year = (metadata.get("year") or "").strip()
    doi = (metadata.get("doi") or "").strip()
    url = (metadata.get("url") or "").strip()
    file_path = (metadata.get("file_path") or "").strip()
    raw = (metadata.get("raw") or "").strip()

    doi_match = re.search(r"(10\.\d{4,9}/[^\s,;]+)", doi, flags=re.IGNORECASE)
    doi = doi_match.group(1).rstrip(".)],;") if doi_match else ""
    url = url.rstrip(".)],;")
    year = (
        re.search(r"\b(19|20)\d{2}\b", year).group(0)
        if re.search(r"\b(19|20)\d{2}\b", year)
        else ""
    )
    title = re.sub(r"\bDOI\b\s*:?.*$", "", title, flags=re.IGNORECASE).strip(" .;,")
    title = re.sub(r"\s+", " ", title).strip()

    if not year:
        year_in_title = re.search(r"\b(19|20)\d{2}\b", title)
        if year_in_title:
            year = year_in_title.group(0)
            title = re.sub(rf"\b{re.escape(year)}\b", "", title).strip(" .;,")
            title = re.sub(r"\s+", " ", title).strip()
    if raw and not title:
        m_author_year = re.match(r"^([^,]{2,80}),\s*((?:19|20)\d{2})$", raw)
        if m_author_year:
            author_stub = m_author_year.group(1).strip().upper()
            year = year or m_author_year.group(2)
            title = "TÍTULO NÃO IDENTIFICADO"
            raw = f"{author_stub}."

    prefix = f"[{number}] " if isinstance(number, int) else ""
    core = title or "TÍTULO NÃO IDENTIFICADO"

    fragments: list[str] = []
    fragments.append(core.rstrip(".;,") + ".")

    if year and year not in core:
        fragments.append(f"{year}.")
    else:
        fragments.append("[s.d.].")

    if doi:
        fragments.append(f"DOI: {doi}.")

    if url:
        fragments.append(f"Disponível em: {url.rstrip('.,;')}.")
    elif file_path:
        fragments.append(f"Documento local: {file_path}.")

    output = " ".join(fragment.strip() for fragment in fragments if fragment.strip())
    output = re.sub(r"\bDOI:\s*DOI:\s*", "DOI: ", output, flags=re.IGNORECASE)
    output = re.sub(r"\bDOI:\s*\.(?=\s|$)", "", output, flags=re.IGNORECASE)
    output = re.sub(r"(\[s\.d\.\]\.\s*){2,}", "[s.d.]. ", output, flags=re.IGNORECASE)
    output = re.sub(r"\.{2,}", ".", output)
    output = re.sub(r"\s+", " ", output).strip()
    return f"{prefix}{output}" if not output.startswith(prefix) else output


def _merge_metadata(base: dict, extra: dict) -> dict:
    """Merge two metadata dicts, preferring non-empty values from *base*.

    Only the keys ``title``, ``doi``, ``url``, ``year``, and ``file_path``
    are considered for merging.  All other keys are copied from *base*
    unchanged.

    Args:
        base: Primary metadata dict whose existing non-empty values are kept.
        extra: Supplementary metadata dict used to fill empty keys in *base*.

    Returns:
        A new dict combining both sources.
    """
    merged = dict(base)
    for key in ("title", "doi", "url", "year", "file_path"):
        if not merged.get(key) and extra.get(key):
            merged[key] = extra[key]
    return merged


def _extract_non_numbered_mentions(markdown: str) -> list[str]:
    """Extract in-text author-year citations that are not numbered references.

    Scans the markdown for parenthesised ``(Author, Year)`` patterns and
    for bare reference lines inside section-level reference blocks that lack
    a leading ``[n]`` number tag.

    Args:
        markdown: Full markdown document text.

    Returns:
        Deduplicated list of citation strings in author-year or raw-text
        format.  Each entry has been normalised and bounded to 180
        characters.
    """
    mentions: list[str] = []

    patterns = [
        r"\(([A-Z][A-Za-zÀ-ÿ'’\-]+(?:\s+et\s+al\.)?(?:\s*&\s*[A-Z][A-Za-zÀ-ÿ'’\-]+)?\s*,\s*(?:19|20)\d{2})\)",
        r"\(([A-Z][^()\n]{6,120},\s*(?:19|20)\d{2})\)",
    ]

    for pattern in patterns:
        for match in re.findall(pattern, markdown):
            text = re.sub(r"\s+", " ", match).strip(" .;,")
            if text:
                mentions.append(text)

    lines = markdown.splitlines()
    in_refs = False
    for line in lines:
        stripped = line.strip()
        if re.match(
            r"^###\s+(?:References\s+for\s+this\s+section|Refer[êe]ncias\s+desta\s+se[çc][ãa]o)\s*$",
            stripped,
            flags=re.IGNORECASE,
        ):
            in_refs = True
            continue
        if in_refs and re.match(r"^##\s+", stripped):
            in_refs = False
        if not in_refs:
            continue
        if not stripped or stripped.startswith("<!--"):
            continue
        if re.match(r"^\[\d+\]", stripped):
            continue
        cleaned = re.sub(r"^[-*]\s+", "", stripped)
        if (
            cleaned
            and len(cleaned) <= 180
            and "http" not in cleaned.lower()
            and "doi" not in cleaned.lower()
        ):
            mentions.append(cleaned)

    dedup: list[str] = []
    seen = set()
    for mention in mentions:
        key = _normalize_reference_key(mention)
        if not key or key in seen:
            continue
        seen.add(key)
        dedup.append(mention)
    return dedup


def _collect_reference_inventory(markdown: str) -> dict:
    """Build a structured inventory of all references and citations in the document.

    Args:
        markdown: Full markdown document text.

    Returns:
        A dict with five keys:

        - ``references_by_number`` (dict[int, str]): Maps each citation
          number to its raw reference line.
        - ``citation_paragraphs`` (dict[int, list[str]]): Maps each citation
          number to the body paragraphs that contain it.
        - ``unique_references`` (list[str]): Deduplicated reference lines
          ordered by citation number.
        - ``cited_numbers`` (list[int]): Sorted list of all citation numbers
          found in body paragraphs.
        - ``non_numbered_mentions`` (list[str]): Author-year citations not
          expressed as numbered references.
    """
    sections = _split_sections(markdown)
    references_by_number: dict[int, str] = {}
    citation_paragraphs: dict[int, list[str]] = {}

    for section in sections:
        for ref in section.get("references", []):
            match = re.match(r"^\[(\d+)\]\s*(.+)$", ref.strip())
            if not match:
                continue
            number = int(match.group(1))
            text = f"[{number}] {match.group(2).strip()}"
            references_by_number[number] = text

        for paragraph in section.get("paragraphs", []):
            p_text = paragraph.get("text", "")
            for number_token in re.findall(r"\[(\d+)\]", p_text):
                number = int(number_token)
                citation_paragraphs.setdefault(number, []).append(p_text)

    unique_refs: list[str] = []
    seen_keys: set[str] = set()
    for number in sorted(references_by_number.keys()):
        ref = references_by_number[number]
        key = _normalize_reference_key(ref)
        if key and key in seen_keys:
            continue
        if key:
            seen_keys.add(key)
        unique_refs.append(ref)

    cited_numbers = sorted(citation_paragraphs.keys())
    non_numbered_mentions = _extract_non_numbered_mentions(markdown)
    return {
        "references_by_number": references_by_number,
        "citation_paragraphs": citation_paragraphs,
        "unique_references": unique_refs,
        "cited_numbers": cited_numbers,
        "non_numbered_mentions": non_numbered_mentions,
    }


def _enrich_reference_from_mongo(number: int, paragraphs: list[str]) -> tuple[dict, dict]:
    """Enrich reference metadata by querying the local MongoDB vector store.

    Runs up to four paragraph-based queries, keeping the highest-scored
    record as the best candidate.

    Args:
        number: Citation number being resolved, stored in the returned
            metadata under the ``"number"`` key.
        paragraphs: Body paragraphs that cite this number, used as query
            seeds for the vector search.

    Returns:
        A 2-tuple of:

        - A metadata dict with keys ``number``, ``title``, ``doi``, ``url``,
          ``file_path``.  Empty dict when no records are found.
        - A stats dict with keys ``mongo_queries`` (int) and
          ``mongo_hits`` (int).
    """
    if not paragraphs:
        return {}, {"mongo_queries": 0, "mongo_hits": 0}

    mongo_queries = 0
    best: dict | None = None
    for paragraph in paragraphs[:4]:
        query = paragraph[:600]
        mongo_queries += 1
        records = search_chunk_records(query, k=6)
        if not records:
            continue
        candidate = records[0]
        if best is None or float(candidate.get("score", 0.0) or 0.0) > float(
            best.get("score", 0.0) or 0.0
        ):
            best = candidate

    if not best:
        return {}, {"mongo_queries": mongo_queries, "mongo_hits": 0}

    title = best.get("source_title", "") or "(untitled source)"
    doi = best.get("doi", "")
    url = best.get("source_url", "")
    file_path = best.get("file_path", "")

    metadata = {
        "number": number,
        "title": title,
        "doi": doi,
        "url": url,
        "file_path": file_path,
    }
    return metadata, {"mongo_queries": mongo_queries, "mongo_hits": 1}


def _enrich_reference_from_web(number: int, query: str) -> tuple[dict, dict]:
    """Enrich reference metadata by fetching and extracting pages via Tavily.

    Args:
        number: Citation number being resolved, stored in the returned
            metadata under the ``"number"`` key.
        query: Search query string sent to Tavily.

    Returns:
        A 2-tuple of:

        - A metadata dict with keys ``number``, ``title``, ``doi``, ``url``,
          ``year``.  Empty dict when Tavily returns no usable results.
        - A stats dict with keys ``web_queries`` (int) and ``web_hits``
          (int).
    """
    if not query.strip():
        return {}, {"web_queries": 0, "web_hits": 0}

    web = search_tavily_incremental(query=query[:400], previous_urls=[], max_results=5)
    urls = web.get("new_urls", [])[:2]
    if not urls:
        return {}, {"web_queries": 1, "web_hits": 0}

    extracted = extract_tavily.invoke({"urls": urls, "include_images": False})
    items = extracted.get("extracted", []) if isinstance(extracted, dict) else []
    if not items:
        return {}, {"web_queries": 1, "web_hits": 0}

    first = items[0]
    title = first.get("title", "") or "(untitled source)"
    url = first.get("url", "")
    content = str(first.get("content", ""))
    doi_match = re.search(r"(10\.\d{4,9}/[^\s)]+)", content)
    doi = doi_match.group(1) if doi_match else ""

    year_match = re.search(r"\b(19|20)\d{2}\b", content)
    metadata = {
        "number": number,
        "title": title,
        "doi": doi,
        "url": url,
        "year": year_match.group(0) if year_match else "",
    }
    return metadata, {"web_queries": 1, "web_hits": 1}


def _collect_all_raw_references_text(markdown: str) -> list[str]:
    """Extract every non-empty line from all reference sections in a markdown document.

    Detects reference section headings (e.g. *References*, *Referências*,
    *Bibliography*) and collects lines until the next heading of the same or
    higher level.

    Args:
        markdown: Full markdown document text.

    Returns:
        List of raw reference lines preserving order of appearance.
    """
    ref_heading_re = re.compile(
        r"^(#+)\s+(?:[\d]+[\s\.]+)?(refer[eê]ncias|references|bibliography|bibliograf\w+|bibliog\w+)\b",
        re.IGNORECASE,
    )
    any_heading_re = re.compile(r"^(#+)\s+")

    lines = markdown.splitlines()
    collected: list[str] = []
    collecting = False
    current_depth = 0

    for line in lines:
        stripped = line.strip()
        ref_match = ref_heading_re.match(stripped)
        if ref_match:
            collecting = True
            current_depth = len(ref_match.group(1))
            continue

        if collecting:
            any_match = any_heading_re.match(stripped)
            if any_match and len(any_match.group(1)) <= current_depth:
                collecting = False
            elif stripped:
                collected.append(stripped)

    return collected


def _collect_all_citation_paragraphs(markdown: str) -> dict[int, list[str]]:
    """Scan the document body for paragraphs that contain numbered citation tokens.

    Non-paragraph lines (headings, HTML comments) are skipped.  At most two
    paragraphs per citation number are stored to keep memory bounded.

    Args:
        markdown: Full markdown document text.

    Returns:
        Dict mapping each citation number ``n`` to a list of up to two body
        lines that contain the token ``[n]``.
    """
    result: dict[int, list[str]] = {}
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("<!--"):
            continue
        nums = {int(n) for n in re.findall(r"\[(\d+)\]", stripped)}
        for num in nums:
            paragraphs = result.setdefault(num, [])
            if len(paragraphs) < 2:
                paragraphs.append(stripped)
    return result


def _handle_resolve_numbers_request(
    markdown: str, user_text: str, allow_web: bool = True
) -> tuple[str, dict]:
    """Resolve specific numbered references using the extractor-to-formatter agent pipeline.

    When no specific numbers are requested, all references found in the
    document are processed.

    Args:
        markdown: Working-copy document text from which references are read.
        user_text: User's request message, parsed for citation numbers and
            used for language detection.
        allow_web: Whether external web search is permitted during extraction
            and formatting.

    Returns:
        A 2-tuple of:

        - A localized markdown reply with an ABNT-formatted reference list.
        - A metadata dict with keys ``intent`` (``"resolve_numbers"``),
          ``count`` (int), and ``agent`` (str).
    """
    language = _detect_user_language(user_text)
    requested = _extract_requested_citation_numbers(user_text)
    inventory = _collect_reference_inventory(markdown)
    references_by_number: dict[int, str] = inventory.get("references_by_number", {})

    entries = (
        {n: references_by_number[n] for n in requested if n in references_by_number}
        if requested
        else references_by_number
    )

    if not entries:
        msg = _localized_text(
            language,
            "Nenhuma referência encontrada para os números solicitados.",
            "No references found for the requested numbers.",
        )
        return msg, {"intent": "resolve_numbers", "count": 0, "agent": "none"}

    raw_block = "\n".join(entries.values())
    citation_context = _collect_all_citation_paragraphs(markdown)

    enriched = run_reference_extractor_agent(
        raw_block, citation_context=citation_context, allow_web=allow_web
    )
    abnt_list = run_reference_formatter_agent(enriched, allow_web=allow_web)

    heading = _localized_text(language, "### Referências (ABNT)", "### References (ABNT)")
    reply = f"{heading}\n\n{abnt_list}"
    return reply, {
        "intent": "resolve_numbers",
        "count": len(entries),
        "agent": "reference_extractor+reference_formatter",
    }


def _handle_list_all_references_request(
    markdown: str, user_text: str, allow_web: bool = True
) -> tuple[str, dict]:
    """Collect every reference from the document and format them in ABNT.

    Combines numbered references from the inventory with any additional raw
    lines found in reference section blocks.  All entries are normalised to
    ``[n]`` prefixes before being passed to the agent pipeline.

    Args:
        markdown: Working-copy document text from which references are read.
        user_text: User's request message, used for language detection.
        allow_web: Whether external web search is permitted during extraction
            and formatting.

    Returns:
        A 2-tuple of:

        - A localized markdown reply with the ABNT-formatted reference list.
        - A metadata dict with keys ``intent`` (``"list_all"``), ``count``
          (int), and ``agent`` (str).
    """
    language = _detect_user_language(user_text)

    inventory = _collect_reference_inventory(markdown)
    primary_refs: list[str] = list(inventory.get("references_by_number", {}).values())

    extra_lines = _collect_all_raw_references_text(markdown)
    primary_refs.extend(extra_lines)

    primary_refs = [r for r in primary_refs if not r.strip().startswith("<!--")]

    if not primary_refs:
        msg = _localized_text(
            language,
            "Nenhuma referência encontrada no documento. Verifique se o arquivo contém seções de referências.",
            "No references found in the document. Check that the file contains reference sections.",
        )
        return msg, {"intent": "list_all", "count": 0, "agent": "none"}

    numbered_lines: list[str] = []
    counter = 1
    for ref in primary_refs:
        if re.match(r"^\[\d+\]", ref):
            numbered_lines.append(ref)
        else:
            numbered_lines.append(f"[{counter}] {ref}")
        counter += 1
    raw_block = "\n".join(numbered_lines)

    citation_context: dict[int, list[str]] = _collect_all_citation_paragraphs(markdown)
    for num, paras in inventory.get("citation_paragraphs", {}).items():
        existing = citation_context.setdefault(num, [])
        for para in paras:
            if para not in existing and len(existing) < 2:
                existing.append(para)

    enriched = run_reference_extractor_agent(
        raw_block, citation_context=citation_context, allow_web=allow_web
    )

    abnt_list = run_reference_formatter_agent(enriched, allow_web=allow_web)

    heading = _localized_text(
        language,
        "### Referências do documento (ABNT)",
        "### Document references (ABNT)",
    )
    reply = f"{heading}\n\n{abnt_list}"
    meta = {
        "intent": "list_all",
        "count": len(primary_refs),
        "agent": "reference_extractor+reference_formatter",
    }
    return reply, meta


def _enrich_metadata_doi_first(metadata: dict, allow_web: bool) -> tuple[dict, dict]:
    """Enrich reference metadata using a DOI-first resolution strategy.

    Resolution order:

    1. MongoDB vector store lookup for title, DOI, and URL.
    2. DOI extraction from the stored URL or raw text.
    3. CrossRef title-to-DOI lookup (web only).
    4. BibTeX fetch via the resolved DOI (web only).
    5. Tavily page extraction as a fallback for still-incomplete metadata.

    Args:
        metadata: Initial metadata dict; must contain at least one of
            ``"raw"`` or ``"title"`` for a meaningful query.
        allow_web: Whether CrossRef, BibTeX-DOI, and Tavily lookups are
            allowed.

    Returns:
        A 2-tuple of:

        - The enriched metadata dict.
        - A stats dict with keys ``mongo_queries``, ``mongo_hits``,
          ``web_queries``, ``web_hits`` (all int).
    """
    mongo_queries = 0
    mongo_hits = 0
    web_queries = 0
    web_hits = 0

    query = (metadata.get("raw") or metadata.get("title") or "").strip()
    if query:
        mongo_queries += 1
        records = search_chunk_records(query[:500], k=4)
        if records:
            best = records[0]
            mongo_hits += 1
            metadata = _merge_metadata(
                metadata,
                {
                    "title": best.get("source_title", ""),
                    "doi": best.get("doi", ""),
                    "url": best.get("source_url", ""),
                    "file_path": best.get("file_path", ""),
                },
            )

    doi = (metadata.get("doi") or "").strip()
    if not doi:
        doi = extract_doi_from_url(metadata.get("url", "") or "") or ""
    if not doi:
        doi = search_doi_in_text(metadata.get("raw", "") or "") or ""

    if allow_web and not doi and (metadata.get("title") or ""):
        web_queries += 1
        doi = search_crossref_by_title((metadata.get("title") or "")[:200]) or ""
        if doi:
            web_hits += 1
            metadata["doi"] = doi

    bibtex_success = False
    if allow_web and doi:
        web_queries += 1
        bibtex = get_bibtex_from_doi(doi, timeout=10)
        if bibtex:
            web_hits += 1
            metadata = _merge_metadata(
                metadata, _metadata_from_bibtex(metadata.get("number"), bibtex)
            )
            metadata["doi"] = metadata.get("doi") or doi
            bibtex_success = True

    weak_metadata = not (
        metadata.get("title")
        and (metadata.get("year") or metadata.get("doi") or metadata.get("url"))
    )
    should_try_tavily = allow_web and (
        not bibtex_success or not _is_metadata_complete(metadata) or weak_metadata
    )

    if should_try_tavily:
        query_seed = (
            query or metadata.get("title") or metadata.get("url") or metadata.get("doi") or ""
        )
        web_ref, web_meta = _enrich_reference_from_web(metadata.get("number") or 0, str(query_seed))
        web_queries += int(web_meta.get("web_queries", 0))
        web_hits += int(web_meta.get("web_hits", 0))
        if web_ref:
            metadata = _merge_metadata(metadata, web_ref)

    return metadata, {
        "mongo_queries": mongo_queries,
        "mongo_hits": mongo_hits,
        "web_queries": web_queries,
        "web_hits": web_hits,
    }


def _handle_format_provided_references_request(user_text: str, allow_web: bool) -> tuple[str, dict]:
    """Format a user-provided reference list in ABNT using the agent pipeline.

    Delegates directly to :func:`run_reference_extractor_agent` followed by
    :func:`run_reference_formatter_agent` without additional enrichment steps.

    Args:
        user_text: User's message containing the reference list to format.
        allow_web: Whether external web search is permitted during extraction
            and formatting.

    Returns:
        A 2-tuple of:

        - A localized markdown reply with the ABNT-formatted sources.
        - A metadata dict with keys ``intent`` (``"format_provided"``) and
          ``agent`` (str).
    """
    language = _detect_user_language(user_text)

    enriched = run_reference_extractor_agent(user_text, allow_web=allow_web)

    abnt_list = run_reference_formatter_agent(enriched, allow_web=allow_web)

    heading = _localized_text(
        language,
        "### Fontes formatadas (ABNT)",
        "### Formatted sources (ABNT)",
    )
    reply = f"{heading}\n\n{abnt_list}"
    meta = {
        "intent": "format_provided",
        "agent": "reference_extractor+reference_formatter",
    }
    return reply, meta


def _handle_reference_request(markdown: str, user_text: str, allow_web: bool) -> tuple[str, dict]:
    """Enumerate, enrich, and display all targeted numbered references in ABNT format.

    For each target citation number the function:

    1. Parses the raw reference line from the document inventory.
    2. Enriches via MongoDB paragraph-based vector search.
    3. Optionally enriches via Tavily when metadata remains incomplete.
    4. Formats the result as an ABNT entry.

    The reply contains four labelled sections: unique deduplicated
    references, non-numbered mentions, numbered references, and search
    traceability stats.

    Args:
        markdown: Working-copy document text.
        user_text: User's request message; parsed for explicit citation
            numbers and language detection.  When no numbers are found all
            cited numbers are processed.
        allow_web: Whether Tavily web enrichment is permitted.

    Returns:
        A 2-tuple of:

        - A localized markdown reply string.
        - A stats dict with keys ``requested_numbers``,
          ``unresolved_numbers``, ``mongo_queries``, ``mongo_hits``,
          ``web_queries``, ``web_hits``.
    """
    language = _detect_user_language(user_text)
    inventory = _collect_reference_inventory(markdown)
    references_by_number = inventory["references_by_number"]
    citation_paragraphs = inventory["citation_paragraphs"]
    unique_references = inventory["unique_references"]
    cited_numbers = inventory["cited_numbers"]
    non_numbered_mentions = inventory.get("non_numbered_mentions", [])

    requested_numbers = _extract_requested_citation_numbers(user_text)
    target_numbers = requested_numbers or cited_numbers

    mongo_queries = 0
    mongo_hits = 0
    web_queries = 0
    web_hits = 0

    resolved_numbered: list[str] = []
    unresolved: list[int] = []

    complete_count = 0
    for number in target_numbers:
        raw_ref = references_by_number.get(number, f"[{number}]")
        metadata = _metadata_from_raw_reference(number, raw_ref)

        mongo_ref, mongo_meta = _enrich_reference_from_mongo(
            number, citation_paragraphs.get(number, [])
        )
        mongo_queries += int(mongo_meta.get("mongo_queries", 0))
        mongo_hits += int(mongo_meta.get("mongo_hits", 0))
        if mongo_ref:
            metadata = _merge_metadata(metadata, mongo_ref)

        need_web = allow_web and (not _is_metadata_complete(metadata))
        if need_web:
            query_seed = " ".join(citation_paragraphs.get(number, [])[:1])
            if not query_seed:
                query_seed = metadata.get("title", "")
            web_ref, web_meta = _enrich_reference_from_web(number, query_seed)
            web_queries += int(web_meta.get("web_queries", 0))
            web_hits += int(web_meta.get("web_hits", 0))
            if web_ref:
                metadata = _merge_metadata(metadata, web_ref)

        formatted = _format_abnt_entry(metadata)
        if formatted.strip():
            resolved_numbered.append(formatted)
            if _is_metadata_complete(metadata):
                complete_count += 1
            else:
                unresolved.append(number)
        else:
            unresolved.append(number)

    dedup_resolved: list[str] = []
    seen = set()
    for ref in resolved_numbered:
        key = _normalize_reference_key(ref)
        if key in seen:
            continue
        seen.add(key)
        dedup_resolved.append(ref)

    lines: list[str] = []
    lines.append(
        _localized_text(
            language,
            "### Referências únicas (deduplicadas) — padrão ABNT",
            "### Unique references (deduplicated) — ABNT style",
        )
    )
    lines.append("")
    if unique_references:
        unique_abnt: list[str] = []
        unique_seen: set[str] = set()
        for ref in unique_references:
            metadata = _metadata_from_raw_reference(None, ref)
            formatted = _format_abnt_entry(metadata)
            key = _normalize_reference_key(formatted)
            if key and key in unique_seen:
                continue
            if key:
                unique_seen.add(key)
            unique_abnt.append(formatted)
        lines.extend(f"- {ref}" for ref in unique_abnt)
    else:
        lines.append(
            _localized_text(
                language,
                "- Nenhuma referência explícita detectada no bloco de referências.",
                "- No explicit references were detected in references blocks.",
            )
        )

    lines += [
        "",
        _localized_text(
            language,
            "### Referências não numeradas detectadas",
            "### Detected non-numbered references",
        ),
        "",
    ]
    if non_numbered_mentions:
        non_numbered_abnt = [
            _format_abnt_entry(_metadata_from_raw_reference(None, mention))
            for mention in non_numbered_mentions
        ]
        lines.extend(f"- {ref}" for ref in non_numbered_abnt)
    else:
        lines.append(
            _localized_text(
                language,
                "- Nenhuma referência não numerada detectada no texto.",
                "- No non-numbered references detected in the text.",
            )
        )

    lines += [
        "",
        _localized_text(language, "### Referências numeradas [n]", "### Numbered references [n]"),
        "",
    ]
    if dedup_resolved:
        lines.extend(f"- {ref}" for ref in dedup_resolved)
    else:
        lines.append(
            _localized_text(
                language,
                "- Nenhuma referência numerada foi resolvida.",
                "- No numbered references were resolved.",
            )
        )

    if unresolved:
        lines += ["", _localized_text(language, "### Pendências", "### Pending")]
        lines.append(
            _localized_text(
                language,
                f"- Não foi possível resolver completamente: {', '.join(f'[{n}]' for n in unresolved)}",
                f"- Could not fully resolve: {', '.join(f'[{n}]' for n in unresolved)}",
            )
        )
        if not allow_web:
            lines.append(
                _localized_text(
                    language,
                    "- Para completar essas referências no padrão ABNT, ative **Allow web search** e repita o comando.",
                    "- To complete these references in ABNT format, enable **Allow web search** and run the command again.",
                )
            )

    lines += [
        "",
        _localized_text(language, "### Rastreabilidade da busca", "### Search traceability"),
        _localized_text(
            language,
            f"- MongoDB: {mongo_queries} consulta(s), {mongo_hits} item(ns) resolvido(s)",
            f"- MongoDB: {mongo_queries} query(ies), {mongo_hits} item(s) resolved",
        ),
        _localized_text(
            language,
            f"- Tavily: {web_queries} consulta(s), {web_hits} item(ns) resolvido(s)",
            f"- Tavily: {web_queries} query(ies), {web_hits} item(s) resolved",
        ),
        _localized_text(
            language,
            f"- Cobertura de [n] solicitados: {len(target_numbers) - len(unresolved)}/{len(target_numbers)}",
            f"- Requested [n] coverage: {len(target_numbers) - len(unresolved)}/{len(target_numbers)}",
        ),
        _localized_text(
            language,
            f"- Completude ABNT de [n]: {complete_count}/{len(target_numbers)}",
            f"- ABNT completeness for [n]: {complete_count}/{len(target_numbers)}",
        ),
    ]

    meta = {
        "requested_numbers": target_numbers,
        "unresolved_numbers": unresolved,
        "mongo_queries": mongo_queries,
        "mongo_hits": mongo_hits,
        "web_queries": web_queries,
        "web_hits": web_hits,
    }
    return "\n".join(lines), meta


def _list_paragraphs_using_citation(markdown: str, user_text: str) -> str:
    """List all paragraphs in the working copy that contain a specific citation token.

    Args:
        markdown: Working-copy document text.
        user_text: User's query; must contain a citation number such as
            ``[2]`` for the lookup to succeed.

    Returns:
        A localized markdown string with section-level paragraph matches and
        any matching reference lines.  Returns an error message when no
        citation number can be extracted from *user_text*, or when no
        paragraphs match.
    """
    language = _detect_user_language(user_text)
    citation_number = _extract_citation_number(user_text)
    if citation_number is None:
        return _localized_text(
            language,
            "Não consegui identificar a citação pedida. Use algo como [2].",
            "I couldn't identify the requested citation. Use something like [2].",
        )

    sections = _split_sections(markdown)
    token = f"[{citation_number}]"
    matches: list[str] = []
    reference_hits: list[str] = []

    for section in sections:
        refs = section.get("references", [])
        for ref in refs:
            if ref.startswith(token):
                reference_hits.append(f"- **{section['title']}**: {ref}")

        for paragraph_index, paragraph in enumerate(section.get("paragraphs", []), start=1):
            text = paragraph.get("text", "")
            if token not in text:
                continue
            snippet = re.sub(r"\s+", " ", text).strip()
            if len(snippet) > 280:
                snippet = snippet[:277].rstrip() + "..."
            matches.append(
                _localized_text(
                    language,
                    f"- **{section['title']}**, parágrafo **{paragraph_index}**: {snippet}",
                    f"- **{section['title']}**, paragraph **{paragraph_index}**: {snippet}",
                )
            )

    if not matches:
        return _localized_text(
            language,
            f"Nenhum parágrafo na cópia de trabalho usa a citação **{token}**.",
            f"No paragraph in the working copy uses citation **{token}**.",
        )

    lines = [
        _localized_text(
            language,
            f"### Parágrafos que usam {token}",
            f"### Paragraphs using {token}",
        ),
        "",
        *matches,
    ]
    if reference_hits:
        lines += [
            "",
            _localized_text(language, "### Referência detectada", "### Detected reference"),
            "",
            *reference_hits[:8],
        ]
    return "\n".join(lines)


def _confirm_paragraph(markdown: str, user_text: str) -> tuple[str, dict]:
    """Retrieve MongoDB evidence chunks and context for a specific paragraph.

    Target paragraph resolution order:

    1. Quoted snippet match inside *user_text*.
    2. Section index + paragraph index parsed from *user_text*.

    Args:
        markdown: Working-copy document text.
        user_text: User's query specifying the target paragraph by a quoted
            snippet, or by section and paragraph references.

    Returns:
        A 2-tuple of:

        - A localized markdown verification report showing MongoDB chunk
          evidence, source approximations, and the target paragraph text.
        - A stats dict with keys ``section`` (str), ``chunks`` (int), and
          ``references`` (int).  Empty dict when the paragraph cannot be
          resolved.
    """
    language = _detect_user_language(user_text)
    sections = _split_sections(markdown)
    snippet = _extract_quoted_snippet(user_text)

    target_para = None
    target_sec = None
    if snippet:
        for section in sections:
            for paragraph in section.get("paragraphs", []):
                if snippet.lower() in paragraph["text"].lower():
                    target_para = paragraph
                    target_sec = section
                    break
            if target_para:
                break

    if target_para is None:
        sec_idx = _resolve_section_index(user_text, sections)
        if sec_idx is not None:
            p_idx = _resolve_paragraph_index(
                user_text, len(sections[sec_idx].get("paragraphs", []))
            )
            if p_idx is not None:
                target_sec = sections[sec_idx]
                target_para = target_sec["paragraphs"][p_idx]

    if target_para is None:
        return (
            _localized_text(
                language,
                "Não consegui resolver o parágrafo alvo. Informe seção + parágrafo ou envie o trecho entre aspas.",
                "I couldn't resolve the target paragraph. Provide section + paragraph or send the excerpt in quotes.",
            ),
            {},
        )

    chunks = search_chunks(target_para["text"][:600], k=6)
    refs = target_sec.get("references", []) if target_sec else []
    ref_labels = [re.sub(r"^\[(\d+)\]\s*", "", r) for r in refs[:5]]
    authors_hint = [os.path.basename(r).replace(".pdf", "") for r in ref_labels]
    evidence = (
        "\n\n".join(chunks[:3])
        if chunks
        else _localized_text(
            language,
            "Sem chunks relevantes retornados no momento.",
            "No relevant chunks were returned at the moment.",
        )
    )

    msg = (
        _localized_text(language, "### Verificação do parágrafo\n", "### Paragraph verification\n")
        + _localized_text(
            language,
            f"- Seção: **{target_sec['title'] if target_sec else 'N/A'}**\n",
            f"- Section: **{target_sec['title'] if target_sec else 'N/A'}**\n",
        )
        + _localized_text(
            language,
            f"- Evidências MongoDB: **{len(chunks)} chunks**\n",
            f"- MongoDB evidence: **{len(chunks)} chunks**\n",
        )
        + _localized_text(
            language,
            f"- Fontes/autores (aproximação pelos arquivos/links citados): {', '.join(authors_hint[:6]) if authors_hint else 'não identificado'}\n\n",
            f"- Sources/authors (approximated from cited files/links): {', '.join(authors_hint[:6]) if authors_hint else 'not identified'}\n\n",
        )
        + _localized_text(
            language,
            f"**Trecho alvo:**\n{target_para['text'][:700]}\n\n",
            f"**Target excerpt:**\n{target_para['text'][:700]}\n\n",
        )
        + _localized_text(
            language,
            f"**Evidência principal:**\n{evidence[:1800]}",
            f"**Primary evidence:**\n{evidence[:1800]}",
        )
    )
    return msg, {
        "section": target_sec["title"] if target_sec else "",
        "chunks": len(chunks),
        "references": len(refs),
    }


def _suggest_more_documents(user_text: str, allow_web: bool) -> tuple[str, dict]:
    """Suggest related documents using local MongoDB chunks and optional Tavily web search.

    Args:
        user_text: User's query or a textual excerpt (optionally in quotes),
            used directly as the search query.
        allow_web: When ``True``, Tavily is called to supplement MongoDB
            results with live web pages.

    Returns:
        A 2-tuple of:

        - A localized markdown reply listing up to three local evidence
          chunks and, when *allow_web* is ``True``, up to three extracted
          web page titles and URLs.
        - A stats dict with keys ``source`` (str), ``chunks`` (int), and
          ``web_urls`` (int, present only when *allow_web* is ``True``).
    """
    language = _detect_user_language(user_text)
    snippet = _extract_quoted_snippet(user_text)
    query = snippet or user_text

    local_chunks = search_chunks(query[:600], k=5)
    local_msg = "\n".join(
        f"- Local evidence chunk {i + 1}: {chunk[:180]}..."
        for i, chunk in enumerate(local_chunks[:3])
    )

    if not allow_web:
        msg = (
            _localized_text(
                language,
                "### Documentos relacionados (modo local)\n",
                "### Related documents (local mode)\n",
            )
            + _localized_text(
                language,
                "Use 'search on internet' na pergunta para incluir documentos web.\n\n",
                "Use 'search on internet' in your request to include web documents.\n\n",
            )
            + f"{local_msg or _localized_text(language, '- Sem evidência local retornada.', '- No local evidence returned.')}"
        )
        return msg, {"source": "mongo", "chunks": len(local_chunks)}

    web = search_tavily_incremental(query=query[:400], previous_urls=[], max_results=5)
    urls = web.get("new_urls", [])[:3]
    extracted = (
        extract_tavily.invoke({"urls": urls, "include_images": False})
        if urls
        else {"extracted": []}
    )

    lines = [
        _localized_text(
            language,
            "### Documentos relacionados (local + web)",
            "### Related documents (local + web)",
        )
    ]
    if local_msg:
        lines += ["**MongoDB**", local_msg]
    if urls:
        lines += ["\n**Web (Tavily)**"]
        for idx, item in enumerate(extracted.get("extracted", [])[:3], start=1):
            lines.append(f"- [{idx}] {item.get('title', '(sem título)')} — {item.get('url', '')}")
    else:
        lines.append(
            _localized_text(
                language,
                "- Nenhum novo URL web encontrado.",
                "- No new web URL was found.",
            )
        )

    return "\n".join(lines), {
        "source": "mongo+web",
        "chunks": len(local_chunks),
        "web_urls": len(urls),
    }


def _build_edit_proposal(markdown: str, user_text: str, allow_web: bool) -> tuple[str, dict]:
    """Generate an AI-assisted paragraph edit proposal for a target section.

    Resolves the target section and paragraph from *user_text*, gathers
    MongoDB evidence chunks and optional Tavily web context, then calls the
    LLM to produce a revised paragraph.

    Args:
        markdown: Working-copy document text.
        user_text: User's edit instruction; must reference a section and
            optionally a paragraph number.
        allow_web: Whether Tavily context pages are fetched to ground the
            edit suggestion.

    Returns:
        A 2-tuple of:

        - A localized markdown preview showing the before/after text with
          instructions for applying the edit via the **Confirm Edit** button.
        - A proposal dict with keys ``section_title``, ``paragraph_index``,
          ``start``, ``end``, ``before``, ``after``, ``created_at``.  Empty
          dict when the target section or paragraph cannot be resolved.
    """
    language = _detect_user_language(user_text)
    sections = _split_sections(markdown)
    sec_idx = _resolve_section_index(user_text, sections)
    if sec_idx is None:
        return (
            _localized_text(
                language,
                "Não consegui identificar a seção alvo para edição.",
                "I couldn't identify the target section for editing.",
            ),
            {},
        )

    section = sections[sec_idx]
    p_idx = _resolve_paragraph_index(user_text, len(section.get("paragraphs", [])))
    if p_idx is None:
        p_idx = 0 if section.get("paragraphs") else None
    if p_idx is None:
        return (
            _localized_text(
                language,
                "A seção alvo não possui parágrafos editáveis.",
                "The target section has no editable paragraphs.",
            ),
            {},
        )

    paragraph = section["paragraphs"][p_idx]
    evidence_chunks = search_chunks(paragraph["text"][:600], k=5)

    web_context = ""
    if allow_web:
        web = search_tavily_incremental(
            query=paragraph["text"][:350], previous_urls=[], max_results=3
        )
        urls = web.get("new_urls", [])[:2]
        if urls:
            ext = extract_tavily.invoke({"urls": urls, "include_images": False})
            web_context = "\n\nWEB SOURCES:\n" + "\n\n".join(
                f"URL: {item.get('url', '')}\nTITLE: {item.get('title', '')}\nCONTENT: {str(item.get('content', ''))[:1200]}"
                for item in ext.get("extracted", [])
            )

    prompt_obj = load_prompt(
        "academic/edit_paragraph_suggestions",
        user_instruction=user_text,
        current_year=datetime.now().year,
        original_paragraph=paragraph["text"],
        evidence_chunks="\n".join(evidence_chunks[:3]),
        web_context=web_context,
    )

    try:
        proposed = str(llm_call(prompt=prompt_obj.text, temperature=0.2)).strip()
    except Exception:
        proposed = paragraph["text"]

    proposal = {
        "section_title": section["title"],
        "paragraph_index": p_idx,
        "start": paragraph["start"],
        "end": paragraph["end"],
        "before": paragraph["text"],
        "after": proposed,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    preview = (
        _localized_text(
            language,
            "### Proposta de edição (pendente)\n",
            "### Edit proposal (pending)\n",
        )
        + _localized_text(
            language,
            f"- Alvo: **{section['title']}**, parágrafo **{p_idx + 1}**\n",
            f"- Target: **{section['title']}**, paragraph **{p_idx + 1}**\n",
        )
        + _localized_text(
            language,
            "- Ação necessária: clique em **Confirm Edit** para aplicar.\n\n",
            "- Required action: click **Confirm Edit** to apply it.\n\n",
        )
        + _localized_text(
            language,
            f"**Antes**\n{proposal['before'][:1200]}\n\n",
            f"**Before**\n{proposal['before'][:1200]}\n\n",
        )
        + _localized_text(
            language,
            f"**Depois (proposto)**\n{proposal['after'][:1200]}",
            f"**After (proposed)**\n{proposal['after'][:1200]}",
        )
    )
    return preview, proposal
