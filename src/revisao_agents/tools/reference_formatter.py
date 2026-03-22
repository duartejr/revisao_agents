"""
reference_formatter.py — User-driven reference formatting agent.

Reads a YAML (or JSON) file supplied by the user that lists references and a
desired citation pattern.  For each reference the agent resolves bibliographic
data using the same REACT strategies already available in the codebase
(DOI → Crossref, ArXiv, MongoDB chunks, Tavily, title search) and then formats
the complete list according to the chosen style.

Supported built-in patterns
---------------------------
abnt       ABNT NBR 6023
apa        APA 7th Edition
ieee       IEEE Reference Style
vancouver  Vancouver / NLM
mla        MLA 9th Edition
chicago    Chicago Author-Date

For any other style the user must supply ``pattern_url`` in the YAML file with
the URL of the official formatting rules.  The agent fetches that page and uses
it as context when calling the LLM to format each reference.

Public entry point
------------------
run_reference_formatter()   — interactive CLI (called from __main__.py)
format_references_from_file() — programmatic API
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # PyYAML

logger = logging.getLogger(__name__)

# ── Imports from existing codebase ────────────────────────────────────────────
from ..config import llm_call
from ..utils.llm_utils.prompt_loader import load_prompt
from ..utils.bib_utils.crossref_bibtex import get_reference_data_react
from ..utils.bib_utils.doi_utils import get_bibtex_from_doi
from ..utils.vector_utils.mongodb_corpus import CorpusMongoDB

# ─────────────────────────────────────────────────────────────────────────────
# Built-in pattern formatters
# ─────────────────────────────────────────────────────────────────────────────

#: Patterns that we can format without fetching external rules.
BUILTIN_PATTERNS = {"abnt", "apa", "ieee", "vancouver", "mla", "chicago"}


def _format_abnt(fields: Dict[str, Any]) -> str:
    """ABNT NBR 6023 formatting from a field dict.

    Args:
        fields: Dictionary containing bibliographic information.

    Returns:
        Formatted citation string.
    Example:
        >>> fields = {"author": "Silva, J.", "title": "Exemplo", "year": "2020"}
        >>> _format_abnt(fields)
        'Silva, J.. **Exemplo**, 2020.'
    """
    author = fields.get("author", "")
    title = fields.get("title", "Sem título")
    journal = fields.get("journal", "")
    year = fields.get("year", "s.d.")
    volume = fields.get("volume", "")
    number = fields.get("number", "")
    pages = fields.get("pages", "")
    doi = fields.get("doi", "")
    url = fields.get("url", "")
    publisher = fields.get("publisher", "")

    # Author: LAST, First Name. or keep as-is if already formatted
    citation = f"{author}. " if author else ""
    citation += f"**{title}**"

    if journal:
        citation += f". {journal}"
        parts = []
        if volume:
            parts.append(f"v. {volume}")
        if number:
            parts.append(f"n. {number}")
        if pages:
            parts.append(f"p. {pages}")
        if parts:
            citation += ", " + ", ".join(parts)

    if publisher:
        citation += f". {publisher}"

    citation += f", {year}."

    if doi:
        citation += f" DOI: {doi}"
    elif url and not Path(url).exists():
        citation += f" Disponível em: {url}"

    return citation


def _format_apa(fields: Dict[str, Any]) -> str:
    """APA 7th Edition formatting.

    Args:
        fields: Dictionary containing bibliographic information.

    Returns:
        Formatted citation string.
    """
    author = fields.get("author", "")
    title = fields.get("title", "Sem título")
    journal = fields.get("journal", "")
    year = fields.get("year", "n.d.")
    volume = fields.get("volume", "")
    number = fields.get("number", "")
    pages = fields.get("pages", "")
    doi = fields.get("doi", "")
    url = fields.get("url", "")
    publisher = fields.get("publisher", "")

    citation = f"{author} ({year}). " if author else f"({year}). "
    citation += f"*{title}*"

    if journal:
        citation += f". *{journal}*"
        if volume:
            citation += f", *{volume}*"
        if number:
            citation += f"({number})"
        if pages:
            citation += f", {pages}"
    elif publisher:
        citation += f". {publisher}"

    citation += "."

    if doi:
        citation += f" https://doi.org/{doi}"
    elif url and not Path(url).exists():
        citation += f" {url}"

    return citation


def _format_ieee(fields: Dict[str, Any]) -> str:
    """IEEE reference style formatting.

    Args:
        fields: Dictionary containing bibliographic information.

    Returns:
        Formatted citation string.
    """
    author = fields.get("author", "")
    title = fields.get("title", "Untitled")
    journal = fields.get("journal", "")
    year = fields.get("year", "")
    volume = fields.get("volume", "")
    number = fields.get("number", "")
    pages = fields.get("pages", "")
    doi = fields.get("doi", "")
    url = fields.get("url", "")
    publisher = fields.get("publisher", "")

    citation = f"{author}, " if author else ""
    citation += f'"{title},"'

    if journal:
        citation += f" *{journal}*"
        if volume:
            citation += f", vol. {volume}"
        if number:
            citation += f", no. {number}"
        if pages:
            citation += f", pp. {pages}"
    elif publisher:
        citation += f". {publisher}"

    if year:
        citation += f", {year}"

    citation += "."

    if doi:
        citation += f" doi: {doi}"
    elif url and not Path(url).exists():
        citation += f" [Online]. Available: {url}"

    return citation


def _format_vancouver(fields: Dict[str, Any]) -> str:
    """Vancouver / NLM formatting.

    Args:
        fields: Dictionary containing bibliographic information.

    Returns:
        Formatted citation string.
    """
    author = fields.get("author", "")
    title = fields.get("title", "Untitled")
    journal = fields.get("journal", "")
    year = fields.get("year", "")
    volume = fields.get("volume", "")
    number = fields.get("number", "")
    pages = fields.get("pages", "")
    doi = fields.get("doi", "")
    url = fields.get("url", "")

    citation = f"{author}. " if author else ""
    citation += f"{title}."

    if journal:
        citation += f" {journal}."
        if year:
            citation += f" {year}"
        if volume:
            citation += f";{volume}"
        if number:
            citation += f"({number})"
        if pages:
            citation += f":{pages}"
        citation += "."
    elif year:
        citation += f" {year}."

    if doi:
        citation += f" doi:{doi}"
    elif url and not Path(url).exists():
        citation += f" Available from: {url}"

    return citation


def _format_mla(fields: Dict[str, Any]) -> str:
    """MLA 9th edition formatting.

    Args:
        fields: Dictionary containing bibliographic information.

    Returns:
        Formatted citation string.
    """
    author = fields.get("author", "")
    title = fields.get("title", "Untitled")
    journal = fields.get("journal", "")
    year = fields.get("year", "")
    volume = fields.get("volume", "")
    number = fields.get("number", "")
    pages = fields.get("pages", "")
    doi = fields.get("doi", "")
    url = fields.get("url", "")
    publisher = fields.get("publisher", "")

    citation = f"{author}. " if author else ""
    citation += f'"{title}."'

    if journal:
        citation += f" *{journal}*"
        if volume:
            citation += f", vol. {volume}"
        if number:
            citation += f", no. {number}"
        if year:
            citation += f", {year}"
        if pages:
            citation += f", pp. {pages}"
    elif publisher:
        citation += f" {publisher}, {year}" if year else f" {publisher}"

    citation += "."

    if doi:
        citation += f" https://doi.org/{doi}"
    elif url and not Path(url).exists():
        citation += f" {url}"

    return citation


def _format_chicago(fields: Dict[str, Any]) -> str:
    """Chicago Author-Date formatting.

    Args:
        fields: Dictionary containing bibliographic information.

    Returns:
        Formatted citation string.
    """
    author = fields.get("author", "")
    title = fields.get("title", "Untitled")
    journal = fields.get("journal", "")
    year = fields.get("year", "")
    volume = fields.get("volume", "")
    number = fields.get("number", "")
    pages = fields.get("pages", "")
    doi = fields.get("doi", "")
    url = fields.get("url", "")
    publisher = fields.get("publisher", "")

    citation = f"{author}. " if author else ""
    if year:
        citation += f"{year}. "
    citation += f'"{title}."'

    if journal:
        citation += f" *{journal}*"
        if volume:
            citation += f" {volume}"
        if number:
            citation += f" ({number})"
        if pages:
            citation += f": {pages}"
    elif publisher:
        citation += f" {publisher}"

    citation += "."

    if doi:
        citation += f" https://doi.org/{doi}"
    elif url and not Path(url).exists():
        citation += f" {url}"

    return citation


_BUILTIN_FORMATTERS = {
    "abnt": _format_abnt,
    "apa": _format_apa,
    "ieee": _format_ieee,
    "vancouver": _format_vancouver,
    "mla": _format_mla,
    "chicago": _format_chicago,
}


# ─────────────────────────────────────────────────────────────────────────────
# BibTeX field extraction helper
# ─────────────────────────────────────────────────────────────────────────────

_BIBTEX_FIELD_RE = re.compile(r'(\w+)\s*=\s*["{]([^"}]+)["}]', re.IGNORECASE)


def _parse_bibtex_fields(bibtex: str) -> Dict[str, str]:
    """Extract key=value fields from a BibTeX string into a plain dict.

    Args:
        bibtex: Raw BibTeX entry as a string.

    Returns:
        Dictionary of fields extracted from the BibTeX entry."""
    return {
        m.group(1).lower(): m.group(2).strip()
        for m in _BIBTEX_FIELD_RE.finditer(bibtex)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Custom / unknown pattern support
# ─────────────────────────────────────────────────────────────────────────────


def _fetch_pattern_rules(pattern_url: str) -> str:
    """Fetch the text content of a pattern-rules page (best-effort).

    Args:
        pattern_url: URL of the page containing the formatting rules for a custom pattern.

    Returns:
        Text content of the page, stripped of HTML tags, to be used as context for LLM formatting.
    """
    try:
        req = urllib.request.Request(
            pattern_url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ref-formatter/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        # Strip HTML tags for a cleaner context string
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"\s{2,}", " ", text)
        return text[:8000]  # cap to avoid huge prompts
    except Exception as exc:
        logger.warning(f"Could not fetch pattern rules from {pattern_url}: {exc}")
        return ""


def _format_with_llm(fields: Dict[str, Any], pattern: str, rules_text: str) -> str:
    """Use the LLM to format a reference when no built-in formatter exists.
    The prompt includes the fields and the rules text to guide the LLM in formatting.

    Args:
        fields: Dictionary containing bibliographic information.
        pattern: The citation style pattern to use.
        rules_text: Text content of the formatting rules for the custom pattern.

    Returns:
        Formatted citation string.
    """
    fields_str = "\n".join(f"  {k}: {v}" for k, v in fields.items() if v)

    rules_section = (
        f"\n\nFormatting rules (from {rules_text[:200]}…):\n{rules_text[:3000]}"
        if rules_text
        else ""
    )

    prompt_obj = load_prompt(
        "common/format_reference_bibtex",
        citation_style=pattern.upper(),
        rules_section=rules_section,
        fields_text=fields_str,
    )

    try:
        result = llm_call(prompt_obj.text, temperature=0.0)
        return result.strip()
    except Exception as exc:
        logger.warning(f"LLM formatting failed: {exc}")
        return _format_abnt(fields)  # fallback to ABNT


# ─────────────────────────────────────────────────────────────────────────────
# Core per-reference resolution
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_reference(
    entry: Dict[str, Any],
    mongo_corpus: Optional[Any],
    tavily_enabled: bool,
) -> Dict[str, Any]:
    """Resolve a single reference entry into a fields dict.

    Priority:
    1.  Entry already has enough manual fields → use directly (add bibtex fetch if DOI present).
    2.  DOI provided → Crossref fetch.
    3.  URL/file provided → REACT agent (all strategies).
    4.  Fallback: use whatever is in the entry.

    Args:
        entry: The raw reference entry from the input file.
        mongo_corpus: Optional MongoDB corpus instance for REACT agent.
        tavily_enabled: Whether to allow Tavily web searches in the REACT agent.

    Returns:
        A dictionary of resolved fields for this reference, ready for formatting.
    """
    doi = entry.get("doi", "").strip() if entry.get("doi") else ""
    url = entry.get("url", "").strip() if entry.get("url") else ""

    # Detect manual-fields entries: have at least author + title
    has_manual = bool(entry.get("author") and entry.get("title"))

    # ── Strategy A: pure manual fields (may still have DOI for link) ──────
    if has_manual and not url:
        fields = dict(entry)
        if doi and not fields.get("bibtex"):
            # Try to enrich via Crossref for completeness
            try:
                bibtex = get_bibtex_from_doi(doi, timeout=10)
                if bibtex:
                    crossref_fields = _parse_bibtex_fields(bibtex)
                    # Manual fields take priority; Crossref fills gaps
                    for k, v in crossref_fields.items():
                        if k not in fields or not fields[k]:
                            fields[k] = v
            except Exception:
                pass
        fields["source"] = "manual"
        return fields

    # ── Strategy B: DOI only ──────────────────────────────────────────────
    if doi and not url and not has_manual:
        try:
            bibtex = get_bibtex_from_doi(doi, timeout=10)
            if bibtex:
                fields = _parse_bibtex_fields(bibtex)
                fields["doi"] = doi
                fields["source"] = "crossref_doi"
                return fields
        except Exception as exc:
            logger.warning(f"Crossref fetch for DOI {doi} failed: {exc}")
        return {"doi": doi, "title": f"DOI: {doi}", "source": "fallback"}

    # ── Strategy C: URL / file path → REACT agent ─────────────────────────
    if url:
        ref_data = get_reference_data_react(
            file_path=url,
            mongo_corpus=mongo_corpus,
            tavily_enabled=tavily_enabled,
            max_iterations=5,
            timeout=10,
        )
        if ref_data.get("bibtex"):
            fields = _parse_bibtex_fields(ref_data["bibtex"])
        else:
            fields = {}

        # Overlay manual fields if provided in the entry
        for k, v in entry.items():
            if v and k not in ("url", "doi"):
                fields[k] = v

        fields["url"] = url
        fields["doi"] = fields.get("doi") or ref_data.get("doi") or doi
        fields["source"] = ref_data.get("source", "unknown")
        return fields

    # ── Strategy D: manual fields with URL ────────────────────────────────
    return dict(entry) | {"source": "manual"}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def format_references_from_file(
    input_path: str,
    tavily_enabled: bool = False,
    output_path: Optional[str] = None,
) -> str:
    """Format references from a YAML or JSON input file.

    Args:
        input_path    : Path to the user-supplied YAML or JSON file.
        tavily_enabled: Whether to allow Tavily web searches.
        output_path   : If provided, write the formatted markdown to this path.

    Returns:
        Formatted markdown string (the content of "## Referências").
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # ── Load YAML or JSON ─────────────────────────────────────────────────
    raw_text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(raw_text)
    elif path.suffix.lower() == ".json":
        data = json.loads(raw_text)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix} (use .yaml or .json)")

    if not isinstance(data, dict):
        raise ValueError(
            "Input file must be a mapping with 'pattern' and 'references' keys."
        )

    pattern: str = str(data.get("pattern", "abnt")).lower().strip()
    pattern_url: str = str(data.get("pattern_url", "")).strip()
    entries: List[Any] = data.get("references", [])

    if not entries:
        raise ValueError("No references found under 'references:' in the input file.")

    print(f"\n  📋 Pattern : {pattern.upper()}")
    print(f"  📄 File    : {path.name}")
    print(f"  📚 Entries : {len(entries)}")

    # ── Fetch custom pattern rules if needed ──────────────────────────────
    rules_text = ""
    if pattern not in BUILTIN_PATTERNS:
        if not pattern_url:
            print(
                f"\n  ⚠️  Pattern '{pattern}' is not built-in.  "
                "Add 'pattern_url' to the YAML file with the formatting rules URL."
            )
        else:
            print(f"\n  🌐 Fetching formatting rules from: {pattern_url}")
            rules_text = _fetch_pattern_rules(pattern_url)
            if rules_text:
                print(f"     ✅ Rules fetched ({len(rules_text)} chars)")
            else:
                print("     ⚠️  Could not fetch rules — will use LLM best-effort")

    formatter = _BUILTIN_FORMATTERS.get(pattern)

    # ── Connect MongoDB (best-effort) ─────────────────────────────────────
    mongo_corpus: Optional[Any] = None
    try:
        mongo_corpus = CorpusMongoDB()
        mongo_corpus.connect()
        logger.info("MongoDB connected for reference formatting")
    except Exception as exc:
        logger.warning(f"MongoDB unavailable: {exc}")
        mongo_corpus = None

    # ── Process each reference ────────────────────────────────────────────
    formatted_refs: List[str] = []
    total = len(entries)

    for i, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            print(f"  ⚠️  Entry {i} is not a mapping — skipping")
            continue

        label = entry.get("doi") or entry.get("url") or entry.get("title") or f"#{i}"
        short_label = str(label)[:60]
        print(f"  [{i:2d}/{total}] {short_label}")

        try:
            fields = _resolve_reference(entry, mongo_corpus, tavily_enabled)
        except Exception as exc:
            logger.warning(f"Resolution failed for entry {i}: {exc}")
            fields = dict(entry) | {"source": "error"}

        source = fields.get("source", "?")
        icon = {
            "crossref_doi": "🔗",
            "crossref_title": "🔗",
            "arxiv": "📄",
            "mongo_chunks": "🗄️",
            "tavily": "🌐",
            "manual": "✏️",
            "fallback": "📎",
            "error": "❌",
        }.get(source, "❓")
        print(f"        {icon} source={source}")

        # Format according to pattern
        if formatter:
            citation_text = formatter(fields)
        else:
            # Unknown pattern → use LLM with fetched rules
            citation_text = _format_with_llm(fields, pattern, rules_text)

        formatted_refs.append(f"[{i}] {citation_text}")

    # ── Close MongoDB ─────────────────────────────────────────────────────
    if mongo_corpus:
        with contextlib.suppress(Exception):
            mongo_corpus.close()

    # ── Build markdown output ─────────────────────────────────────────────
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    header = (
        f"# References\n\n"
        f"> **Pattern:** {pattern.upper()} | "
        f"**File:** {path.name} | "
        f"**Generated:** {now} | "
        f"**Total:** {len(formatted_refs)} references\n\n"
        "---\n\n"
    )
    body = "\n\n".join(formatted_refs)
    markdown = header + body + "\n"

    # ── Write output ──────────────────────────────────────────────────────
    if output_path is None:
        slug = re.sub(r"[^\w]", "_", pattern)
        date_stamp = datetime.now().strftime("%Y%m%d_%H%M")
        os.makedirs("reviews", exist_ok=True)
        output_path = f"reviews/referencias_{slug}_{date_stamp}.md"

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        print(f"\n  💾 Output saved: {output_path}")
    except Exception as exc:
        print(f"\n  ⚠️  Could not save output: {exc}")

    return markdown


# ─────────────────────────────────────────────────────────────────────────────
# Interactive CLI (called from __main__.py)
# ─────────────────────────────────────────────────────────────────────────────


def run_reference_formatter() -> None:
    """Interactive menu for the Reference Formatting Agent.

    Args:
        None (all input is via prompts)

    Returns:
        None (prints formatted references and saves to file)
    """
    print("\n" + "=" * 70)
    print("AGENTE DE FORMATAÇÃO DE REFERÊNCIAS")
    print("=" * 70)
    print(
        "\nEste agente lê um arquivo YAML ou JSON com suas referências e as\n"
        "formata no padrão bibliográfico escolhido.\n"
        "\nPadrões embutidos: ABNT, APA, IEEE, Vancouver, MLA, Chicago.\n"
        "Para qualquer outro padrão, inclua 'pattern_url' no arquivo YAML\n"
        "com o link para as regras do padrão.\n"
    )
    print(f"  Exemplos disponíveis em: references/")
    print(f"    • references/example_abnt.yaml")
    print(f"    • references/example_apa.yaml")
    print(f"    • references/example_ieee.yaml")
    print(f"    • references/example_custom_pattern.yaml\n")

    # ── Input file ───────────────────────────────────────────────────────
    input_path = input("📂 Caminho do arquivo YAML/JSON de referências: ").strip()
    if not input_path:
        print("❌ Caminho vazio.")
        return
    input_path = os.path.expanduser(input_path)
    if not os.path.exists(input_path):
        print(f"❌ Arquivo não encontrado: {input_path}")
        return

    # ── Tavily option ─────────────────────────────────────────────────────
    print("\nDeseja permitir busca web via Tavily para referências sem DOI?")
    tavily_opt = input("Permitir Tavily? [y/N]: ").strip().lower() or "n"
    tavily_enabled = tavily_opt == "y"

    # ── Output path (optional) ────────────────────────────────────────────
    print("\n(Opcional) Caminho de saída para o arquivo .md formatado.")
    print("  Pressione Enter para usar o nome padrão em reviews/")
    output_path_raw = input("📝 Saída [Enter = automático]: ").strip()
    output_path: Optional[str] = (
        os.path.expanduser(output_path_raw) if output_path_raw else None
    )

    print("\n" + "=" * 70)

    try:
        markdown = format_references_from_file(
            input_path=input_path,
            tavily_enabled=tavily_enabled,
            output_path=output_path,
        )
        print("\n" + "=" * 70)
        print("✅ FORMATAÇÃO CONCLUÍDA")
        print("=" * 70)
    except Exception as exc:
        print(f"\n❌ Erro: {exc}")
        logger.exception("Reference formatter error")
