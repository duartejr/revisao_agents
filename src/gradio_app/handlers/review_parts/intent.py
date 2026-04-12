"""User intent detection heuristics for the interactive review handler.

Contains keyword-based classifiers that map raw chat messages to review
actions: edit proposals, reference lookups, citation-usage queries, phrase
source requests, image suggestions, and confirmation/cancellation flows.
"""

from __future__ import annotations

import hashlib
import re

from .document import _extract_quoted_snippet


def _intent(user_text: str) -> str:
    """Infer user intent from the input text based on keyword heuristics.

    Args:
        user_text: The input text from the user.

    Returns:
        A string label representing the inferred intent.
    """
    text = user_text.lower().strip()
    if text in {
        "confirm",
        "confirm edit",
        "apply edit",
        "yes apply",
        "yes",
        "confirmar",
        "confirmar edição",
        "aplicar edição",
        "aplicar edicao",
        "sim",
    }:
        return "apply_pending_edit"
    if text in {
        "cancel",
        "cancel edit",
        "discard edit",
        "no",
        "cancelar",
        "cancelar edição",
        "cancelar edicao",
        "descartar edição",
        "descartar edicao",
        "não",
        "nao",
    }:
        return "cancel_pending_edit"
    if any(
        phrase in text
        for phrase in [
            "main finding",
            "main findings",
            "key finding",
            "key findings",
            "principais achados",
            "achado principal",
            "achados principais",
        ]
    ):
        return "summarize_main_findings"
    if any(
        phrase in text
        for phrase in [
            "cited in section",
            "papers are cited",
            "references in section",
            "sources in section",
            "artigos citados",
            "referências na seção",
            "referencias na secao",
            "fontes na seção",
            "fontes na secao",
        ]
    ):
        return "list_section_citations"
    if ("confirmed" in text and "paragraph" in text) or (
        "confirmado" in text and ("parágrafo" in text or "paragrafo" in text)
    ):
        return "confirm_paragraph_by_authors"
    if any(
        phrase in text
        for phrase in [
            "more documents",
            "more sources",
            "additional documents",
            "additional sources",
            "mais documentos",
            "mais fontes",
        ]
    ) and any(phrase in text for phrase in ["phrase", "excerpt", "snippet", "frase", "trecho"]):
        return "suggest_more_documents_for_phrase"
    if any(
        word in text
        for word in [
            "edit",
            "fix",
            "add",
            "rewrite",
            "improve",
            "update",
            "modify",
            "replace",
            "remove",
            "melhore",
            "corrija",
            "adicionar",
            "reescreva",
            "atualize",
            "modifique",
            "substitua",
            "remova",
        ]
    ):
        return "propose_targeted_edit"
    return "summarize_main_findings"


def _explicit_web_request(user_text: str) -> bool:
    """Detect explicit user intent to perform a web search based on keywords.

    Args:
        user_text: The input text from the user.

    Returns:
        True when the message explicitly requests web search.
    """
    text = user_text.lower()
    return any(
        k in text
        for k in [
            "internet",
            "web",
            "online",
            "tavily",
            "search on internet",
            "search the web",
            "busque na internet",
            "pesquise na internet",
            "busque online",
        ]
    )


def _extract_citation_number(user_text: str) -> int | None:
    """Extract a citation number from the user text based on heuristics.

    Args:
        user_text: The input text from the user.

    Returns:
        The citation number if found, otherwise None.
    """
    match = re.search(r"\[(\d+)\]", user_text)
    if match:
        return int(match.group(1))

    text = user_text.lower()
    match = re.search(r"(?:source|citation|reference|refer(?:e|ê)ncia|fonte)\s*#?\s*(\d+)", text)
    if match:
        return int(match.group(1))
    return None


def _is_citation_usage_query(user_text: str) -> bool:
    """Return True only for queries asking which paragraphs use a citation.

    Args:
        user_text: The input text from the user.

    Returns:
        True if the message is about locating citation usage.
    """
    text = user_text.lower()
    if _extract_citation_number(user_text) is None:
        return False

    exclusions = [
        "replace",
        "substitut",
        "alternative",
        "instead",
        "find source",
        "find new",
        "new source",
        "search for",
        "not yet used",
        "not used yet",
        "haven't been used",
        "can be used to",
        "could replace",
        "suggest",
        "recommend",
        "look for",
        "related with",
        "related to",
        "procurar",
        "buscar",
        "busque",
        "pesquise",
        "pesquisar",
        "ainda não usada",
        "ainda nao usada",
        "não usada ainda",
        "nao usada ainda",
        "fontes não usadas",
        "fontes nao usadas",
        "novas fontes",
        "nova fonte",
        "sugerir",
        "recomendar",
        "substituir",
        "alternativa",
        "relacionado com",
        "relacionado a",
        "reescreva",
        "reescrever",
        "melhore",
        "melhorar",
        "melhorar o",
        "adicionando",
        "adicione",
        "adicionar",
        "rewrite",
        "improve",
        "add new",
        "new references",
        "abnt",
    ]
    if any(kw in text for kw in exclusions):
        return False

    listing_words = [
        "paragraph",
        "paragraphs",
        "parágrafo",
        "parágrafos",
        "paragrafo",
        "paragrafos",
        "where",
        "which",
        "what",
        "list",
        "show",
        "onde",
        "qual",
        "quais",
        "listar",
        "mostre",
        "mostrar",
    ]
    usage_words = [
        "using",
        "uses",
        "used",
        "cite",
        "cites",
        "cited",
        "referência",
        "referencia",
        "referências",
        "referencias",
        "mention",
        "mentions",
        "mentioned",
        "menciona",
        "mencionado",
        "usando",
        r"\busa\b",
        "usado",
        "citado",
        "citam",
    ]
    return any(w in text for w in listing_words) and any(re.search(w, text) for w in usage_words)


def _matches_intent_keyword(text: str, keyword: str) -> bool:
    """Match a keyword as a whole token or exact phrase inside text.

    Args:
        text: The input text to search in.
        keyword: The keyword to match.

    Returns:
        True if the keyword is present as a token or exact phrase.
    """
    text = (text or "").lower()
    keyword = (keyword or "").strip().lower()
    if not keyword:
        return False
    if re.search(r"\s", keyword):
        return keyword in text
    return bool(re.search(rf"\b{re.escape(keyword)}\b", text))


def _classify_phrase_reference_intent(user_text: str) -> tuple[bool, dict[str, bool]]:
    """Return whether the user asks for the source of a specific phrase/snippet.

    Args:
        user_text: The input text from the user.

    Returns:
        Tuple containing a boolean and a debug dictionary of heuristic signals.
    """
    text = (user_text or "").lower()
    has_citation_number = _extract_citation_number(user_text) is not None
    has_quoted_snippet = bool(_extract_quoted_snippet(user_text))

    source_markers = [
        "reference",
        "source",
        "citation",
        "referência",
        "referencia",
        "fonte",
        "citação",
        "citacao",
    ]
    phrase_markers = [
        "phrase",
        "snippet",
        "excerpt",
        "trecho",
        "frase",
    ]
    rewrite_exclusions = [
        "rephrase",
        "paraphrase",
        "paráfrase",
        "parafrasear",
        "parafraseie",
        "reescreva",
        "reescrever",
        "rewrite",
    ]

    has_source_marker = any(_matches_intent_keyword(text, marker) for marker in source_markers)
    has_phrase_marker = has_quoted_snippet or any(
        _matches_intent_keyword(text, marker) for marker in phrase_markers
    )
    has_rewrite_exclusion = any(
        _matches_intent_keyword(text, marker) for marker in rewrite_exclusions
    )

    debug = {
        "has_citation_number": has_citation_number,
        "has_source_marker": has_source_marker,
        "has_phrase_marker": has_phrase_marker,
        "has_quoted_snippet": has_quoted_snippet,
        "has_rewrite_exclusion": has_rewrite_exclusion,
    }
    return (
        has_citation_number
        and has_source_marker
        and has_phrase_marker
        and not has_rewrite_exclusion,
        debug,
    )


def _is_phrase_reference_query(user_text: str) -> bool:
    """Detect requests asking for the source/reference of a specific phrase/snippet.

    Args:
        user_text: The input text from the user.

    Returns:
        True if the message asks for a phrase reference.
    """
    return _classify_phrase_reference_intent(user_text)[0]


def _build_phrase_reference_query_seed(user_text: str) -> str:
    """Build the best-effort query seed from quoted text or from the full user message.

    Args:
        user_text: The input text from the user.

    Returns:
        A query seed string used for lookup.
    """
    snippet = _extract_quoted_snippet(user_text)
    if snippet:
        return snippet

    marker_match = re.search(
        r"(?:frase|phrase|trecho)\s*[:?]\s*(.+)$",
        user_text,
        flags=re.IGNORECASE,
    )
    if marker_match:
        candidate = marker_match.group(1).strip()
        if candidate:
            return candidate

    return (user_text or "").strip()


def _extract_requested_citation_numbers(user_text: str) -> list[int]:
    """Extract citation numbers from the user text based on heuristics.

    Args:
        user_text: The input text from the user.

    Returns:
        A sorted list of unique citation numbers.
    """
    numbers = [int(match) for match in re.findall(r"\[(\d+)\]", user_text)]
    if numbers:
        return sorted(dict.fromkeys(numbers))

    fallback = [
        int(match)
        for match in re.findall(
            r"(?:source|citation|reference|refer(?:e|ê)ncia|fonte)\s*#?\s*(\d+)",
            user_text.lower(),
        )
    ]
    return sorted(dict.fromkeys(fallback))


def _is_reference_request(user_text: str) -> bool:
    """Detect if the user is asking for references based on intent classification.

    Args:
        user_text: The input text from the user.

    Returns:
        True if the intent is reference-related.
    """
    intent = _classify_reference_intent(user_text)
    return intent in {"list_all", "format_provided", "resolve_numbers"}


def _contains_keyword(text: str, keyword: str) -> bool:
    """Check if the keyword is present in the text as a whole word or exact phrase.

    Args:
        text: The input text to search in.
        keyword: The keyword to match.

    Returns:
        True if the keyword is present.
    """
    keyword = (keyword or "").strip().lower()
    if not keyword:
        return False
    if re.search(r"\s", keyword):
        return keyword in text
    return bool(re.search(rf"\b{re.escape(keyword)}\b", text))


def _classify_reference_intent(user_text: str) -> str | None:
    """Classify the user's intent regarding references based on heuristics.

    Args:
        user_text: The input text from the user.

    Returns:
        The intent label or None when no reference intent is detected.
    """
    text = user_text.lower()
    numbers = _extract_requested_citation_numbers(user_text)

    format_keywords = [
        "abnt",
        "format",
        "formate",
        "formatar",
        "norma",
        "padrão",
        "padrao",
    ]
    has_format_keyword = any(_contains_keyword(text, keyword) for keyword in format_keywords)
    if has_format_keyword:
        provided_items = _extract_provided_reference_items(user_text)
        if provided_items:
            return "format_provided"

    explicit_list_all_phrases = [
        "todas as referências",
        "todas as referencias",
        "all references",
        "all sources",
        "sem repetição",
        "sem repeticao",
        "without duplicates",
        "deduplicate",
        "used in this document",
        "used in document",
        "usadas neste documento",
        "referências usadas no documento",
        "referencias usadas no documento",
    ]
    list_all_action_words = [
        "liste",
        "listar",
        "list",
        "show",
        "mostre",
        "retorne",
        "return",
    ]
    has_explicit_phrase = any(
        _contains_keyword(text, phrase) for phrase in explicit_list_all_phrases
    )
    has_list_action = any(_contains_keyword(text, keyword) for keyword in list_all_action_words)
    has_reference_word = any(
        _contains_keyword(text, keyword)
        for keyword in [
            "referência",
            "referencias",
            "referências",
            "references",
            "fontes",
            "sources",
        ]
    )
    if has_explicit_phrase or (has_list_action and has_reference_word and "document" in text):
        return "list_all"

    if numbers:
        return "resolve_numbers"

    return None


def _extract_provided_reference_items(user_text: str) -> list[str]:
    """Extract reference items provided by the user for formatting.

    Args:
        user_text: The input text from the user.

    Returns:
        A list of reference strings provided by the user.
    """
    lines = [line.strip() for line in (user_text or "").splitlines() if line.strip()]
    items: list[str] = []

    for line in lines:
        stripped = re.sub(r"^(?:[-*]|\d+[\).]|\[\d+\])\s*", "", line).strip()
        if not stripped:
            continue
        if re.search(
            r"\b(formate|formatar|abnt|liste|listar|all references|todas as refer)\b",
            stripped,
            flags=re.IGNORECASE,
        ):
            continue
        if ";" in stripped and len(stripped) > 30 and "http" not in stripped.lower():
            parts = [p.strip() for p in stripped.split(";") if p.strip()]
            items.extend(parts)
            continue
        items.append(stripped)

    if len(items) >= 2:
        return items

    body_after_colon = user_text.split(":", 1)[1].strip() if ":" in user_text else ""
    if body_after_colon:
        chunks = [p.strip() for p in re.split(r"\n+|;", body_after_colon) if p.strip()]
        filtered = [p for p in chunks if len(p) > 6]
        if len(filtered) >= 1:
            return filtered
    return []


def _reference_request_fingerprint(user_text: str) -> str:
    """Return a stable fingerprint for a reference request.

    Args:
        user_text: The input text from the user.

    Returns:
        A short hash fingerprint.
    """
    normalized = re.sub(r"\s+", " ", (user_text or "").strip().lower())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


def _is_affirmative_confirmation(user_text: str) -> bool:
    """Return True when the user confirms an action.

    Args:
        user_text: The input text from the user.

    Returns:
        True if the message confirms the action.
    """
    text = (user_text or "").strip().lower()
    patterns = [
        r"^(sim|s|yes|y|ok|okay|confirmo|confirmar|pode|prosseguir|continue|go ahead)\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _is_negative_confirmation(user_text: str) -> bool:
    """Return True when the user cancels an action.

    Args:
        user_text: The input text from the user.

    Returns:
        True if the message cancels the action.
    """
    text = (user_text or "").strip().lower()
    patterns = [
        r"^(nao|não|n|no|cancelar|cancela|pare|stop|cancel)\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)
