import os
import re
import difflib
from typing import List
from ...config import HIST_MAX_TURNS, PLAN_MAX_CHARS


def fmt_chunks(chunks: List[str], max_chars: int = 1200) -> str:
    """Formats a list of text chunks into a single string, truncating if necessary.

    Args:
        chunks: List of text chunks to format.
        max_chars: Maximum total characters in the output string.

    Returns:
        A single string with numbered chunks, truncated to max_chars."""
    block = ""
    for i, c in enumerate(chunks, 1):
        row = f"[{i}] {c}\n"
        if len(block) + len(row) > max_chars:
            break
        block += row
    return block.strip()


def fmt_snippets(results: List[dict], max_chars: int = 1200) -> str:
    """Formats search results with title, snippet, and URL into a single string.

    Args:
        results: List of search result dicts with 'title', 'snippet', and 'url' keys.
        max_chars: Maximum total characters in the output string.

    Returns:
        A formatted string with numbered search results, truncated to max_chars.
    """
    block = ""
    for i, r in enumerate(results, 1):
        title = r.get("title", "")[:60]
        snippet = r.get("snippet", "")[:250]
        url = r.get("url", "")[:120]
        row = f"[{i}] {title}\n    {snippet}\n    {url}\n\n"
        if len(block) + len(row) > max_chars:
            break
        block += row
    return block.strip()


def summarize_hist(history: List[tuple], max_turns: int = HIST_MAX_TURNS) -> str:
    """Summarizes a conversation history into a concise string with recent turns.

    Args:
        history: List of (role, content) tuples representing the conversation history.
        max_turns: Maximum number of recent turns to include in the summary.
    Returns:
        A formatted string summarizing the recent conversation history."""
    if not history:
        return "(without conversation history)"
    recent = history[-(max_turns * 2) :]
    lines = []
    for role, c in recent:
        label = "Agent" if role == "assistant" else "User"
        resumo = c[:300] + "..." if len(c) > 300 else c
        lines.append(f"{label}: {resumo}")
    return "\n".join(lines)


def truncate(s: str, n: int = PLAN_MAX_CHARS) -> str:
    """Truncates a string to n characters, adding an ellipsis if truncated.

    Args:
        s: The input string to truncate.
        n: The maximum number of characters allowed in the output string.

    Returns:
        The original string if its length is less than or equal to n, otherwise a truncated version with an ellipsis.
    """
    return s if len(s) <= n else s[:n] + "\n...[truncated]"


def save_md(content: str, prefix: str, theme: str) -> str:
    """Saves content to a Markdown file with a name based on the theme.

    Args:
        content: The content to save in the Markdown file.
        prefix: The prefix for the file name.
        theme: The theme to include in the file name.

    Returns:
        The path to the saved Markdown file.
    """
    slug = re.sub(r"[^\w\s-]", "", theme[:40]).strip().replace(" ", "_").lower()
    path = f"{prefix}_{slug}.md"
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print("\nSaved in:", path)
    except Exception as e:
        print("Could not save:", str(e))
    return path


def normalize(text: str) -> str:
    """Lowercase, no punctuation, simple spaces.

    Args:
        text: The input text to normalize.
    Returns:
        A normalized version of the input text, with lowercase letters, no punctuation, and single spaces.
    """
    t = text.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def fuzzy_sim(a: str, b: str) -> float:
    """SequenceMatcher ratio of two strings, a simple fuzzy similarity score between 0 and 1.

    Args:
        a: First string to compare.
        b: Second string to compare.

    Returns:
        A float between 0 and 1 representing the similarity between the two strings, where 1 means identical and 0 means completely different.
    """
    return difflib.SequenceMatcher(None, a, b).ratio()


def fuzzy_search_in_text(anchor_norm: str, corpus_norm: str) -> tuple:
    """
    Slides a window over the corpus trying to find the anchor using fuzzy matching.

    Args:
        anchor_norm: The normalized anchor string to search for.
        corpus_norm: The normalized corpus string to search within.

    Returns:
        A tuple containing the best similarity score and the corresponding snippet from the corpus (up to 120 characters).
    """
    anchor_words = anchor_norm.split()
    corpus_words = corpus_norm.split()
    n = len(anchor_words)
    if n == 0:
        return 0.0, ""

    best = 0.0
    best_snippet = ""
    step = max(1, n // 4)

    for i in range(0, max(1, len(corpus_words) - n), step):
        window = " ".join(corpus_words[i : i + n + n // 3])
        score = fuzzy_sim(anchor_norm, window)
        if score > best:
            best = score
            best_snippet = window[:120]

    return best, best_snippet


def summarize_section(title: str, text: str) -> str:
    """Generates a short summary of a section using LLM.

    Args:
        title: The title of the section.
        text: The content of the section.

    Returns:
        A concise summary of the section, limited to 400 characters.
    """
    from ...config import llm_call

    resp = llm_call(
        f"Summarize the central technical concepts of in 3-4 concise sentences."
        f"'{title}'. Highlight the fundamentals, key formulas, and conclusions.\n\n"
        f"SECTION:\n{text[:2500]}",
        temperature=0.1,
    )
    return resp[:400]


def parse_technical_plan(text: str) -> tuple:
    """
    Extracts theme, summary, and section list from a technical plan in Markdown.
    Returns (theme, summary, sections) where sections is a list of dicts with 'indice', 'titulo', 'conteudo_esperado', and 'recursos'.

    Args:
        text: The input Markdown text of the technical plan.

    Returns:
        A tuple containing the theme (str), summary (str), and a list of sections (list of dicts).
    """
    theme = "Technical Review"
    m = re.search(
        r"\*\*(?:Theme|Topic|Tema|T[óo]pico):\*\*\s*(.+)", text, re.IGNORECASE
    )
    if m:
        theme = m.group(1).replace("*", "").strip()
    summary = text[:1200].strip()
    sections = []
    pattern = r"\|\s*([0-9\.]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]*)\s*\|"
    for level, title, cont_esp, resources in re.findall(pattern, text):
        level_clean = level.strip()
        title_clean = title.strip().replace("**", "")
        if (
            not level_clean
            or re.search(
                r"(?:^|\b)(Level|N[ií]vel)(?:\b|$)", level_clean, re.IGNORECASE
            )
            or "---" in level_clean
            or re.search(
                r"(?:Title|T[ií]tulo|Expected\s+Content|Conte[uú]do\s+Esperado|Resources|Recursos)",
                title_clean,
                re.IGNORECASE,
            )
        ):
            continue
        sections.append(
            {
                "index": len(sections),
                "title": f"{level_clean} {title_clean}",
                "expected_content": cont_esp.strip(),
                "resources": resources.strip(),
            }
        )
    if not sections:
        for i, t in enumerate(re.findall(r"^##\s+([0-9]+\..+)$", text, re.MULTILINE)):
            sections.append(
                {"index": i, "title": t, "expected_content": t, "resources": ""}
            )
    if not sections:
        raise ValueError("❌ Nenhuma seção encontrada no plano.")
    return theme, summary, sections


def parse_academic_plan(text: str) -> tuple:
    """
    Extract theme, summary, and section list from an academic plan Markdown file.

    Academic plans use a 3-column table:
      | N. Title | Objective | Topics |

    Returns (theme, summary, sections) in the same shape as parse_technical_plan so the
    entire downstream writer pipeline is unaffected.

    Args:
        text: The input Markdown text of the academic plan.

    Returns:
        A tuple containing the theme (str), summary (str), and a list of sections (list of dicts).
    """
    theme = "Academic Review"
    m = re.search(
        r"\*\*(?:Theme|Topic|Tema|T[óo]pico):\*\*\s*(.+)", text, re.IGNORECASE
    )
    if m:
        theme = m.group(1).replace("*", "").strip()

    # Strip a fenced code block wrapper added by the planner (``` markdown ... ```)
    inner = re.search(r"```(?:markdown)?\n([\s\S]+?)\n```", text)
    content = inner.group(1) if inner else text

    summary = content[:1200].strip()
    sections = []

    # Primary: 3-column table  | N. Title | Objective | TTopics |
    pattern = (
        r"\|\s*\*?\*?(\d[\d\.]*\.?\s+[^|*]+?)\*?\*?\s*\|\s*([^|]+)\s*\|\s*([^|]*)\s*\|"
    )
    for title_raw, objective, topics in re.findall(pattern, content):
        title_clean = title_raw.strip().replace("**", "")
        if (
            not title_clean
            or re.search(
                r"(?:Title|T[ií]tulo|Objective|Objetivo|Topics|T[óo]picos?)",
                title_clean,
                re.IGNORECASE,
            )
            or "---" in title_clean
        ):
            continue
        sections.append(
            {
                "index": len(sections),
                "title": title_clean,
                "expected_content": objective.strip(),
                "resources": topics.strip(),
            }
        )

    # Fallback: H2 / H3 numbered headings  (## 1. Title)
    if not sections:
        for i, t in enumerate(
            re.findall(r"^#{2,3}\s+(\d[\d\.]*\s+.+)$", content, re.MULTILINE)
        ):
            sections.append(
                {
                    "index": i,
                    "title": t.strip(),
                    "expected_content": t.strip(),
                    "resources": "",
                }
            )

    if not sections:
        raise ValueError("❌ No section found in the academic plan..")

    return theme, summary, sections


def extract_anchors(text: str) -> list:
    """Extracts anchor texts [ANCHOR: "..."] from a text block.

    Args:
        text: The input text containing anchor patterns.

    Returns:
        A list of anchor texts extracted from the input text, where each anchor is the content inside the [ANCHOR: "..."] pattern, stripped of leading and trailing whitespace.
    """
    pattern = re.compile(r'\[ANCHOR:\s*"((?:[^"\\]|\\.)*)"\]', re.DOTALL)
    return [m.strip() for m in pattern.findall(text)]


import re


def contains_assertion_verbs(text: str) -> bool:
    """
    Checks if a string contains common English or Portuguese narrative/assertive verbs.

    This is useful for filtering out titles, fragments, or non-sentential
    strings by ensuring the text contains at least one descriptive action.

    Args:
        text (str): The string or paragraph to analyze.

    Returns:
        bool: True if at least one tracked verb is found, False otherwise.
    """
    # Combined list of Portuguese and English assertive/descriptive verbs
    # PT: foi, é, são, demonstra, prova, mostra, evidencia, encontrou, observou, descobriu, propôs, definiu
    # EN: was, is, are, demonstrates, proves, shows, evidences, found, observed, discovered, proposed, defined
    pattern = (
        r"\b("
        r"foi|é|são|demonstra|prova|mostra|evidencia|encontrou|observou|descobriu|propôs|definiu|"
        r"was|is|are|demonstrates|proves|shows|evidences|found|observed|discovered|proposed|defined"
        r")\b"
    )

    return bool(re.search(pattern, text, re.IGNORECASE))


def is_paragraph_verifiable(paragraph: str) -> bool:
    """
    Returns True if the paragraph contains verifiable claims
    that require anchor/source support.

    Args:
        paragraph: The input text paragraph to evaluate.
    Returns:
        A boolean indicating whether the paragraph is likely to contain verifiable claims, based on heuristics such as the presence of numbers, citations, anchors, or strong verbs.
    """
    p = paragraph.strip()
    if len(p) < 60:
        return False
    if p.startswith("#"):
        return False
    if re.match(r"^\s*[-*]\s", p):
        return False
    if p.startswith("```"):
        return False
    if p.startswith("$$") or re.match(r"^\s*\$[^$]+\$", p):
        return False
    if re.match(
        r"^\*?(?:Figur(?:a|e)|Quadro|Table|Tabela|Graph|Gr[áa]fico)", p, re.IGNORECASE
    ) or p.startswith("!["):
        return False
    # Has numbers, citations, or strong verbs → likely verifiable
    has_numbers = bool(re.search(r"\b\d+[\d.,]*\b", p))
    has_citations = bool(re.search(r"\[\d+\]", p))
    has_anchors = bool(re.search(r"\[ANCHOR:", p))
    has_verbs = contains_assertion_verbs(p)
    return has_numbers or has_citations or has_anchors or has_verbs
