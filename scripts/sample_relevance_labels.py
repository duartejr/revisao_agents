"""
sample_relevance_labels.py - W10-STORY-05 Manual Relevance Labeling Sample

Samples query-result pairs from existing Tavily search logs
(``runtime/search_logs/*.md``) into a CSV template for human relevance
labeling, and fills in the existing LLM-as-judge relevance score for each
sampled pair (re-run via ``evaluate_search_snippets``, since raw search logs
only record Tavily's own relevance ``Score``, not the LLM judge's verdict).
This produces the first calibration check between human judgment and the
LLM-as-judge relevance scores used throughout the A/B experiments.

Design notes:
    - The LLM judge's relevance scale is binary in practice (see
      ``docs/EVALUATION_METRICS_GUIDE.md`` and ``data/eval/README.md``):
      "Perfectly relevant" (1.0) or "Not relevant" (0.0). The
      ``relevance_level`` type technically also allows "Partially relevant",
      but the underlying MLflow ``RelevanceToQuery`` judge is a yes/no
      classifier, so that middle value never actually occurs. Human labels
      should use the same binary scale for a fair comparison — this is
      documented explicitly as a limitation rather than silently ignored.
    - Sampled pairs are grouped by (source_file, query) before calling the
      judge, since ``evaluate_search_snippets`` evaluates a whole batch of
      snippets for one query in a single call — this keeps the number of
      LLM calls proportional to distinct queries sampled, not to the sample
      size itself.
    - The ``human_label`` column is intentionally left blank: filling it in
      requires an actual human's (or domain expert's) judgment, which this
      script cannot fabricate. See ``data/eval/README.md`` for the labeling
      guideline.

Usage
-----
    uv run python scripts/sample_relevance_labels.py

Requirements
------------
- ``runtime/search_logs/`` populated with at least a few dozen result-bearing
  search logs (empty ``*_no_results.md`` files are skipped).
- ``OPENAI_API_KEY`` set (the relevance judge hardcodes ``openai:/gpt-4o-mini``
  regardless of ``LLM_PROVIDER``, per CLAUDE.md).
"""

import asyncio
import csv
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

# Ensure repo root is on sys.path when running directly with `uv run python`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from revisao_agents.config import SEARCH_LOGS_DIR  # noqa: E402
from revisao_agents.evaluation.evaluators import evaluate_search_snippets  # noqa: E402

OUTPUT_PATH = _ROOT / "data" / "eval" / "relevance_labels_template.csv"

#: Target sample size (Sprint success criterion: 30-50 labeled pairs).
SAMPLE_SIZE = 40

#: Fixed seed so re-running this script (e.g. to regenerate after new search
#: logs accumulate) produces a reproducible sample rather than a new random
#: draw each time.
RANDOM_SEED = 10

_QUERY_PATTERN = re.compile(r"\*\*Query:\*\*\s*`([^`]*)`")

# One result block spans from a "## [N] Title" heading to the next "---"
# separator. Its internal fields (URL/Score/Language/Content) are each
# written *conditionally* by the generator (`_save_search_md` in
# tools/tavily_web_search.py uses `if url:`/`if score:`/etc.), so a
# genuinely falsy value — most importantly a Score of exactly 0.0, a
# meaningful "not relevant" result, not a missing one — omits that line
# entirely. Matching the whole block first and extracting each field with
# its own optional regex (rather than one rigid all-fields-required
# pattern) avoids silently dropping those rows, which would otherwise bias
# the sample toward already-higher-scoring results.
_RESULT_BLOCK_PATTERN = re.compile(
    r"## \[(?P<idx>\d+)\] (?P<title>.*?)\n\n(?P<body>.*?)\n---", re.DOTALL
)
_URL_PATTERN = re.compile(r"\*\*URL:\*\* (.*)")
_SCORE_PATTERN = re.compile(r"\*\*Score:\*\* (.*)")
_LANGUAGE_PATTERN = re.compile(r"\*\*Language:\*\* (\w+)")
_CONTENT_PATTERN = re.compile(r"\*\*Content:\*\*\n\n(.*?)(?:\n\n\*\*Images|\Z)", re.DOTALL)

CSV_FIELDNAMES = [
    "source_file",
    "query",
    "result_index",
    "title",
    "url",
    "tavily_score",
    "language",
    "snippet",
    "llm_judge_relevance_level",
    "llm_judge_relevance_score",
    "human_label",
]


def _extract_pairs_from_file(path: Path) -> list[dict]:
    """Parse one search-log Markdown file into its query-result pair rows.

    Each of a result block's fields (URL/Score/Language/Content) is
    extracted independently and defaults to ``""`` if absent, since the log
    generator omits any field that was falsy at write time (including a
    genuine ``Score`` of ``0.0``) — see the module-level comment above
    ``_RESULT_BLOCK_PATTERN``. A row is only skipped if it has no content at
    all, since a query-result pair with no snippet text can't be labeled for
    relevance either way.

    Args:
        path: Path to a single ``runtime/search_logs/*.md`` file.

    Returns:
        A list of row dicts (empty if the file has no ``Query:`` line, no
        result blocks — e.g. a ``*_no_results.md`` file — or no block with
        actual content text).
    """
    text = path.read_text(encoding="utf-8")
    query_match = _QUERY_PATTERN.search(text)
    if not query_match:
        return []
    query = query_match.group(1)

    pairs = []
    for match in _RESULT_BLOCK_PATTERN.finditer(text):
        body = match.group("body")

        content_match = _CONTENT_PATTERN.search(body)
        if not content_match:
            continue
        content = " ".join(content_match.group(1).strip().split())
        if not content:
            continue

        url_match = _URL_PATTERN.search(body)
        score_match = _SCORE_PATTERN.search(body)
        language_match = _LANGUAGE_PATTERN.search(body)

        pairs.append(
            {
                "source_file": path.name,
                "query": query,
                "result_index": match.group("idx"),
                "title": match.group("title").strip(),
                "url": url_match.group(1).strip() if url_match else "",
                "tavily_score": score_match.group(1).strip() if score_match else "",
                "language": language_match.group(1) if language_match else "",
                "snippet": content[:500],
            }
        )
    return pairs


def collect_all_pairs(search_logs_dir: Path) -> list[dict]:
    """Parse every search log file into individual query-result pairs.

    Args:
        search_logs_dir: Directory containing ``*.md`` Tavily search logs.

    Returns:
        A flat list of query-result pair dicts (empty-result log files
        contribute nothing).
    """
    all_pairs: list[dict] = []
    for path in sorted(search_logs_dir.glob("*.md")):
        all_pairs.extend(_extract_pairs_from_file(path))
    return all_pairs


def sample_pairs(pairs: list[dict], sample_size: int, seed: int) -> list[dict]:
    """Deterministically sample up to ``sample_size`` pairs.

    Args:
        pairs: Output of :func:`collect_all_pairs`.
        sample_size: Maximum number of pairs to return.
        seed: Random seed for reproducible sampling.

    Returns:
        A list of at most ``sample_size`` pairs, in random (but seeded) order.
    """
    rng = random.Random(seed)
    shuffled = pairs.copy()
    rng.shuffle(shuffled)
    return shuffled[:sample_size]


async def annotate_with_judge_scores(pairs: list[dict]) -> list[dict]:
    """Fill in each pair's LLM-judge relevance level/score by re-running the judge.

    Groups pairs by ``(source_file, query)`` so ``evaluate_search_snippets``
    is called once per distinct query batch rather than once per row.

    Args:
        pairs: Rows from :func:`sample_pairs` (mutated in place with the two
            new ``llm_judge_relevance_*`` keys, and also returned).

    Returns:
        The same list of dicts, each now including the judge's verdict.
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in pairs:
        groups[(row["source_file"], row["query"])].append(row)

    for (_source_file, query), rows in groups.items():
        snippets = [row["snippet"] for row in rows]
        urls = [row["url"] for row in rows]
        try:
            evaluations = await evaluate_search_snippets(
                query=query,
                snippets=snippets,
                urls=urls,
                interview_metadata={"user_goals": query, "depth_setting": "unknown"},
            )
        except Exception as exc:
            print(f"Judge evaluation failed for query {query!r}: {exc}")
            for row in rows:
                row["llm_judge_relevance_level"] = "ERROR"
                row["llm_judge_relevance_score"] = ""
            continue

        for row, evaluation in zip(rows, evaluations, strict=True):
            row["llm_judge_relevance_level"] = evaluation.relevance_level
            row["llm_judge_relevance_score"] = evaluation.relevance_score

    for row in pairs:
        row.setdefault("human_label", "")

    return pairs


def write_csv(pairs: list[dict], output_path: Path) -> None:
    """Write sampled, judge-annotated pairs to the labeling CSV template.

    Args:
        pairs: Rows to write (see :data:`CSV_FIELDNAMES` for the schema).
        output_path: Destination CSV path; parent directories are created.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(pairs)


async def main_async() -> None:
    """Sample query-result pairs, annotate with judge scores, and write the CSV template."""
    all_pairs = collect_all_pairs(Path(SEARCH_LOGS_DIR))
    print(f"Found {len(all_pairs)} query-result pairs across search logs.")

    sampled = sample_pairs(all_pairs, SAMPLE_SIZE, RANDOM_SEED)
    print(f"Sampled {len(sampled)} pairs; running the LLM judge on each...")

    annotated = await annotate_with_judge_scores(sampled)

    write_csv(annotated, OUTPUT_PATH)
    print(f"Saved {len(annotated)} rows to {OUTPUT_PATH.relative_to(_ROOT)}")
    print(
        "Next step: a human (ideally a domain expert) fills in the 'human_label' "
        "column — see data/eval/README.md for the labeling guideline — then run "
        "scripts/compute_relevance_agreement.py."
    )


def main() -> None:
    """Entry point for the script."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
