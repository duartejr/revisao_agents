"""Unit tests for ``scripts/sample_relevance_labels.py``.

``scripts/`` is not an installed package, so the module under test is loaded
via ``importlib`` from its file path rather than a normal import.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "sample_relevance_labels.py"

_SAMPLE_LOG_WITH_RESULTS = """# Tavily Search — INCREMENTAL_TECHNICAL

- **Date/Time:** 2026-06-27 18:24:25
- **Type:** incremental_technical
- **Query:** `how to use ai in education`
- **Total Results:** 2
- **Credits Used:** 1
- **Request ID:** abc123

---

## [1] First Result Title

**URL:** https://example.com/a
**Score:** 0.9500
**Language:** en

**Content:**

This is the first snippet content, spanning
multiple lines of text.

---

## [2] Second Result Title

**URL:** https://example.com/b
**Score:** 0.8000
**Language:** pt

**Content:**

This is the second snippet content.

---
"""

_SAMPLE_LOG_WITH_PARTIAL_FIELDS = """# Tavily Search — INCREMENTAL_TECHNICAL

- **Date/Time:** 2026-06-27 18:24:25
- **Type:** incremental_technical
- **Query:** `edge case query`
- **Total Results:** 2
- **Credits Used:** 1
- **Request ID:** xyz789

---

## [1] Zero-Score Result

**URL:** https://example.com/zero-score

**Content:**

A not-relevant snippet. Its Score was exactly 0.0, which is falsy in
Python, so the generator (``_save_search_md``) omitted the Score line
entirely under ``if score:`` — this must not cause the whole row to be
dropped.

---

## [2] No Content Result

**URL:** https://example.com/empty

---
"""

_SAMPLE_LOG_NO_RESULTS = """# Tavily Search — ACADEMIC

- **Date/Time:** 2026-06-21 11:15:34
- **Type:** academic
- **Query:** `some query with zero hits`
- **Total Results:** 0
- **Credits Used:** 0
- **Request ID:** def456

---
"""


def _load_module():
    """Import ``sample_relevance_labels.py`` fresh, bypassing module caching."""
    spec = importlib.util.spec_from_file_location("sample_relevance_labels", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["sample_relevance_labels"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sampler():
    """Load the script module (no MLflow dependency, so no store fixture needed)."""
    return _load_module()


# ── _extract_pairs_from_file / collect_all_pairs ────────────────────────────


def test_extract_pairs_from_file_parses_both_results(sampler, tmp_path):
    log_path = tmp_path / "log_with_results.md"
    log_path.write_text(_SAMPLE_LOG_WITH_RESULTS, encoding="utf-8")

    pairs = sampler._extract_pairs_from_file(log_path)

    assert len(pairs) == 2
    assert pairs[0]["query"] == "how to use ai in education"
    assert pairs[0]["title"] == "First Result Title"
    assert pairs[0]["url"] == "https://example.com/a"
    assert pairs[0]["tavily_score"] == "0.9500"
    assert pairs[0]["language"] == "en"
    assert "first snippet content, spanning multiple lines" in pairs[0]["snippet"]
    assert pairs[1]["url"] == "https://example.com/b"


def test_extract_pairs_from_file_keeps_rows_with_missing_optional_fields(sampler, tmp_path):
    """A block with a falsy (omitted) Score/URL/Language field must still be
    kept as long as it has Content — dropping it would bias the sample
    toward already-higher-scoring results, since a Score of exactly 0.0 is
    the specific value the generator's `if score:` check treats as absent."""
    log_path = tmp_path / "log_partial_fields.md"
    log_path.write_text(_SAMPLE_LOG_WITH_PARTIAL_FIELDS, encoding="utf-8")

    pairs = sampler._extract_pairs_from_file(log_path)

    assert len(pairs) == 1  # the no-content block must be excluded
    assert pairs[0]["title"] == "Zero-Score Result"
    assert pairs[0]["url"] == "https://example.com/zero-score"
    assert pairs[0]["tavily_score"] == ""  # omitted in the source log, not fabricated
    assert pairs[0]["language"] == ""
    assert "not-relevant snippet" in pairs[0]["snippet"]


def test_extract_pairs_from_file_no_results_returns_empty(sampler, tmp_path):
    log_path = tmp_path / "log_no_results.md"
    log_path.write_text(_SAMPLE_LOG_NO_RESULTS, encoding="utf-8")

    assert sampler._extract_pairs_from_file(log_path) == []


def test_collect_all_pairs_across_multiple_files(sampler, tmp_path):
    (tmp_path / "a_with_results.md").write_text(_SAMPLE_LOG_WITH_RESULTS, encoding="utf-8")
    (tmp_path / "b_no_results.md").write_text(_SAMPLE_LOG_NO_RESULTS, encoding="utf-8")

    pairs = sampler.collect_all_pairs(tmp_path)

    assert len(pairs) == 2


# ── sample_pairs ─────────────────────────────────────────────────────────────


def test_sample_pairs_caps_at_sample_size(sampler):
    pairs = [{"id": i} for i in range(10)]
    sampled = sampler.sample_pairs(pairs, sample_size=3, seed=1)
    assert len(sampled) == 3


def test_sample_pairs_is_deterministic_for_a_fixed_seed(sampler):
    pairs = [{"id": i} for i in range(10)]
    first = sampler.sample_pairs(pairs, sample_size=5, seed=42)
    second = sampler.sample_pairs(pairs, sample_size=5, seed=42)
    assert first == second


# ── annotate_with_judge_scores ───────────────────────────────────────────────


async def test_annotate_groups_by_source_file_and_query(sampler):
    """Two rows sharing (source_file, query) must be evaluated in a single
    evaluate_search_snippets call, not two separate calls."""
    rows = [
        {
            "source_file": "log1.md",
            "query": "q1",
            "snippet": "snippet A",
            "url": "https://a.com",
        },
        {
            "source_file": "log1.md",
            "query": "q1",
            "snippet": "snippet B",
            "url": "https://b.com",
        },
    ]
    eval_a = MagicMock(relevance_level="Perfectly relevant", relevance_score=1.0)
    eval_b = MagicMock(relevance_level="Not relevant", relevance_score=0.0)
    fake_evaluate = AsyncMock(return_value=[eval_a, eval_b])

    with patch.object(sampler, "evaluate_search_snippets", fake_evaluate):
        result = await sampler.annotate_with_judge_scores(rows)

    fake_evaluate.assert_awaited_once()
    assert result[0]["llm_judge_relevance_level"] == "Perfectly relevant"
    assert result[0]["llm_judge_relevance_score"] == 1.0
    assert result[1]["llm_judge_relevance_level"] == "Not relevant"
    assert result[1]["human_label"] == ""


async def test_annotate_judge_failure_marks_error_without_raising(sampler):
    rows = [{"source_file": "log1.md", "query": "q1", "snippet": "x", "url": "https://a.com"}]
    fake_evaluate = AsyncMock(side_effect=RuntimeError("LLM API unavailable"))

    with patch.object(sampler, "evaluate_search_snippets", fake_evaluate):
        result = await sampler.annotate_with_judge_scores(rows)

    assert result[0]["llm_judge_relevance_level"] == "ERROR"


# ── write_csv ─────────────────────────────────────────────────────────────────


def test_write_csv_round_trips_all_fieldnames(sampler, tmp_path):
    output_path = tmp_path / "nested" / "labels.csv"
    row = dict.fromkeys(sampler.CSV_FIELDNAMES, "")
    row["query"] = "q1"

    sampler.write_csv([row], output_path)

    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "query" in content.splitlines()[0]
    assert "q1" in content
