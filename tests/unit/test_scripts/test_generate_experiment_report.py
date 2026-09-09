"""Unit tests for ``scripts/generate_experiment_report.py``.

``scripts/`` is not an installed package, so the module under test is loaded
via ``importlib`` from its file path rather than a normal import.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "generate_experiment_report.py"


def _load_module():
    """Import ``generate_experiment_report.py`` fresh, bypassing module caching."""
    spec = importlib.util.spec_from_file_location("generate_experiment_report", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_experiment_report"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ger(mlflow_local_store):
    """Load the script module against the isolated MLflow tracking store."""
    return _load_module()


def _depth_runs_df(rows: list[dict]) -> pd.DataFrame:
    base = {
        "status": "FINISHED",
        "params.depth": None,
        "metrics.average_latency": None,
        "metrics.total_credits": None,
        "metrics.average_urls_found": None,
        "metrics.average_relevance_score": None,
        "metrics.average_academic_quality_pct": None,
        "metrics.average_citation_potential_pct": None,
    }
    return pd.DataFrame([{**base, **row} for row in rows])


# ── _fetch_runs ──────────────────────────────────────────────────────────────


def test_fetch_runs_empty_search_runs_returns_empty_df(ger):
    with patch.object(ger.mlflow, "search_runs", return_value=pd.DataFrame()):
        result = ger._fetch_runs(ger.EXP_AB_DEPTH_EXPERIMENTS, ger.DEPTH_METRICS, "depth")
    assert result.empty


def test_fetch_runs_missing_required_columns_returns_empty_df(ger):
    """Runs that never logged the expected columns at all (not just NaN) must
    not raise KeyError."""
    runs = pd.DataFrame([{"run_id": "r1", "status": "FINISHED"}])
    with patch.object(ger.mlflow, "search_runs", return_value=runs):
        result = ger._fetch_runs(ger.EXP_AB_DEPTH_EXPERIMENTS, ger.DEPTH_METRICS, "depth")
    assert result.empty


def test_fetch_runs_filters_unfinished_and_incomplete(ger):
    runs = _depth_runs_df(
        [
            {
                "params.depth": "fast",
                "metrics.average_latency": 1.0,
                "metrics.total_credits": 5.0,
                "metrics.average_urls_found": 3.0,
                "metrics.average_relevance_score": 0.8,
                "metrics.average_academic_quality_pct": 60.0,
                "metrics.average_citation_potential_pct": 50.0,
            },
            {  # unfinished run — excluded
                "status": "RUNNING",
                "params.depth": "fast",
                "metrics.average_latency": 1.0,
                "metrics.total_credits": 5.0,
                "metrics.average_urls_found": 3.0,
                "metrics.average_relevance_score": 0.8,
                "metrics.average_academic_quality_pct": 60.0,
                "metrics.average_citation_potential_pct": 50.0,
            },
            {  # missing a metric value — excluded via dropna
                "params.depth": "advanced",
                "metrics.average_latency": None,
                "metrics.total_credits": 9.0,
                "metrics.average_urls_found": 4.0,
                "metrics.average_relevance_score": 0.9,
                "metrics.average_academic_quality_pct": 70.0,
                "metrics.average_citation_potential_pct": 60.0,
            },
        ]
    )
    with patch.object(ger.mlflow, "search_runs", return_value=runs):
        result = ger._fetch_runs(ger.EXP_AB_DEPTH_EXPERIMENTS, ger.DEPTH_METRICS, "depth")

    assert len(result) == 1
    assert result.iloc[0]["depth"] == "fast"


# ── summarize ────────────────────────────────────────────────────────────────


def test_summarize_computes_count_mean_min_max(ger):
    runs = pd.DataFrame(
        {
            "depth": ["fast", "fast", "advanced"],
            "average_relevance_score": [0.6, 0.8, 0.9],
            "total_credits": [5.0, 7.0, 20.0],
        }
    )
    summary = ger.summarize(runs, "depth", ["average_relevance_score", "total_credits"])

    fast_row = summary[summary["depth"] == "fast"].iloc[0]
    assert fast_row["runs"] == 2
    assert fast_row["average_relevance_score_mean"] == pytest.approx(0.7)
    assert fast_row["average_relevance_score_min"] == pytest.approx(0.6)
    assert fast_row["average_relevance_score_max"] == pytest.approx(0.8)


def test_summarize_empty_returns_empty(ger):
    assert ger.summarize(pd.DataFrame(), "depth", ["x"]).empty


# ── recommend_depth ──────────────────────────────────────────────────────────


def test_recommend_depth_no_data_says_insufficient(ger):
    rec = ger.recommend_depth(pd.DataFrame(), ("fast", "basic", "advanced"))
    assert "Insufficient data" in rec


def test_recommend_depth_missing_variant_says_insufficient(ger):
    runs = pd.DataFrame(
        {
            "depth": ["fast"],
            "average_relevance_score": [0.8],
            "total_credits": [5.0],
        }
    )
    summary = ger.summarize(runs, "depth", ["average_relevance_score", "total_credits"])
    rec = ger.recommend_depth(summary, ("fast", "basic", "advanced"))
    assert "Insufficient data" in rec
    assert "basic" in rec and "advanced" in rec


def test_recommend_depth_picks_best_relevance_and_cheapest(ger):
    runs = pd.DataFrame(
        {
            "depth": ["fast", "basic", "advanced"],
            "average_relevance_score": [0.6, 0.8, 0.95],
            "total_credits": [5.0, 8.0, 20.0],
        }
    )
    summary = ger.summarize(runs, "depth", ["average_relevance_score", "total_credits"])
    rec = ger.recommend_depth(summary, ("fast", "basic", "advanced"))

    assert "advanced" in rec  # highest relevance
    assert "fast" in rec  # cheapest


# ── recommend_refinement ─────────────────────────────────────────────────────


def test_recommend_refinement_no_data_says_insufficient(ger):
    rec = ger.recommend_refinement(pd.DataFrame(), ("refined", "bypassed"))
    assert "Insufficient data" in rec


def test_recommend_refinement_higher_quality_when_refined(ger):
    runs = pd.DataFrame(
        {
            "arm": ["refined", "refined", "bypassed", "bypassed"],
            "session_duration": [10.0, 12.0, 8.0, 9.0],
            "refinement_rounds": [2, 2, 1, 1],
            "plan_section_count": [5, 6, 3, 4],
            "plan_quality_score": [8.0, 9.0, 5.0, 6.0],
        }
    )
    summary = ger.summarize(runs, "arm", ger.REFINEMENT_METRICS)
    rec = ger.recommend_refinement(summary, ("refined", "bypassed"))

    assert "keep it mandatory" in rec


def test_recommend_refinement_similar_quality_defers(ger):
    runs = pd.DataFrame(
        {
            "arm": ["refined", "bypassed"],
            "session_duration": [10.0, 9.0],
            "refinement_rounds": [2, 1],
            "plan_section_count": [5, 5],
            "plan_quality_score": [7.0, 7.1],
        }
    )
    summary = ger.summarize(runs, "arm", ger.REFINEMENT_METRICS)
    rec = ger.recommend_refinement(summary, ("refined", "bypassed"))

    assert "no clear quality difference" in rec


# ── build_report / log_report_to_mlflow ──────────────────────────────────────


def test_build_report_handles_empty_summaries(ger):
    report = ger.build_report(
        pd.DataFrame(), "**Insufficient data**", pd.DataFrame(), "**Insufficient data**"
    )
    assert "No data." in report
    assert report.count("Insufficient data") == 2


def test_log_report_to_mlflow_logs_text_artifact(ger):
    mock_mlflow = MagicMock()
    mock_mlflow.start_run.return_value.__enter__.return_value.info.run_id = "abc123"
    mock_mlflow.start_run.return_value.__exit__.return_value = False

    with patch.object(ger, "mlflow", mock_mlflow):
        run_id = ger.log_report_to_mlflow("# report body")

    assert run_id == "abc123"
    mock_mlflow.set_experiment.assert_called_once_with(ger.EXP_EXPERIMENT_REPORTS)
    mock_mlflow.log_text.assert_called_once_with("# report body", "week9_experiment_report.md")
