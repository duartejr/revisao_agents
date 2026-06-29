"""Unit tests for ``scripts/generate_cost_report.py``.

``scripts/`` is not an installed package, so the module under test is loaded
via ``importlib`` from its file path rather than a normal import.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "generate_cost_report.py"


def _load_module():
    """Import ``generate_cost_report.py`` fresh, bypassing module caching."""
    spec = importlib.util.spec_from_file_location("generate_cost_report", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_cost_report"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def gcr(mlflow_local_store):
    """Load the script module against the isolated MLflow tracking store."""
    return _load_module()


def _make_runs_df(rows: list[dict]) -> pd.DataFrame:
    """Build a fake ``mlflow.search_runs()`` result from row dicts."""
    return pd.DataFrame(rows)


# ── fetch_session_cost_data ────────────────────────────────────────────────


def test_fetch_session_cost_data_empty_search_runs_returns_empty_df(gcr):
    """No runs in the tracking store must return an empty, well-shaped DataFrame."""
    with patch.object(gcr.mlflow, "search_runs", return_value=pd.DataFrame()):
        result = gcr.fetch_session_cost_data()

    assert result.empty
    assert list(result.columns) == ["review_type", "search_depth", "total_credits_used"]


def test_fetch_session_cost_data_missing_required_columns_returns_empty_df(gcr):
    """Runs that never logged ``review_type``/``search_depth``/``total_credits_used``
    as a column (not just as NaN) must not raise ``KeyError`` — they should be
    treated the same as "no instrumented data yet".
    """
    runs = _make_runs_df(
        [
            {
                "run_id": "r1",
                "status": "FINISHED",
                "tags.mlflow.parentRunId": None,
                # Note: no "params.review_type", "params.search_depth", or
                # "metrics.total_credits_used" columns at all.
            }
        ]
    )
    with patch.object(gcr.mlflow, "search_runs", return_value=runs):
        result = gcr.fetch_session_cost_data()

    assert result.empty
    assert list(result.columns) == ["review_type", "search_depth", "total_credits_used"]


def test_fetch_session_cost_data_filters_failed_baseline_and_children(gcr):
    """Only FINISHED, non-baseline, parent runs with complete cost data survive."""
    runs = _make_runs_df(
        [
            {  # valid session
                "run_id": "good",
                "status": "FINISHED",
                "tags.mlflow.parentRunId": None,
                "tags.baseline": None,
                "params.review_type": "tecnico",
                "params.search_depth": "basic",
                "metrics.total_credits_used": 3.0,
            },
            {  # failed session — excluded
                "run_id": "failed",
                "status": "FAILED",
                "tags.mlflow.parentRunId": None,
                "tags.baseline": None,
                "params.review_type": "tecnico",
                "params.search_depth": "basic",
                "metrics.total_credits_used": None,
            },
            {  # baseline run — excluded
                "run_id": "baseline",
                "status": "FINISHED",
                "tags.mlflow.parentRunId": None,
                "tags.baseline": "true",
                "params.review_type": "tecnico",
                "params.search_depth": "basic",
                "metrics.total_credits_used": 0.0,
            },
            {  # per-search child run — excluded
                "run_id": "child",
                "status": "FINISHED",
                "tags.mlflow.parentRunId": "good",
                "tags.baseline": None,
                "params.review_type": None,
                "params.search_depth": "basic",
                "metrics.total_credits_used": None,
            },
            {  # legacy run predating cost instrumentation — excluded
                "run_id": "legacy",
                "status": "FINISHED",
                "tags.mlflow.parentRunId": None,
                "tags.baseline": None,
                "params.review_type": "tecnico",
                "params.search_depth": None,
                "metrics.total_credits_used": None,
            },
        ]
    )
    with patch.object(gcr.mlflow, "search_runs", return_value=runs):
        result = gcr.fetch_session_cost_data()

    assert len(result) == 1
    assert result.iloc[0]["review_type"] == "tecnico"
    assert result.iloc[0]["total_credits_used"] == 3.0


# ── summarize ───────────────────────────────────────────────────────────────


def test_summarize_aggregates_by_review_type_and_search_depth(gcr):
    sessions = pd.DataFrame(
        {
            "review_type": ["tecnico", "tecnico", "academico"],
            "search_depth": ["basic", "basic", "advanced"],
            "total_credits_used": [3.0, 5.0, 10.0],
        }
    )

    summary = gcr.summarize(sessions)

    tecnico_row = summary[summary["review_type"] == "tecnico"].iloc[0]
    assert tecnico_row["sessions"] == 2
    assert tecnico_row["total_credits"] == 8.0
    assert tecnico_row["avg_credits_per_session"] == 4.0


# ── build_report ─────────────────────────────────────────────────────────────


def test_build_report_empty_sessions_returns_placeholder_message(gcr):
    report = gcr.build_report(pd.DataFrame(), pd.DataFrame())
    assert "No instrumented, completed planning sessions found" in report


def test_build_report_includes_summary_table(gcr):
    sessions = pd.DataFrame(
        {
            "review_type": ["tecnico"],
            "search_depth": ["basic"],
            "total_credits_used": [3.0],
        }
    )
    summary = gcr.summarize(sessions)

    report = gcr.build_report(sessions, summary)

    assert "Sessions analyzed: 1" in report
    assert "tecnico" in report
    assert "basic" in report


# ── log_report_to_mlflow ─────────────────────────────────────────────────────


def test_log_report_to_mlflow_logs_artifacts_and_group_metrics(gcr):
    sessions = pd.DataFrame(
        {
            "review_type": ["tecnico"],
            "search_depth": ["basic"],
            "total_credits_used": [3.0],
        }
    )
    summary = gcr.summarize(sessions)
    report = "# Tavily Cost Report\n"

    mock_mlflow = MagicMock()
    mock_mlflow.start_run.return_value.__enter__.return_value.info.run_id = "abc123"
    mock_mlflow.start_run.return_value.__exit__.return_value = False

    with patch.object(gcr, "mlflow", mock_mlflow):
        run_id = gcr.log_report_to_mlflow(sessions, summary, report)

    assert run_id == "abc123"
    mock_mlflow.set_experiment.assert_called_once_with(gcr.EXP_COST_REPORTS)
    mock_mlflow.log_param.assert_called_once_with("sessions_analyzed", 1)
    mock_mlflow.log_metric.assert_any_call("total_credits_used", 3.0)
    mock_mlflow.log_metric.assert_any_call("avg_credits_per_session__tecnico_basic", 3.0)

    text_artifact_names = {call.args[1] for call in mock_mlflow.log_text.call_args_list}
    assert text_artifact_names == {"cost_report.md", "cost_summary.csv"}
