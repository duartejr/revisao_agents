"""Unit tests for ``scripts/run_ab_depth_experiment.py``.

``scripts/`` is not an installed package, so the module under test is loaded
via ``importlib`` from its file path rather than a normal import.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "run_ab_depth_experiment.py"


def _load_module():
    """Import ``run_ab_depth_experiment.py`` fresh, bypassing module caching."""
    spec = importlib.util.spec_from_file_location("run_ab_depth_experiment", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_ab_depth_experiment"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ab_script(mlflow_local_store):
    """Load the script module against the isolated MLflow tracking store."""
    return _load_module()


# ── run_single_depth ──────────────────────────────────────────────────────────


async def test_run_single_depth_credit_guard_stops_before_limit(ab_script):
    """When the first query exhausts the credit budget, the second must be skipped.

    MAX_CREDITS_PER_DEPTH = 30. First query costs 31 credits, pushing
    total_credits_used above the threshold. The guard check fires at the top of
    the loop for the second query, so ``TavilyClient.search`` must be called
    exactly once.
    """
    fake_client = MagicMock()
    fake_client.search.side_effect = [
        # first call: overshoots the 30-credit budget
        {
            "results": [{"url": "https://arxiv.org/p1", "content": "attention text"}],
            "usage": {"credits": 31},
        },
        # second call: must never be consumed
        {
            "results": [{"url": "https://arxiv.org/p2", "content": "RAG text"}],
            "usage": {"credits": 5},
        },
    ]

    fake_evaluation = MagicMock(relevance_score=1.0, academic_quality=True, citation_potential=True)

    two_queries = [
        {"query": "transformer attention", "type": "academic"},
        {"query": "RAG evaluation", "type": "academic"},
    ]

    with (
        patch.object(ab_script, "TavilyClient", return_value=fake_client),
        patch.object(
            ab_script, "evaluate_search_snippets", AsyncMock(return_value=[fake_evaluation])
        ),
        patch.object(ab_script, "get_clean_key", return_value="fake-key"),
    ):
        result = await ab_script.run_single_depth("fast", two_queries)

    assert fake_client.search.call_count == 1, "guard must stop before the second query"
    assert result["total_credits"] == 31
    assert len(result["queries"]) == 1


# ── main_async ────────────────────────────────────────────────────────────────


async def test_main_async_logs_experiment_params_and_metrics(ab_script):
    """main_async() must configure the experiment, log per-depth params, and
    log all aggregated metric keys for each depth variant.

    DEPTHS is reduced to ["fast"] via patch.object to avoid running the same
    assertion logic three times without adding coverage value.
    """
    fake_client = MagicMock()
    fake_client.search.return_value = {
        "results": [{"url": "https://example.com", "content": "some content"}],
        "usage": {"credits": 2},
    }

    fake_evaluation = MagicMock(
        relevance_score=0.8, academic_quality=True, citation_potential=False
    )

    mock_mlflow = MagicMock()
    mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()
    mock_mlflow.start_run.return_value.__exit__.return_value = False

    with (
        patch.object(ab_script, "DEPTHS", ["fast"]),
        patch.object(ab_script, "mlflow", mock_mlflow),
        patch.object(ab_script, "TavilyClient", return_value=fake_client),
        patch.object(
            ab_script, "evaluate_search_snippets", AsyncMock(return_value=[fake_evaluation])
        ),
        patch.object(ab_script, "get_tracking_uri", return_value="file:///tmp/mlruns-test"),
        patch.object(ab_script, "get_clean_key", return_value="fake-key"),
    ):
        await ab_script.main_async()

    # experiment setup
    mock_mlflow.set_experiment.assert_called_once_with("ab_depth_experiments")
    mock_mlflow.start_run.assert_called_once_with(run_name="depth_fast")

    # per-depth params
    mock_mlflow.log_param.assert_any_call("depth", "fast")
    mock_mlflow.log_param.assert_any_call("num_queries", len(ab_script.TEST_QUERIES))

    # all expected metric keys were logged
    logged_metrics: dict = mock_mlflow.log_metrics.call_args.args[0]
    assert "average_latency" in logged_metrics
    assert "total_credits" in logged_metrics
    assert "average_urls_found" in logged_metrics
    assert "average_relevance_score" in logged_metrics
    assert "average_academic_quality_pct" in logged_metrics
    assert "average_citation_potential_pct" in logged_metrics

    # JSON artifact saved with the right name
    artifact_filename = mock_mlflow.log_dict.call_args.args[1]
    assert artifact_filename == "ab_results_fast.json"
