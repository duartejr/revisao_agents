"""Unit tests for ``scripts/run_ab_depth_experiment.py``.

``scripts/`` is not an installed package, so the module under test is loaded
via ``importlib`` from its file path rather than a normal import.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

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


async def test_main_async_runs_per_depth_creates_distinct_indexed_runs(ab_script):
    """When ``runs_per_depth`` > 1, each repetition must be its own MLflow run
    with an index-suffixed name and a logged ``run_index`` param.

    DEPTHS is reduced to ["fast"] so the test only needs to reason about the
    repetition axis, not the depth axis.
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
        await ab_script.main_async(runs_per_depth=3)

    run_names = [call.kwargs["run_name"] for call in mock_mlflow.start_run.call_args_list]
    assert run_names == ["depth_fast_run_0", "depth_fast_run_1", "depth_fast_run_2"]

    run_index_calls = [
        call for call in mock_mlflow.log_param.call_args_list if call.args[0] == "run_index"
    ]
    assert [call.args[1] for call in run_index_calls] == [0, 1, 2]


# ── CLI wiring (Typer) ───────────────────────────────────────────────────────


def test_cli_runs_per_depth_flag_is_wired_to_main_async(ab_script):
    """The --runs-per-depth/-n flag must reach main_async with the parsed int."""
    runner = CliRunner()
    fake_main_async = AsyncMock()

    with patch.object(ab_script, "main_async", fake_main_async):
        result = runner.invoke(ab_script.app, ["--runs-per-depth", "5"])

    assert result.exit_code == 0
    fake_main_async.assert_called_once_with(runs_per_depth=5, depths=ab_script.DEPTHS)


def test_cli_runs_per_depth_short_flag(ab_script):
    """The -n short alias must behave identically to --runs-per-depth."""
    runner = CliRunner()
    fake_main_async = AsyncMock()

    with patch.object(ab_script, "main_async", fake_main_async):
        result = runner.invoke(ab_script.app, ["-n", "3"])

    assert result.exit_code == 0
    fake_main_async.assert_called_once_with(runs_per_depth=3, depths=ab_script.DEPTHS)


def test_cli_rejects_runs_per_depth_below_one(ab_script):
    """The min=1 constraint must reject 0 or negative values at the CLI layer."""
    runner = CliRunner()
    result = runner.invoke(ab_script.app, ["--runs-per-depth", "0"])
    assert result.exit_code != 0


def test_cli_default_runs_per_depth_is_one(ab_script):
    """With no flag, main_async must be called with the documented default of 1."""
    runner = CliRunner()
    fake_main_async = AsyncMock()

    with patch.object(ab_script, "main_async", fake_main_async):
        result = runner.invoke(ab_script.app, [])

    assert result.exit_code == 0
    fake_main_async.assert_called_once_with(runs_per_depth=1, depths=ab_script.DEPTHS)


def test_cli_depths_flag_restricts_to_selected_subset(ab_script):
    """--depths must parse a comma-separated subset and pass it through as a list."""
    runner = CliRunner()
    fake_main_async = AsyncMock()

    with patch.object(ab_script, "main_async", fake_main_async):
        result = runner.invoke(ab_script.app, ["--depths", "advanced"])

    assert result.exit_code == 0
    fake_main_async.assert_called_once_with(runs_per_depth=1, depths=["advanced"])


def test_cli_depths_short_flag_parses_multiple_values(ab_script):
    """The -d short alias must split multiple comma-separated depths, trimming whitespace."""
    runner = CliRunner()
    fake_main_async = AsyncMock()

    with patch.object(ab_script, "main_async", fake_main_async):
        result = runner.invoke(ab_script.app, ["-d", "fast, basic"])

    assert result.exit_code == 0
    fake_main_async.assert_called_once_with(runs_per_depth=1, depths=["fast", "basic"])


# ── main_async depths validation ────────────────────────────────────────────


async def test_main_async_rejects_unknown_depth(ab_script):
    """An unrecognized depth value must fail loudly rather than silently
    running zero iterations for it (a plain `for depth in []`-style typo
    would otherwise produce an empty, silently-incomplete experiment)."""
    with pytest.raises(ValueError, match="Unknown depth"):
        await ab_script.main_async(depths=["not_a_real_depth"])


async def test_main_async_rejects_empty_depths_list(ab_script):
    """An explicit empty list must fail loudly, not silently run zero
    iterations and exit 0 with no signal that nothing happened — a real gap
    found in review: `--depths ""` (or a comma-only value) parses to `[]`
    at the CLI layer and would otherwise reach this function unnoticed."""
    with pytest.raises(ValueError, match="must not be empty"):
        await ab_script.main_async(depths=[])


async def test_main_async_normalizes_depth_case(ab_script):
    """Depths are matched case-insensitively, mirroring the sibling
    refinement script's workflow_type normalization — "ADVANCED" must
    behave identically to "advanced", not raise."""
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
        patch.object(ab_script, "mlflow", mock_mlflow),
        patch.object(ab_script, "TavilyClient", return_value=fake_client),
        patch.object(
            ab_script, "evaluate_search_snippets", AsyncMock(return_value=[fake_evaluation])
        ),
        patch.object(ab_script, "get_tracking_uri", return_value="file:///tmp/mlruns-test"),
        patch.object(ab_script, "get_clean_key", return_value="fake-key"),
    ):
        await ab_script.main_async(depths=["  ADVANCED  "])

    mock_mlflow.log_param.assert_any_call("depth", "advanced")


async def test_main_async_deduplicates_repeated_depths(ab_script):
    """A duplicated depth value must run once, not once per occurrence — a
    copy-paste/shell-quoting mistake (e.g. `--depths fast,fast`) must not
    silently double the Tavily credit spend for that depth."""
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
        patch.object(ab_script, "mlflow", mock_mlflow),
        patch.object(ab_script, "TavilyClient", return_value=fake_client),
        patch.object(
            ab_script, "evaluate_search_snippets", AsyncMock(return_value=[fake_evaluation])
        ),
        patch.object(ab_script, "get_tracking_uri", return_value="file:///tmp/mlruns-test"),
        patch.object(ab_script, "get_clean_key", return_value="fake-key"),
    ):
        await ab_script.main_async(depths=["fast", "fast", "advanced"])

    depth_calls = [
        call.args[1] for call in mock_mlflow.log_param.call_args_list if call.args[0] == "depth"
    ]
    assert depth_calls == ["fast", "advanced"]


async def test_main_async_depths_restricts_which_variants_run(ab_script):
    """Passing depths=["advanced"] must only execute that depth, leaving
    fast/basic untouched — this is what makes resuming a partial run
    possible without re-spending credits on already-finished depths."""
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
        patch.object(ab_script, "mlflow", mock_mlflow),
        patch.object(ab_script, "TavilyClient", return_value=fake_client),
        patch.object(
            ab_script, "evaluate_search_snippets", AsyncMock(return_value=[fake_evaluation])
        ),
        patch.object(ab_script, "get_tracking_uri", return_value="file:///tmp/mlruns-test"),
        patch.object(ab_script, "get_clean_key", return_value="fake-key"),
    ):
        await ab_script.main_async(depths=["advanced"])

    mock_mlflow.log_param.assert_any_call("depth", "advanced")
    depth_calls = [
        call.args[1] for call in mock_mlflow.log_param.call_args_list if call.args[0] == "depth"
    ]
    assert depth_calls == ["advanced"]
