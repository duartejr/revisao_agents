"""Unit tests for ``scripts/analyze_experiment_significance.py``.

``scripts/`` is not an installed package, so the module under test is loaded
via ``importlib`` from its file path rather than a normal import. It in turn
imports its sibling ``generate_experiment_report.py`` module directly (both
scripts add the ``scripts/`` directory to ``sys.path``), so that module is
loaded first via the same mechanism.
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
_SCRIPT_PATH = _SCRIPTS_DIR / "analyze_experiment_significance.py"


def _load_module():
    """Import ``analyze_experiment_significance.py`` fresh, bypassing module caching."""
    spec = importlib.util.spec_from_file_location("analyze_experiment_significance", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyze_experiment_significance"] = module
    spec.loader.exec_module(module)
    return module


class _NullRunCtx:
    """Minimal ``mlflow.start_run(...)`` context-manager stand-in for ``main()`` tests."""

    def __enter__(self):
        class _Info:
            run_id = "fake_run_id"

        class _Run:
            info = _Info()

        return _Run()

    def __exit__(self, *exc):
        return False


@pytest.fixture
def sig(mlflow_local_store):
    """Load the script module against the isolated MLflow tracking store."""
    return _load_module()


# ── welch_ttest_ci ─────────────────────────────────────────────────────────


def test_welch_ttest_ci_returns_none_below_two_samples(sig):
    assert sig.welch_ttest_ci(pd.Series([1.0]), pd.Series([1.0, 2.0])) is None
    assert sig.welch_ttest_ci(pd.Series([1.0, 2.0]), pd.Series([1.0])) is None


def test_welch_ttest_ci_computes_expected_fields(sig):
    sample_a = pd.Series([0.6, 0.65, 0.7, 0.62, 0.68, 0.71, 0.66, 0.69])
    sample_b = pd.Series([0.5, 0.55, 0.52, 0.58, 0.51, 0.56, 0.53, 0.54])

    result = sig.welch_ttest_ci(sample_a, sample_b)

    assert result["n_a"] == 8
    assert result["n_b"] == 8
    assert result["mean_diff"] == pytest.approx(sample_a.mean() - sample_b.mean())
    assert result["p_value"] < 0.05  # clearly separated samples
    assert result["ci_low"] < result["mean_diff"] < result["ci_high"]


def test_welch_ttest_ci_identical_zero_variance_samples_collapses_ci(sig):
    """Two groups with zero variance AND equal means (e.g. a metric that's
    deterministic given the variant, like Tavily's fixed per-depth credit
    cost) hits scipy's 0/0 case, which returns nan for t/p — a real bug this
    project shipped in its first report run. Must report t=0, p=1 (no
    evidence of a difference) instead of a bare "nan" in the output."""
    sample_a = pd.Series([5.0, 5.0, 5.0])
    sample_b = pd.Series([5.0, 5.0, 5.0])

    result = sig.welch_ttest_ci(sample_a, sample_b)

    assert result["mean_diff"] == 0.0
    assert result["ci_low"] == result["ci_high"] == 0.0
    assert result["t_stat"] == 0.0
    assert result["p_value"] == 1.0


def test_welch_ttest_ci_constant_but_different_groups_gives_zero_pvalue(sig):
    """Two groups that are each internally constant but differ from each
    other (e.g. fast always costs 5 credits, advanced always costs 10) is
    the well-defined limiting case (t=+/-inf) — scipy already returns a
    clean p=0.0 for this, not nan, so no override should fire here."""
    sample_a = pd.Series([5.0, 5.0, 5.0])
    sample_b = pd.Series([10.0, 10.0, 10.0])

    result = sig.welch_ttest_ci(sample_a, sample_b)

    assert result["mean_diff"] == -5.0
    assert result["p_value"] == 0.0
    assert result["t_stat"] == float("-inf")  # discriminates this branch from the equal-means one
    assert result["ci_low"] == result["ci_high"] == -5.0


def test_welch_ttest_ci_uses_tolerance_not_exact_equality_for_zero_variance(sig):
    """Two samples that are conceptually constant and equal but differ by a
    float64 representation artifact (~1e-16) must still hit the zero-variance
    override, not silently fall through to scipy's raw computation."""
    sample_a = pd.Series([0.1 + 0.2] * 5)  # 0.30000000000000004
    sample_b = pd.Series([0.3] * 5)  # 0.3

    result = sig.welch_ttest_ci(sample_a, sample_b)

    assert result["t_stat"] == 0.0
    assert result["p_value"] == 1.0


# ── fetch_runs_by_two_params ─────────────────────────────────────────────────


def test_fetch_runs_by_two_params_keeps_both_group_columns(sig, monkeypatch):
    runs = pd.DataFrame(
        {
            "status": ["FINISHED"] * 4,
            "params.workflow_type": ["technical", "technical", "academic", "academic"],
            "params.arm": ["refined", "bypassed", "refined", "bypassed"],
            "metrics.plan_quality_score": [8.0, 9.0, 8.5, 9.5],
        }
    )
    monkeypatch.setattr(sig.mlflow, "search_runs", lambda **kwargs: runs)

    result = sig.fetch_runs_by_two_params(
        "planning_refinement_ab", ["plan_quality_score"], "workflow_type", "arm"
    )

    assert list(result.columns) == ["workflow_type", "arm", "plan_quality_score"]
    assert len(result) == 4
    assert set(result["workflow_type"]) == {"technical", "academic"}


def test_fetch_runs_by_two_params_missing_column_returns_empty(sig, monkeypatch):
    runs = pd.DataFrame({"status": ["FINISHED"], "params.arm": ["refined"]})
    monkeypatch.setattr(sig.mlflow, "search_runs", lambda **kwargs: runs)

    result = sig.fetch_runs_by_two_params(
        "planning_refinement_ab", ["plan_quality_score"], "workflow_type", "arm"
    )
    assert result.empty


def test_fetch_runs_by_two_params_empty_search_returns_empty(sig, monkeypatch):
    monkeypatch.setattr(sig.mlflow, "search_runs", lambda **kwargs: pd.DataFrame())
    result = sig.fetch_runs_by_two_params("x", ["m"], "a", "b")
    assert result.empty


# ── format_stratified_tables ─────────────────────────────────────────────────


def test_format_stratified_tables_empty_dict_says_no_data(sig):
    assert "No stratified data" in sig.format_stratified_tables({})


def test_format_stratified_tables_renders_one_section_per_stratum_sorted(sig):
    stratified = {
        "academic": [
            {"comparison": "refined vs bypassed", "metric": "m", "insufficient_data": True}
        ],
        "technical": [
            {"comparison": "refined vs bypassed", "metric": "m", "insufficient_data": True}
        ],
    }
    rendered = sig.format_stratified_tables(stratified)
    assert rendered.index("#### academic") < rendered.index("#### technical")


# ── compare_all_pairs / format_significance_table ───────────────────────────


def test_compare_all_pairs_empty_runs_returns_empty_list(sig):
    assert sig.compare_all_pairs(pd.DataFrame(), "depth", ["m"], (("a", "b"),)) == []


def test_compare_all_pairs_marks_insufficient_data_when_variant_undersampled(sig):
    runs = pd.DataFrame({"depth": ["fast"], "average_relevance_score": [0.8]})
    rows = sig.compare_all_pairs(runs, "depth", ["average_relevance_score"], (("fast", "basic"),))
    assert len(rows) == 1
    assert rows[0]["insufficient_data"] is True


def test_compare_all_pairs_computes_result_for_well_sampled_pair(sig):
    runs = pd.DataFrame(
        {
            "depth": ["fast"] * 4 + ["basic"] * 4,
            "average_relevance_score": [0.6, 0.62, 0.64, 0.61, 0.8, 0.82, 0.79, 0.81],
        }
    )
    rows = sig.compare_all_pairs(runs, "depth", ["average_relevance_score"], (("fast", "basic"),))
    assert len(rows) == 1
    assert rows[0]["insufficient_data"] is False
    assert rows[0]["comparison"] == "fast vs basic"
    assert "p_value" in rows[0]


def test_format_significance_table_empty_rows_says_no_data(sig):
    assert "No finished runs" in sig.format_significance_table([])


def test_format_significance_table_renders_insufficient_and_computed_rows(sig):
    rows = [
        {"comparison": "fast vs basic", "metric": "m", "insufficient_data": True},
        {
            "comparison": "basic vs advanced",
            "metric": "m",
            "insufficient_data": False,
            "n_a": 4,
            "n_b": 4,
            "mean_a": 0.6,
            "mean_b": 0.8,
            "mean_diff": -0.2,
            "p_value": 0.01,
            "ci_low": -0.3,
            "ci_high": -0.1,
        },
    ]
    table = sig.format_significance_table(rows)
    assert "insufficient data" in table
    assert "basic vs advanced" in table
    assert "0.0100" in table


# ── main() integration (mocked mlflow) ──────────────────────────────────────


def test_main_writes_report_with_significance_section(sig, tmp_path, monkeypatch):
    """main() must fetch both experiments, compute significance, and write the
    Week 10 report file at the (redirected) REPORT_PATH."""
    depth_runs = pd.DataFrame(
        {
            "depth": ["fast"] * 3 + ["advanced"] * 3,
            **{metric: [1.0, 1.1, 0.9, 2.0, 2.1, 1.9] for metric in sig.ger.DEPTH_METRICS},
        }
    )
    refinement_runs_by_arm = pd.DataFrame(
        {
            "arm": ["refined"] * 3 + ["bypassed"] * 3,
            **{metric: [5.0, 5.1, 4.9, 3.0, 3.1, 2.9] for metric in sig.ger.REFINEMENT_METRICS},
        }
    )
    refinement_runs_by_workflow = pd.DataFrame(
        {
            "workflow_type": ["technical"] * 3 + ["academic"] * 3,
            **{metric: [5.0, 5.1, 4.9, 3.0, 3.1, 2.9] for metric in sig.ger.REFINEMENT_METRICS},
        }
    )

    refinement_runs_stratified = pd.DataFrame(
        {
            "workflow_type": ["technical"] * 3 + ["academic"] * 3,
            "arm": ["refined", "refined", "bypassed"] * 2,
            **{metric: [5.0, 5.1, 3.0, 5.2, 5.3, 3.1] for metric in sig.ger.REFINEMENT_METRICS},
        }
    )

    def fake_fetch_runs(experiment_name, metric_names, group_param):
        if experiment_name == sig.ger.EXP_AB_DEPTH_EXPERIMENTS:
            return depth_runs
        if group_param == "workflow_type":
            return refinement_runs_by_workflow
        return refinement_runs_by_arm

    report_path = tmp_path / "week10_experiment_report.md"
    monkeypatch.setattr(sig, "REPORT_PATH", report_path)
    monkeypatch.setattr(sig, "_ROOT", tmp_path)
    monkeypatch.setattr(sig.ger, "fetch_runs", fake_fetch_runs)
    monkeypatch.setattr(sig, "fetch_runs_by_two_params", lambda *a, **k: refinement_runs_stratified)
    monkeypatch.setattr(sig.mlflow, "start_run", lambda **kwargs: _NullRunCtx())
    monkeypatch.setattr(sig.mlflow, "set_experiment", lambda *a, **k: None)
    monkeypatch.setattr(sig.mlflow, "set_tag", lambda *a, **k: None)
    monkeypatch.setattr(sig.mlflow, "log_text", lambda *a, **k: None)
    monkeypatch.setattr(sig.mlflow, "set_tracking_uri", lambda *a, **k: None)

    sig.main()

    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "Week 10 Experiment Statistical Significance Report" in content
    assert "Significance (Welch's t-test, 95% CI)" in content
    assert "Technical vs. academic workflow cross-check" in content
    assert "Statistical power limitations" in content
    assert "Significance, stratified by workflow type" in content
    assert "#### academic" in content
    assert "#### technical" in content
    # With real technical/academic data present (n=3 each), the table itself
    # must contain a computed row (n counts, mean, p-value), not just the
    # explanatory prose that mentions "insufficient data" as a description
    # of the *other* possible outcome.
    workflow_section = content.split("Technical vs. academic")[1].split("## Statistical")[0]
    assert "technical vs academic | session_duration | 3/3" in workflow_section


def test_main_reports_insufficient_data_when_academic_arm_never_ran(sig, tmp_path, monkeypatch):
    """When only one workflow_type has data (the historical, pre-Week-10
    default), the technical-vs-academic section must say so via the
    existing insufficient-data path rather than a hardcoded 'deferred' claim
    that could go stale once the academic cross-check actually runs."""
    depth_runs = pd.DataFrame(
        {
            "depth": ["fast"] * 3 + ["advanced"] * 3,
            **{metric: [1.0, 1.1, 0.9, 2.0, 2.1, 1.9] for metric in sig.ger.DEPTH_METRICS},
        }
    )
    refinement_runs_by_arm = pd.DataFrame(
        {
            "arm": ["refined"] * 3 + ["bypassed"] * 3,
            **{metric: [5.0, 5.1, 4.9, 3.0, 3.1, 2.9] for metric in sig.ger.REFINEMENT_METRICS},
        }
    )
    # Only "technical" present — the academic arm never ran.
    refinement_runs_by_workflow = pd.DataFrame(
        {
            "workflow_type": ["technical"] * 6,
            **{metric: [5.0, 5.1, 4.9, 3.0, 3.1, 2.9] for metric in sig.ger.REFINEMENT_METRICS},
        }
    )

    def fake_fetch_runs(experiment_name, metric_names, group_param):
        if experiment_name == sig.ger.EXP_AB_DEPTH_EXPERIMENTS:
            return depth_runs
        if group_param == "workflow_type":
            return refinement_runs_by_workflow
        return refinement_runs_by_arm

    report_path = tmp_path / "week10_experiment_report.md"
    monkeypatch.setattr(sig, "REPORT_PATH", report_path)
    monkeypatch.setattr(sig, "_ROOT", tmp_path)
    monkeypatch.setattr(sig.ger, "fetch_runs", fake_fetch_runs)
    monkeypatch.setattr(sig, "fetch_runs_by_two_params", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(sig.mlflow, "start_run", lambda **kwargs: _NullRunCtx())
    monkeypatch.setattr(sig.mlflow, "set_experiment", lambda *a, **k: None)
    monkeypatch.setattr(sig.mlflow, "set_tag", lambda *a, **k: None)
    monkeypatch.setattr(sig.mlflow, "log_text", lambda *a, **k: None)
    monkeypatch.setattr(sig.mlflow, "set_tracking_uri", lambda *a, **k: None)

    sig.main()

    content = report_path.read_text(encoding="utf-8")
    section = content.split("Technical vs. academic")[1].split("## Statistical")[0]
    assert "insufficient data" in section
    stratified_section = content.split("Significance, stratified by workflow type")[1].split(
        "### Technical"
    )[0]
    assert "No stratified data" in stratified_section
