"""
analyze_experiment_significance.py - W10-STORY-04 Statistical Significance Analysis

Extends Week 9's directional-only reporting (``scripts/generate_experiment_report.py``,
means/min/max only) with a proper two-sample comparison per metric per variant
pair: Welch's t-test p-value and a 95% confidence interval on the mean
difference, computed with ``scipy.stats`` (added as a direct dependency this
sprint — previously only pulled in transitively via ``mlflow``).

Reuses ``scripts/generate_experiment_report.py``'s ``fetch_runs``/``summarize``
helpers and experiment-name constants rather than duplicating the MLflow
query logic — that script is left untouched (its own tests and Week 9
report identity are frozen), while this script produces a distinct,
correctly-labeled Week 10 report at
``management/reports/week10/week10_experiment_report.md``.

Design decision on scipy vs. hand-rolled Welch's t-test (per the sprint doc's
own open question, W10-STORY-04 sub-task 4.1): scipy was chosen — it is the
standard, well-tested implementation, and pandas (already a dependency) is
its natural neighbor. This is a dependency-surface change and goes through
the ``python-review`` loop per ``.claude/rules/commit-pr-python-review-loop.md``
before being considered final.

Statistical power caveat (explicitly not omitted, per the project's existing
"insufficient data is an acceptable conclusion" convention): at these
sample sizes (see each table's own "n (A/B)" column), classical
significance testing is underpowered to detect anything but a large effect
size. A high p-value here means "not enough evidence to distinguish the
variants at this sample size," not "the variants are equivalent."

Pooling caveat: the refinement A/B's "arm" comparison pools runs across
both ``workflow_type`` values (technical and academic). Since
``workflow_type`` is itself a confirmed large, significant covariate for
some metrics (see the technical-vs-academic cross-check), the pooled
comparison is also reported stratified by ``workflow_type`` — a pooled
result that doesn't hold up in either homogeneous stratum is a pooling
artifact, not a robust finding.

Usage
-----
    uv run python scripts/analyze_experiment_significance.py

Requirements
------------
- MLflow tracking URI reachable, with the Week 10 scaled-up runs already
  logged by ``scripts/run_ab_depth_experiment.py --runs-per-depth 10`` and
  ``scripts/run_refinement_ab_experiment.py``.
"""

import math
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

# Ensure repo root AND this scripts/ directory are on sys.path: the former so
# `generate_experiment_report`'s own `revisao_agents...` imports resolve, the
# latter so this script can import that (non-package) sibling module directly.
_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
for _path in (_ROOT, _SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import generate_experiment_report as ger  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from scipy import stats  # noqa: E402

# Load explicitly rather than relying on `generate_experiment_report`'s own
# module-level `load_dotenv()` call as a side effect of the import above —
# that import-time call is an implementation detail of a sibling script this
# one doesn't own, and could reasonably move (e.g. into that script's own
# `main()`) without considering this caller.
load_dotenv()

REPORT_PATH = _ROOT / "management" / "reports" / "week10" / "week10_experiment_report.md"

#: (variant_a, variant_b) pairs to compare for each experiment.
DEPTH_PAIRS: tuple[tuple[str, str], ...] = (
    ("fast", "basic"),
    ("basic", "advanced"),
    ("fast", "advanced"),
)
REFINEMENT_PAIRS: tuple[tuple[str, str], ...] = (("refined", "bypassed"),)
#: Technical (Tavily-backed) vs. academic (MongoDB-backed) workflow
#: comparison — pools both arms (refined+bypassed) per workflow type, to
#: check whether the underlying planning workflow itself shifts these
#: metrics independent of the refinement layer.
WORKFLOW_TYPE_PAIRS: tuple[tuple[str, str], ...] = (("technical", "academic"),)

CONFIDENCE_LEVEL = 0.95


#: Absolute tolerance for treating a variance or mean-difference as "zero"
#: in :func:`welch_ttest_ci`. Exact ``== 0`` comparisons are unsafe here:
#: two samples that are conceptually identical/constant (e.g. two floating-
#: point sums that should both equal 0.3) can differ by ~1e-16 due to
#: float64 representation, which would otherwise silently fall through to
#: scipy's raw (and, for the equal-means case, nan-producing) computation
#: instead of this function's explicit zero-variance handling. All metrics
#: in this project (latency, credits, percentages, 1-10 quality scores) are
#: many orders of magnitude larger than this tolerance.
_ZERO_TOL = 1e-9


def welch_ttest_ci(sample_a: pd.Series, sample_b: pd.Series) -> dict | None:
    """Compute Welch's t-test p-value and a 95% CI on the mean difference.

    Args:
        sample_a: Numeric per-run metric values for variant A.
        sample_b: Numeric per-run metric values for variant B.

    Returns:
        A dict with ``n_a``, ``n_b``, ``mean_a``, ``mean_b``, ``mean_diff``,
        ``t_stat``, ``p_value``, ``ci_low``, ``ci_high``, or ``None`` if
        either sample has fewer than 2 observations (Welch's t-test requires
        a within-group variance estimate, which needs n >= 2 per group).

        Note: when both samples are constant (zero variance, within
        :data:`_ZERO_TOL`) and their means are also equal, ``scipy.stats.
        ttest_ind`` returns ``nan`` for both ``t_stat`` and ``p_value`` (a
        0/0 division) — this function overrides that case to ``t_stat=0.0,
        p_value=1.0`` (there is trivially no evidence of a difference)
        rather than surfacing a bare ``nan``. The other zero-variance case
        (both constant, but at different values) is already well-defined
        (``t_stat=+/-inf``, ``p_value=0.0``) and is not overridden.
    """
    n_a, n_b = len(sample_a), len(sample_b)
    if n_a < 2 or n_b < 2:
        return None

    mean_a, mean_b = sample_a.mean(), sample_b.mean()
    var_a, var_b = sample_a.var(ddof=1), sample_b.var(ddof=1)
    mean_diff = mean_a - mean_b
    se = (var_a / n_a + var_b / n_b) ** 0.5
    is_zero_variance = math.isclose(se, 0.0, abs_tol=_ZERO_TOL)

    if is_zero_variance and math.isclose(mean_diff, 0.0, abs_tol=_ZERO_TOL):
        t_stat, p_value = 0.0, 1.0
        ci_low = ci_high = mean_diff
    elif is_zero_variance:
        # Both groups constant but at different values: the well-defined
        # limiting case (t=+/-inf). scipy already handles this correctly
        # (not nan), but still emits a "precision loss" RuntimeWarning for
        # the degenerate input — expected and harmless here, so suppressed.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t_stat, p_value = stats.ttest_ind(sample_a, sample_b, equal_var=False)
        ci_low = ci_high = mean_diff
    else:
        t_stat, p_value = stats.ttest_ind(sample_a, sample_b, equal_var=False)
        df = (var_a / n_a + var_b / n_b) ** 2 / (
            (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        )
        t_crit = stats.t.ppf(1 - (1 - CONFIDENCE_LEVEL) / 2, df)
        ci_low = mean_diff - t_crit * se
        ci_high = mean_diff + t_crit * se

    return {
        "n_a": n_a,
        "n_b": n_b,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_diff": mean_diff,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def fetch_runs_by_two_params(
    experiment_name: str, metric_names: list[str], group_param_a: str, group_param_b: str
) -> pd.DataFrame:
    """Fetch finished runs for one experiment, keeping two param columns.

    Mirrors ``generate_experiment_report.fetch_runs`` but preserves both
    grouping columns instead of just one, so a metric can be stratified by
    a second dimension (e.g. ``arm`` within each ``workflow_type``) before
    comparison — needed because pooling across a confounding covariate can
    produce a "significant" result that doesn't hold up in either
    homogeneous stratum (or vice versa, mask one that does).

    Args:
        experiment_name: MLflow experiment to query.
        metric_names: Metric column suffixes expected on each run.
        group_param_a: First param column to keep (e.g. ``"workflow_type"``).
        group_param_b: Second param column to keep (e.g. ``"arm"``).

    Returns:
        A DataFrame with columns ``[group_param_a, group_param_b, *metric_names]``,
        or empty if the experiment has no finished runs or is missing any
        expected column (e.g. a run predating ``group_param_a``'s introduction).
    """
    runs = cast("pd.DataFrame", mlflow.search_runs(experiment_names=[experiment_name]))
    if runs.empty:
        return pd.DataFrame()

    runs = runs[runs["status"] == "FINISHED"]

    param_col_a = f"params.{group_param_a}"
    param_col_b = f"params.{group_param_b}"
    metric_cols = [f"metrics.{m}" for m in metric_names]
    required = [param_col_a, param_col_b, *metric_cols]
    if any(col not in runs for col in required):
        return pd.DataFrame()

    runs = runs.dropna(subset=required)
    renamed = runs.rename(
        columns={
            param_col_a: group_param_a,
            param_col_b: group_param_b,
            **dict(zip(metric_cols, metric_names, strict=True)),
        }
    )
    return renamed[[group_param_a, group_param_b, *metric_names]]


def compare_all_pairs(
    runs: pd.DataFrame,
    group_col: str,
    metric_names: list[str],
    pairs: tuple[tuple[str, str], ...],
) -> list[dict]:
    """Run Welch's t-test + 95% CI for every (metric, variant pair) combination.

    Args:
        runs: Run-level DataFrame from ``generate_experiment_report.fetch_runs``
            (one row per finished MLflow run, columns ``[group_col, *metric_names]``).
        group_col: Column identifying the variant (e.g. ``"depth"`` or ``"arm"``).
        metric_names: Metric columns to test.
        pairs: Variant pairs to compare.

    Returns:
        One dict per (metric, pair) combination with the comparison result,
        or an ``"insufficient_data"`` marker row when either variant has
        fewer than 2 finished runs or is entirely missing.
    """
    rows: list[dict] = []
    if runs.empty:
        return rows

    for variant_a, variant_b in pairs:
        sample_group_a = runs[runs[group_col] == variant_a]
        sample_group_b = runs[runs[group_col] == variant_b]
        for metric in metric_names:
            result = None
            if len(sample_group_a) >= 2 and len(sample_group_b) >= 2:
                result = welch_ttest_ci(sample_group_a[metric], sample_group_b[metric])

            row: dict[str, str | bool | int | float] = {
                "comparison": f"{variant_a} vs {variant_b}",
                "metric": metric,
            }
            if result is None:
                row["insufficient_data"] = True
            else:
                row["insufficient_data"] = False
                row.update(result)
            rows.append(row)
    return rows


def format_significance_table(rows: list[dict]) -> str:
    """Render significance comparison rows as a Markdown table.

    Args:
        rows: Output of :func:`compare_all_pairs`.

    Returns:
        Markdown table text, or an explicit "no data" note if ``rows`` is empty.
    """
    if not rows:
        return "_No finished runs available for significance testing._"

    lines = [
        "| Comparison | Metric | n (A/B) | Mean A | Mean B | Δ (A-B) | p-value | 95% CI (Δ) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row["insufficient_data"]:
            lines.append(
                f"| {row['comparison']} | {row['metric']} | insufficient data (n<2 per group) "
                "| — | — | — | — | — |"
            )
            continue
        sig_marker = " **" if row["p_value"] < 0.05 else ""
        lines.append(
            f"| {row['comparison']} | {row['metric']} | {row['n_a']}/{row['n_b']} | "
            f"{row['mean_a']:.3f} | {row['mean_b']:.3f} | {row['mean_diff']:+.3f} | "
            f"{row['p_value']:.4f}{sig_marker} | [{row['ci_low']:.3f}, {row['ci_high']:.3f}] |"
        )
    return "\n".join(lines)


def format_stratified_tables(stratified: dict[str, list[dict]]) -> str:
    """Render one significance table per stratum (e.g. per ``workflow_type``).

    Args:
        stratified: Mapping from stratum name (e.g. ``"technical"``,
            ``"academic"``) to that stratum's :func:`compare_all_pairs` output.

    Returns:
        Markdown text with one ``####`` subsection per stratum, or an
        explicit "no data" note if ``stratified`` is empty.
    """
    if not stratified:
        return "_No stratified data available (no `workflow_type`-tagged runs found)._"

    sections = []
    for stratum in sorted(stratified):
        sections.append(f"#### {stratum}\n\n{format_significance_table(stratified[stratum])}")
    return "\n\n".join(sections)


def build_significance_report(
    depth_summary: pd.DataFrame,
    depth_significance: list[dict],
    refinement_summary: pd.DataFrame,
    refinement_significance: list[dict],
    refinement_significance_stratified: dict[str, list[dict]],
    workflow_type_significance: list[dict],
) -> str:
    """Render the full Week 10 statistical significance report as Markdown.

    Args:
        depth_summary: Output of ``generate_experiment_report.summarize`` for
            the depth A/B experiment (descriptive mean/min/max, for context).
        depth_significance: Output of :func:`compare_all_pairs` for the depth
            A/B experiment.
        refinement_summary: Same as ``depth_summary``, for the refinement A/B.
        refinement_significance: Same as ``depth_significance``, for the
            refinement A/B (refined vs. bypassed, POOLING both workflow
            types). Reported alongside, not instead of,
            ``refinement_significance_stratified`` — see the pooling caveat
            in this module's docstring for why the pooled number alone can
            mislead when ``workflow_type`` is itself a significant covariate.
        refinement_significance_stratified: Output of :func:`compare_all_pairs`
            for refined vs. bypassed, computed separately within each
            ``workflow_type`` value present in the data (via
            :func:`fetch_runs_by_two_params`), so a pooled "significant"
            result can be checked against whether it actually holds in
            either homogeneous stratum.
        workflow_type_significance: Output of :func:`compare_all_pairs`
            comparing the technical vs. academic workflow (pooling both
            arms) — empty/insufficient-data rows if the academic cross-check
            didn't run this sprint (e.g. MongoDB unreachable).

    Returns:
        Markdown-formatted report text.
    """
    lines = [
        "# Week 10 Experiment Statistical Significance Report",
        "",
        f"Generated: {datetime.now(tz=UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "Extends the Week 9 directional-signal report "
        "(`management/reports/week9/week9_experiment_report.md`) with a proper "
        "two-sample comparison — Welch's t-test p-value and a 95% confidence "
        "interval on the mean difference — for every metric/variant pair, "
        "computed via `scipy.stats` over the Week 10 scaled-up sample.",
        "",
        "## 1. Tavily Depth A/B (`ab_depth_experiments`)",
        "",
        "### Descriptive summary",
        "",
        "```" if not depth_summary.empty else "_No data._",
        depth_summary.to_string(index=False) if not depth_summary.empty else "",
        "```" if not depth_summary.empty else "",
        "",
        "### Significance (Welch's t-test, 95% CI)",
        "",
        format_significance_table(depth_significance),
        "",
        "## 2. Planning Refinement A/B (`planning_refinement_ab`)",
        "",
        "### Descriptive summary",
        "",
        "```" if not refinement_summary.empty else "_No data._",
        refinement_summary.to_string(index=False) if not refinement_summary.empty else "",
        "```" if not refinement_summary.empty else "",
        "",
        "### Significance (Welch's t-test, 95% CI) — pooled across workflow types",
        "",
        "**Caveat:** pools runs across both `workflow_type` values (technical "
        "+ academic). The technical-vs-academic cross-check below shows "
        "`workflow_type` is itself a significant covariate for some "
        "metrics — a pooled result here that does not also hold in the "
        "stratified breakdown immediately following is a pooling artifact, "
        "not a robust finding; conversely a stratified effect that the "
        "pooled number dilutes away is not evidence of 'no effect'.",
        "",
        format_significance_table(refinement_significance),
        "",
        "### Significance, stratified by workflow type",
        "",
        "The same refined-vs-bypassed comparison, computed separately "
        "within each workflow type, to check the pooled result above "
        "against each homogeneous population.",
        "",
        format_stratified_tables(refinement_significance_stratified),
        "",
        "### Technical vs. academic workflow cross-check",
        "",
        "Pools both arms (refined + bypassed) per workflow type, to check "
        "whether the underlying planning workflow itself (Tavily-backed vs. "
        "MongoDB-backed) shifts these metrics independent of the refinement "
        "layer. If this shows only 'insufficient data' rows, the "
        "academic-workflow cross-check did not run this sprint (e.g. "
        "MongoDB Atlas was unreachable) and remains deferred, per the "
        "sprint's own Day-2 escalation trigger — not silently dropped.",
        "",
        format_significance_table(workflow_type_significance),
        "",
        "## Statistical power limitations",
        "",
        "- Sample sizes here (see each table's own 'n (A/B)' column — the "
        "stratified breakdown in particular runs at roughly half the "
        "pooled n per stratum) are small for classical significance "
        "testing. A t-test at this scale can reliably detect only large "
        "effect sizes — a non-significant p-value (p >= 0.05) means "
        "*insufficient evidence to distinguish the variants at this sample "
        "size*, not that the variants are equivalent.",
        "- Do not treat any single comparison here as a final production "
        "decision. Per the project's existing convention (see "
        "`scripts/generate_experiment_report.py`'s docstring and the Week 9 "
        "risk register), 'insufficient data' / 'no significant difference "
        "detected' is an accepted, honest conclusion — not a failure of "
        "this analysis.",
        "- These are two-sided Welch's t-tests (unequal-variance, "
        "appropriate since the two arms/depths are not assumed to have "
        "equal variance). No multiple-comparison correction (e.g. "
        "Bonferroni) has been applied across the several metrics/pairs "
        "tested per experiment — with 6 depth metrics x 3 pairs = 18 tests, "
        "some nominally-significant results are expected by chance alone; "
        "treat isolated significant metrics with proportionate skepticism.",
        "",
    ]
    return "\n".join(lines)


def log_report_to_mlflow(report: str) -> str:
    """Log the report as an MLflow artifact on a dedicated run.

    Args:
        report: Output of :func:`build_significance_report`.

    Returns:
        The MLflow run ID of the created run.
    """
    mlflow.set_experiment(ger.EXP_EXPERIMENT_REPORTS)
    with mlflow.start_run(
        run_name=f"week10_significance_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}"
    ) as run:
        mlflow.set_tag("source", "scripts/analyze_experiment_significance.py")
        mlflow.log_text(report, "week10_experiment_report.md")
        return run.info.run_id


def main() -> None:
    """Generate the Week 10 statistical significance report and log/save it."""
    uri = ger.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    print(f"MLflow tracking URI: {uri}\n")

    depth_runs = ger.fetch_runs(ger.EXP_AB_DEPTH_EXPERIMENTS, ger.DEPTH_METRICS, "depth")
    depth_summary = ger.summarize(depth_runs, "depth", ger.DEPTH_METRICS)
    depth_significance = compare_all_pairs(depth_runs, "depth", ger.DEPTH_METRICS, DEPTH_PAIRS)

    refinement_runs = ger.fetch_runs(ger.EXP_PLANNING_REFINEMENT_AB, ger.REFINEMENT_METRICS, "arm")
    refinement_summary = ger.summarize(refinement_runs, "arm", ger.REFINEMENT_METRICS)
    refinement_significance = compare_all_pairs(
        refinement_runs, "arm", ger.REFINEMENT_METRICS, REFINEMENT_PAIRS
    )

    refinement_runs_by_workflow = ger.fetch_runs(
        ger.EXP_PLANNING_REFINEMENT_AB, ger.REFINEMENT_METRICS, "workflow_type"
    )
    workflow_type_significance = compare_all_pairs(
        refinement_runs_by_workflow, "workflow_type", ger.REFINEMENT_METRICS, WORKFLOW_TYPE_PAIRS
    )

    refinement_runs_stratified = fetch_runs_by_two_params(
        ger.EXP_PLANNING_REFINEMENT_AB, ger.REFINEMENT_METRICS, "workflow_type", "arm"
    )
    refinement_significance_stratified: dict[str, list[dict]] = {}
    if not refinement_runs_stratified.empty:
        for workflow_type in refinement_runs_stratified["workflow_type"].unique():
            subset = refinement_runs_stratified[
                refinement_runs_stratified["workflow_type"] == workflow_type
            ]
            refinement_significance_stratified[workflow_type] = compare_all_pairs(
                subset, "arm", ger.REFINEMENT_METRICS, REFINEMENT_PAIRS
            )

    report = build_significance_report(
        depth_summary,
        depth_significance,
        refinement_summary,
        refinement_significance,
        refinement_significance_stratified,
        workflow_type_significance,
    )
    print(report)

    run_id = log_report_to_mlflow(report)
    print(f"Logged to MLflow experiment '{ger.EXP_EXPERIMENT_REPORTS}', run_id={run_id}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Saved to {REPORT_PATH.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
