"""
generate_experiment_report.py - W9-STORY-04 Experiment Decision Report

Queries MLflow for runs logged by the two Week 9 A/B experiments:

    - ``ab_depth_experiments``     (W9-STORY-02, scripts/run_ab_depth_experiment.py)
    - ``planning_refinement_ab``   (W9-STORY-03, scripts/run_refinement_ab_experiment.py)

aggregates per-variant statistics (run count, mean, min, max for each
metric), and renders a Markdown decision report with a recommendation per
experiment — or an explicit "insufficient data" statement when the sample
is too small to support a confident call. Per the roadmap's own risk
register (week9_tasks.md, W9-STORY-04 risks), "insufficient data" is an
accepted, non-escalating conclusion at this stage, not a failure of this
script.

Note on the "fast" depth value (resolves the W9-STORY-02 naming note):
    ``week9_tasks.md`` (sub-task 2.4) flagged that ``run_ab_depth_experiment.py``
    uses depth values ``fast`` / ``basic`` / ``advanced`` and asked whether
    ``fast`` was a "non-standard value" before trusting the results. Per the
    Tavily API reference (https://docs.tavily.com/documentation/api-reference/endpoint/search,
    consulted for this report per CLAUDE.md's integration-docs rule), the
    documented values for ``search_depth`` are ``basic``, ``advanced``,
    ``fast``, and ``ultra-fast`` — ``fast`` is real and documented, not a
    typo or placeholder. This matches
    ``revisao_agents.config._TAVILY_VALID_DEPTHS``, which already validates
    against all four. The only actual naming mismatch was the roadmap
    calling the middle tier "balanced" where the code and the Tavily API
    call it "basic" — a documentation wording gap, not a data-quality issue.
    The depth results below are treated as decision-grade with respect to
    which depth values were exercised; sample size is still small (see
    Limitations).

Output:
    - Printed to stdout
    - Logged as an MLflow artifact (``week9_experiment_report.md``) on a
      dedicated run in the ``experiment_reports`` experiment
    - Saved to ``management/reports/week9/week9_experiment_report.md``

Usage
-----
    uv run python scripts/generate_experiment_report.py

Requirements
------------
- MLflow tracking URI reachable (``make mlflow-start`` if using the default
  local server) with data already logged by the two experiment scripts above.
"""

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

# Ensure repo root is on sys.path when running directly with `uv run python`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from revisao_agents.observability.mlflow_config import (  # noqa: E402
    EXP_AB_DEPTH_EXPERIMENTS,
    EXP_EXPERIMENT_REPORTS,
    EXP_PLANNING_REFINEMENT_AB,
    get_tracking_uri,
)

# `mlflow_config.get_tracking_uri()` only reads `os.environ` — load `.env`
# explicitly so `MLFLOW_TRACKING_URI` is honored regardless of import order
# (same reasoning as scripts/generate_cost_report.py).
load_dotenv()

REPORT_PATH = _ROOT / "management" / "reports" / "week9" / "week9_experiment_report.md"

#: Minimum runs for a given variant before its numbers feed a recommendation
#: rather than an "insufficient data" note. Deliberately low (this is a
#: directional-signal report, not a statistical-significance test — that is
#: explicitly deferred to Week 10 per the roadmap).
#:
#: At the current value of 1, the "under-sampled variant(s)" branches below
#: are unreachable dead code by construction: a variant only appears in
#: `summary` at all if `groupby` found >= 1 row for it, so `runs < 1` can
#: never be true. This is intentional future-proofing, not a live check —
#: raise this constant once a real per-variant minimum is wanted (e.g. in
#: Week 10) and the branches activate with no other code change needed.
MIN_RUNS_PER_VARIANT = 1

DEPTH_METRICS = [
    "average_latency",
    "total_credits",
    "average_urls_found",
    "average_relevance_score",
    "average_academic_quality_pct",
    "average_citation_potential_pct",
]
REFINEMENT_METRICS = [
    "session_duration",
    "refinement_rounds",
    "plan_section_count",
    "plan_quality_score",
]


def fetch_runs(experiment_name: str, metric_names: list[str], group_param: str) -> pd.DataFrame:
    """Fetch finished runs for one experiment, normalized to plain metric/group columns.

    Args:
        experiment_name: MLflow experiment to query.
        metric_names: Metric column suffixes expected on each run (without
            the ``metrics.`` prefix ``mlflow.search_runs`` adds).
        group_param: Param column (without the ``params.`` prefix) used to
            group runs into variants (e.g. ``"depth"`` or ``"arm"``).

    Returns:
        A DataFrame with one row per finished run and columns
        ``[group_param, *metric_names]``, or empty if the experiment has no
        finished runs or is missing any expected column (e.g. no runs
        logged yet).
    """
    runs = cast("pd.DataFrame", mlflow.search_runs(experiment_names=[experiment_name]))
    if runs.empty:
        return pd.DataFrame()

    runs = runs[runs["status"] == "FINISHED"]

    param_col = f"params.{group_param}"
    metric_cols = [f"metrics.{m}" for m in metric_names]
    required = [param_col, *metric_cols]
    if any(col not in runs for col in required):
        return pd.DataFrame()

    runs = runs.dropna(subset=required)
    renamed = runs.rename(
        columns={param_col: group_param, **dict(zip(metric_cols, metric_names, strict=True))}
    )
    return renamed[[group_param, *metric_names]]


def summarize(runs: pd.DataFrame, group_col: str, metric_names: list[str]) -> pd.DataFrame:
    """Aggregate run-level metrics into per-variant count/mean/min/max.

    Args:
        runs: Output of :func:`fetch_runs`.
        group_col: Column to group by (e.g. ``"depth"`` or ``"arm"``).
        metric_names: Metric columns to aggregate.

    Returns:
        One row per variant, with a ``runs`` count column plus
        ``<metric>_mean`` / ``<metric>_min`` / ``<metric>_max`` per metric.
        Empty DataFrame if ``runs`` is empty.
    """
    if runs.empty:
        return pd.DataFrame()

    grouped = runs.groupby(group_col)[metric_names].agg(["mean", "min", "max"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped.insert(0, "runs", runs.groupby(group_col).size())
    return grouped.round(3).reset_index()


def recommend_depth(summary: pd.DataFrame, expected_depths: tuple[str, ...]) -> str:
    """Build a directional recommendation (or insufficient-data note) for the depth A/B.

    Args:
        summary: Output of ``summarize(depth_runs, "depth", DEPTH_METRICS)``.
        expected_depths: Depth values the experiment was designed to cover.

    Returns:
        A short recommendation string.
    """
    if summary.empty:
        return (
            "**Insufficient data.** No finished runs found in "
            f"`{EXP_AB_DEPTH_EXPERIMENTS}`. Run `scripts/run_ab_depth_experiment.py` "
            "before generating this report."
        )

    missing = [d for d in expected_depths if d not in set(summary["depth"])]
    under_sampled = summary[summary["runs"] < MIN_RUNS_PER_VARIANT]["depth"].tolist()
    if missing or under_sampled:
        note = []
        if missing:
            note.append(f"missing depth variant(s): {', '.join(missing)}")
        if under_sampled:
            note.append(f"under-sampled variant(s): {', '.join(under_sampled)}")
        return f"**Insufficient data** — {'; '.join(note)}. Re-run the experiment before deciding a default."

    best = summary.loc[summary["average_relevance_score_mean"].idxmax()]
    cheapest = summary.loc[summary["total_credits_mean"].idxmin()]
    lines = [
        f"Directional signal (n={int(summary['runs'].sum())} run(s) across "
        f"{len(summary)} depth variants, {DEPTH_METRICS[0].split('_')[0]}-query "
        "aggregates per run — not a statistically significant sample; treat as "
        "directional only, per the roadmap's Week 10 follow-up plan):",
        f"- Highest average relevance score: **{best['depth']}** "
        f"({best['average_relevance_score_mean']:.3f})",
        f"- Lowest average credit cost: **{cheapest['depth']}** "
        f"({cheapest['total_credits_mean']:.2f} credits)",
    ]
    if best["depth"] == cheapest["depth"]:
        lines.append(
            f"- **Recommendation:** `{best['depth']}` is both the highest-relevance and "
            "cheapest variant observed — a reasonable default pending Week 10 confirmation."
        )
    else:
        lines.append(
            f"- **Recommendation:** no single variant dominates on both relevance and cost. "
            f"`{best['depth']}` for relevance-sensitive use, `{cheapest['depth']}` for "
            "cost-sensitive use. Defer a single default to Week 10, once a larger sample "
            "can support a statistical comparison."
        )
    return "\n".join(lines)


def recommend_refinement(summary: pd.DataFrame, expected_arms: tuple[str, ...]) -> str:
    """Build a directional recommendation (or insufficient-data note) for the refinement A/B.

    Args:
        summary: Output of ``summarize(refinement_runs, "arm", REFINEMENT_METRICS)``.
        expected_arms: Arm values the experiment was designed to cover.

    Returns:
        A short recommendation string.
    """
    if summary.empty:
        return (
            "**Insufficient data.** No finished runs found in "
            f"`{EXP_PLANNING_REFINEMENT_AB}`. Run `scripts/run_refinement_ab_experiment.py` "
            "before generating this report."
        )

    missing = [a for a in expected_arms if a not in set(summary["arm"])]
    under_sampled = summary[summary["runs"] < MIN_RUNS_PER_VARIANT]["arm"].tolist()
    if missing or under_sampled:
        note = []
        if missing:
            note.append(f"missing arm(s): {', '.join(missing)}")
        if under_sampled:
            note.append(f"under-sampled arm(s): {', '.join(under_sampled)}")
        return f"**Insufficient data** — {'; '.join(note)}. Re-run the experiment before deciding."

    refined = summary[summary["arm"] == "refined"].iloc[0]
    bypassed = summary[summary["arm"] == "bypassed"].iloc[0]
    quality_delta = refined["plan_quality_score_mean"] - bypassed["plan_quality_score_mean"]
    duration_delta = refined["session_duration_mean"] - bypassed["session_duration_mean"]

    lines = [
        f"Directional signal (n={int(summary['runs'].sum())} run(s) across "
        f"{len(summary)} arms — small sample; see script docstring Limitations. "
        "'bypassed' is a synthetic short-circuit, not a real user opting out):",
        f"- Plan quality (LLM judge, 1-10): refined={refined['plan_quality_score_mean']:.2f}, "
        f"bypassed={bypassed['plan_quality_score_mean']:.2f} (Δ={quality_delta:+.2f})",
        f"- Session duration (s): refined={refined['session_duration_mean']:.1f}, "
        f"bypassed={bypassed['session_duration_mean']:.1f} (Δ={duration_delta:+.1f})",
    ]
    if quality_delta > 0.5:
        lines.append(
            "- **Recommendation:** refinement layer measurably improves plan quality in this "
            "sample — keep it mandatory, do not make it optional yet. Re-confirm with a larger "
            "sample in Week 10 before treating this as final."
        )
    elif quality_delta < -0.5:
        lines.append(
            "- **Recommendation:** unexpected — bypassed plans scored higher in this sample. "
            "Do not act on this without Week 10 confirmation; investigate before changing "
            "the layer's default-on behavior."
        )
    else:
        lines.append(
            "- **Recommendation:** no clear quality difference in this sample. Insufficient "
            "signal to justify making the layer optional for power users yet — revisit with "
            "a larger sample in Week 10."
        )
    return "\n".join(lines)


def build_report(
    depth_summary: pd.DataFrame,
    depth_recommendation: str,
    refinement_summary: pd.DataFrame,
    refinement_recommendation: str,
) -> str:
    """Render both experiments' summaries and recommendations as one Markdown report.

    Args:
        depth_summary: Output of ``summarize(depth_runs, "depth", DEPTH_METRICS)``.
        depth_recommendation: Output of :func:`recommend_depth`.
        refinement_summary: Output of ``summarize(refinement_runs, "arm", REFINEMENT_METRICS)``.
        refinement_recommendation: Output of :func:`recommend_refinement`.

    Returns:
        Markdown-formatted report text.
    """
    lines = [
        "# Week 9 Experiment Decision Report",
        "",
        f"Generated: {datetime.now(tz=UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## 1. Tavily Depth A/B (`ab_depth_experiments`, W9-STORY-02)",
        "",
        "```" if not depth_summary.empty else "_No data._",
        depth_summary.to_string(index=False) if not depth_summary.empty else "",
        "```" if not depth_summary.empty else "",
        "",
        depth_recommendation,
        "",
        "## 2. Planning Refinement A/B (`planning_refinement_ab`, W9-STORY-03)",
        "",
        "```" if not refinement_summary.empty else "_No data._",
        refinement_summary.to_string(index=False) if not refinement_summary.empty else "",
        "```" if not refinement_summary.empty else "",
        "",
        refinement_recommendation,
        "",
        "## Limitations",
        "",
        "- Both experiments are directional-signal reports over a small sample "
        "(see each script's docstring for the exact design). Statistical "
        "significance testing is deferred to Week 10 per the roadmap.",
        "- The refinement A/B ran against the technical (Tavily-backed) planning "
        "workflow rather than academic (MongoDB-backed) — see "
        "`scripts/run_refinement_ab_experiment.py`'s Design note for why.",
        "- For full run-level detail (individual queries/themes), use the MLflow "
        "UI's native run comparison view rather than this aggregated report — "
        "see `docs/mlflow_guide.md`.",
        "",
    ]
    return "\n".join(lines)


def log_report_to_mlflow(report: str) -> str:
    """Log the report as an MLflow artifact on a dedicated run.

    Args:
        report: Output of :func:`build_report`.

    Returns:
        The MLflow run ID of the created run.
    """
    mlflow.set_experiment(EXP_EXPERIMENT_REPORTS)
    with mlflow.start_run(
        run_name=f"week9_report_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}"
    ) as run:
        mlflow.set_tag("source", "scripts/generate_experiment_report.py")
        mlflow.log_text(report, "week9_experiment_report.md")
        return run.info.run_id


def main() -> None:
    """Generate the Week 9 experiment decision report and log/save it."""
    uri = get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    print(f"MLflow tracking URI: {uri}\n")

    depth_runs = fetch_runs(EXP_AB_DEPTH_EXPERIMENTS, DEPTH_METRICS, "depth")
    depth_summary = summarize(depth_runs, "depth", DEPTH_METRICS)
    depth_recommendation = recommend_depth(depth_summary, ("fast", "basic", "advanced"))

    refinement_runs = fetch_runs(EXP_PLANNING_REFINEMENT_AB, REFINEMENT_METRICS, "arm")
    refinement_summary = summarize(refinement_runs, "arm", REFINEMENT_METRICS)
    refinement_recommendation = recommend_refinement(refinement_summary, ("refined", "bypassed"))

    report = build_report(
        depth_summary, depth_recommendation, refinement_summary, refinement_recommendation
    )
    print(report)

    run_id = log_report_to_mlflow(report)
    print(f"Logged to MLflow experiment '{EXP_EXPERIMENT_REPORTS}', run_id={run_id}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Saved to {REPORT_PATH.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
