"""
Tavily Cost Analysis Report (MLflow)

Aggregates Tavily credit spend per planning session, grouped by workflow type
(``review_type``) and search depth (``search_depth``), using the MLflow
metrics/params logged on each *parent* workflow run (see
``observability.mlflow_tracking.workflow_run``).

Only parent runs are read — per-search child runs (logged via
``_tavily_search_span`` in ``tools/tavily_web_search.py``) are excluded via
the ``tags.mlflow.parentRunId`` tag, since ``search_depth`` is a session-wide
config value already mirrored onto the parent run's params.

Excluded from the aggregation, and why:

- Runs with ``status != "FINISHED"``: a session that didn't complete has no
  reliable, final ``total_credits_used`` value.
- Runs missing ``metrics.total_credits_used`` or ``params.search_depth``:
  predate the MLflow cost instrumentation added in W8-STORY-04 (W8-STORY-04's
  4.1 sub-task) — not a quality issue, just data from before the metric
  existed.
- Runs tagged ``baseline=true`` (see ``scripts/run_baseline_mlflow.py``):
  reference runs with no real Tavily usage, would skew the average to zero.

By design, ``EXP_PLANNING_ACADEMIC`` rows will always be excluded by the
``metrics.total_credits_used`` filter above: the academic planning workflow
(``nodes/academic.py``) retrieves sources from the MongoDB-backed corpus via
vector search, not Tavily, so it never logs that metric. This is expected,
not a data-quality bug — the cost report only ever reports real Tavily
spend, which only the technical workflow (``nodes/technical.py``) incurs.

No local files are written. The report and the aggregated table are logged
as MLflow artifacts on a dedicated run in the ``cost_reports`` experiment, so
they show up in the MLflow UI's **Artifacts** tab — not as loose files on
disk. Visual depth-vs-cost comparison (e.g. parallel coordinates across
sessions) is left to the MLflow UI's native compare view; see
``docs/mlflow_guide.md``.

Usage
-----
    uv run python scripts/generate_cost_report.py

Requirements
------------
- ``MLFLOW_TRACKING_URI`` (optional): defaults to ``sqlite:///./runtime/mlruns/mlflow.db``
  via ``observability.mlflow_config.get_tracking_uri``.

Output
------
A new run in the ``cost_reports`` MLflow experiment, with:

- params/metrics per (``review_type``, ``search_depth``) group, for
  filtering/sorting in the MLflow UI's run table
- ``cost_report.md`` and ``cost_summary.csv`` artifacts with the full report
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
    EXP_COST_REPORTS,
    EXP_PLANNING_ACADEMIC,
    EXP_PLANNING_TECHNICAL,
    get_tracking_uri,
)

# `mlflow_config.get_tracking_uri()` only reads `os.environ` — it never loads
# `.env` itself (that side effect normally comes from importing
# `revisao_agents.config`, which this script does not need otherwise). Load
# it explicitly so `MLFLOW_TRACKING_URI` is honored regardless of import order.
load_dotenv()

_PLANNING_EXPERIMENTS = [EXP_PLANNING_ACADEMIC, EXP_PLANNING_TECHNICAL]


def fetch_session_cost_data() -> pd.DataFrame:
    """Fetch and filter parent-run cost data from the planning experiments.

    Note:
        Academic planning sessions (``EXP_PLANNING_ACADEMIC``) are expected
        to be entirely absent from the result, by design: that workflow uses
        MongoDB vector search instead of Tavily and therefore never logs
        ``metrics.total_credits_used``, so the ``dropna`` filter below always
        excludes its rows. Only technical planning sessions
        (``EXP_PLANNING_TECHNICAL``) report real Tavily credit spend.

    Returns:
        A DataFrame with one row per completed, instrumented planning
        session, with columns ``review_type``, ``search_depth``, and
        ``total_credits_used``.
    """
    # `search_runs`'s return type is `list[Run] | DataFrame` in the stub, with
    # no literal-dispatched overload on `output_format` for mypy to narrow —
    # the default (and what we pass) always returns a DataFrame at runtime.
    runs = cast("pd.DataFrame", mlflow.search_runs(experiment_names=_PLANNING_EXPERIMENTS))
    if runs.empty:
        return pd.DataFrame(columns=["review_type", "search_depth", "total_credits_used"])

    is_parent = (
        runs["tags.mlflow.parentRunId"].isna() if "tags.mlflow.parentRunId" in runs else True
    )
    is_finished = runs["status"] == "FINISHED"
    is_not_baseline = (
        runs["tags.baseline"].fillna("false") != "true" if "tags.baseline" in runs else True
    )

    sessions = runs[is_parent & is_finished & is_not_baseline].copy()

    # `search_runs` only adds a `metrics.<key>`/`params.<key>` column at all
    # if at least one fetched run logged that key. With no instrumented runs
    # yet (e.g. a fresh tracking store, or only baseline/legacy runs), these
    # columns can be missing entirely rather than just NaN-valued — guard
    # each one explicitly instead of assuming they exist.
    required = ["params.review_type", "params.search_depth", "metrics.total_credits_used"]
    if any(col not in sessions for col in required):
        return pd.DataFrame(columns=["review_type", "search_depth", "total_credits_used"])

    sessions = sessions.dropna(subset=["metrics.total_credits_used", "params.search_depth"])

    return sessions.rename(
        columns={
            "params.review_type": "review_type",
            "params.search_depth": "search_depth",
            "metrics.total_credits_used": "total_credits_used",
        }
    )[["review_type", "search_depth", "total_credits_used"]]


def summarize(sessions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate session-level cost by (review_type, search_depth).

    Args:
        sessions: Output of :func:`fetch_session_cost_data`.

    Returns:
        One row per (review_type, search_depth) group, with ``sessions``,
        ``total_credits``, and ``avg_credits_per_session`` columns.
    """
    return (
        sessions.groupby(["review_type", "search_depth"])["total_credits_used"]
        .agg(sessions="count", total_credits="sum", avg_credits_per_session="mean")
        .round(2)
        .reset_index()
    )


def build_report(sessions: pd.DataFrame, summary: pd.DataFrame) -> str:
    """Render the cost-by-depth aggregation as a Markdown report.

    Args:
        sessions: Output of :func:`fetch_session_cost_data`.
        summary: Output of :func:`summarize`.

    Returns:
        Markdown-formatted report text.
    """
    if sessions.empty:
        return "# Tavily Cost Report\n\nNo instrumented, completed planning sessions found.\n"

    return "\n".join(
        [
            "# Tavily Cost Report",
            "",
            f"Generated: {datetime.now(tz=UTC).strftime('%Y-%m-%d %H:%M UTC')}",
            f"Sessions analyzed: {len(sessions)}",
            "",
            "```",
            summary.to_string(index=False),
            "```",
            "",
            "For depth-vs-cost comparison across individual sessions (e.g. "
            "parallel coordinates), use the MLflow UI compare view on the "
            "planning experiments — see `docs/mlflow_guide.md`.",
            "",
        ]
    )


def log_report_to_mlflow(sessions: pd.DataFrame, summary: pd.DataFrame, report: str) -> str:
    """Log the cost report as MLflow artifacts/metrics on a dedicated run.

    No file is written to disk by this function — ``mlflow.log_text``/
    ``log_table`` stream the content straight into the MLflow artifact
    store, where it shows up under the run's **Artifacts** tab.

    Args:
        sessions: Output of :func:`fetch_session_cost_data`.
        summary: Output of :func:`summarize`.
        report: Output of :func:`build_report`.

    Returns:
        The MLflow run ID of the created run.
    """
    mlflow.set_experiment(EXP_COST_REPORTS)
    with mlflow.start_run(
        run_name=f"cost_report_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}"
    ) as run:
        mlflow.set_tag("source", "scripts/generate_cost_report.py")
        mlflow.log_param("sessions_analyzed", len(sessions))
        mlflow.log_metric("total_credits_used", float(sessions["total_credits_used"].sum()))

        for row in summary.itertuples():
            group = f"{row.review_type}_{row.search_depth}"
            mlflow.log_metric(f"sessions__{group}", row.sessions)
            mlflow.log_metric(f"total_credits__{group}", row.total_credits)
            mlflow.log_metric(f"avg_credits_per_session__{group}", row.avg_credits_per_session)

        mlflow.log_text(report, "cost_report.md")
        mlflow.log_text(summary.to_csv(index=False), "cost_summary.csv")
        return run.info.run_id


def main() -> None:
    """Generate the Tavily cost analysis report and log it to MLflow."""
    uri = get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    print(f"MLflow tracking URI: {uri}\n")

    sessions = fetch_session_cost_data()
    summary = summarize(sessions) if not sessions.empty else pd.DataFrame()
    report = build_report(sessions, summary)
    print(report)

    run_id = log_report_to_mlflow(sessions, summary, report)
    print(f"Logged to MLflow experiment '{EXP_COST_REPORTS}', run_id={run_id}")
    print("Open the MLflow UI with: make mlflow-start")


if __name__ == "__main__":
    main()
