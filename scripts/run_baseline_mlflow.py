"""
Baseline MLflow Runs

This script logs a baseline run to every canonical MLflow experiment so that
experiments are populated with reference data before Week 9's comparative runs.

Baseline runs record the default configuration parameters only — **no metrics
are logged** to avoid skewing dashboard averages computed over real runs.
They are NOT full workflow executions; they establish a reference point for
comparing future runs.

Usage
-----
    uv run python scripts/run_baseline_mlflow.py

Requirements
------------
- ``MLFLOW_TRACKING_URI`` (optional): defaults to ``sqlite:///./mlruns/mlflow.db``
- ``TAVILY_SEARCH_DEPTH`` / ``TAVILY_NUM_RESULTS`` (optional): read from env or
  their own defaults so the script works without a full ``.env`` file.

Output
------
One MLflow run per experiment, tagged ``baseline=true``.  Open the MLflow UI
(``make mlflow-start``) to inspect the runs.
"""

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Ensure repo root is on sys.path when running directly with `uv run python`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mlflow  # noqa: E402
from observability.mlflow_config import (  # noqa: E402
    EXPERIMENTS,
    get_tracking_uri,
)

# ---------------------------------------------------------------------------
# Baseline parameter defaults (mirrors TavilySearchConfig env-var defaults)
# ---------------------------------------------------------------------------

_BASELINE_PARAMS: dict[str, object] = {
    "tavily_search_depth": os.getenv("TAVILY_SEARCH_DEPTH", "basic"),
    "tavily_num_results": int(os.getenv("TAVILY_NUM_RESULTS", "5")),
    "checkpoint_type": os.getenv("CHECKPOINT_TYPE", "memory"),
}


def log_baseline_run(experiment_name: str, description: str) -> str:
    """Log a single baseline run to ``experiment_name``.

    The run records only the default configuration parameters and is tagged
    ``baseline=true`` so it can be excluded from metric aggregations via::

        mlflow.search_runs(filter_string="tags.baseline = 'true'")

    No metrics are logged — zero-valued placeholders would skew dashboard
    averages computed over real runs.

    Args:
        experiment_name: Canonical experiment name (key from ``EXPERIMENTS``).
        description: Human-readable description stored as an MLflow tag.

    Returns:
        The MLflow run ID of the created run.
    """
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="baseline") as run:
        mlflow.set_tag("baseline", "true")
        mlflow.set_tag("baseline_date", datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
        mlflow.set_tag("description", description)
        mlflow.set_tag("source", "scripts/run_baseline_mlflow.py")
        mlflow.log_params(_BASELINE_PARAMS)
        return run.info.run_id


def main() -> None:
    """Log baseline runs to all canonical MLflow experiments."""
    uri = get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    print(f"MLflow tracking URI: {uri}\n")

    results: list[tuple[str, str]] = []
    for exp_name, description in EXPERIMENTS.items():
        run_id = log_baseline_run(exp_name, description)
        results.append((exp_name, run_id))
        print(f"  ✅  {exp_name:<30}  run_id={run_id[:8]}…")

    print(f"\n{len(results)} baseline runs logged.")
    print("Open the MLflow UI with:  make mlflow-start")


if __name__ == "__main__":
    main()
