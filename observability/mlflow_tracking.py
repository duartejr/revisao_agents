"""MLflow experiment initialization and tracking helpers for revisao_agents workflows."""

import logging
from collections.abc import Generator
from contextlib import contextmanager

import mlflow

from .mlflow_config import EXPERIMENTS, MLFLOW_TRACKING_URI

_logger = logging.getLogger(__name__)


def initialize_experiments() -> None:
    """Initialize MLflow tracking URI and create all canonical experiments.

    This function is idempotent — safe to call multiple times. It should be
    called once at application startup (CLI or UI entrypoint) before any
    workflow runs.

    The tracking URI is read from ``MLFLOW_TRACKING_URI`` (env var) via
    :mod:`observability.mlflow_config`. Defaults to a local SQLite backend at
    ``sqlite:///./mlruns/mlflow.db``.

    Experiments created (if not already present):

    - ``planning_academic``
    - ``planning_technical``
    - ``writing_academic``
    - ``writing_technical``
    - ``review_chat``
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    for exp_name in EXPERIMENTS:
        mlflow.set_experiment(exp_name)
    _logger.info("MLflow initialized — experiments: %s", ", ".join(EXPERIMENTS))


@contextmanager
def workflow_run(
    experiment_name: str,
    run_name: str,
    params: dict | None = None,
) -> Generator[mlflow.ActiveRun, None, None]:
    """Context manager that wraps a workflow execution in an MLflow run.

    Sets the experiment, starts a run, logs any provided ``params``, and
    ensures the run is ended even if an exception is raised.

    Args:
        experiment_name: Canonical experiment name (use constants from
            :mod:`observability.mlflow_config`, e.g. ``EXP_PLANNING_ACADEMIC``).
        run_name: Human-readable label for the run (e.g. ``"academic/<theme>"``).
        params: Optional dict of parameters to log with
            :func:`mlflow.log_params`.

    Yields:
        The active :class:`mlflow.ActiveRun` object so callers can log
        additional metrics inside the ``with`` block.

    Example::

        from observability.mlflow_config import EXP_PLANNING_ACADEMIC
        from observability.mlflow_tracking import workflow_run

        with workflow_run(EXP_PLANNING_ACADEMIC, "academic/my-theme", params={"rounds": 3}):
            result = run_graph(...)
            mlflow.log_metric("nodes_executed", result["steps"])
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as active_run:
        if params:
            mlflow.log_params(params)
        yield active_run
