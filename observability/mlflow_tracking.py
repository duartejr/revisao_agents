"""MLflow experiment initialization and tracking helpers for revisao_agents workflows."""

import logging
from collections.abc import Generator
from contextlib import contextmanager

import mlflow

from .mlflow_config import EXPERIMENTS, get_tracking_uri

_logger = logging.getLogger(__name__)


def enable_tracing() -> None:
    """Enable MLflow LangChain auto-tracing for LLM call visibility.

    Activates ``mlflow.langchain.autolog()`` to automatically capture spans for
    all LangChain LLM invocations: model name, prompt tokens, completion tokens,
    and latency. No changes to individual call sites are required.

    This function is idempotent — safe to call multiple times. It is called
    automatically by :func:`initialize_experiments` at startup.

    Note:
        MLflow 3.x removed the ``log_models`` and ``log_input_examples`` parameters.
        Only ``log_traces`` is passed to ``autolog()`` for compatibility.
    """
    mlflow.set_tracking_uri(get_tracking_uri())
    try:
        mlflow.langchain.autolog(
            log_traces=True,  # MLflow 3.x: log_models / log_input_examples removed
        )
        _logger.info("MLflow LangChain auto-tracing enabled.")
    except Exception as exc:  # pragma: no cover
        _logger.warning("MLflow LangChain auto-tracing could not be enabled: %s", exc)


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
    mlflow.set_tracking_uri(get_tracking_uri())
    for exp_name in EXPERIMENTS:
        mlflow.set_experiment(exp_name)
    _logger.info("MLflow initialized — experiments: %s", ", ".join(EXPERIMENTS))
    try:
        enable_tracing()
    except Exception as exc:  # pragma: no cover
        _logger.warning("MLflow tracing could not be enabled during initialization: %s", exc)


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
    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as active_run:
        if params:
            mlflow.log_params(params)
        yield active_run
