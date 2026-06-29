"""Unit-test shared fixtures.

Redirects MLflow to a per-session local filesystem store so that unit tests
never need a running MLflow server.  The ``MLFLOW_TRACKING_URI`` env var is
patched at session scope to avoid repeated directory creation.
"""

import pytest


@pytest.fixture(autouse=True, scope="session")
def mlflow_local_store(tmp_path_factory):
    """Point MLflow at a local file store for the entire unit-test session.

    This prevents any unit test from accidentally hitting ``localhost:5000``
    (or whatever ``MLFLOW_TRACKING_URI`` is set to in the environment) when
    the production code calls ``mlflow.start_run``, ``mlflow.set_experiment``,
    or ``mlflow.set_tracking_uri``.

    It also pre-creates and selects a ``unit-tests`` experiment so that real
    (non-mocked) calls to ``mlflow.start_run`` succeed: a freshly created file
    store has no "Default" experiment (ID 0) on disk, so the very first
    ``start_run`` against it would otherwise fail with
    ``Could not find experiment with ID 0``.
    """
    import os

    import mlflow

    mlruns_dir = tmp_path_factory.mktemp("mlruns", numbered=False)
    tracking_uri = f"file://{mlruns_dir}"

    original_uri = os.environ.get("MLFLOW_TRACKING_URI")
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("unit-tests")

    yield

    # Restore original value (or remove if it wasn't set)
    if original_uri is None:
        os.environ.pop("MLFLOW_TRACKING_URI", None)
    else:
        os.environ["MLFLOW_TRACKING_URI"] = original_uri
