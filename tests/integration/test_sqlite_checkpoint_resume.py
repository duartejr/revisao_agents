import os

import pytest
from langgraph.graph.state import CompiledStateGraph

from revisao_agents.graphs.checkpoints import get_checkpointer
from revisao_agents.workflows.academic_workflow import build_academic_workflow
from revisao_agents.workflows.technical_workflow import build_technical_workflow


@pytest.mark.parametrize(
    "env_vars",
    [
        {"CHECKPOINT_TYPE": "memory"},
        {"CHECKPOINT_TYPE": "sqlite", "CHECKPOINT_PATH": "./test_checkpoints/test.db"},
    ],
)
def test_sqlite_valid_checkpoint_resume(env_vars: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test build_academic_workflow with valid checkpoint configurations.

    Args:
        env_vars: Environment variables for the test.
        monkeypatch: Pytest fixture for environment manipulation.
    """
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    app = build_academic_workflow(checkpointer=get_checkpointer())
    assert isinstance(app, CompiledStateGraph)

    if env_vars.get("CHECKPOINT_TYPE") == "sqlite":
        assert os.path.exists(env_vars["CHECKPOINT_PATH"])

    if env_vars.get("CHECKPOINT_TYPE") == "sqlite":
        os.remove(env_vars["CHECKPOINT_PATH"])
        dir_ = os.path.dirname(env_vars["CHECKPOINT_PATH"])
        if dir_ and os.path.exists(dir_) and not os.listdir(dir_):
            os.rmdir(dir_)


@pytest.mark.parametrize(
    "env_vars",
    [
        {"CHECKPOINT_TYPE": "memory"},
        {"CHECKPOINT_TYPE": "sqlite", "CHECKPOINT_PATH": "./test_checkpoints/test.db"},
    ],
)
def test_build_technical_workflow_with_valid_checkpointers(
    env_vars: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test build_technical_workflow with valid checkpointer configurations.

    Args:
        env_vars: Environment variables for the test.
        monkeypatch: Pytest fixture for environment manipulation.
    """
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    app = build_technical_workflow(checkpointer=get_checkpointer())
    assert isinstance(app, CompiledStateGraph)

    if env_vars.get("CHECKPOINT_TYPE") == "sqlite":
        assert os.path.exists(env_vars["CHECKPOINT_PATH"])

    if env_vars.get("CHECKPOINT_TYPE") == "sqlite":
        os.remove(env_vars["CHECKPOINT_PATH"])
        dir_ = os.path.dirname(env_vars["CHECKPOINT_PATH"])
        if dir_ and os.path.exists(dir_) and not os.listdir(dir_):
            os.rmdir(dir_)


@pytest.mark.parametrize(
    "env_vars,expected_exception",
    [
        ({"CHECKPOINT_TYPE": "banana"}, ValueError),
        ({"CHECKPOINT_TYPE": "sqlite", "CHECKPOINT_PATH": "/invalid/path/?.db"}, ValueError),
    ],
)
def test_sqlite_invalid_checkpoint(monkeypatch, env_vars, expected_exception):
    """Test build_academic_workflow raises exceptions for invalid checkpoint configurations.

    Args:
        monkeypatch: Pytest fixture for environment manipulation.
        env_vars: Invalid environment variables for the test.
        expected_exception: Expected exception type.
    """
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(expected_exception):
        build_academic_workflow(checkpointer=get_checkpointer())


@pytest.mark.parametrize(
    "env_vars,expected_exception",
    [
        ({"CHECKPOINT_TYPE": "banana"}, ValueError),
        ({"CHECKPOINT_TYPE": "sqlite", "CHECKPOINT_PATH": "/invalid/path/?.db"}, ValueError),
    ],
)
def test_build_technical_workflow_with_invalid_checkpointer(
    monkeypatch, env_vars, expected_exception
):
    """Test build_technical_workflow raises exceptions for invalid checkpoint configurations.

    Args:
        monkeypatch: Pytest fixture for environment manipulation.
        env_vars: Invalid environment variables for the test.
        expected_exception: Expected exception type.
    """
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(expected_exception):
        build_technical_workflow(checkpointer=get_checkpointer())


def test_sqlite_no_checkpoint_type(monkeypatch):
    """Test build_academic_workflow defaults to memory checkpointer when CHECKPOINT_TYPE is not set.

    Args:
        monkeypatch: Pytest fixture for environment manipulation.
    """
    monkeypatch.delenv("CHECKPOINT_TYPE", raising=False)
    app = build_academic_workflow(checkpointer=get_checkpointer())
    assert isinstance(app, CompiledStateGraph)
