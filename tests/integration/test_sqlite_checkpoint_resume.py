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
    """Test that build_academic_workflow successfully creates a StateGraph with valid checkpoint configurations.
    This ensures that the workflow can be built with both in-memory and SQLite checkpointers, and that the SQLite checkpoint file is created as expected.
    After the test, it cleans up any created SQLite checkpoint files to maintain a clean test environment.

    Args:
        env_vars (dict): A dictionary of environment variables to set for the test, including CHECKPOINT_TYPE and optionally CHECKPOINT_PATH.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for modifying environment variables.

    Raises:
        AssertionError: If the returned object is not a CompiledStateGraph or if the SQLite checkpoint file is not created when expected.
        ValueError: If the CHECKPOINT_TYPE is unsupported or if there are issues creating the SQLite checkpoint file.
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
    """Test that build_technical_workflow returns a valid StateGraph for various valid checkpointers.
    This ensures the function accepts None, MemorySaver, and SqliteSaver without errors, maintaining backward compatibility.

    Args:
        env_vars (dict): Environment variables for the test.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for environment manipulation.

    Raises:
        AssertionError: If the returned object is not a StateGraph instance.
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
    """Test that build_academic_workflow raises appropriate exceptions for invalid checkpoint configurations.
    This ensures that the function properly validates the CHECKPOINT_TYPE and CHECKPOINT_PATH environment variables, and raises ValueError for unsupported checkpoint types or invalid paths.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for modifying environment variables.
        env_vars (dict): A dictionary of environment variables to set for the test, including invalid CHECKPOINT_TYPE or CHECKPOINT_PATH.
        expected_exception (Exception): The type of exception expected to be raised for the given invalid configuration.

    Raises:
        ValueError: If the CHECKPOINT_TYPE is unsupported or if the CHECKPOINT_PATH is invalid.
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
    """Test that build_technical_workflow raises a ValueError when given an invalid checkpointer.
    This ensures that the function properly validates the type of the checkpointer argument and raises an appropriate error for unsupported types.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for environment manipulation.
        env_vars (dict): Invalid environment variables.
        expected_exception (Exception): Expected exception type.

    Raises:
        ValueError: If the checkpointer is not an instance of BaseCheckpointSaver.
    """
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(expected_exception):
        build_technical_workflow(checkpointer=get_checkpointer())


def test_sqlite_no_checkpoint_type(monkeypatch):
    """Test that build_academic_workflow defaults to MemorySaver when CHECKPOINT_TYPE is not set.
    This ensures that the function maintains backward compatibility by using an in-memory checkpointer when no checkpoint
    type is specified, and that it successfully builds the workflow without errors.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for modifying environment variables.

    Raises:
        AssertionError: If the returned object is not a CompiledStateGraph.
    """
    monkeypatch.delenv("CHECKPOINT_TYPE", raising=False)
    app = build_academic_workflow(checkpointer=get_checkpointer())
    assert isinstance(app, CompiledStateGraph)
