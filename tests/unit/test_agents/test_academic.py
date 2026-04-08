"""
tests/unit/test_agents/test_academic.py

Unit tests for the academic review agents.
"""

import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph

from revisao_agents.state import ReviewState
from revisao_agents.workflows.academic_workflow import build_academic_workflow
from revisao_agents.workflows.technical_workflow import build_technical_workflow


def _make_state(**overrides) -> ReviewState:
    """
    Helper function to create a base ReviewState with optional overrides.

    Args:
        **overrides: Key-value pairs to override the default state values.

    Returns:
        ReviewState: A state object with the specified overrides.
    """
    base: ReviewState = {
        "theme": "Test topic",
        "review_type": "academico",
        "relevant_chunks": [],
        "technical_snippets": [],
        "technical_urls": [],
        "current_plan": "",
        "interview_history": [],
        "questions_asked": 0,
        "max_questions": 3,
        "final_plan": "",
        "final_plan_path": "",
        "status": "starting",
    }
    base.update(overrides)
    return base


@pytest.mark.skip(reason="consulta_vetorial_node not found in academic nodes")
@patch("revisao_agents.nodes.academic.search_chunks", return_value=[])
@patch("revisao_agents.nodes.academic.accumulate_chunks", return_value=[])
def test_consulta_vetorial_node_returns_dict(mock_acc: MagicMock, mock_busca: MagicMock) -> None:
    """Test that the consulta_vetorial_node returns a dictionary, even when no relevant chunks are found.
    This test ensures that the node handles cases where the search returns no results gracefully, without throwing errors.

    Args:
        mock_acc (MagicMock): Mock for the accumulate_chunks function, returning an empty list.
        mock_busca (MagicMock): Mock for the search_chunks function, returning an empty list

    Raises:
        AssertionError: If the result is not a dictionary, or if the function does not handle empty search results properly.
    """
    from revisao_agents.nodes.academic import consulta_vetorial_node

    state = _make_state()
    result = consulta_vetorial_node(state)
    assert isinstance(result, dict)


@pytest.mark.skip(reason="plano_inicial_academico_node not found in academic nodes")
@patch("revisao_agents.nodes.academic.load_prompt")
@patch("revisao_agents.nodes.academic.get_llm")
def test_plano_inicial_returns_dict(mock_llm: MagicMock, mock_load_prompt: MagicMock) -> None:
    """Test that the plano_inicial_academico_node returns a dictionary with a 'current_plan' key.
    This test verifies that the plano_inicial_academico_node correctly processes the input state and generates an initial academic plan, even when provided with dummy relevant chunks. The test mocks the prompt loading and LLM invocation to ensure that the node's logic is tested in isolation.

    Args:
        mock_llm (MagicMock): Mock for the get_llm function, allowing us to simulate the LLM's response.
        mock_load_prompt (MagicMock): Mock for the load_prompt function, providing a dummy prompt for the test.
    Raises:
        AssertionError: If the result is not a dictionary, or if the 'current_plan' key is missing from the result.
    """
    mock_load_prompt.return_value = SimpleNamespace(text="prompt", temperature=0.1)
    mock_llm.return_value.invoke.return_value = MagicMock(content="Plano de teste")
    from revisao_agents.nodes.academic import plano_inicial_academico_node

    state = _make_state(relevant_chunks=["dummy chunk"])
    result = plano_inicial_academico_node(state)
    assert isinstance(result, dict)
    assert "current_plan" in result


@pytest.mark.parametrize(
    "checkpointer",
    [
        ("None", None),
        ("memory", MemorySaver()),
        ("sqlite", SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))),
    ],
)
def test_build_workflow_with_valid_checkpointers(
    checkpointer: tuple[str, MemorySaver | SqliteSaver | None],
) -> None:
    """Test that build_academic_workflow returns a valid StateGraph for various valid checkpointers.
    This ensures the function accepts None, MemorySaver, and SqliteSaver without errors, maintaining backward compatibility.

    Args:
        checkpointer (tuple[str, MemorySaver | SqliteSaver | None]): Name and instance of the checkpointer.

    Raises:
        AssertionError: If the returned object is not a StateGraph instance.
    """
    name, saver = checkpointer

    app = build_academic_workflow(checkpointer=saver)
    assert isinstance(app, CompiledStateGraph)


@pytest.mark.parametrize(
    "invalid_checkpointer",
    [
        ("string", "not a saver"),
        ("integer", 123),
        ("list", []),
        ("dict", {}),
    ],
)
def test_build_workflow_with_invalid_checkpointer(invalid_checkpointer: tuple[str, object]) -> None:
    """Test that build_academic_workflow raises a ValueError when given an invalid checkpointer.
    This ensures that the function properly validates the type of the checkpointer argument and raises an appropriate error for unsupported types.

    Args:
        invalid_checkpointer (tuple[str, object]): Name and instance of the invalid checkpointer.

    Raises:
        ValueError: If the checkpointer is not an instance of BaseCheckpointSaver.
    """
    name, saver = invalid_checkpointer
    with pytest.raises(ValueError):
        build_academic_workflow(checkpointer=saver)


@pytest.mark.parametrize(
    "checkpointer",
    [
        ("None", None),
        ("memory", MemorySaver()),
        ("sqlite", SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))),
    ],
)
def test_build_technical_workflow_with_valid_checkpointers(
    checkpointer: tuple[str, MemorySaver | SqliteSaver | None],
) -> None:
    """Test that build_technical_workflow returns a valid StateGraph for various valid checkpointers.
    This ensures the function accepts None, MemorySaver, and SqliteSaver without errors, maintaining backward compatibility.

    Args:
        checkpointer (tuple[str, MemorySaver | SqliteSaver | None]): Name and instance of the checkpointer.

    Raises:
        AssertionError: If the returned object is not a StateGraph instance.
    """
    name, saver = checkpointer

    app = build_technical_workflow(checkpointer=saver)
    assert isinstance(app, CompiledStateGraph)


@pytest.mark.parametrize(
    "invalid_checkpointer",
    [
        ("string", "not a saver"),
        ("integer", 123),
        ("list", []),
        ("dict", {}),
    ],
)
def test_build_technical_workflow_with_invalid_checkpointer(
    invalid_checkpointer: tuple[str, object],
) -> None:
    """Test that build_technical_workflow raises a ValueError when given an invalid checkpointer.
    This ensures that the function properly validates the type of the checkpointer argument and raises an appropriate error for unsupported types.

    Args:
        invalid_checkpointer (tuple[str, object]): Name and instance of the invalid checkpointer.

    Raises:
        ValueError: If the checkpointer is not an instance of BaseCheckpointSaver.
    """
    name, saver = invalid_checkpointer
    with pytest.raises(ValueError):
        build_technical_workflow(checkpointer=saver)
