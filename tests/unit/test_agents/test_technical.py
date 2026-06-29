"""
tests/unit/test_agents/test_technical.py

Unit tests for the technical review agents in ``revisao_agents.nodes.technical``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from revisao_agents.state import ReviewState


def _make_state(**overrides) -> ReviewState:
    """Build a minimal ``ReviewState`` for ``finalize_technical_plan_node`` tests.

    Args:
        **overrides: Key-value pairs to override the default state values.

    Returns:
        ReviewState: A state object with the specified overrides applied.
    """
    base: ReviewState = {
        "theme": "Test topic",
        "review_type": "tecnico",
        "technical_snippets": [],
        "technical_urls": [],
        "current_plan": "",
        "interview_history": [],
        "final_plan": "",
        "final_plan_path": "",
        "total_credits_used": 0.0,
        "status": "starting",
    }
    base.update(overrides)
    return base


@patch("revisao_agents.nodes.technical.mlflow")
@patch("revisao_agents.nodes.technical.save_md", return_value="runtime/plans/fake.md")
@patch("revisao_agents.nodes.technical.get_llm")
@patch("revisao_agents.nodes.technical.load_prompt")
def test_finalize_technical_plan_node_logs_total_credits_used(
    mock_load_prompt: MagicMock,
    mock_get_llm: MagicMock,
    mock_save_md: MagicMock,
    mock_mlflow: MagicMock,
) -> None:
    """Test that the final node logs ``total_credits_used`` to the active MLflow run.

    Args:
        mock_load_prompt: Mock for ``load_prompt``, returning a dummy prompt.
        mock_get_llm: Mock for ``get_llm``, returning a dummy LLM response.
        mock_save_md: Mock for ``save_md``, avoiding real filesystem writes.
        mock_mlflow: Mock for the ``mlflow`` module used inside ``technical.py``,
            with ``active_run()`` simulated as truthy (a run is active).

    Raises:
        AssertionError: If ``mlflow.log_metric`` is not called with the expected
            metric name and value, or if the returned state is malformed.
    """
    mock_load_prompt.return_value = SimpleNamespace(text="prompt", temperature=0.1)
    mock_get_llm.return_value.invoke.return_value = MagicMock(content="Final plan content")
    mock_mlflow.active_run.return_value = MagicMock()  # simulate an active run

    from revisao_agents.nodes.technical import finalize_technical_plan_node

    state = _make_state(total_credits_used=12.5)
    result = finalize_technical_plan_node(state)

    mock_mlflow.log_metric.assert_called_once_with("total_credits_used", 12.5)
    assert isinstance(result, dict)
    assert result["status"] == "completed"
    assert result["final_plan"] == "Final plan content"


@patch("revisao_agents.nodes.technical.mlflow")
@patch("revisao_agents.nodes.technical.save_md", return_value="runtime/plans/fake.md")
@patch("revisao_agents.nodes.technical.get_llm")
@patch("revisao_agents.nodes.technical.load_prompt")
def test_finalize_technical_plan_node_skips_metric_without_active_run(
    mock_load_prompt: MagicMock,
    mock_get_llm: MagicMock,
    mock_save_md: MagicMock,
    mock_mlflow: MagicMock,
) -> None:
    """Test that no metric is logged when there is no active MLflow run.

    Args:
        mock_load_prompt: Mock for ``load_prompt``, returning a dummy prompt.
        mock_get_llm: Mock for ``get_llm``, returning a dummy LLM response.
        mock_save_md: Mock for ``save_md``, avoiding real filesystem writes.
        mock_mlflow: Mock for the ``mlflow`` module, with ``active_run()``
            simulated as ``None`` (no run active).

    Raises:
        AssertionError: If ``mlflow.log_metric`` is called despite no active run.
    """
    mock_load_prompt.return_value = SimpleNamespace(text="prompt", temperature=0.1)
    mock_get_llm.return_value.invoke.return_value = MagicMock(content="Final plan content")
    mock_mlflow.active_run.return_value = None

    from revisao_agents.nodes.technical import finalize_technical_plan_node

    state = _make_state(total_credits_used=7.0)
    finalize_technical_plan_node(state)

    mock_mlflow.log_metric.assert_not_called()
