"""
tests/unit/test_agents/test_academic.py

Unit tests for the academic review agents.
"""

from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from revisao_agents.state import ReviewState


def _make_state(**overrides) -> ReviewState:
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


@patch("revisao_agents.nodes.academic.search_chunks", return_value=[])
@patch("revisao_agents.nodes.academic.accumulate_chunks", return_value=[])
def test_consulta_vetorial_node_returns_dict(mock_acc, mock_busca):
    from revisao_agents.nodes.academic import consulta_vetorial_node

    state = _make_state()
    result = consulta_vetorial_node(state)
    assert isinstance(result, dict)


@patch("revisao_agents.nodes.academic.load_prompt")
@patch("revisao_agents.nodes.academic.get_llm")
def test_plano_inicial_returns_dict(mock_llm, mock_load_prompt):
    mock_load_prompt.return_value = SimpleNamespace(text="prompt", temperature=0.1)
    mock_llm.return_value.invoke.return_value = MagicMock(content="Plano de teste")
    from revisao_agents.nodes.academic import plano_inicial_academico_node

    state = _make_state(relevant_chunks=["dummy chunk"])
    result = plano_inicial_academico_node(state)
    assert isinstance(result, dict)
    assert "current_plan" in result
