"""
tests/unit/test_agents/test_academic.py

Unit tests for the academic review agents.
"""

from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from revisao_agents.state import RevisaoState


def _make_state(**overrides) -> RevisaoState:
    base: RevisaoState = {
        "tema": "Test topic",
        "tipo_revisao": "academico",
        "chunks_relevantes": [],
        "snippets_tecnicos": [],
        "urls_tecnicos": [],
        "plano_atual": "",
        "historico_entrevista": [],
        "perguntas_feitas": 0,
        "max_perguntas": 3,
        "plano_final": "",
        "plano_final_path": "",
        "status": "iniciando",
    }
    base.update(overrides)
    return base


@patch("revisao_agents.nodes.academic.buscar_chunks", return_value=[])
@patch("revisao_agents.nodes.academic.acumular_chunks", return_value=[])
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

    state = _make_state(chunks_relevantes=["dummy chunk"])
    result = plano_inicial_academico_node(state)
    assert isinstance(result, dict)
    assert "plano_atual" in result
