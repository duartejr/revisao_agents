"""
tests/unit/test_agents/test_academic.py

Unit tests for the academic review agents.
"""

import pytest
from unittest.mock import patch, MagicMock

from revisao_agents.state import RevisaoState


def _make_state(**overrides) -> RevisaoState:
    base: RevisaoState = {
        "tema": "Test topic",
        "tipo": "academico",
        "chunks_relevantes": [],
        "snippets_tecnicos": [],
        "urls_tecnicos": [],
        "plano_atual": "",
        "historico_entrevista": [],
        "pergunta_numero": 0,
        "max_perguntas": 3,
        "finalizado": False,
    }
    base.update(overrides)
    return base


@patch("revisao_agents.agents.academic.buscar_chunks", return_value=[])
@patch("revisao_agents.agents.academic.acumular_chunks", return_value=[])
def test_consulta_vetorial_node_returns_dict(mock_acc, mock_busca):
    from revisao_agents.agents.academic import consulta_vetorial_node

    state = _make_state()
    result = consulta_vetorial_node(state)
    assert isinstance(result, dict)


@patch("revisao_agents.agents.academic.get_llm")
def test_plano_inicial_returns_dict(mock_llm):
    mock_llm.return_value.invoke.return_value = MagicMock(content="Plano de teste")
    from revisao_agents.agents.academic import plano_inicial_academico_node

    state = _make_state(chunks_relevantes=[{"texto": "dummy", "fonte": "x"}])
    result = plano_inicial_academico_node(state)
    assert isinstance(result, dict)
    assert "plano_atual" in result
