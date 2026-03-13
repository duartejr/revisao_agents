"""
Unit tests for Tavily search robustness contracts.
"""

from unittest.mock import patch


def test_search_tavily_empty_results_no_crash():
    from revisao_agents.tools import tavily_web_search as tws

    class _FakeClient:
        def search(self, **kwargs):
            return {"results": []}

    with patch.object(tws, "_get_client", return_value=_FakeClient()):
        result = tws.search_tavily.invoke({"queries": ["no results"], "max_results": 3})

    assert isinstance(result, dict)
    assert result["urls_encontrados"] == []
    assert result["resultados"] == []


def test_search_tavily_incremental_error_contract_includes_resultados():
    from revisao_agents.tools import tavily_web_search as tws

    with patch.object(tws, "_get_client", side_effect=RuntimeError("boom")):
        result = tws.search_tavily_incremental("q", ["https://old"], max_results=3)

    assert result["urls_novos"] == []
    assert result["total_acumulado"] == ["https://old"]
    assert result["resultados"] == []


def test_search_tavily_incremental_success_contract_includes_resultados():
    from revisao_agents.tools import tavily_web_search as tws

    class _FakeClient:
        def search(self, **kwargs):
            return {
                "results": [
                    {
                        "url": "https://example.org/paper",
                        "title": "Paper",
                        "content": "the and with results",
                        "score": 0.91,
                    }
                ]
            }

    with patch.object(tws, "_get_client", return_value=_FakeClient()):
        result = tws.search_tavily_incremental("q", [], max_results=3)

    assert "urls_novos" in result
    assert "total_acumulado" in result
    assert "resultados" in result
    assert isinstance(result["resultados"], list)
