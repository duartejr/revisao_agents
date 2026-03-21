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
    assert result["urls_found"] == []
    assert result["results"] == []


def test_search_tavily_incremental_error_contract_includes_results():
    from revisao_agents.tools import tavily_web_search as tws

    with patch.object(tws, "_get_client", side_effect=RuntimeError("boom")):
        result = tws.search_tavily_incremental("q", ["https://old"], max_results=3)

    assert result["new_urls"] == []
    assert result["total_accumulated"] == ["https://old"]
    assert result["results"] == []


def test_search_tavily_incremental_success_contract_includes_results():
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

    assert "new_urls" in result
    assert "total_accumulated" in result
    assert "results" in result
    assert isinstance(result["results"], list)
