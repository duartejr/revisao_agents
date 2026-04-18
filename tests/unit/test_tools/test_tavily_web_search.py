"""
Unit tests for Tavily search robustness contracts.
"""

import os
from unittest.mock import MagicMock, patch


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


# ── TavilySearchConfig: include_usage env var ─────────────────────────────


def test_tavily_config_include_usage_default_is_true():
    """TAVILY_INCLUDE_USAGE defaults to True when the env var is absent."""
    from revisao_agents.config import TavilySearchConfig

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("TAVILY_INCLUDE_USAGE", None)
        cfg = TavilySearchConfig.load_from_env()

    assert cfg.include_usage is True


def test_tavily_config_include_usage_false_from_env():
    """TAVILY_INCLUDE_USAGE=false sets include_usage to False."""
    from revisao_agents.config import TavilySearchConfig

    with patch.dict(os.environ, {"TAVILY_INCLUDE_USAGE": "false"}):
        cfg = TavilySearchConfig.load_from_env()

    assert cfg.include_usage is False


def test_tavily_config_include_usage_truthy_variants():
    """TAVILY_INCLUDE_USAGE accepts '1' and 'yes' as truthy values."""
    from revisao_agents.config import TavilySearchConfig

    for val in ("1", "yes", "true", "True", "YES"):
        with patch.dict(os.environ, {"TAVILY_INCLUDE_USAGE": val}):
            cfg = TavilySearchConfig.load_from_env()
        assert cfg.include_usage is True, f"Expected True for TAVILY_INCLUDE_USAGE={val!r}"


def test_search_tavily_forwards_include_usage():
    """search_tavily passes include_usage from TAVILY_CONFIG to client.search()."""
    from revisao_agents.tools import tavily_web_search as tws

    fake_client = MagicMock()
    fake_client.search.return_value = {
        "results": [],
        "usage": {"credits": 1},
        "request_id": "req-1",
    }

    with (
        patch.object(tws, "_get_client", return_value=fake_client),
        patch.object(tws, "_save_search_md"),
    ):
        tws.search_tavily.invoke({"queries": ["test"], "max_results": 3})

    _, kwargs = fake_client.search.call_args
    assert "include_usage" in kwargs
    assert kwargs["include_usage"] == tws.TAVILY_CONFIG.include_usage
