"""
Unit tests for Tavily search robustness contracts.
"""

import os
from unittest.mock import MagicMock, patch

import pytest


def test_search_tavily_empty_results_no_crash():
    """Verify ``search_tavily`` returns empty lists when the client returns no results."""
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
    """Verify ``search_tavily_incremental`` preserves accumulated URLs when
    the Tavily ``client.search`` call raises a real Tavily/requests error.
    """
    from tavily.errors import BadRequestError

    from revisao_agents.tools import tavily_web_search as tws

    class _FailingClient:
        def search(self, **kwargs):
            raise BadRequestError("boom")

    with patch.object(tws, "_get_client", return_value=_FailingClient()):
        result = tws.search_tavily_incremental("q", ["https://old"], max_results=3)

    assert result["new_urls"] == []
    assert result["total_accumulated"] == ["https://old"]
    assert result["results"] == []


def test_search_tavily_incremental_success_contract_includes_results():
    """Verify ``search_tavily_incremental`` returns discovered URLs and result objects on success."""
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

    with (
        patch.object(tws, "_get_client", return_value=_FakeClient()),
        patch.object(tws, "_save_search_md"),
    ):
        result = tws.search_tavily_incremental("q", [], max_results=3)

    assert "https://example.org/paper" in result["total_accumulated"]
    assert isinstance(result["new_urls"], list)
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


# ── search_tavily_incremental: deduplication ──────────────────────────────


def test_search_tavily_incremental_deduplicates_urls():
    """URLs already in previous_urls must not appear in new_urls."""
    from revisao_agents.tools import tavily_web_search as tws

    existing = "https://example.org/paper"

    class _FakeClient:
        def search(self, **kwargs):
            return {
                "results": [
                    {"url": existing, "title": "Paper", "content": "content here", "score": 0.95},
                ]
            }

    with (
        patch.object(tws, "_get_client", return_value=_FakeClient()),
        patch.object(tws, "_save_search_md"),
    ):
        result = tws.search_tavily_incremental("q", [existing], max_results=5)

    assert existing not in result["new_urls"]


def test_search_tavily_incremental_total_accumulated_has_no_duplicates():
    """total_accumulated must contain each URL exactly once."""
    from revisao_agents.tools import tavily_web_search as tws

    url = "https://example.org/paper"

    class _FakeClient:
        def search(self, **kwargs):
            return {
                "results": [
                    {"url": url, "title": "Paper", "content": "content here", "score": 0.95},
                ]
            }

    with (
        patch.object(tws, "_get_client", return_value=_FakeClient()),
        patch.object(tws, "_save_search_md"),
    ):
        result = tws.search_tavily_incremental("q", [url], max_results=5)

    assert result["total_accumulated"].count(url) == 1


def test_search_tavily_incremental_mlflow_error_propagates():
    """An MLflow failure while opening the span must propagate out of
    ``search_tavily_incremental``, not be swallowed by the search-error
    fallback (the except clause only catches specific Tavily/requests
    errors raised by ``client.search``, so an unrelated ``MlflowException``-
    like error is never caught there).
    """
    from revisao_agents.tools import tavily_web_search as tws

    class _FakeClient:
        def search(self, **kwargs):
            return {"results": [], "usage": {"credits": 0}}

    mock_mlflow = MagicMock()
    mock_mlflow.start_run.side_effect = RuntimeError("mlflow down")

    with (
        patch.object(tws, "_get_client", return_value=_FakeClient()),
        patch.object(tws, "mlflow", mock_mlflow),
        pytest.raises(RuntimeError, match="mlflow down"),
    ):
        tws.search_tavily_incremental("q", ["https://old"], max_results=3)


def test_search_tavily_incremental_unrelated_search_error_propagates():
    """A ``client.search`` failure of a type NOT covered by
    ``TAVILY_SEARCH_ERRORS`` (e.g. an unexpected bug surfacing as
    ``RuntimeError``) must propagate rather than be swallowed by the
    search-error fallback, since the except clause is intentionally narrow.
    """
    from revisao_agents.tools import tavily_web_search as tws

    class _FailingClient:
        def search(self, **kwargs):
            raise RuntimeError("unexpected bug")

    with (
        patch.object(tws, "_get_client", return_value=_FailingClient()),
        pytest.raises(RuntimeError, match="unexpected bug"),
    ):
        tws.search_tavily_incremental("q", ["https://old"], max_results=3)


def test_tavily_search_span_starts_nested_run_and_logs_search_depth():
    """``_tavily_search_span`` must open a nested MLflow run and log search_depth on entry."""
    from revisao_agents.tools import tavily_web_search as tws

    mock_mlflow = MagicMock()
    mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()
    mock_mlflow.start_run.return_value.__exit__.return_value = False

    with (
        patch.object(tws, "mlflow", mock_mlflow),
        tws._tavily_search_span("my query", "advanced") as span,
    ):
        assert span is not None

    mock_mlflow.start_run.assert_called_once_with(run_name="search:my query", nested=True)
    mock_mlflow.log_param.assert_called_once_with("search_depth", "advanced")


def test_tavily_search_span_logs_latency_on_exit():
    """``_tavily_search_span`` must log a ``latency`` metric when the block exits normally."""
    from revisao_agents.tools import tavily_web_search as tws

    mock_mlflow = MagicMock()
    mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()
    mock_mlflow.start_run.return_value.__exit__.return_value = False

    with patch.object(tws, "mlflow", mock_mlflow), tws._tavily_search_span("q", "basic"):
        pass

    latency_calls = [
        call for call in mock_mlflow.log_metric.call_args_list if call.args[0] == "latency"
    ]
    assert len(latency_calls) == 1
    assert isinstance(latency_calls[0].args[1], float)


def test_tavily_search_span_log_metrics_callback_forwards_to_mlflow():
    """The span's ``log_metrics`` callback must forward kwargs to ``mlflow.log_metrics``."""
    from revisao_agents.tools import tavily_web_search as tws

    mock_mlflow = MagicMock()
    mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()
    mock_mlflow.start_run.return_value.__exit__.return_value = False

    with (
        patch.object(tws, "mlflow", mock_mlflow),
        tws._tavily_search_span("q", "basic") as span,
    ):
        span.log_metrics(credits_used=3, urls_found=5, valid_academic_urls_found=2)

    mock_mlflow.log_metrics.assert_called_once_with(
        {"credits_used": 3, "urls_found": 5, "valid_academic_urls_found": 2}
    )


# ── search_tavily_incremental_technician: MLflow instrumentation ──────────


def test_search_tavily_incremental_technician_error_contract_includes_results():
    """Verify the function preserves accumulated URLs when the Tavily
    ``client.search`` call raises a real Tavily/requests error.
    """
    from tavily.errors import BadRequestError

    from revisao_agents.tools import tavily_web_search as tws

    class _FailingClient:
        def search(self, **kwargs):
            raise BadRequestError("boom")

    with patch.object(tws, "_get_client", return_value=_FailingClient()):
        result = tws.search_tavily_incremental_technician("q", ["https://old"], max_results=3)

    assert result["new_urls"] == []
    assert result["total_accumulated"] == ["https://old"]
    assert result["results"] == []
    assert result["urls_found"] == []


def test_search_tavily_incremental_technician_success_contract_includes_results():
    """Verify the function returns discovered URLs and result objects on success."""
    from revisao_agents.tools import tavily_web_search as tws

    class _FakeClient:
        def search(self, **kwargs):
            return {
                "results": [
                    {
                        "url": "https://example.org/spec",
                        "title": "Spec",
                        "content": "the and with results",
                        "score": 0.91,
                    }
                ]
            }

    with (
        patch.object(tws, "_get_client", return_value=_FakeClient()),
        patch.object(tws, "_save_search_md"),
        patch.object(tws, "filter_technical_urls", side_effect=lambda urls: urls),
    ):
        result = tws.search_tavily_incremental_technician("q", [], max_results=3)

    assert "https://example.org/spec" in result["total_accumulated"]
    assert isinstance(result["new_urls"], list)
    assert isinstance(result["results"], list)


def test_search_tavily_incremental_technician_logs_mlflow_metrics():
    """``search_tavily_incremental_technician`` must log credits/urls/valid-urls via the span."""
    from revisao_agents.tools import tavily_web_search as tws

    class _FakeClient:
        def search(self, **kwargs):
            return {
                "results": [
                    {
                        "url": "https://example.org/spec",
                        "title": "Spec",
                        "content": "content here",
                        "score": 0.95,
                    }
                ],
                "usage": {"credits": 2},
            }

    mock_mlflow = MagicMock()
    mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()
    mock_mlflow.start_run.return_value.__exit__.return_value = False
    mock_mlflow.active_run.return_value = None

    with (
        patch.object(tws, "_get_client", return_value=_FakeClient()),
        patch.object(tws, "_save_search_md"),
        patch.object(tws, "filter_technical_urls", side_effect=lambda urls: urls),
        patch.object(tws, "mlflow", mock_mlflow),
    ):
        tws.search_tavily_incremental_technician("q", [], max_results=3)

    mock_mlflow.log_metrics.assert_called_once_with(
        {"credits_used": 2, "urls_found": 1, "valid_technical_urls_found": 1}
    )


def test_search_tavily_incremental_technician_mlflow_error_propagates():
    """An MLflow failure inside the span must propagate out of the function,
    not be swallowed by the search-error fallback (the broad except now only
    wraps the Tavily ``client.search`` call, not the instrumentation block).
    """
    from revisao_agents.tools import tavily_web_search as tws

    class _FakeClient:
        def search(self, **kwargs):
            return {"results": [], "usage": {"credits": 0}}

    mock_mlflow = MagicMock()
    mock_mlflow.start_run.side_effect = RuntimeError("mlflow down")

    with (
        patch.object(tws, "_get_client", return_value=_FakeClient()),
        patch.object(tws, "mlflow", mock_mlflow),
        pytest.raises(RuntimeError, match="mlflow down"),
    ):
        tws.search_tavily_incremental_technician("q", ["https://old"], max_results=3)


def test_tavily_search_span_propagates_exceptions():
    """An exception raised inside the ``with`` block must propagate, not be swallowed."""
    from revisao_agents.tools import tavily_web_search as tws

    mock_mlflow = MagicMock()
    mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()
    mock_mlflow.start_run.return_value.__exit__.return_value = False

    with (
        patch.object(tws, "mlflow", mock_mlflow),
        pytest.raises(RuntimeError, match="boom"),
        tws._tavily_search_span("q", "basic"),
    ):
        raise RuntimeError("boom")

    # Latency must still be logged via the finally block even on exception.
    latency_calls = [
        call for call in mock_mlflow.log_metric.call_args_list if call.args[0] == "latency"
    ]
    assert len(latency_calls) == 1


# ── search_tavily_technical: MLflow instrumentation ───────────────────────


def test_search_tavily_technical_query_error_is_isolated():
    """A real Tavily/requests search error on one query must not abort the
    others, and must not appear in ``results``/``found_urls`` (matching the
    pre-existing per-query fault isolation contract).
    """
    from tavily.errors import BadRequestError

    from revisao_agents.tools import tavily_web_search as tws

    class _FakeClient:
        def search(self, **kwargs):
            if kwargs["query"].startswith("bad"):
                raise BadRequestError("boom")
            return {
                "results": [
                    {
                        "url": "https://example.org/doc",
                        "title": "Doc",
                        "content": "the and with results",
                        "score": 0.91,
                    }
                ],
                "usage": {"credits": 1},
            }

    with (
        patch.object(tws, "_get_client", return_value=_FakeClient()),
        patch.object(tws, "_save_search_md"),
    ):
        result = tws.search_tavily_technical.invoke(
            {"queries": ["bad query", "good query"], "max_results": 3}
        )

    assert result["found_urls"] == ["https://example.org/doc"]
    assert len(result["results"]) == 1


def test_search_tavily_technical_logs_mlflow_metrics_per_query():
    """``search_tavily_technical`` must log credits/urls metrics via the span
    for each query processed in the loop.
    """
    from revisao_agents.tools import tavily_web_search as tws

    class _FakeClient:
        def search(self, **kwargs):
            return {
                "results": [
                    {
                        "url": "https://example.org/doc",
                        "title": "Doc",
                        "content": "content here",
                        "score": 0.95,
                    }
                ],
                "usage": {"credits": 2},
            }

    mock_mlflow = MagicMock()
    mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()
    mock_mlflow.start_run.return_value.__exit__.return_value = False

    with (
        patch.object(tws, "_get_client", return_value=_FakeClient()),
        patch.object(tws, "_save_search_md"),
        patch.object(tws, "mlflow", mock_mlflow),
    ):
        tws.search_tavily_technical.invoke({"queries": ["q1"], "max_results": 3})

    mock_mlflow.log_metrics.assert_called_once_with({"credits_used": 2, "urls_found": 1})


def test_search_tavily_technical_mlflow_error_propagates():
    """An MLflow failure while instrumenting a query's span must propagate
    out of ``search_tavily_technical`` rather than be swallowed, consistent
    with the contract already validated for ``search_tavily_incremental``
    (see ``test_search_tavily_incremental_unrelated_search_error_propagates``).

    The failure happens on the first query's span setup, so the second query
    must never be reached.
    """
    from revisao_agents.tools import tavily_web_search as tws

    class _FakeClient:
        def search(self, **kwargs):
            return {
                "results": [
                    {
                        "url": "https://example.org/doc",
                        "title": "Doc",
                        "content": "content here",
                        "score": 0.95,
                    }
                ],
                "usage": {"credits": 1},
            }

    fake_client = _FakeClient()
    fake_client.search = MagicMock(wraps=fake_client.search)

    mock_mlflow = MagicMock()
    mock_mlflow.start_run.side_effect = RuntimeError("mlflow down")

    with (
        patch.object(tws, "_get_client", return_value=fake_client),
        patch.object(tws, "_save_search_md"),
        patch.object(tws, "mlflow", mock_mlflow),
        pytest.raises(RuntimeError, match="mlflow down"),
    ):
        tws.search_tavily_technical.invoke(
            {"queries": ["bad query", "good query"], "max_results": 3}
        )

    # The exception must propagate before any query is searched: the span is
    # opened (and fails) before `client.search` is ever called.
    fake_client.search.assert_not_called()


def test_search_tavily_technical_span_includes_search_latency():
    """The ``_tavily_search_span`` block must wrap ``client.search`` itself,
    so the logged ``latency`` metric includes search time, not just
    post-processing — keeping this function consistent with
    ``search_tavily_incremental``/``search_tavily_incremental_technician``.
    """
    import time

    from revisao_agents.tools import tavily_web_search as tws

    sleep_seconds = 0.05

    class _SlowClient:
        def search(self, **kwargs):
            time.sleep(sleep_seconds)
            return {"results": [], "usage": {"credits": 0}}

    with (
        patch.object(tws, "_get_client", return_value=_SlowClient()),
        patch.object(tws, "_save_search_md"),
        patch("mlflow.log_metric") as mock_log_metric,
    ):
        tws.search_tavily_technical.invoke({"queries": ["q1"], "max_results": 3})

    latency_calls = [call for call in mock_log_metric.call_args_list if call.args[0] == "latency"]
    assert len(latency_calls) == 1
    assert latency_calls[0].args[1] >= sleep_seconds
