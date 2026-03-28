"""
tests/unit/test_tools/test_image_tools.py

Unit tests for the image_tools module.

Uses mocks so no real network calls or installed LangChain providers are needed.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Build stub modules (kept as module-level objects for reuse in tests,
# but NOT permanently inserted into sys.modules here).
# ---------------------------------------------------------------------------

_SENTINEL = object()

# langchain_core.tools stub – pass-through @tool decorator
_lc_tools_mod = types.ModuleType("langchain_core.tools")
_lc_tools_mod.tool = lambda f: f  # @tool becomes identity

_lc_mod = types.ModuleType("langchain_core")
_lc_mod.tools = _lc_tools_mod

# tavily_web_search stub – search_tavily_images / _get_client replaced per test
_tws_mod = types.ModuleType("revisao_agents.tools.tavily_web_search")
_tws_mod.search_tavily_images = mock.MagicMock()
_tws_mod._get_client = mock.MagicMock()

_ra_mod = types.ModuleType("revisao_agents")
_ra_tools_mod = types.ModuleType("revisao_agents.tools")

_STUBS: dict[str, types.ModuleType] = {
    "langchain_core": _lc_mod,
    "langchain_core.tools": _lc_tools_mod,
    "revisao_agents": _ra_mod,
    "revisao_agents.tools": _ra_tools_mod,
    "revisao_agents.tools.tavily_web_search": _tws_mod,
}

_TOOLS_PATH = (
    pathlib.Path(__file__).parents[3] / "src" / "revisao_agents" / "tools" / "image_tools.py"
)


def _load_module() -> types.ModuleType:
    """Load image_tools.py in isolation via a temporary sys.modules injection.

    _SENTINEL distinguishes "key was absent from sys.modules" from "key mapped
    to None" so the finally block can remove newly added keys rather than
    setting them to None (which would leave a broken entry in sys.modules).
    """
    saved = {k: sys.modules.get(k, _SENTINEL) for k in _STUBS}
    sys.modules.update(_STUBS)
    try:
        spec = importlib.util.spec_from_file_location("image_tools_under_test", _TOOLS_PATH)
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "revisao_agents.tools"
        spec.loader.exec_module(mod)
        return mod
    finally:
        # Restore sys.modules so later imports (e.g. test_tavily_web_search.py)
        # see the real packages.
        for k, v in saved.items():
            if v is _SENTINEL:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


_mod = _load_module()

# Grab symbols under test
search_images_with_queries = _mod.search_images_with_queries
lookup_page_metadata = _mod.lookup_page_metadata
search_paper_reference = _mod.search_paper_reference
format_image_markdown = _mod.format_image_markdown
get_image_tools = _mod.get_image_tools
_cache_key = _mod._cache_key
_load_cache = _mod._load_cache
_save_cache = _mod._save_cache


# ---------------------------------------------------------------------------
# Autouse fixture: ensure the TWS stub is in sys.modules for every test so
# that lookup_page_metadata / search_paper_reference can do their lazy
# `from .tavily_web_search import _get_client` import correctly.
# monkeypatch reverts the override automatically after each test.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _inject_tws_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "revisao_agents.tools.tavily_web_search", _tws_mod)


# ===========================================================================
# get_image_tools
# ===========================================================================


class TestGetImageTools:
    def test_returns_four_tools(self):
        tools = get_image_tools()
        assert len(tools) == 4

    def test_tool_names(self):
        names = {t.__name__ if callable(t) else t.name for t in get_image_tools()}
        assert "search_images_with_queries" in names
        assert "lookup_page_metadata" in names
        assert "search_paper_reference" in names
        assert "format_image_markdown" in names

    def test_old_tools_absent(self):
        names = {
            t.__name__ if callable(t) else getattr(t, "name", str(t)) for t in get_image_tools()
        }
        assert "search_images_for_section" not in names
        assert "search_images_for_paragraph" not in names


# ===========================================================================
# search_images_with_queries
# ===========================================================================


class TestSearchImagesWithQueries:
    def _make_tavily_result(self, urls):
        return {
            "images": [
                {
                    "image_url": u,
                    "description": f"desc {i}",
                    "source_url": f"https://page{i}.com",
                    "page_title": f"Title {i}",
                }
                for i, u in enumerate(urls)
            ],
            "total": len(urls),
        }

    def test_empty_queries_returns_error(self):
        result = search_images_with_queries([])
        assert isinstance(result, list)
        assert result[0].get("error")

    def test_passes_queries_to_tavily(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, "_CACHE_DIR", str(tmp_path))
        _tws_mod.search_tavily_images.return_value = self._make_tavily_result(
            ["https://a.com/img.png", "https://b.com/fig.jpg"]
        )
        monkeypatch.setattr(
            _tws_mod.search_tavily_images,
            "invoke",
            lambda kw: self._make_tavily_result(["https://a.com/img.png"]),
        )

        # Patch at module level
        with mock.patch.object(_mod, "search_tavily_images") as mock_tavily:
            mock_tavily.invoke.return_value = self._make_tavily_result(
                ["https://a.com/img.png", "https://b.com/fig.jpg"]
            )
            search_images_with_queries(
                ["Chronos transformer architecture", "LSTM streamflow diagram"]
            )

        call_args = mock_tavily.invoke.call_args
        sent_queries = call_args[0][0]["queries"]
        assert "Chronos transformer architecture" in sent_queries
        assert "LSTM streamflow diagram" in sent_queries

    def test_returns_image_dicts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, "_CACHE_DIR", str(tmp_path))
        with mock.patch.object(_mod, "search_tavily_images") as mock_tavily:
            mock_tavily.invoke.return_value = self._make_tavily_result(
                ["https://a.com/img.png", "https://b.com/fig.jpg"]
            )
            result = search_images_with_queries(["some specific query"])

        assert len(result) >= 1
        assert "image_url" in result[0]
        assert "source_url" in result[0]
        assert "page_title" in result[0]
        assert "source_note_en" in result[0]
        assert "source_note_pt" in result[0]

    def test_falls_back_to_url_field(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, "_CACHE_DIR", str(tmp_path))
        with mock.patch.object(_mod, "search_tavily_images") as mock_tavily:
            mock_tavily.invoke.return_value = {
                "images": [
                    {
                        "url": "https://a.com/img.png",
                        "description": "diagram",
                        "source_url": "",
                        "page_title": "",
                    }
                ],
                "total": 1,
            }
            result = search_images_with_queries(["query"])

        assert result[0]["image_url"] == "https://a.com/img.png"

    def test_max_results_respected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, "_CACHE_DIR", str(tmp_path))
        with mock.patch.object(_mod, "search_tavily_images") as mock_tavily:
            mock_tavily.invoke.return_value = self._make_tavily_result(
                [f"https://x.com/img{i}.png" for i in range(10)]
            )
            result = search_images_with_queries(["query"], max_results=3)

        assert len(result) <= 3

    def test_cache_hit_skips_tavily(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, "_CACHE_DIR", str(tmp_path))
        cached_data = [
            {
                "image_url": "https://cached.com/img.png",
                "description": "",
                "source_url": "",
                "page_title": "",
            }
        ]
        key = _cache_key(["cached query"])
        _save_cache.__wrapped__(key, cached_data) if hasattr(
            _save_cache, "__wrapped__"
        ) else _mod._save_cache(key, cached_data)

        with mock.patch.object(_mod, "search_tavily_images") as mock_tavily:
            result = search_images_with_queries(["cached query"])
            mock_tavily.invoke.assert_not_called()

        assert result[0]["image_url"] == "https://cached.com/img.png"


# ===========================================================================
# lookup_page_metadata
# ===========================================================================


class TestLookupPageMetadata:
    def _fake_extract(self, page_url, title, raw_content):
        """Build a fake Tavily Extract response."""
        fake_client = mock.MagicMock()
        fake_client.extract.return_value = {
            "results": [{"title": title, "raw_content": raw_content, "url": page_url}],
            "failed_results": [],
        }
        return fake_client

    def test_no_url_returns_error(self):
        result = lookup_page_metadata("", "")
        assert result["error"]

    def test_derives_page_url_from_image_url(self):
        image_url = "https://hess.copernicus.org/articles/26/1673/2022/hess-26-img.png"
        with mock.patch.object(_mod, "search_tavily_images"):
            client = self._fake_extract(
                "https://hess.copernicus.org/articles/26/1673/2022/",
                "Deep learning for rainfall-runoff modelling",
                "Authors: Smith, J.; Lee, K. Vol. 26, No. 7, pp. 1673-1693, 2022. DOI: 10.5194/hess-26-1673-2022",
            )
            _tws_mod._get_client.return_value = client
            result = lookup_page_metadata(image_url=image_url, source_url="")

        assert result["page_url"].startswith("https://hess.copernicus.org/articles/26/1673/2022")
        assert result["domain"] == "HESS.COPERNICUS.ORG"

    def test_source_url_takes_priority(self):
        client = self._fake_extract(
            "https://example.com/paper",
            "Paper Title",
            "Authors: Hochreiter, S. 2022. DOI: 10.1234/test",
        )
        _tws_mod._get_client.return_value = client
        result = lookup_page_metadata(
            image_url="https://other.com/img.png",
            source_url="https://example.com/paper",
        )
        assert result["page_url"] == "https://example.com/paper"

    def test_extracts_doi(self):
        client = self._fake_extract(
            "https://journal.com/paper/",
            "My Paper",
            "Some content. DOI: 10.5194/hess-26-1673-2022. More content.",
        )
        _tws_mod._get_client.return_value = client
        result = lookup_page_metadata(image_url="", source_url="https://journal.com/paper/")
        assert result["doi"] == "10.5194/hess-26-1673-2022"

    def test_extracts_year(self):
        client = self._fake_extract(
            "https://journal.com/2022/paper",
            "A Paper From 2022",
            "Published 2022. Authors: Doe, J.",
        )
        _tws_mod._get_client.return_value = client
        result = lookup_page_metadata(image_url="", source_url="https://journal.com/2022/paper")
        assert result["year"] == "2022"

    def test_no_abnt_reference_key(self):
        """The old pre-formatted abnt_reference key must NOT be present."""
        client = self._fake_extract("https://x.com/", "Title", "content 2022")
        _tws_mod._get_client.return_value = client
        result = lookup_page_metadata(image_url="", source_url="https://x.com/")
        assert "abnt_reference" not in result

    def test_result_has_all_keys(self):
        client = self._fake_extract("https://x.com/", "Title", "content")
        _tws_mod._get_client.return_value = client
        result = lookup_page_metadata(image_url="", source_url="https://x.com/")
        for key in (
            "page_url",
            "domain",
            "title",
            "authors",
            "journal",
            "year",
            "volume",
            "issue",
            "pages",
            "doi",
            "content_excerpt",
            "error",
        ):
            assert key in result, f"Missing key: {key}"

    def test_no_hardcoded_portuguese(self):
        """No Portuguese strings must appear when there is no URL."""
        result = lookup_page_metadata("", "")
        for v in result.values():
            if isinstance(v, str):
                assert "FONTE NÃO IDENTIFICADA" not in v
                assert "Disponível em" not in v
                assert "Acesso em" not in v


# ===========================================================================
# search_paper_reference
# ===========================================================================


class TestSearchPaperReference:
    def _fake_search(self, raw_content):
        fake_client = mock.MagicMock()
        fake_client.search.return_value = {
            "results": [
                {"url": "https://doi.org/test", "raw_content": raw_content, "content": raw_content}
            ]
        }
        return fake_client

    def test_returns_expected_keys(self):
        _tws_mod._get_client.return_value = self._fake_search(
            "Authors: Smith, J. DOI: 10.1234/abc. Journal of Water, Vol. 5, No. 2, pp. 100-120, 2022."
        )
        result = search_paper_reference("Some Paper Title", year="2022")
        for key in (
            "title",
            "authors",
            "journal",
            "volume",
            "issue",
            "pages",
            "year",
            "doi",
            "source_url",
            "error",
        ):
            assert key in result, f"Missing key: {key}"

    def test_extracts_doi(self):
        _tws_mod._get_client.return_value = self._fake_search("DOI: 10.5194/hess-26-1673-2022")
        result = search_paper_reference("rainfall runoff deep learning")
        assert result["doi"] == "10.5194/hess-26-1673-2022"

    def test_preserves_title_input(self):
        _tws_mod._get_client.return_value = self._fake_search("no metadata here")
        result = search_paper_reference("My Specific Paper Title", year="2021")
        assert result["title"] == "My Specific Paper Title"
        assert result["year"] == "2021"

    def test_api_error_sets_error_field(self):
        fake_client = mock.MagicMock()
        fake_client.search.side_effect = RuntimeError("API down")
        _tws_mod._get_client.return_value = fake_client
        result = search_paper_reference("any title")
        assert result["error"]


# ===========================================================================
# format_image_markdown
# ===========================================================================


class TestFormatImageMarkdown:
    def test_contains_image_url(self):
        out = format_image_markdown("https://x.com/img.png", "Figure 1: Arch", "Smith 2022")
        assert "https://x.com/img.png" in out

    def test_contains_caption(self):
        out = format_image_markdown("https://x.com/img.png", "Figure 1: Arch", "Smith 2022")
        assert "Figure 1: Arch" in out

    def test_contains_attribution(self):
        out = format_image_markdown("https://x.com/img.png", "Figure 1", "Smith, J. 2022.")
        assert "Smith, J. 2022." in out
