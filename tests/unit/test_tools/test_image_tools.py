"""
tests/unit/test_tools/test_image_tools.py

Unit tests for the image_tools module.

Uses mocks so no real network calls or installed LangChain providers are needed.
"""

from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys
import types
from unittest import mock

# ---------------------------------------------------------------------------
# Stub out heavy transitive dependencies before importing image_tools
# ---------------------------------------------------------------------------


def _make_stub_module(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


# langchain_core.tools – just expose a pass-through @tool decorator
lc_tools = _make_stub_module("langchain_core")
lc_tools.tools = _make_stub_module("langchain_core.tools")
lc_tools.tools.tool = lambda f: f  # @tool becomes identity

# tavily_web_search stub – provides search_tavily_images (replaced in tests)
_tws_mod = _make_stub_module("revisao_agents.tools.tavily_web_search")
_tws_mod.search_tavily_images = mock.MagicMock()
_tws_mod._get_client = mock.MagicMock()

# Insert parent package stubs so Python can resolve relative imports
for pkg in [
    "revisao_agents",
    "revisao_agents.tools",
]:
    if pkg not in sys.modules:
        _make_stub_module(pkg)

_TOOLS_PATH = (
    pathlib.Path(__file__).parents[3] / "src" / "revisao_agents" / "tools" / "image_tools.py"
)

spec = importlib.util.spec_from_file_location("image_tools_under_test", _TOOLS_PATH)
_mod = importlib.util.module_from_spec(spec)
# Point relative imports at our stubs
_mod.__package__ = "revisao_agents.tools"
spec.loader.exec_module(_mod)

# Grab symbols under test
search_images_with_queries = _mod.search_images_with_queries
lookup_page_metadata = _mod.lookup_page_metadata
search_paper_reference = _mod.search_paper_reference
format_image_markdown = _mod.format_image_markdown
get_image_tools = _mod.get_image_tools
_cache_key = _mod._cache_key
_load_cache = _mod._load_cache
_save_cache = _mod._save_cache


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
        assert "source_note" in result[0]

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
