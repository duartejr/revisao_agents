"""
Unit tests for review_tools.py – tool contracts and error handling.
"""

from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import os

import pytest


# ── search_evidence ──────────────────────────────────────────────────────

class TestSearchEvidence:
    """Tool: search_evidence"""

    def test_returns_formatted_chunks(self):
        fake_chunks = ["chunk A text", "chunk B text"]
        with patch(
            "revisao_agents.tools.review_tools.search_chunks",
            return_value=fake_chunks,
        ):
            from revisao_agents.tools.review_tools import search_evidence

            result = search_evidence.invoke({"query": "climate model", "k": 2})

        assert "[Chunk 1]" in result
        assert "chunk A text" in result
        assert "[Chunk 2]" in result
        assert "chunk B text" in result

    def test_returns_no_results_message_on_empty(self):
        with patch(
            "revisao_agents.tools.review_tools.search_chunks",
            return_value=[],
        ):
            from revisao_agents.tools.review_tools import search_evidence

            result = search_evidence.invoke({"query": "not found"})

        assert "No relevant evidence" in result

    def test_k_is_clamped_to_max_10(self):
        with patch(
            "revisao_agents.tools.review_tools.search_chunks",
            return_value=["c"],
        ) as mock_sc:
            from revisao_agents.tools.review_tools import search_evidence

            search_evidence.invoke({"query": "test", "k": 50})

        # search_chunks should have been called with k <= 10
        call_args = mock_sc.call_args
        # k may be positional (arg[1]) or keyword
        k_val = call_args.kwargs.get("k") if "k" in call_args.kwargs else call_args.args[1]
        assert k_val <= 10


# ── search_web_sources ───────────────────────────────────────────────────

class TestSearchWebSources:
    """Tool: search_web_sources"""

    def test_returns_formatted_web_results(self):
        fake_tavily = {
            "new_urls": ["https://example.com/paper1"],
            "total_accumulated": ["https://example.com/paper1"],
            "results": [],
        }
        fake_extract_tool = SimpleNamespace(
            invoke=lambda _: {
                "extracted": [
                    {
                        "title": "Paper One",
                        "url": "https://example.com/paper1",
                        "content": "some content about climate",
                    },
                ],
                "failed": [],
            }
        )
        with patch(
            "revisao_agents.tools.review_tools.search_tavily_incremental",
            return_value=fake_tavily,
        ), patch(
            "revisao_agents.tools.review_tools.extract_tavily",
            fake_extract_tool,
        ):
            from revisao_agents.tools.review_tools import search_web_sources

            result = search_web_sources.invoke({"query": "climate prediction"})

        assert "Paper One" in result
        assert "https://example.com/paper1" in result

    def test_returns_no_results_when_empty(self):
        fake_tavily = {
            "new_urls": [],
            "total_accumulated": [],
            "results": [],
        }
        with patch(
            "revisao_agents.tools.review_tools.search_tavily_incremental",
            return_value=fake_tavily,
        ):
            from revisao_agents.tools.review_tools import search_web_sources

            result = search_web_sources.invoke({"query": "nothing here"})

        assert "No web results" in result


# ── search_evidence_sources ──────────────────────────────────────────────

class TestSearchEvidenceSources:
    """Tool: search_evidence_sources"""

    def test_returns_source_metadata(self):
        fake_records = [
            {
                "chunk": "evidence snippet",
                "file_path": "chunks_cache/a1.md",
                "source_title": "Chronos-2 Paper",
                "source_url": "https://example.org/chronos2",
                "doi": "10.1234/chronos.2024",
                "score": 0.91,
            }
        ]
        with patch(
            "revisao_agents.tools.review_tools.search_chunk_records",
            return_value=fake_records,
        ):
            from revisao_agents.tools.review_tools import search_evidence_sources

            result = search_evidence_sources.invoke({"query": "chronos transformer"})

        assert "Chronos-2 Paper" in result
        assert "https://example.org/chronos2" in result
        assert "10.1234/chronos.2024" in result

    def test_returns_no_results_message_on_empty(self):
        with patch(
            "revisao_agents.tools.review_tools.search_chunk_records",
            return_value=[],
        ):
            from revisao_agents.tools.review_tools import search_evidence_sources

            result = search_evidence_sources.invoke({"query": "unknown"})

        assert "No evidence sources" in result


class TestSearchNearChunks:
    def test_returns_neighbor_window(self):
        with tempfile.TemporaryDirectory() as tmpd:
            anchor_path = os.path.join(tmpd, "abc123_2.txt")
            for idx in range(0, 5):
                with open(os.path.join(tmpd, f"abc123_{idx}.txt"), "w", encoding="utf-8") as f:
                    f.write(f"chunk-{idx}")

            with patch(
                "revisao_agents.tools.review_tools.search_chunk_records",
                return_value=[
                    {
                        "file_path": anchor_path,
                        "source_title": "Paper X",
                        "chunk": "anchor text",
                    }
                ],
            ):
                from revisao_agents.tools.review_tools import search_near_chunks

                result = search_near_chunks.invoke({"query": "topic", "n": 1})

        assert "Anchor source: Paper X" in result
        assert "[Chunk 1]" in result
        assert "[Chunk 2] (ANCHOR)" in result
        assert "[Chunk 3]" in result


class TestSearchWebImages:
    def test_formats_image_results(self):
        mock_tool = SimpleNamespace(
            invoke=lambda _: {
                "images": [
                    {
                        "image_url": "https://img.example/a.png",
                        "source_url": "https://paper.example",
                        "page_title": "Paper",
                        "description": "figure",
                    }
                ]
            }
        )
        with patch("revisao_agents.tools.review_tools.search_tavily_images", mock_tool):
            from revisao_agents.tools.review_tools import search_web_images

            result = search_web_images.invoke({"query": "chronos architecture"})

        assert "https://img.example/a.png" in result
        assert "https://paper.example" in result


class TestExtractWebTextFromUrl:
    def test_extracts_single_url_text(self):
        mock_tool = SimpleNamespace(
            invoke=lambda _: {
                "extracted": [
                    {
                        "title": "Chronos Paper",
                        "url": "https://paper.example",
                        "content": "full text content",
                    }
                ]
            }
        )
        with patch("revisao_agents.tools.review_tools.extract_tavily", mock_tool):
            from revisao_agents.tools.review_tools import extract_web_text_from_url

            result = extract_web_text_from_url.invoke({"url": "https://paper.example"})

        assert "Chronos Paper" in result
        assert "full text content" in result


class TestGetBibtexForReference:
    def test_gets_bibtex_from_title_lookup(self):
        with patch(
            "revisao_agents.tools.review_tools.search_crossref_by_title",
            return_value="10.1234/abcd",
        ), patch(
            "revisao_agents.tools.review_tools.get_bibtex_from_doi",
            return_value="@article{key,title={Paper}}",
        ):
            from revisao_agents.tools.review_tools import get_bibtex_for_reference

            result = get_bibtex_for_reference.invoke({"query_or_doi": "Chronos 2 paper"})

        assert "10.1234/abcd" in result
        assert "@article" in result

    def test_returns_message_when_no_doi(self):
        with patch(
            "revisao_agents.tools.review_tools.search_crossref_by_title",
            return_value=None,
        ):
            from revisao_agents.tools.review_tools import get_bibtex_for_reference

            result = get_bibtex_for_reference.invoke({"query_or_doi": "unknown paper"})

        assert "No DOI found" in result


# ── get_review_tools ─────────────────────────────────────────────────────

class TestGetReviewTools:
    def test_local_only_excludes_web_tool(self):
        from revisao_agents.tools.review_tools import get_review_tools

        tools = get_review_tools(allow_web=False)
        names = [t.name for t in tools]
        assert "search_evidence" in names
        assert "search_evidence_sources" in names
        assert "search_near_chunks" in names
        assert "search_web_sources" not in names
        assert "search_web_images" not in names
        assert "extract_web_text_from_url" not in names
        assert "get_bibtex_for_reference" not in names

    def test_web_enabled_includes_web_tool(self):
        from revisao_agents.tools.review_tools import get_review_tools

        tools = get_review_tools(allow_web=True)
        names = [t.name for t in tools]
        assert "search_evidence" in names
        assert "search_evidence_sources" in names
        assert "search_near_chunks" in names
        assert "search_web_sources" in names
        assert "search_web_images" in names
        assert "extract_web_text_from_url" in names
        assert "get_bibtex_for_reference" in names
