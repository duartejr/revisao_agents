"""
tests/unit/test_agents/test_writing_nodes.py

Unit tests for the previously-untested writing-pipeline nodes
(``parse_plan_node``, ``consolidate_node``), added alongside their new
``@mlflow.trace`` instrumentation (W10-STORY-06). These nodes had zero test
coverage before this sprint.
"""

from unittest.mock import MagicMock, patch

from revisao_agents.nodes.writing.consolidate_node import consolidate_node
from revisao_agents.nodes.writing.parse_plan_node import parse_plan_node


class TestParsePlanNode:
    """Unit tests for ``parse_plan_node``'s technical/academic dispatch."""

    def test_technical_mode_extracts_theme_and_sections(self, tmp_path):
        plan_text = (
            "**Theme:** Test Theme\n\n"
            "| Level | Title | Expected Content | Resources |\n"
            "|---|---|---|---|\n"
            "| 1 | Introduction | Basic overview | some resource |\n"
        )
        plan_path = tmp_path / "plan.md"
        plan_path.write_text(plan_text, encoding="utf-8")

        state = {"plan_path": str(plan_path), "writer_config": {"mode": "technical"}}

        result = parse_plan_node(state)

        assert result["theme"] == "Test Theme"
        assert len(result["sections"]) == 1
        assert result["sections"][0]["title"] == "1 Introduction"
        assert result["written_sections"] == []
        assert result["status"] == "plan_parsed"
        assert result["plan_path"] == str(plan_path)

    def test_academic_mode_extracts_theme_and_sections(self, tmp_path):
        plan_text = "**Theme:** Test Academic Theme\n\n## 1 Introduction\nSome content.\n"
        plan_path = tmp_path / "plan.md"
        plan_path.write_text(plan_text, encoding="utf-8")

        state = {"plan_path": str(plan_path), "writer_config": {"mode": "academic"}}

        result = parse_plan_node(state)

        assert result["theme"] == "Test Academic Theme"
        assert len(result["sections"]) == 1
        assert result["status"] == "plan_parsed"


class TestConsolidateNode:
    """Unit tests for ``consolidate_node``'s document assembly and file output."""

    def test_builds_document_and_writes_files(self, tmp_path):
        state = {
            "theme": "Test Theme",
            "written_sections": [
                {
                    "index": 0,
                    "title": "Introduction",
                    "text": "Some intro text [1].",
                    "source_map": {1: "https://example.com/a"},
                },
            ],
            "refs_urls": ["https://example.com/a"],
            "react_log": ["log line 1"],
            "verification_stats": [
                {
                    "section": "Introduction",
                    "total": 2,
                    "approved": 1,
                    "adjusted": 1,
                    "corrected": 0,
                }
            ],
            "cumulative_summary": "summary text",
            "writer_config": {"mode": "technical", "language": "en"},
        }

        with (
            patch(
                "revisao_agents.nodes.writing.consolidate_node.llm_call",
                MagicMock(side_effect=["Intro text.", "Conclusion text."]),
            ),
            patch("revisao_agents.nodes.writing.consolidate_node.REVIEWS_DIR", str(tmp_path)),
        ):
            result = consolidate_node(state)

        assert result == {"status": "completed"}
        written_files = list(tmp_path.glob("*.md"))
        assert len(written_files) == 1
        content = written_files[0].read_text(encoding="utf-8")
        assert "Test Theme" in content
        assert "Intro text." in content
        assert "Conclusion text." in content
