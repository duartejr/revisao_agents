"""
tests/integration/test_review_graph.py

Integration tests for the full review graph.

These tests require valid API keys in the environment and will be
skipped automatically when keys are not present.
"""

import os
import pytest

openai_key = os.getenv("OPENAI_API_KEY", "")
requires_api = pytest.mark.skipif(
    not openai_key or openai_key.startswith("sk-fake"),
    reason="OPENAI_API_KEY not set — skipping integration tests",
)


@requires_api
def test_academic_graph_builds():
    from revisao_agents.graphs.review_graph import build_academic_graph

    graph = build_academic_graph()
    assert graph is not None


@requires_api
def test_technical_graph_builds():
    from revisao_agents.graphs.review_graph import build_technical_graph

    graph = build_technical_graph()
    assert graph is not None
