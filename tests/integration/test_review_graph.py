"""
tests/integration/test_review_graph.py

Integration smoke tests for graph/workflow construction.

These tests verify buildability of the canonical planning stack and
the compatibility graph wrappers. They do not require external API keys.
"""

import importlib.util

import pytest

has_langgraph = importlib.util.find_spec("langgraph") is not None
requires_langgraph = pytest.mark.skipif(
    not has_langgraph,
    reason="langgraph not installed — skipping graph/workflow build tests",
)


@requires_langgraph
def test_academic_graph_builds():
    """Verify that ``build_review_graph`` constructs the academic workflow without errors.

    Confirms that all components and node dependencies required for the
    ``"academico"`` review type are correctly wired and the returned graph
    object is not ``None``.
    """
    from revisao_agents.graphs import build_review_graph

    graph = build_review_graph(review_type="academico")
    assert graph is not None


@requires_langgraph
def test_technical_graph_builds():
    """Verify that ``build_review_graph`` constructs the technical workflow without errors.

    Confirms that all components and node dependencies required for the
    ``"tecnico"`` review type are correctly wired and the returned graph
    object is not ``None``.
    """
    from revisao_agents.graphs import build_review_graph

    graph = build_review_graph(review_type="tecnico")
    assert graph is not None


@requires_langgraph
def test_academic_workflow_builds():
    """Verify that ``build_academic_workflow`` constructs the workflow object without errors."""
    from revisao_agents.workflows.academic_workflow import build_academic_workflow

    workflow = build_academic_workflow()
    assert workflow is not None


@requires_langgraph
def test_technical_workflow_builds():
    """Verify that ``build_technical_workflow`` constructs the workflow object without errors."""
    from revisao_agents.workflows.technical_workflow import build_technical_workflow

    workflow = build_technical_workflow()
    assert workflow is not None
