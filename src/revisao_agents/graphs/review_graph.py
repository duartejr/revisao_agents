"""
Compatibility wrapper for planning graph entry points.

Canonical planning execution lives in `workflows/academic_workflow.py`
and `workflows/technical_workflow.py`. This module keeps backward-compatible
function names (`build_academic_graph`, `build_technical_graph`, etc.) and
delegates directly to those workflow builders.
"""

from __future__ import annotations

from ..workflows import build_academic_workflow, build_technical_workflow


def _normalize_review_type(review_type: str | None) -> str:
    """Normalize review type string to canonical values.

    Args:
        review_type: Input string indicating review type (e.g., "academic", "technical", "academico", "tecnico")
    Returns:
        Normalized review type: "academic" or "technical"
    """
    value = (review_type or "academic").strip().lower()
    if value in {"technical", "tecnico"}:
        return "technical"
    return "academic"


# ---------------------------------------------------------------------------
# Academic review graph
# ---------------------------------------------------------------------------


def build_academic_graph(checkpointer=None) -> object:
    """Build academic planning graph using canonical workflow implementation.

    Args:
        checkpointer: optional LangGraph checkpointer (defaults to MemorySaver)

    Returns:
        compiled LangGraph graph instance for academic review workflow
    """
    return build_academic_workflow(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Technical chapter graph
# ---------------------------------------------------------------------------


def build_technical_graph(checkpointer=None) -> object:
    """Build technical planning graph using canonical workflow implementation.

    Args:
        checkpointer: optional LangGraph checkpointer (defaults to MemorySaver)

    Returns:
        compiled LangGraph graph instance for technical review workflow
    """
    return build_technical_workflow(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Convenience helpers (used by cli.py)
# ---------------------------------------------------------------------------


def build_review_graph(
    review_type: str = "academic",
    checkpointer=None,
    tipo: str | None = None,
):
    """
    Factory that returns the appropriate compiled graph.

    Args:
        review_type: "academic" | "technical"
        tipo: legacy alias for review_type ("academico" | "tecnico")
        checkpointer: optional LangGraph checkpointer (defaults to MemorySaver)

    Returns:
        Compiled LangGraph graph instance for the specified review type.
    """
    normalized = _normalize_review_type(tipo if tipo is not None else review_type)
    if normalized == "technical":
        return build_technical_graph(checkpointer=checkpointer)
    return build_academic_graph(checkpointer=checkpointer)


def run_review_graph(graph, input_text: str, debug: bool = False) -> dict:
    """
    Run a compiled review graph to completion and return the final state.

    Args:
        graph: compiled LangGraph graph
        input_text: the paper text or topic string
        debug: whether to print intermediate states

    Returns:
        final state dict
    """
    config = {"configurable": {"thread_id": "cli-run"}}
    state = {
        "theme": input_text,
        "review_type": "academic",
        "relevant_chunks": [],
        "technical_snippets": [],
        "technical_urls": [],
        "current_plan": "",
        "interview_history": [],
        "questions_asked": 0,
        "max_questions": 1,
        "final_plan": "",
        "final_plan_path": "",
        "status": "starting",
    }

    result = state
    for step in graph.stream(state, config=config):
        if debug:
            print(step)
        for node_data in step.values():
            if isinstance(node_data, dict):
                result.update(node_data)

    return result
