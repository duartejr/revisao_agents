"""
Compatibility wrapper for planning graph entry points.

Canonical planning execution lives in `workflows/academic_workflow.py`
and `workflows/technical_workflow.py`. This module keeps backward-compatible
function names (`build_academic_graph`, `build_technical_graph`, etc.) and
delegates directly to those workflow builders.
"""

from __future__ import annotations

from ..workflows import build_academico_workflow, build_tecnico_workflow


# ---------------------------------------------------------------------------
# Academic review graph
# ---------------------------------------------------------------------------

def build_academic_graph(checkpointer=None):
    """Build academic planning graph using canonical workflow implementation."""
    return build_academico_workflow()


# ---------------------------------------------------------------------------
# Technical chapter graph
# ---------------------------------------------------------------------------

def build_technical_graph(checkpointer=None):
    """Build technical planning graph using canonical workflow implementation."""
    return build_tecnico_workflow()


# ---------------------------------------------------------------------------
# Convenience helpers (used by cli.py)
# ---------------------------------------------------------------------------

def build_review_graph(tipo: str = "academico", checkpointer=None):
    """
    Factory that returns the appropriate compiled graph.

    Args:
        tipo: "academico" | "tecnico"
        checkpointer: optional LangGraph checkpointer (defaults to MemorySaver)
    """
    if tipo == "tecnico":
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
        "review_type": "academico",
        "relevant_chunks": [],
        "technical_snippets": [],
        "technical_urls": [],
        "current_plan": "",
        "interview_history": [],
        "questions_asked": 0,
        "max_questions": 1,
        "final_plan": "",
        "final_plan_path": "",
        "status": "iniciando",
    }

    result = state
    for step in graph.stream(state, config=config):
        if debug:
            print(step)
        for node_data in step.values():
            if isinstance(node_data, dict):
                result.update(node_data)

    return result
