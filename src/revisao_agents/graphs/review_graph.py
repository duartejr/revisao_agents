"""
review_graph.py - Main LangGraph StateGraph definitions.

Contains the academic and technical review graphs, and the
technical-writing graph, all wired from agents/. 
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..state import RevisaoState, EscritaTecnicaState
from ..agents.academic import (
    consulta_vetorial_node,
    plano_inicial_academico_node,
    refinar_consulta_academico_node,
    refinar_plano_academico_node,
    finalizar_plano_academico_node,
)
from ..agents.technical import (
    busca_tecnica_inicial_node,
    plano_inicial_tecnico_node,
    refinar_busca_tecnica_node,
    refinar_plano_tecnico_node,
    finalizar_plano_tecnico_node,
)
from ..agents.common import (
    pausa_humana_node,
    entrevista_node,
    roteador_entrevista,
)
from .checkpoints import make_checkpointer


# ---------------------------------------------------------------------------
# Academic review graph
# ---------------------------------------------------------------------------

def build_academic_graph(checkpointer=None):
    """Build and compile the academic literature-review graph."""
    g = StateGraph(RevisaoState)

    g.add_node("consulta_vetorial",        consulta_vetorial_node)
    g.add_node("plano_inicial_academico",   plano_inicial_academico_node)
    g.add_node("pausa_humana",              pausa_humana_node)
    g.add_node("entrevista",                entrevista_node)
    g.add_node("refinar_consulta_academico",refinar_consulta_academico_node)
    g.add_node("refinar_plano_academico",   refinar_plano_academico_node)
    g.add_node("finalizar_plano_academico", finalizar_plano_academico_node)

    g.set_entry_point("consulta_vetorial")
    g.add_edge("consulta_vetorial",        "plano_inicial_academico")
    g.add_edge("plano_inicial_academico",  "pausa_humana")
    g.add_edge("pausa_humana",             "entrevista")
    g.add_conditional_edges(
        "entrevista",
        roteador_entrevista,
        {
            "refinar": "refinar_consulta_academico",
            "finalizar": "finalizar_plano_academico",
        },
    )
    g.add_edge("refinar_consulta_academico", "refinar_plano_academico")
    g.add_edge("refinar_plano_academico",    "pausa_humana")
    g.add_edge("finalizar_plano_academico",  END)

    cp = checkpointer or make_checkpointer()
    return g.compile(checkpointer=cp, interrupt_before=["pausa_humana"])


# ---------------------------------------------------------------------------
# Technical chapter graph
# ---------------------------------------------------------------------------

def build_technical_graph(checkpointer=None):
    """Build and compile the technical chapter-planning graph."""
    g = StateGraph(RevisaoState)

    g.add_node("busca_tecnica_inicial",   busca_tecnica_inicial_node)
    g.add_node("plano_inicial_tecnico",   plano_inicial_tecnico_node)
    g.add_node("pausa_humana",            pausa_humana_node)
    g.add_node("entrevista",              entrevista_node)
    g.add_node("refinar_busca_tecnica",   refinar_busca_tecnica_node)
    g.add_node("refinar_plano_tecnico",   refinar_plano_tecnico_node)
    g.add_node("finalizar_plano_tecnico", finalizar_plano_tecnico_node)

    g.set_entry_point("busca_tecnica_inicial")
    g.add_edge("busca_tecnica_inicial",  "plano_inicial_tecnico")
    g.add_edge("plano_inicial_tecnico",  "pausa_humana")
    g.add_edge("pausa_humana",           "entrevista")
    g.add_conditional_edges(
        "entrevista",
        roteador_entrevista,
        {
            "refinar": "refinar_busca_tecnica",
            "finalizar": "finalizar_plano_tecnico",
        },
    )
    g.add_edge("refinar_busca_tecnica",   "refinar_plano_tecnico")
    g.add_edge("refinar_plano_tecnico",   "pausa_humana")
    g.add_edge("finalizar_plano_tecnico", END)

    cp = checkpointer or make_checkpointer()
    return g.compile(checkpointer=cp, interrupt_before=["pausa_humana"])


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
    state = {"tema": input_text}

    result = {}
    for step in graph.stream(state, config=config):
        if debug:
            print(step)
        result = step

    return result
