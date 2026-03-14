from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..state import ReviewState
from ..nodes import (
    consulta_vetorial_node,
    plano_inicial_academico_node,
    entrevista_node,
    human_pause_node,
    refinar_consulta_academico_node,
    refinar_plano_academico_node,
    finalizar_plano_academico_node,
    roteador_entrevista,
)


def build_academic_workflow():
    builder = StateGraph(ReviewState)
    builder.add_node("consulta_vetorial",  consulta_vetorial_node)
    builder.add_node("plano_inicial",      plano_inicial_academico_node)
    builder.add_node("entrevista",         entrevista_node)
    builder.add_node("human_pause",        human_pause_node)
    builder.add_node("refinar_consulta",   refinar_consulta_academico_node)
    builder.add_node("refinar_plano",      refinar_plano_academico_node)
    builder.add_node("finalizar_plano",    finalizar_plano_academico_node)

    builder.set_entry_point("consulta_vetorial")
    builder.add_edge("consulta_vetorial", "plano_inicial")
    builder.add_edge("plano_inicial",     "entrevista")
    builder.add_edge("entrevista",        "human_pause")
    builder.add_edge("human_pause",       "refinar_consulta")
    builder.add_edge("refinar_consulta",  "refinar_plano")
    builder.add_conditional_edges("refinar_plano", roteador_entrevista,
        {"continuar": "entrevista", "finalizar": "finalizar_plano"})
    builder.add_edge("finalizar_plano", END)

    return builder.compile(checkpointer=MemorySaver(), interrupt_before=["human_pause"])


def build_academico_workflow():
    """Backward-compatible alias for build_academic_workflow."""
    return build_academic_workflow()
