from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..state import ReviewState
from ..nodes import (
    busca_tecnica_inicial_node,
    plano_inicial_tecnico_node,
    entrevista_node,
    human_pause_node,
    refinar_busca_tecnica_node,
    refinar_plano_tecnico_node,
    finalizar_plano_tecnico_node,
    roteador_entrevista,
)


def build_tecnico_workflow():
    builder = StateGraph(ReviewState)
    builder.add_node("busca_tecnica_inicial",    busca_tecnica_inicial_node)
    builder.add_node("plano_inicial_tecnico",    plano_inicial_tecnico_node)
    builder.add_node("entrevista",               entrevista_node)
    builder.add_node("human_pause",              human_pause_node)
    builder.add_node("refinar_busca_tecnica",    refinar_busca_tecnica_node)
    builder.add_node("refinar_plano_tecnico",    refinar_plano_tecnico_node)
    builder.add_node("finalizar_plano_tecnico",  finalizar_plano_tecnico_node)

    builder.set_entry_point("busca_tecnica_inicial")
    builder.add_edge("busca_tecnica_inicial",   "plano_inicial_tecnico")
    builder.add_edge("plano_inicial_tecnico",   "entrevista")
    builder.add_edge("entrevista",              "human_pause")
    builder.add_edge("human_pause",             "refinar_busca_tecnica")
    builder.add_edge("refinar_busca_tecnica",   "refinar_plano_tecnico")
    builder.add_conditional_edges("refinar_plano_tecnico", roteador_entrevista,
        {"continuar": "entrevista", "finalizar": "finalizar_plano_tecnico"})
    builder.add_edge("finalizar_plano_tecnico", END)

    return builder.compile(checkpointer=MemorySaver(), interrupt_before=["human_pause"])
