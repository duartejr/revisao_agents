from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import RevisaoState
from nodes import (
    consulta_vetorial_node,
    plano_inicial_academico_node,
    entrevista_node,
    pausa_humana_node,
    refinar_consulta_academico_node,
    refinar_plano_academico_node,
    finalizar_plano_academico_node,
    roteador_entrevista,
)

def build_academico_workflow():
    builder = StateGraph(RevisaoState)
    builder.add_node("consulta_vetorial",  consulta_vetorial_node)
    builder.add_node("plano_inicial",      plano_inicial_academico_node)
    builder.add_node("entrevista",         entrevista_node)
    builder.add_node("pausa_humana",       pausa_humana_node)
    builder.add_node("refinar_consulta",   refinar_consulta_academico_node)
    builder.add_node("refinar_plano",      refinar_plano_academico_node)
    builder.add_node("finalizar_plano",    finalizar_plano_academico_node)

    builder.set_entry_point("consulta_vetorial")
    builder.add_edge("consulta_vetorial", "plano_inicial")
    builder.add_edge("plano_inicial",     "entrevista")
    builder.add_edge("entrevista",        "pausa_humana")
    builder.add_edge("pausa_humana",      "refinar_consulta")
    builder.add_edge("refinar_consulta",  "refinar_plano")
    builder.add_conditional_edges("refinar_plano", roteador_entrevista,
        {"continuar": "entrevista", "finalizar": "finalizar_plano"})
    builder.add_edge("finalizar_plano", END)

    return builder.compile(checkpointer=MemorySaver(), interrupt_before=["pausa_humana"])