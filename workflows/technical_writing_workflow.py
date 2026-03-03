from langgraph.graph import StateGraph, END
from state import EscritaTecnicaState
from nodes.technical_writing import parsear_plano_node, escrever_secoes_node, consolidar_node

def build_workflow():
    builder = StateGraph(EscritaTecnicaState)
    builder.add_node("parsear_plano", parsear_plano_node)
    builder.add_node("escrever_secoes", escrever_secoes_node)
    builder.add_node("consolidar", consolidar_node)
    builder.set_entry_point("parsear_plano")
    builder.add_edge("parsear_plano", "escrever_secoes")
    builder.add_edge("escrever_secoes", "consolidar")
    builder.add_edge("consolidar", END)
    return builder.compile()