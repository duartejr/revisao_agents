from state import RevisaoState
from config import get_llm, ENCERRAMENTO
from utils.helpers import fmt_chunks, fmt_snippets, resumir_hist, truncar

def pausa_humana_node(state: RevisaoState) -> dict:
    """Nó de pausa para interação humana (vazio)."""
    return {}

def entrevista_node(state: RevisaoState) -> dict:
    """Gera uma pergunta para o usuário baseada no plano atual."""
    tema      = state["tema"]
    tipo      = state.get("tipo_revisao", "academico")
    p_n       = state["perguntas_feitas"]
    max_p     = state["max_perguntas"]
    plano_c   = truncar(state["plano_atual"], 500)
    hist_c    = resumir_hist(state["historico_entrevista"], 1)
    restantes = max_p - p_n

    if tipo == "tecnico":
        ctx_extra = fmt_snippets(state.get("snippets_tecnicos", [])[-3:], 400)
        instrucoes = (
            "O plano e de um CAPITULO TECNICO. Pergunte sobre:\n"
            "- Nivel de detalhe matematico desejado\n"
            "- Quais algoritmos precisam de pseudocodigo\n"
            "- Quais formulas ou metricas especificas incluir\n"
            "- Fontes tecnicas especificas a consultar\n"
            "- Quais recursos visuais sao prioritarios\n"
        )
        tipo_label = "capitulo tecnico"
    else:
        ctx_extra = fmt_chunks(state.get("chunks_relevantes", [])[-16:], 400)
        instrucoes = (
            "O plano e de uma REVISAO ACADEMICA. Pergunte sobre:\n"
            "- Recorte temporal, geografico ou metodologico\n"
            "- Secoes tecnicas ou comparativas ausentes\n"
            "- Profundidade em algum ponto especifico\n"
            "- Lacunas na literatura identificadas\n"
        )
        tipo_label = "revisao da literatura"

    prompt = (
        f"Orientador refinando plano de {tipo_label} sobre {repr(tema)}\n\n"
        f"Plano atual:\n{plano_c}\n\n"
        f"Dialogo anterior:\n{hist_c}\n\n"
        f"Material encontrado:\n{ctx_extra}\n\n"
        f"Esta e a pergunta {p_n+1} de {max_p} ({restantes} restante(s)).\n\n"
        + instrucoes +
        "Identifique O ASPECTO MAIS IMPORTANTE ainda ausente. "
        "Formule UMA pergunta direta com 2-3 opcoes concretas. Maximo 6 linhas."
    )
    resp     = get_llm(0.7).invoke(prompt)
    pergunta = resp.content if hasattr(resp, "content") else str(resp)
    return {
        "historico_entrevista": [("assistant", pergunta)],
        "perguntas_feitas": p_n + 1,
        "status": "aguardando_usuario",
    }

def roteador_entrevista(state: RevisaoState) -> str:
    """Decide se continua entrevista ou finaliza."""
    if state.get("perguntas_feitas", 0) >= state.get("max_perguntas", 3):
        return "finalizar"
    for role, c in reversed(state.get("historico_entrevista", [])):
        if role == "user":
            if set(c.lower().strip().split()) & ENCERRAMENTO:
                return "finalizar"
            break
    return "continuar"