"""
Common agents - LangGraph nodes shared across different review workflows.

Nodes for interview, pausing, and routing decisions:
- Human pause for interaction
- Interview node for user questions
- Router for conditional flow

Prompts are loaded from YAML files in prompts/common/.
"""

from ..state import RevisaoState
from ..utils.llm_providers import get_llm
from ..utils.helpers import fmt_chunks, fmt_snippets, resumir_hist, truncar
from ..utils.prompt_loader import load_prompt, get_prompt_field

# Constants (may need to be moved to config)
ENCERRAMENTO = {"fim", "terminar", "sair", "encerrar", "pronto", "acabar"}


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
        ctx_extra  = fmt_snippets(state.get("snippets_tecnicos", [])[-3:], 400)
        instrucoes = get_prompt_field("common/entrevista", "instrucoes_tecnico")
        tipo_label = "capitulo tecnico"
    else:
        ctx_extra  = fmt_chunks(state.get("chunks_relevantes", [])[-16:], 400)
        instrucoes = get_prompt_field("common/entrevista", "instrucoes_academico")
        tipo_label = "revisao da literatura"

    prompt = load_prompt(
        "common/entrevista",
        tema=repr(tema),
        tipo_label=tipo_label,
        plano_c=plano_c,
        hist_c=hist_c,
        ctx_extra=ctx_extra,
        pergunta_num=p_n + 1,
        max_p=max_p,
        restantes=restantes,
        instrucoes=instrucoes,
    )
    resp     = get_llm(prompt.temperature).invoke(prompt.text)
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
