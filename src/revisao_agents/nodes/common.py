"""
Common agents - LangGraph nodes shared across different review workflows.

Nodes for interview, pausing, and routing decisions:
- Human pause for interaction
- Interview node for user questions
- Router for conditional flow

Prompts are loaded from YAML files in prompts/common/.
"""

from ..state import ReviewState
from ..utils.llm_utils.llm_providers import get_llm
from ..utils.file_utils.helpers import fmt_chunks, fmt_snippets, summarize_hist, truncate
from ..utils.llm_utils.prompt_loader import load_prompt, get_prompt_field

# Constants (may need to be moved to config)
ENCERRAMENTO = {"fim", "terminar", "sair", "encerrar", "pronto", "acabar"}


def human_pause_node(state: ReviewState) -> dict:
    """Human-in-the-loop pause node."""
    return {}


def entrevista_node(state: ReviewState) -> dict:
    """Gera uma pergunta para o usuário baseada no plano atual."""
    theme              = state["theme"]
    review_type        = state.get("review_type", "academic")
    questions_asked    = state["questions_asked"]
    max_questions      = state["max_questions"]
    current_plan       = truncate(state["current_plan"], 500)
    current_history    = summarize_hist(state["interview_history"], 1)
    restantes = max_questions - questions_asked

    if review_type in {"tecnico", "technical"}:
        ctx_extra  = fmt_snippets(state.get("technical_snippets", [])[-3:], 400)
        instrucoes = get_prompt_field("common/entrevista", "instrucoes_tecnico")
        tipo_label = "capitulo tecnico"
    else:
        ctx_extra  = fmt_chunks(state.get("relevant_chunks", [])[-16:], 400)
        instrucoes = get_prompt_field("common/entrevista", "instrucoes_academico")
        tipo_label = "revisao da literatura"

    prompt = load_prompt(
        "common/entrevista",
        tema=repr(theme),
        tipo_label=tipo_label,
        plano_c=current_plan,
        hist_c=current_history,
        ctx_extra=ctx_extra,
        pergunta_num=questions_asked + 1,
        max_p=max_questions,
        restantes=restantes,
        instrucoes=instrucoes,
    )
    resp     = get_llm(temperature=prompt.temperature).invoke(prompt.text)
    pergunta = resp.content if hasattr(resp, "content") else str(resp)
    return {
        "interview_history": [("assistant", pergunta)],
        "questions_asked": questions_asked + 1,
        "status": "aguardando_usuario",
    }


def roteador_entrevista(state: ReviewState) -> str:
    """Decide se continua entrevista ou finaliza."""
    if state.get("questions_asked", 0) >= state.get("max_questions", 3):
        return "finalizar"
    for role, c in reversed(state.get("interview_history", [])):
        if role == "user":
            if set(c.lower().strip().split()) & ENCERRAMENTO:
                return "finalizar"
            break
    return "continuar"


def interview_node(state: ReviewState) -> dict:
    return entrevista_node(state)


def route_interview(state: ReviewState) -> str:
    decision = roteador_entrevista(state)
    if decision == "continuar":
        return "continue"
    return "finalize"
