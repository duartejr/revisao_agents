"""
Academic review agents - LangGraph nodes for literature review planning.

Nodes for the academic review workflow:
- Vector search for relevant papers
- Initial academic plan generation
- Plan refinement based on user feedback
- Final academic review plan

Prompts are loaded from YAML files in prompts/academic/.
"""

from ..state import ReviewState
from ..utils.llm_utils.llm_providers import get_llm
from ..utils.vector_utils.vector_store import search_chunks, accumulate_chunks
from ..utils.file_utils.helpers import fmt_chunks, truncate, save_md
from ..utils.llm_utils.prompt_loader import load_prompt

# Constants (may need to be moved to config)
CHUNKS_PER_QUERY = 10  # TODO: Move to config if it should be configurable


def consulta_vetorial_node(state: ReviewState) -> dict:
    """Busca chunks iniciais sobre o theme."""
    theme = state["theme"]
    print("\n[MONGODB] query:", repr(theme))
    chunks = search_chunks(theme)
    print("   ", len(chunks), "chunks recuperados")
    return {"relevant_chunks": chunks, "status": "chunks_ok"}


def plano_inicial_academico_node(state: ReviewState) -> dict:
    """Gera o primeiro rascunho do plano acadêmico."""
    theme = state["theme"]
    ctx  = fmt_chunks(state["relevant_chunks"], 900)
    prompt = load_prompt("academic/plano_inicial", tema=repr(theme), ctx=ctx)
    resp  = get_llm(temperature=prompt.temperature).invoke(prompt.text)
    plan = resp.content if hasattr(resp, "content") else str(resp)
    print("\nPlano academico inicial elaborado.")
    return {"current_plan": plan, "status": "plano_inicial_pronto"}


def refinar_consulta_academico_node(state: ReviewState) -> dict:
    """Refaz busca vetorial com base na última pergunta do usuário."""
    query = state["theme"]
    for role, c in reversed(state["interview_history"]):
        if role == "user":
            query = c[:150]
            break
    print("\n[MONGODB re-query] query:", repr(query[:60]))
    novos = search_chunks(query)
    acum  = accumulate_chunks(state["relevant_chunks"], novos)
    print("   ", len(novos), "recuperados | total:", len(acum))
    return {"relevant_chunks": acum, "status": "chunks_refinados"}


def refinar_plano_academico_node(state: ReviewState) -> dict:
    """Atualiza o plano acadêmico com novos chunks e feedback."""
    theme       = state["theme"]
    current_plan = truncate(state["current_plan"], 700)
    ultima     = ""
    for role, c in reversed(state["interview_history"]):
        if role == "user":
            ultima = c[:300]
            break
    ctx_novo = fmt_chunks(state["relevant_chunks"][-CHUNKS_PER_QUERY:], 600)
    prompt = load_prompt(
        "academic/refinar_plano",
        tema=repr(theme),
        plano_curr=current_plan,
        ultima=ultima,
        ctx_novo=ctx_novo,
    )
    resp  = get_llm(temperature=prompt.temperature).invoke(prompt.text)
    plan = resp.content if hasattr(resp, "content") else str(resp)
    print("   Plano academico atualizado.")
    return {"current_plan": plan, "status": "plano_refinado"}


def finalizar_plano_academico_node(state: ReviewState) -> dict:
    """Gera o plano acadêmico final e salva em Markdown."""
    theme       = state["theme"]
    current_plan = truncate(state["current_plan"], 1000)
    ctx        = fmt_chunks(state["relevant_chunks"], 800)
    prompt = load_prompt(
        "academic/finalizar_plano",
        tema=repr(theme),
        plano_curr=current_plan,
        ctx=ctx,
    )
    resp        = get_llm(temperature=prompt.temperature).invoke(prompt.text)
    final_plan = resp.content if hasattr(resp, "content") else str(resp)
    print("\n" + "=" * 70)
    print("PLANO FINAL — REVISAO ACADEMICA")
    print("=" * 70)
    print(final_plan)
    print("=" * 70)
    md   = "# Plano de Revisao da Literatura\n\n**Tema:** " + theme + "\n\n" + final_plan
    path = save_md(md, "plans/plano_revisao", theme)
    return {"final_plan": final_plan, "final_plan_path": path, "status": "concluido"}


def vector_search_node(state: ReviewState) -> dict:
    return consulta_vetorial_node(state)


def initial_academic_plan_node(state: ReviewState) -> dict:
    return plano_inicial_academico_node(state)


def refine_academic_search_node(state: ReviewState) -> dict:
    return refinar_consulta_academico_node(state)


def refine_academic_plan_node(state: ReviewState) -> dict:
    return refinar_plano_academico_node(state)


def finalize_academic_plan_node(state: ReviewState) -> dict:
    return finalizar_plano_academico_node(state)
