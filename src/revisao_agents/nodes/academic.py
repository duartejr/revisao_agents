"""
Academic review agents - LangGraph nodes for literature review planning.

Nodes for the academic review workflow:
- Vector search for relevant papers
- Initial academic plan generation
- Plan refinement based on user feedback
- Final academic review plan

Prompts are loaded from YAML files in prompts/academic/.
"""

from ..state import RevisaoState
from ..utils.llm_utils.llm_providers import get_llm
from ..utils.vector_utils.vector_store import buscar_chunks, acumular_chunks
from ..utils.file_utils.helpers import fmt_chunks, truncar, salvar_md
from ..utils.llm_utils.prompt_loader import load_prompt

# Constants (may need to be moved to config)
CHUNKS_PER_QUERY = 10  # TODO: Move to config if it should be configurable


def consulta_vetorial_node(state: RevisaoState) -> dict:
    """Busca chunks iniciais sobre o tema."""
    tema = state["tema"]
    print("\n[FAISS] query:", repr(tema))
    chunks = buscar_chunks(tema)
    print("   ", len(chunks), "chunks recuperados")
    return {"chunks_relevantes": chunks, "status": "chunks_ok"}


def plano_inicial_academico_node(state: RevisaoState) -> dict:
    """Gera o primeiro rascunho do plano acadêmico."""
    tema = state["tema"]
    ctx  = fmt_chunks(state["chunks_relevantes"], 900)
    prompt = load_prompt("academic/plano_inicial", tema=repr(tema), ctx=ctx)
    resp  = get_llm(temperature=prompt.temperature).invoke(prompt.text)
    plano = resp.content if hasattr(resp, "content") else str(resp)
    print("\nPlano academico inicial elaborado.")
    return {"plano_atual": plano, "status": "plano_inicial_pronto"}


def refinar_consulta_academico_node(state: RevisaoState) -> dict:
    """Refaz busca vetorial com base na última pergunta do usuário."""
    query = state["tema"]
    for role, c in reversed(state["historico_entrevista"]):
        if role == "user":
            query = c[:150]
            break
    print("\n[FAISS re-consulta] query:", repr(query[:60]))
    novos = buscar_chunks(query)
    acum  = acumular_chunks(state["chunks_relevantes"], novos)
    print("   ", len(novos), "recuperados | total:", len(acum))
    return {"chunks_relevantes": acum, "status": "chunks_refinados"}


def refinar_plano_academico_node(state: RevisaoState) -> dict:
    """Atualiza o plano acadêmico com novos chunks e feedback."""
    tema       = state["tema"]
    plano_curr = truncar(state["plano_atual"], 700)
    ultima     = ""
    for role, c in reversed(state["historico_entrevista"]):
        if role == "user":
            ultima = c[:300]
            break
    ctx_novo = fmt_chunks(state["chunks_relevantes"][-CHUNKS_PER_QUERY:], 600)
    prompt = load_prompt(
        "academic/refinar_plano",
        tema=repr(tema),
        plano_curr=plano_curr,
        ultima=ultima,
        ctx_novo=ctx_novo,
    )
    resp  = get_llm(temperature=prompt.temperature).invoke(prompt.text)
    plano = resp.content if hasattr(resp, "content") else str(resp)
    print("   Plano academico atualizado.")
    return {"plano_atual": plano, "status": "plano_refinado"}


def finalizar_plano_academico_node(state: RevisaoState) -> dict:
    """Gera o plano acadêmico final e salva em Markdown."""
    tema       = state["tema"]
    plano_curr = truncar(state["plano_atual"], 1000)
    ctx        = fmt_chunks(state["chunks_relevantes"], 800)
    prompt = load_prompt(
        "academic/finalizar_plano",
        tema=tema,
        plano_curr=plano_curr,
        ctx=ctx,
    )
    resp        = get_llm(temperature=prompt.temperature).invoke(prompt.text)
    plano_final = resp.content if hasattr(resp, "content") else str(resp)
    print("\n" + "=" * 70)
    print("PLANO FINAL — REVISAO ACADEMICA")
    print("=" * 70)
    print(plano_final)
    print("=" * 70)
    md   = "# Plano de Revisao da Literatura\n\n**Tema:** " + tema + "\n\n" + plano_final
    path = salvar_md(md, "plans/plano_revisao", tema)
    return {"plano_final": plano_final, "plano_final_path": path, "status": "concluido"}
