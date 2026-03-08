"""
Technical review agents - LangGraph nodes for technical chapter planning.

Nodes for the technical review workflow:
- Web search for technical sources
- Initial technical plan generation
- Plan refinement based on user feedback
- Final technical review plan

Prompts are loaded from YAML files in prompts/technical/.
"""

from ..state import RevisaoState
from ..utils.llm_utils.llm_providers import get_llm
from ..utils.search_utils.tavily_client import buscar_conteudo_tecnico
from ..utils.file_utils.helpers import fmt_snippets, truncar, salvar_md
from ..utils.llm_utils.prompt_loader import load_prompt


def busca_tecnica_inicial_node(state: RevisaoState) -> dict:
    """Busca inicial de conteúdo técnico via Tavily."""
    tema = state["tema"]
    print("\n[Busca tecnica inicial] tema:", repr(tema))
    res      = buscar_conteudo_tecnico(tema, [])
    urls     = res.get("total_acumulado", [])
    snippets = res.get("resultados", [])
    print("\n" + "=" * 70)
    print("FONTES TECNICAS ENCONTRADAS")
    print("=" * 70)
    for i, r in enumerate(snippets, 1):
        print("\n  [" + str(i).rjust(2) + "] " + r.get("title", "")[:70])
        print("       " + r.get("url",   "")[:80])
        print("       " + r.get("snippet", "")[:120] + "...")
    print("=" * 70)
    return {"urls_tecnicos": urls, "snippets_tecnicos": snippets,
            "status": "busca_tecnica_ok"}


def plano_inicial_tecnico_node(state: RevisaoState) -> dict:
    """Gera o rascunho inicial do plano técnico."""
    tema     = state["tema"]
    snippets = fmt_snippets(state.get("snippets_tecnicos", []), 1200)
    prompt = load_prompt("technical/plano_inicial", tema=tema, snippets=snippets)
    resp  = get_llm(prompt.temperature).invoke(prompt.text)
    plano = resp.content if hasattr(resp, "content") else str(resp)
    print("\nPlano tecnico inicial elaborado.")
    return {"plano_atual": plano, "status": "plano_tecnico_inicial_pronto"}


def refinar_busca_tecnica_node(state: RevisaoState) -> dict:
    """Refaz busca técnica com base na última pergunta."""
    query = state["tema"]
    for role, c in reversed(state["historico_entrevista"]):
        if role == "user":
            query = state["tema"] + " " + c[:100]
            break
    query = query.strip()
    print("\n[Re-busca tecnica] query:", repr(query[:70]))
    urls_ant = state.get("urls_tecnicos", [])
    res      = buscar_conteudo_tecnico(query, urls_ant)
    novos    = res.get("urls_novos", [])
    total    = res.get("total_acumulado", urls_ant)
    snips_n  = res.get("resultados", [])
    if novos:
        print("\nNovas fontes (" + str(len(novos)) + "):")
        for r in snips_n[:4]:
            print("  * " + r.get("title", "")[:60])
            print("    " + r.get("url",   "")[:70])
    snips_acum = (snips_n + state.get("snippets_tecnicos", []))[:20]
    return {"urls_tecnicos": total, "snippets_tecnicos": snips_acum,
            "status": "busca_tecnica_refinada"}


def refinar_plano_tecnico_node(state: RevisaoState) -> dict:
    """Atualiza o plano técnico com novas fontes e feedback."""
    tema       = state["tema"]
    plano_curr = truncar(state["plano_atual"], 700)
    ultima     = ""
    for role, c in reversed(state["historico_entrevista"]):
        if role == "user":
            ultima = c[:300]
            break
    snips = fmt_snippets(state.get("snippets_tecnicos", [])[:5], 800)
    prompt = load_prompt(
        "technical/refinar_plano",
        tema=tema,
        plano_curr=plano_curr,
        ultima=ultima,
        snips=snips,
    )
    resp  = get_llm(prompt.temperature).invoke(prompt.text)
    plano = resp.content if hasattr(resp, "content") else str(resp)
    print("   Plano tecnico atualizado.")
    return {"plano_atual": plano, "status": "plano_tecnico_refinado"}


def finalizar_plano_tecnico_node(state: RevisaoState) -> dict:
    """Gera o plano técnico final e salva em Markdown."""
    tema       = state["tema"]
    plano_curr = truncar(state["plano_atual"], 1000)
    snips      = fmt_snippets(state.get("snippets_tecnicos", [])[:8], 800)
    urls       = state.get("urls_tecnicos", [])
    prompt = load_prompt("technical/finalizar_plano", snips=snips)
    resp        = get_llm(prompt.temperature).invoke(prompt.text)
    plano_final = resp.content if hasattr(resp, "content") else str(resp)
    print("\n" + "=" * 70)
    print("PLANO FINAL — REVISAO TECNICA")
    print("=" * 70)
    print(plano_final)
    print("=" * 70)
    urls_md = "\n".join("- " + u for u in urls[:30])
    md = (
        "# Plano de Revisao Tecnica\n\n"
        "**Tema:** " + tema + "\n\n"
        + plano_final +
        "\n\n## URLs Tecnicas Identificadas\n\n" + urls_md + "\n"
    )
    path = salvar_md(md, "plans/plano_revisao_tecnica", tema)
    return {"plano_final": plano_final, "plano_final_path": path, "status": "concluido"}
