from state import RevisaoState
from config import get_llm, CHUNKS_PER_QUERY
from utils.vector_store import buscar_chunks, acumular_chunks
from utils.helpers import fmt_chunks, truncar, salvar_md

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
    prompt = (
        "Pesquisador senior elaborando plano inicial de revisao da literatura sobre:\n"
        + repr(tema) + "\n\n"
        "Trechos dos artigos:\n" + ctx + "\n\n"
        "Proponha plano de revisao narrativa baseado no conteudo real dos artigos.\n"
        "Estrutura:\n"
        "- Objetivo (1 frase)\n"
        "- 4 a 6 secoes (titulo + 1 frase de justificativa)\n"
        "- 2 lacunas visiveis\n"
        "- 2 perguntas de pesquisa\n\n"
        "Use Markdown. Este e um rascunho a ser refinado."
    )
    resp  = get_llm(0.5).invoke(prompt)
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
    prompt = (
        "Refinando plano de revisao da literatura sobre " + repr(tema) + "\n\n"
        "Plano atual:\n" + plano_curr + "\n\n"
        "Pesquisador respondeu:\n\"" + ultima + "\"\n\n"
        "Novos trechos dos artigos:\n" + ctx_novo + "\n\n"
        "Atualize o plano atendendo a instrucao. Retorne plano completo em Markdown.\n"
        "Ultima linha obrigatoria: Alteracao: <resumo do que mudou>"
    )
    resp  = get_llm(0.4).invoke(prompt)
    plano = resp.content if hasattr(resp, "content") else str(resp)
    print("   Plano academico atualizado.")
    return {"plano_atual": plano, "status": "plano_refinado"}

def finalizar_plano_academico_node(state: RevisaoState) -> dict:
    """Gera o plano acadêmico final e salva em Markdown."""
    tema       = state["tema"]
    plano_curr = truncar(state["plano_atual"], 1000)
    ctx        = fmt_chunks(state["chunks_relevantes"], 800)
    prompt = (
        "Tema: " + tema + "\n\n"
        "Plano refinado:\n" + plano_curr + "\n\n"
        "Base de evidencias:\n" + ctx + "\n\n"
        "Gere o PLANO FINAL estruturado em Markdown com:\n"
        "## Objetivo\n"
        "## Perguntas de Pesquisa\n"
        "## Estrategia de Busca\n"
        "## Estrutura da Revisao\n"
        "(tabela: | **N. Titulo** | Objetivo | Topicos |)\n"
        "## Lacunas Identificadas\n"
        "## Notas Metodologicas\n\n"
        "Seja detalhado e cientifico."
    )
    resp        = get_llm(0.2).invoke(prompt)
    plano_final = resp.content if hasattr(resp, "content") else str(resp)
    print("\n" + "=" * 70)
    print("PLANO FINAL — REVISAO ACADEMICA")
    print("=" * 70)
    print(plano_final)
    print("=" * 70)
    md   = "# Plano de Revisao da Literatura\n\n**Tema:** " + tema + "\n\n" + plano_final
    path = salvar_md(md, "plano_revisao", tema)
    return {"plano_final": plano_final, "plano_final_path": path, "status": "concluido"}