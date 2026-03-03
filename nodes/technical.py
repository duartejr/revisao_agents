from state import RevisaoState
from config import get_llm
from utils.tavily_client import buscar_conteudo_tecnico
from utils.helpers import fmt_snippets, truncar, salvar_md

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
    prompt = f"""
        # PERSONA
    Atue como um Editor-Chefe Técnico e Arquiteto de Conteúdo Acadêmico. Sua tarefa é criar o "Blueprint" (Plano Detalhado) para um capítulo de nível de pós-graduação/profissional sobre o tema abaixo.

    # INPUTS
    - TEMA: {tema}
    - REFERÊNCIAS: {snippets}

    # OBJETIVO DO PLANO
    Este plano deve servir como um guia rigoroso para a redação posterior. O capítulo não deve ser uma introdução, mas um aprofundamento técnico.

    # ESTRUTURA DO PLANO (O que você deve entregar):

    1. OBJETIVO PEDAGÓGICO: Qual competência técnica avançada o leitor terá ao final? (1 parágrafo).
    2. PRÉ-REQUISITOS: O que o leitor já deve saber (Cálculo, Física, Programação, etc.) para não ser interrompido.
    3. TAXONOMIA DO CAPÍTULO (O CORAÇÃO DO PLANO):
    - Divida o tema em 5 a 8 seções principais.
    - Para CADA seção, detalhe:
        * Sub-itens (H3) que cobrem a teoria, o mecanismo e a aplicação.
        * Elementos Formais: Liste quais equações (ex: Navier-Stokes, Black-Scholes), algoritmos ou normas técnicas (ISO/ABNT) DEVEM ser incluídos.
        * Dados/Tabelas: Descreva quais dados comparativos a seção deve trazer.
    4. ESTRATÉGIA VISUAL: Descreva detalhadamente quais figuras, diagramas de blocos ou gráficos de dispersão são necessários para a compreensão.
    5. CASO DE ESTUDO/APLICAÇÃO: Proponha um problema prático complexo que será resolvido ao final do capítulo para consolidar o conhecimento.

    # DIRETRIZES POR ÁREA (ADAPTABILIDADE)
    - Se for Engenharia/Química: Foque em balanço de massa/energia, propriedades termodinâmicas e diagramas de processo.
    - Se for TI/Algoritmos: Foque em complexidade assintótica, estruturas de dados e provas de convergência.
    - Se for Finanças: Foque em modelos estocásticos, análise de risco e séries temporais.

    Use Markdown. Seja exaustivo na subdivisão dos tópicos para evitar lacunas de conhecimento.
    """
    resp  = get_llm(0.4).invoke(prompt)
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
    prompt = f"""
    # PERSONA
    Atue como Editor Sênior de Publicações Técnicas. Sua missão é REFINAR e APROFUNDAR o blueprint de um capítulo técnico, integrando feedbacks específicos e novas evidências científicas.

    # CONTEXTO ATUAL
    - TEMA: {tema}
    - PLANO ANTERIOR: {plano_curr}
    - FEEDBACK DO PESQUISADOR: "{ultima}"
    - NOVAS FONTES TÉCNICAS (SNIPPETS): {snips}

    # DIRETRIZES DE REFINAMENTO
    1. INTEGRAÇÃO ORGÂNICA: Não apenas "anexe" as novas informações. Reorganize as seções se o feedback ou as novas fontes alterarem a hierarquia lógica do tema.
    2. ESPECIFICAÇÃO DE RIGOR:
    - Se o feedback exigir MATEMÁTICA: Nomeie as equações específicas (ex: "Incluir Derivação da Transformada de Fourier") e descreva as variáveis principais em LaTeX.
    - Se exigir ALGORITMOS: Defina a lógica (ex: "Inserir Pseudocódigo de Otimização por Enxame de Partículas") e complexidade (Big O).
    - Se exigir PROCESSOS: Detalhe etapas de entrada/saída, normas técnicas (ISO, ASTM, IEEE) ou métricas de performance.
    3. CONSISTÊNCIA: Mantenha o tom acadêmico de alto nível e a estrutura de sub-itens (H3) detalhados.

    # ESTRUTURA DE SAÍDA
    1. PLANO DE CAPÍTULO ATUALIZADO (Markdown completo, incluindo as seções que não mudaram para manter o contexto).
    2. Justificativa Técnica: Explique brevemente por que a nova estrutura é superior.
    3. ÚLTIMA LINHA OBRIGATÓRIA: Alteração: <resumo executivo do que foi modificado ou adicionado>.
    """
    resp  = get_llm(0.4).invoke(prompt)
    plano = resp.content if hasattr(resp, "content") else str(resp)
    print("   Plano tecnico atualizado.")
    return {"plano_atual": plano, "status": "plano_tecnico_refinado"}

def finalizar_plano_tecnico_node(state: RevisaoState) -> dict:
    """Gera o plano técnico final e salva em Markdown."""
    tema       = state["tema"]
    plano_curr = truncar(state["plano_atual"], 1000)
    snips      = fmt_snippets(state.get("snippets_tecnicos", [])[:8], 800)
    urls       = state.get("urls_tecnicos", [])
    prompt = f"""
    # PERSONA
    Diretor Editorial de uma Editora Científica. Seu objetivo é criar um "MAPA DE EXECUÇÃO GRANULAR".

    # TAREFA
    Gere o PLANO FINAL em Markdown. Para garantir a extensão do livro, você deve quebrar cada capítulo em SUBSEÇÕES detalhadas.

    ## 1. Estrutura Hierárquica (Matriz de Conteúdo)
    | Nível | Título Técnico | Conteúdo Detalhado e Objetivos de Escrita | Recursos (Equações/Algoritmos) |
    | :--- | :--- | :--- | :--- |
    | 1.0 | Introdução | Contextualização e importância... | - |
    | 2.0 | [Título Seção] | Visão geral do conceito... | - |
    | 2.1 | [Subseção] | Derivação detalhada de... (Mínimo 5 parágrafos) | Equação de [X] |
    | 2.2 | [Subseção] | Análise comparativa de... (Mínimo 5 parágrafos) | Tabela de [Y] |

    ## 2. Protocolo de Citação e Fontes
    - Utilize APENAS as fontes abaixo: {snips}
    - Cada subseção DEVE prever o uso de citações numeradas [1], [2] baseadas nestas fontes.

    ## 3. Inventário de Recursos (Checklist para o Escritor)
    - **Equações:** (Liste em LaTeX)
    - **Algoritmos:** (Descreva a lógica do pseudocódigo)
    - **Figuras:** (Descreva o que o gráfico/diagrama deve mostrar)

    **Instrução de Extensão:** Planeje pelo menos 3 subseções (X.1, X.2, X.3) para cada seção principal. O objetivo é que cada subseção se torne um texto exaustivo.
    """
    resp        = get_llm(0.2).invoke(prompt)
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
    path = salvar_md(md, "plano_revisao_tecnica", tema)
    return {"plano_final": plano_final, "plano_final_path": path, "status": "concluido"}