# Architecture

## Overview

```
revisao_agents/
├── run_ui.py              ← Ponto de entrada da UI Gradio (porta 7860)
├── scripts/
│   ├── bootstrap.sh       ← Bootstrap interativo Linux/macOS
│   └── bootstrap.ps1      ← Bootstrap interativo Windows PowerShell
├── src/
│   ├── gradio_app/        ← Interface gráfica Gradio
│   │   ├── app.py         ← Definição das abas e componentes
│   │   └── handlers/      ← Lógica de negócio das abas (subpacote)
│   │       ├── __init__.py          ← Re-exports de todos os handlers públicos
│   │       ├── base.py              ← Utilitários LLM e I/O de arquivos
│   │       ├── planning.py          ← Planejamento e gerenciamento de threads
│   │       ├── review.py            ← Orquestração da revisão interativa
│   │       ├── writing.py           ← Workflow de escrita
│   │       ├── tools.py             ← Indexação de PDFs e formatação de referências
│   │       └── review_parts/        ← Sub-módulos da revisão
│   │           ├── document.py      ← Listagem de arquivos de revisão
│   │           ├── intent.py        ← Detecção de intenção do usuário
│   │           ├── images.py        ← Máquina de estados de sugestão de imagens
│   │           ├── reviewer.py      ← Journal Reviewer Mode (relatório + follow-up, somente leitura)
│   │           └── references.py   ← Pipeline de enriquecimento de referências
│   └── revisao_agents/    ← Pacote principal
│       ├── agents/        ← Agentes de chamada de ferramentas e de saída estruturada (review_agent, image_suggestion, reference_*, journal_reviewer_agent); não são nós do LangGraph
│       ├── graphs/        ← StateGraph definitions
│       ├── nodes/         ← Nós especializados (escrita por seções, verificação)
│       ├── workflows/     ← Montagem de workflows (academic, technical, writing)
│       ├── tools/         ← LangChain @tool wrappers (busca, referências, web)
│       ├── prompts/       ← Templates YAML de prompts versionados
│       ├── core/
│       │   └── schemas/   ← Pydantic models compartilhados (inclui journal_review.py)
│       ├── utils/         ← Utilitários (llm_providers, vector_store, tavily, …)
│       ├── observability/ ← Rastreamento MLflow (experimentos, métricas)
│       ├── evaluation/    ← Avaliadores de qualidade de busca/snippets
│       ├── config.py      ← Configuração via pydantic-settings + .env
│       ├── state.py       ← TypedDict de estado do LangGraph
│       ├── hitl.py        ← Nó Human-in-the-Loop
│       ├── cli.py         ← CLI Typer (entrypoint: revisao-agents)
│       └── __main__.py    ← Menu interativo (python -m revisao_agents)
├── tests/
├── docs/
├── learning/               ← Prática/aprendizado: examples/, notebooks/, scripts manuais
├── management/             ← Roadmap, sprints, relatórios (não versionado)
├── runtime/                ← Saída gerada em runtime: plans/, reviews/, caches, checkpoints (não versionado)
└── .env.example           ← Template de configuração
```

## Pontos de entrada

| Modo | Comando | Porta/Saída |
|------|---------|-------------|
| UI Gradio | `uv run python run_ui.py` | http://localhost:7860 |
| CLI script | `uv run revisao-agents [TEMA]` | stdout + `$PLANS_DIR` (padrão: `runtime/plans/`) |
| Menu interativo | `uv run python -m revisao_agents` | stdout + `$PLANS_DIR` + `$REVIEWS_DIR` |

## Abas da UI

| Aba | Workflow Acionado | Saída |
|-----|-------------------|-------|
| 📋 Plan | `build_review_graph(academic/technical)` | `runtime/plans/*.md` |
| ✍️ Write | `build_technical_writing_workflow()` | `runtime/reviews/*.md` |
| 🤖 Revisão Interativa | `ReviewAgent` (ReAct loop) | edição do arquivo |
| 📁 Index PDFs | `ingest_pdf_folder()` | MongoDB chunks |
| 📚 References | `run_reference_formatter()` | markdown formatado |
| 📄 View | leitura direta de arquivo | renderização local |

## Persistência e Checkpointing

Os workflows de planejamento (acadêmico e técnico) suportam checkpointing opcional via parâmetro `checkpointer` nas funções `build_academic_workflow()` e `build_technical_workflow()`. Isso permite:

- **Persistência de estado**: Salvar o progresso do workflow em memória, SQLite ou PostgreSQL.
- **Resumo de execuções interrompidas**: Capacidade de retomar workflows pausados (devido a interrupções manuais ou erros).
- **Configuração via `.env`**: O tipo de checkpointer é controlado pela variável `CHECKPOINT_TYPE` (memory/sqlite/postgres).

Por padrão, usa `MemorySaver` (não persistente). Para persistência, instale dependências extras e configure `CHECKPOINT_TYPE=sqlite` ou `postgres`.

---

## Fluxo de dados — Planejamento

```
UI (📋 Plan) / CLI revisao-agents
        │
        ▼
handlers/planning.py  (start_planning / continue_planning / load_thread_state)
        │
        ▼
graphs/review_graph.py  ←─ build_review_graph(tipo)
        │
        ▼ LangGraph StateGraph (HITL)
        │
        ├─► workflows/academic_workflow.py
        │       identify_and_refine → [clarify: human_pause → post_pause_router]
        │       → consulta_vetorial → plano_inicial → interview_router
        │       → refinamento HITL → finalizar_plano
        │
        └─► workflows/technical_workflow.py
                identify_and_refine → [clarify: human_pause → post_pause_router]
                → busca_tecnica → plano_inicial → interview_router
                → refinamento HITL → finalizar_plano
                     │
                     └─► tools/ (busca vetorial, Tavily, referências)
```

## Fluxo de dados — Escrita

```
UI (✍️ Write) / python -m revisao_agents [3]
        │
        ▼
workflows/technical_writing_workflow.py
        │
        ▼ LangGraph StateGraph (sem HITL)
        │
        ├─► nodes/writing/parse_plan_node.py     (lê plan.md → lista de seções)
        ├─► nodes/writing/write_sections_node.py (ReAct por seção)
        │       │
        │       ├─► tools/review_tools.py        (busca vetorial MongoDB)
        │       ├─► tools/review_tools.py        (Tavily web search, se ativado)
        │       └─► utils/llm_utils/             (LLM provider)
        │
        └─► nodes/writing/verification.py        (validação de fontes)
```

## Fluxo de dados — Revisão Interativa

```
UI (🤖 Revisão Interativa)
        │
        ▼
handlers/review.py  (review_chat_turn / start_review_session)
        │
        ├─► review_parts/intent.py     (detecção de intenção)
        ├─► review_parts/images.py     (sugestão de imagens)
        ├─► review_parts/references.py (enriquecimento de referências)
        ├─► review_parts/reviewer.py   (Journal Reviewer Mode — ramo antecipado, ver abaixo)
        ├─► agents/review_agent.py     (ReAct loop, MAX_ITERATIONS=6)
        └─► tools/review_tools.py      (Tavily, se web ativado)
```

### Journal Reviewer Mode

Com o checkbox **🧑‍⚖️ Journal Reviewer Mode** ligado, `review_chat_turn(..., reviewer_mode=True)` desvia para
`review_parts/reviewer.py::handle_reviewer_turn` **antes** de qualquer classificador de referência/edição
(retorno único). O ramo é somente leitura: não escreve a cópia de trabalho nem toca em `pending_edit`.

```
handle_reviewer_turn
   ├─ sem relatório guardado (ou "gerar novamente")
   │     └─► agents/journal_reviewer_agent.run_journal_review
   │            ├─ build_review_context  (orçamento JOURNAL_REVIEW_MAX_CHARS + marcador de truncamento)
   │            ├─ prompts/common/journal_reviewer.yaml
   │            └─ config.llm_call(response_schema=JournalReviewReport)   ← sem ferramentas
   │         → session_state["journal_review_report"] / ["journal_review_markdown"] / ["journal_review_doc_hash"]
   └─ com relatório guardado (follow-up)
         └─► agents/review_agent.run_review_agent(system_prompt_override=journal_reviewer_followup.yaml)
              (ferramentas de evidência ativas; marcadores de edição descartados;
               aviso de relatório desatualizado se o hash do documento mudou)
```

A detecção de "gerar novamente" é uma regex que exige que a **mensagem inteira** seja o pedido (verbo imperativo + substantivo de relatório, ou "novamente/again", com "please/por favor" opcional),
de modo que perguntas *sobre* o relatório não o regeneram.

O esquema (`core/schemas/journal_review.py`) é o contrato; cada geração gera um run no experimento MLflow
`journal_review` (ver [guia do MLflow](mlflow_guide.md)). Os relatórios vivem apenas no `session_state`.

## Provedores LLM suportados

| Provider | `LLM_PROVIDER` | Variável de chave | Modelo padrão |
|---------|---------|-----------|---------|
| OpenAI | `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| Google Gemini | `google` | `GOOGLE_API_KEY` | `gemini-2.5-flash` |
| Groq | `groq` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` | configurável |

> OpenAI é **sempre** necessária para embeddings (`text-embedding-3-small`), independente do provedor LLM.

## Prompt versioning

All prompts live as YAML files under `prompts/`. Each file has:

```yaml
name: plano_inicial_academico
version: "1.0"
temperature: 0.5
system: |
  … {tema} … {ctx} …
```

Loaded at runtime via `utils/prompt_loader.load_prompt(path, **vars)`.
