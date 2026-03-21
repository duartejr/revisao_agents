# Architecture

## Overview

```
revisao_agent/
├── run_ui.py              ← Ponto de entrada da UI Gradio (porta 7860)
├── scripts/
│   ├── bootstrap.sh       ← Bootstrap interativo Linux/macOS
│   └── bootstrap.ps1      ← Bootstrap interativo Windows PowerShell
├── src/
│   ├── gradio_app/        ← Interface gráfica Gradio
│   │   ├── app.py         ← Definição das abas e componentes
│   │   └── handlers.py    ← Lógica de negócio das abas
│   └── revisao_agents/    ← Pacote principal
│       ├── agents/        ← Nós do LangGraph (funções por workflow)
│       ├── graphs/        ← StateGraph definitions
│       ├── nodes/         ← Nós especializados (escrita por seções, verificação)
│       ├── workflows/     ← Montagem de workflows (academic, technical, writing)
│       ├── tools/         ← LangChain @tool wrappers (busca, referências, web)
│       ├── prompts/       ← Templates YAML de prompts versionados
│       ├── core/
│       │   └── schemas/   ← Pydantic models compartilhados
│       ├── utils/         ← Utilitários (llm_providers, vector_store, tavily, …)
│       ├── config.py      ← Configuração via pydantic-settings + .env
│       ├── state.py       ← TypedDict de estado do LangGraph
│       ├── hitl.py        ← Nó Human-in-the-Loop
│       ├── cli.py         ← CLI Typer (entrypoint: revisao-agents)
│       └── __main__.py    ← Menu interativo (python -m revisao_agents)
├── tests/
├── docs/
├── plans/                 ← Planos gerados pelos workflows
├── reviews/               ← Revisões/capítulos gerados
└── .env.example           ← Template de configuração
```

## Pontos de entrada

| Modo | Comando | Porta/Saída |
|------|---------|-------------|
| UI Gradio | `uv run python run_ui.py` | http://localhost:7860 |
| CLI script | `uv run revisao-agents [TEMA]` | stdout + plans/ |
| Menu interativo | `uv run python -m revisao_agents` | stdout + plans/ + reviews/ |

## Abas da UI

| Aba | Workflow Acionado | Saída |
|-----|-------------------|-------|
| 📋 Plan | `build_review_graph(academic/technical)` | `plans/*.md` |
| ✍️ Write | `build_technical_writing_workflow()` | `reviews/*.md` |
| 🤖 Revisão Interativa | `ReviewAgent` (ReAct loop) | edição do arquivo |
| 📁 Index PDFs | `ingest_pdf_folder()` | MongoDB chunks |
| 📚 References | `run_reference_formatter()` | markdown formatado |
| 📄 View | leitura direta de arquivo | renderização local |

## Fluxo de dados — Planejamento

```
UI (📋 Plan) / CLI revisao-agents
        │
        ▼
graphs/review_graph.py  ←─ build_review_graph(tipo)
        │
        ▼ LangGraph StateGraph (HITL)
        │
        ├─► workflows/academic_workflow.py
        │       consulta_vetorial → plano_inicial → interview_router
        │       → refinamento HITL → finalizar_plano
        │
        └─► workflows/technical_workflow.py
                busca_tecnica → plano_inicial → interview_router
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
agents/review_agent.py  (ReAct loop, MAX_ITERATIONS=6)
        │
        ├─► detecção de intenção (summarize / cite / edit / search)
        ├─► tools/review_tools.py (busca no documento, citações)
        └─► tools/review_tools.py (Tavily, se web ativado)
```

## Provedores LLM suportados

| Provider | `LLM_PROVIDER` | Variável de chave | Modelo padrão |
|---------|---------|-----------|---------|
| OpenAI | `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| Google Gemini | `google` | `GOOGLE_API_KEY` | `gemini-1.5-flash` |
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
