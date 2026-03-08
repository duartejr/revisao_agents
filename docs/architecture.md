# Architecture

## Overview

```
revisao_agents/
├── src/revisao_agents/
│   ├── agents/          ← LangGraph node functions (one file per workflow type)
│   ├── graphs/          ← StateGraph definitions that wire agents together
│   ├── prompts/         ← YAML prompt templates (versioned, readable)
│   ├── tools/           ← LangChain tools (search, retrieval, …)
│   ├── core/
│   │   └── schemas/     ← Pydantic models shared across layers
│   ├── utils/           ← Pure helpers (logging, LLM factory, vector store, …)
│   ├── config.py        ← pydantic-settings + .env
│   ├── state.py         ← TypedDict state definitions
│   ├── hitl.py          ← Human-in-the-Loop node
│   └── cli.py           ← Typer CLI entry point
├── tests/
│   ├── unit/
│   └── integration/
├── examples/
└── docs/
```

## Data flow

```
CLI / __main__
     │
     ▼
graphs/review_graph.py  ←─ build_review_graph(tipo)
     │                           │
     │  StateGraph.compile()     │
     ▼                           ▼
LangGraph runtime          graphs/checkpoints.py
     │
     ├─► agents/academic.py     (consulta_vetorial → plano_inicial → refinar → finalizar)
     ├─► agents/technical.py    (busca_tecnica → plano_inicial → refinar → finalizar)
     ├─► agents/common.py       (pausa_humana ↔ entrevista ↔ roteador)
     └─► agents/technical_writing.py
              │
              ├─► tools/           (LangChain @tool wrappers)
              ├─► prompts/*.yaml   (loaded via utils/prompt_loader.py)
              └─► utils/           (llm_providers, vector_store, tavily_client, …)
```

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
