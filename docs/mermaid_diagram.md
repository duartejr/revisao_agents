# Mermaid Diagrams

## Academic Review Graph

```mermaid
flowchart TD
    A([Start]) --> B[consulta_vetorial]
    B --> C[plano_inicial_academico]
    C --> D{pausa_humana\nHITL}
    D --> E[entrevista]
    E -->|refinar| F[refinar_consulta_academico]
    E -->|finalizar| G[finalizar_plano_academico]
    F --> H[refinar_plano_academico]
    H --> D
    G --> Z([END])
```

## Technical Chapter Graph

```mermaid
flowchart TD
    A([Start]) --> B[busca_tecnica_inicial]
    B --> C[plano_inicial_tecnico]
    C --> D{pausa_humana\nHITL}
    D --> E[entrevista]
    E -->|refinar| F[refinar_busca_tecnica]
    E -->|finalizar| G[finalizar_plano_tecnico]
    F --> H[refinar_plano_tecnico]
    H --> D
    G --> Z([END])
```

## Package dependency graph

```mermaid
graph LR
    CLI[cli.py] --> GR[graphs/review_graph]
    GR --> AG[agents/*]
    GR --> CP[graphs/checkpoints]
    AG --> PL[utils/prompt_loader]
    AG --> LLM[utils/llm_providers]
    AG --> TS[tools/*]
    PL --> PR[prompts/**/*.yaml]
    AG --> ST[state.py]
    CLI --> CFG[config.py]
```
