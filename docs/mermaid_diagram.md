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
    APP[gradio_app/app.py] --> H[handlers/__init__]
    H --> HB[handlers/base.py]
    H --> HP[handlers/planning.py]
    H --> HR[handlers/review.py]
    H --> HW[handlers/writing.py]
    H --> HT[handlers/tools.py]
    HR --> RP[handlers/review_parts/]
    RP --> RPD[document.py]
    RP --> RPI[intent.py]
    RP --> RPIM[images.py]
    RP --> RPR[references.py]
    HP --> GR[graphs/review_graph]
    HP --> CP[graphs/checkpoints]
    HR --> AG[agents/review_agent]
    HW --> WF[workflows/*]
    HT --> TS[tools/*]
    GR --> AG2[agents/*]
    AG --> LLM[utils/llm_providers]
    AG --> PL[utils/prompt_loader]
    AG --> TS
    PL --> PR[prompts/**/*.yaml]
    AG --> ST[state.py]
    APP --> CFG[config.py]
```
