# Guia MLflow — Rastreamento de Experimentos

Este guia explica como usar o MLflow para rastrear experimentos nos workflows do Revisão Agents.

---

## O que é rastreado

O MLflow é configurado no pacote `src/revisao_agents/observability/` e rastreia:

| Experimento             | Workflow associado                        |
|-------------------------|-------------------------------------------|
| `planning_academic`     | Planejamento de revisão acadêmica         |
| `planning_technical`    | Planejamento de revisão técnica           |
| `writing_academic`      | Escrita de revisão acadêmica              |
| `writing_technical`     | Escrita de revisão técnica                |
| `review_chat`           | Interações de revisão interativa (chat)   |
| `cost_reports`          | Relatórios agregados de custo Tavily (`scripts/generate_cost_report.py`, ver seção [Relatório de custos](#relatório-de-custos-w8-story-04)) |

Métricas registradas em cada busca Tavily individual (run filho, aninhado via `_tavily_search_span`):

- `latency` — tempo de resposta da busca (segundos)
- `credits_used` — créditos Tavily consumidos nessa busca
- `urls_found` — total de URLs encontradas
- `valid_academic_urls_found` / `valid_technical_urls_found` — URLs que passam pelo filtro de domínio

Params/metrics registrados no run pai (sessão de planejamento completa, via `workflow_run`):

- `params.search_depth` — profundidade de busca da sessão (`basic`/`advanced`), constante durante toda a sessão
- `metrics.total_credits_used` — créditos Tavily acumulados na sessão inteira, logado ao final (`finalize_*_plan_node`)

---

## Iniciando o servidor MLflow

```bash
make mlflow-start
```

O servidor sobe em `http://127.0.0.1:5000` com banco de dados SQLite local.

Para personalizar porta ou backend:

```bash
MLFLOW_PORT=8080 make mlflow-start
MLFLOW_BACKEND_STORE_URI=sqlite:///./meu_backend.db make mlflow-start
```

---

## Variáveis de ambiente

| Variável               | Padrão                            | Descrição                                   |
|------------------------|-----------------------------------|---------------------------------------------|
| `MLFLOW_TRACKING_URI`  | `sqlite:///./runtime/mlruns/mlflow.db`    | URI do backend de rastreamento              |
| `MLFLOW_HOST`          | `127.0.0.1`                       | Host para o servidor MLflow UI              |
| `MLFLOW_PORT`          | `5000`                            | Porta para o servidor MLflow UI             |
| `MLFLOW_BACKEND_STORE_URI` | `sqlite:///./runtime/mlruns/mlflow.db` | URI do backend para o comando `make mlflow-start` |

Configure no `.env`:

```dotenv
MLFLOW_TRACKING_URI=sqlite:///./runtime/mlruns/mlflow.db
```

---

## Estrutura do pacote observability/

```
src/revisao_agents/observability/
├── __init__.py          # re-exporta initialize_experiments
├── mlflow_config.py     # constantes e leitura de variáveis de ambiente
└── mlflow_tracking.py   # inicialização de experimentos
```

**Regra de isolamento:** nenhum módulo dentro de `observability/` importa de outros módulos de `revisao_agents` além de si mesmo.
Toda leitura de variáveis de ambiente é feita via `os.getenv` em `mlflow_config.py`.

---

## Inicialização automática de experimentos

Os experimentos são criados automaticamente ao iniciar a aplicação:

- **CLI** (`uv run revisao-agents`): chama `initialize_experiments()` no início do comando `main`
- **UI** (`python run_ui.py`): chama `initialize_experiments()` antes de iniciar o Gradio

A função é idempotente — segura para chamar múltiplas vezes.

---

## Adicionando rastreamento em novos workflows

Para rastrear métricas em um novo workflow:

```python
import mlflow
from revisao_agents.observability.mlflow_config import EXP_WRITING_ACADEMIC  # use a constante

mlflow.set_experiment(EXP_WRITING_ACADEMIC)
with mlflow.start_run(run_name="nome-do-run"):
    mlflow.log_param("modelo", llm_model)
    mlflow.log_metric("latencia_total", latency)
    mlflow.log_metric("secoes_geradas", num_sections)
```

Use sempre as constantes de `revisao_agents.observability.mlflow_config` (ex.: `EXP_PLANNING_ACADEMIC`) em vez de strings literais.

---

## Visualizando runs

1. Inicie o servidor: `make mlflow-start`
2. Acesse: `http://127.0.0.1:5000`
3. Selecione o experimento desejado no painel lateral
4. Compare runs por métricas ou parâmetros

## Runs de baseline

O script `scripts/run_baseline_mlflow.py` cria um run de referência em cada experimento com os parâmetros de configuração padrão. Nenhuma métrica é registrada — runs de baseline são marcados com a tag `baseline=true` e podem ser excluídos de agregações:

```bash
uv run python scripts/run_baseline_mlflow.py
```

Para excluir baseline de consultas MLflow:

```python
mlflow.search_runs(filter_string="tags.baseline != 'true'")
```

---

## Rastreamento com MLflow Tracing API

O pacote `observability` ativa automaticamente o rastreamento de chamadas LLM via `mlflow.langchain.autolog()`.

### Como funciona

`enable_tracing()` é chamado internamente por `initialize_experiments()` na inicialização da aplicação. Ela instrui o MLflow a interceptar **toda** chamada LangChain automaticamente, capturando:

- Nome do modelo (ex.: `gpt-4o-mini`)
- Tokens de entrada e saída
- Latência da chamada
- Prompt enviado e resposta recebida

Nenhuma alteração nos arquivos de nós é necessária para capturar chamadas LLM — a instrumentação é automática.

### Decoradores `@mlflow.trace` nos nós

Além do autolog, os nós críticos do grafo são instrumentados com `@mlflow.trace` para criar spans hierárquicos visíveis na aba **Traces** do servidor MLflow:

| Nó | `span_type` |
|---|---|
| `interview_node`, `identify_and_refine_node` | `AGENT` |
| `initial_academic_plan_node`, `refine_academic_plan_node`, `finalize_academic_plan_node` | `AGENT` |
| `initial_technical_plan_node`, `refine_technical_plan_node`, `finalize_technical_plan_node` | `AGENT` |
| `write_sections_node` | `AGENT` |
| `_thought_phase`, `_observation_phase`, `_draft_phase` | `CHAIN` |
| `search_tavily_incremental` | `TOOL` |

### Visualizando traces

1. Acesse `http://localhost:5000` com o servidor MLflow ativo.
2. Selecione o experimento desejado (ex.: `planning_academic`).
3. Clique na aba **Traces** para ver a árvore de spans de cada execução.
4. Use o filtro `run_id` para correlacionar traces com métricas do mesmo run.

### Adicionando traces a novos nós

```python
import mlflow

@mlflow.trace(name="nome_do_no", span_type="AGENT")
def meu_novo_no(state: ReviewState) -> dict:
    ...
```

`span_type` deve ser um dos valores: `"AGENT"`, `"CHAIN"`, `"TOOL"`, `"UNKNOWN"`.

---

## Relatório de custos (W8-STORY-04)

O script `scripts/generate_cost_report.py` agrega o gasto de créditos Tavily por sessão de planejamento, agrupado por `review_type` (acadêmico/técnico) e `search_depth` (`basic`/`advanced`):

```bash
uv run python scripts/generate_cost_report.py
```

**De onde vêm os dados:** apenas runs **pai** das sessões de planejamento (`planning_academic`, `planning_technical`) — identificados por não ter a tag `tags.mlflow.parentRunId` — são considerados. Métricas de buscas individuais (runs filhos) não entram na agregação porque `search_depth` é uma config de sessão inteira (constante em todas as buscas daquela execução), já espelhada como param do run pai.

**O que é excluído da agregação, e por quê:**

| Critério | Motivo |
|---|---|
| `status != "FINISHED"` | Sessão que não terminou não tem `total_credits_used` final confiável |
| Falta `metrics.total_credits_used` ou `params.search_depth` | Run anterior à instrumentação de custo (não é problema de qualidade, é dado histórico) |
| `tags.baseline == "true"` | Run de referência sem uso real de Tavily (ver `scripts/run_baseline_mlflow.py`) |

> **Nota:** sessões de `planning_academic` sempre serão excluídas do relatório de custos — esse é o comportamento esperado, não um bug. O workflow acadêmico busca fontes via vector search no MongoDB, não via Tavily, então nunca registra `metrics.total_credits_used`. Apenas sessões de `planning_technical` aparecem na agregação de custos.

**Onde o relatório fica:** nenhum arquivo é salvo em disco. O script cria um run no experimento `cost_reports` e usa `mlflow.log_text` para anexar `cost_report.md` e `cost_summary.csv` como artefatos do run — visíveis na aba **Artifacts** da UI do MLflow. As métricas agregadas por grupo (`avg_credits_per_session__<review_type>_<search_depth>`, etc.) também ficam no run, pra comparar entre execuções do script ao longo do tempo na própria tabela de runs.

**Visualização depth vs. custo:** a UI do MLflow não tem painéis customizados tipo Grafana. Para comparar profundidade de busca vs. custo entre sessões individuais, use o recurso nativo de comparação de runs:

1. Abra `http://localhost:5000` e selecione o experimento `planning_academic` ou `planning_technical`
2. Selecione duas ou mais sessões (checkbox) e clique em **Compare**
3. Use o gráfico de **Parallel Coordinates**, com `search_depth` e `total_credits_used` como eixos

## Próximos passos (Semana 9+)

- Módulo de avaliação em `src/revisao_agents/evaluation/`
