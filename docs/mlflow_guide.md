# Guia MLflow — Rastreamento de Experimentos

Este guia explica como usar o MLflow para rastrear experimentos nos workflows do Revisão Agents.

---

## O que é rastreado

O MLflow é configurado no pacote `observability/` (isolado da lógica de execução) e rastreia:

| Experimento             | Workflow associado                        |
|-------------------------|-------------------------------------------|
| `planning_academic`     | Planejamento de revisão acadêmica         |
| `planning_technical`    | Planejamento de revisão técnica           |
| `writing_academic`      | Escrita de revisão acadêmica              |
| `writing_technical`     | Escrita de revisão técnica                |
| `review_chat`           | Interações de revisão interativa (chat)   |

Métricas registradas atualmente em buscas Tavily incrementais:

- `latency` — tempo de resposta da busca (segundos)
- `credits_used` — créditos Tavily consumidos
- `urls_found` — total de URLs encontradas
- `valid_academic_urls_found` — URLs que passam pelo filtro acadêmico

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
| `MLFLOW_TRACKING_URI`  | `sqlite:///./mlruns/mlflow.db`    | URI do backend de rastreamento              |
| `MLFLOW_HOST`          | `127.0.0.1`                       | Host para o servidor MLflow UI              |
| `MLFLOW_PORT`          | `5000`                            | Porta para o servidor MLflow UI             |
| `MLFLOW_BACKEND_STORE_URI` | `sqlite:///./mlruns/mlflow.db` | URI do backend para o comando `make mlflow-start` |

Configure no `.env`:

```dotenv
MLFLOW_TRACKING_URI=sqlite:///./mlruns/mlflow.db
```

---

## Estrutura do pacote observability/

```
observability/
├── __init__.py          # re-exporta initialize_experiments
├── mlflow_config.py     # constantes e leitura de variáveis de ambiente
└── mlflow_tracking.py   # inicialização de experimentos
```

**Regra de isolamento:** nenhum módulo dentro de `observability/` importa de `src/revisao_agents`.
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
from observability.mlflow_config import EXP_WRITING_ACADEMIC  # use a constante

mlflow.set_experiment(EXP_WRITING_ACADEMIC)
with mlflow.start_run(run_name="nome-do-run"):
    mlflow.log_param("modelo", llm_model)
    mlflow.log_metric("latencia_total", latency)
    mlflow.log_metric("secoes_geradas", num_sections)
```

Use sempre as constantes de `observability.mlflow_config` (ex.: `EXP_PLANNING_ACADEMIC`) em vez de strings literais.

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

## Próximos passos (Semana 9+)

- Integração com nós individuais dos grafos LangGraph
- Módulo de avaliação em `src/revisao_agents/evaluation/`
