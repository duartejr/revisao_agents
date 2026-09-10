# Guia de Métricas de Avaliação

> **Público:** Usuários e contribuidores que precisam interpretar ou estender as métricas de qualidade do Revisão Agents
> **Funcionalidade:** `src/revisao_agents/observability/search_metrics.py` (métricas automáticas) e `src/revisao_agents/evaluation/` (LLM-as-judge)

Este guia documenta as duas famílias de métricas de qualidade do projeto: métricas automáticas (calculadas por fórmula, sem LLM) e métricas de LLM-as-judge (um segundo LLM avalia a saída do agente). Para como o MLflow em si funciona (servidor, tracing, runs), ver o [Guia MLflow](mlflow_guide.md) — este guia cobre apenas o *significado* das métricas, não a infraestrutura de rastreamento.

---

## Sumário

1. [Visão geral do framework de avaliação](#1-visão-geral-do-framework-de-avaliação)
2. [Métricas automáticas (`SearchQualityMetrics`)](#2-métricas-automáticas-searchqualitymetrics)
3. [Métricas LLM-as-judge](#3-métricas-llm-as-judge)
4. [Limiares de filtragem relacionados](#4-limiares-de-filtragem-relacionados)
5. [Executando avaliações localmente](#5-executando-avaliações-localmente)
6. [Visualizando resultados na UI do MLflow](#6-visualizando-resultados-na-ui-do-mlflow)

---

## 1. Visão geral do framework de avaliação

| Família | Onde vive | Custo | Quando roda |
|---|---|---|---|
| Métricas automáticas | `src/revisao_agents/observability/search_metrics.py` | Nenhum (cálculo local) | A cada busca, dentro do próprio workflow de planejamento |
| LLM-as-judge | `src/revisao_agents/evaluation/` | 1 chamada LLM por julgamento (`gpt-4o-mini`) | Sob demanda, tipicamente em experimentos A/B (não a cada busca de produção) |

As métricas automáticas respondem "quanto/quantos" (cobertura, reuso, créditos). As métricas LLM-as-judge respondem "quão bom" — relevância, qualidade acadêmica, potencial de citação e, desde a W9-STORY-03, qualidade de plano.

---

## 2. Métricas automáticas (`SearchQualityMetrics`)

Calculadas em `SearchQualityMetrics` (classe de métodos estáticos, sem estado) e logadas via `SearchQualityMetrics.log_all_metrics_to_mlflow`. Chamadas a partir de `nodes/technical.py` e `tools/tavily_web_search.py` durante buscas reais — nenhuma chamada de LLM extra é feita.

| Métrica | Fórmula | Intervalo | Interpretação |
|---|---|---|---|
| `search_coverage` | `len(new_urls)` | inteiro ≥ 0 | Quantas URLs *novas* (não vistas antes na sessão) essa busca específica trouxe. Alto = a busca está descobrindo território novo. |
| `result_reuse_percent` | `(URLs vistas mais de uma vez) / (total de aparições) × 100` | 0.0–100.0 | Quanto os resultados se repetem entre buscas da mesma sessão. Alto reuso sugere que buscas subsequentes estão convergindo (ou que as queries são pouco diversificadas). |
| `credit_efficiency_individual` | `credits_used` da busca | float ≥ 0 | Créditos Tavily gastos nessa busca específica. |
| `credit_efficiency_aggregated` | `total_credits_used / total_search_queries` | float ≥ 0 | Créditos médios por busca, acumulado na sessão. Útil para comparar `TAVILY_SEARCH_DEPTH` (ver [Guia de Ajuste do Tavily](tavily_tuning_guide.md)). |

**Onde ver:** métricas por busca individual aparecem no run filho aninhado (`search:<query>`, via `_tavily_search_span`); `total_credits_used` agregado aparece como métrica do run pai da sessão de planejamento.

**Nota de implementação:** `calculate_all_search_metrics` é a função de conveniência que calcula as quatro de uma vez — é o ponto de entrada normal a partir de um nó do grafo, em vez de chamar os quatro métodos estáticos individualmente.

---

## 3. Métricas LLM-as-judge

Implementadas como *judges* do MLflow (`mlflow.genai.judges.make_judge` / `RelevanceToQuery`) em `evaluation/snippet_evaluators.py`, orquestradas por `evaluation/evaluators.py`. Cada julgamento é uma chamada real ao modelo configurado (`openai:/gpt-4o-mini`), então essas métricas têm custo e latência — não rodam em toda busca de produção, apenas quando um fluxo de avaliação é explicitamente executado (ver §5).

### 3.1 Avaliação de snippets (`evaluate_search_snippets`)

Roda três judges por snippet e retorna uma lista de `SnippetEvaluation` (`evaluation/types.py`):

| Campo | Tipo | Judge | Critério |
|---|---|---|---|
| `relevance_level` / `relevance_score` | `"Perfectly relevant"` \| `"Not relevant"` / `1.0` \| `0.0` | `RelevanceToQuery` (built-in do MLflow) | O snippet responde à query do usuário? |
| `academic_quality` | `bool` | `academic_quality` (custom) | Informação tecnicamente sólida **e** de fonte reputável **e** alinhada aos objetivos do usuário — as três condições devem ser verdadeiras. |
| `citation_potential` | `bool` | `citation_potential` (custom) | O snippet é específico, atribuível a uma fonte, autoritativo e preserva sentido fora de contexto — utilizável como citação em um artigo. |

`log_snippet_evaluations_to_mlflow` agrega essas avaliações em métricas por lote: `eval_perfectly_relevant_pct`, `eval_academic_quality_pct`, `eval_citation_potential_pct`, `eval_relevance_avg_score`, entre outras — todas em percentual (0–100) ou score médio (0–1).

### 3.2 Avaliação de plano (`evaluate_plan_quality`, W9-STORY-03)

Judge `plan_quality` (`evaluation/snippet_evaluators.py::get_plan_quality_judge`) que recebe o tema e o texto do plano (truncado aos primeiros 4000 caracteres antes de ser enviado ao judge) e retorna uma nota de **1 a 10** avaliando:

- **Coerência estrutural** — as seções constroem logicamente em direção ao tema
- **Completude** — cobre a amplitude esperada para o tema
- **Especificidade** — evita preenchimento genérico, engaja com o tópico real

`evaluate_plan_quality(theme, plan)` retorna `0.0` quando o plano está vazio, quando o judge retorna um valor que não pode ser convertido para `int`, ou quando o valor convertido cai fora do intervalo [1, 10] — em qualquer desses casos, trate o valor como "sem avaliação", não como "pior nota possível". O judge é configurado com `feedback_value_type=int` (MLflow `make_judge`), que força um `feedback.value` inteiro via structured outputs — não há parsing de texto livre. Chamadas ao judge (erros de API/rede) não são capturadas dentro da própria função; scripts não supervisionados (como o experimento A/B) devem envolver a chamada em `try/except`. Criado para o experimento A/B da camada de refinamento (`scripts/run_refinement_ab_experiment.py`); reutilizável em qualquer contexto que precise comparar planos gerados sob condições diferentes.

---

## 4. Limiares de filtragem relacionados

Três limiares em `src/revisao_agents/config.py` controlam **quais** resultados chegam a ser avaliados/usados, antes de qualquer julgamento de qualidade:

| Constante | Padrão | Efeito |
|---|---|---|
| `SNIPPET_MIN_SCORE` | `0.7` | Corta chunks do corpus MongoDB abaixo desse score de similaridade (`utils/vector_utils/mongodb_corpus.py`). |
| `TAVILY_RESULT_MIN_SCORE` | `0.7` | Corta resultados de busca Tavily abaixo desse `score` (`tools/tavily_web_search.py`). |
| `LANGUAGE_BOOST_EN` | `0.3` | Bônus de score para resultados em inglês antes da reordenação por idioma. |

Os três são configuráveis via variável de ambiente (ver `.env.example`), sem precisar editar código. Detalhes de cada um, com exemplos de ajuste, estão no [Guia de Ajuste do Tavily](tavily_tuning_guide.md#tavily_result_min_score).

---

## 5. Executando avaliações localmente

Não há um comando `make` dedicado — as avaliações rodam como parte de scripts de experimento em `scripts/`, que fazem chamadas reais a LLM/Tavily/MongoDB (consomem API):

```bash
# A/B de profundidade de busca Tavily (fast/basic/advanced) — usa evaluate_search_snippets
uv run python scripts/run_ab_depth_experiment.py

# A/B da camada de refinamento de linguagem/ambiguidade — usa evaluate_plan_quality
uv run python scripts/run_refinement_ab_experiment.py
```

Para avaliar snippets ou um plano isoladamente, sem rodar um experimento completo:

```python
import asyncio
from revisao_agents.evaluation import evaluate_search_snippets, evaluate_plan_quality

evaluations = asyncio.run(evaluate_search_snippets(
    query="transformer attention mechanism",
    snippets=["..."],
    urls=["https://example.com"],
    interview_metadata={"user_goals": "..."},
))

score = asyncio.run(evaluate_plan_quality(theme="...", plan="## Introdução\n..."))
```

Ambos exigem `OPENAI_API_KEY` configurado — os judges usam `openai:/gpt-4o-mini` independentemente de `LLM_PROVIDER` (mesma exigência da geração de embeddings, ver CLAUDE.md "Key Gotchas").

Testes unitários (sem custo de API, tudo mockado): `uv run --extra dev pytest tests/unit/test_scripts/ tests/unit/test_agents/test_identify_and_refine.py`.

---

## 6. Visualizando resultados na UI do MLflow

1. `make mlflow-start` e acesse `http://127.0.0.1:5000`
2. Selecione o experimento relevante:
   - `ab_depth_experiments` — resultados de `run_ab_depth_experiment.py`
   - `planning_refinement_ab` — resultados de `run_refinement_ab_experiment.py`
   - `experiment_reports` — relatórios agregados (`scripts/generate_experiment_report.py`)
3. Métricas numéricas (as tabelas acima) aparecem na coluna de métricas de cada run; artefatos JSON por variante ficam na aba **Artifacts**
4. Valores categóricos (ex.: `language_detected`) aparecem como **params**, não métricas — o MLflow só aceita valores numéricos em `log_metric` (ver [Tracking API](https://mlflow.org/docs/latest/tracking/))

Para instruções gerais de navegação na UI (comparar runs, Parallel Coordinates, etc.), ver o [Guia MLflow, seção "Visualizando runs"](mlflow_guide.md#visualizando-runs).
