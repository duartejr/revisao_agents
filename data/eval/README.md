# Dataset de Rotulagem Manual de Relevância (W10-STORY-05)

Primeiro dataset de relevância rotulado manualmente neste projeto, usado como
verificação de calibração em relação aos scores de relevância do LLM-as-judge
usados nos experimentos A/B de profundidade Tavily/refinamento
(`evaluation/evaluators.py::evaluate_search_snippets`, que orquestra o judge
definido em `evaluation/snippet_evaluators.py`).

Não havia uma convenção de armazenamento existente neste repositório para um
dataset de avaliação rotulado (ele não é consumido pelo pytest, então
`tests/fixtures/` não fazia sentido). `data/eval/` é um novo diretório de
nível superior para este e futuros datasets de calibração/avaliação.

## Conteúdo

- `relevance_labels_template.csv` — gerado por `scripts/sample_relevance_labels.py`.
  Amostra ~30-50 pares query-resultado de `runtime/search_logs/`, e preenche o
  veredito do LLM judge de cada linha (reexecutado via `evaluate_search_snippets`,
  já que os search logs brutos não persistem o veredito do judge, apenas o
  `Score` do próprio Tavily). A coluna `human_label` começa em branco.

## Diretriz de rotulagem

**Escala — deve corresponder à escala real do LLM judge, não ao schema teórico:**

Conforme `docs/EVALUATION_METRICS_GUIDE.md` §3.1, o judge subjacente
(`RelevanceToQuery`, um judge nativo do MLflow) é um classificador **binário**
sim/não: "Perfectly relevant" (1.0) ou "Not relevant" (0.0). O campo
`relevance_level` do modelo Pydantic `SnippetEvaluation` é tipado como um
`Literal["Perfectly relevant", "Partially relevant", "Not relevant"]` de 3
valores, mas `"Partially relevant"` nunca ocorre de fato na configuração atual
do judge neste código — é uma superfície de schema não utilizada.

Para obter uma comparação justa, rotule cada linha com exatamente um destes
dois valores na coluna `human_label`:

- `relevant` — o snippet responde à query o suficiente para que você o citaria
  ou usaria ao escrever sobre este tema.
- `not_relevant` — não responde.

**Não** introduza uma terceira categoria "parcialmente relevante" — o judge
não consegue produzir esse valor hoje, então não haveria nada para comparar.
Se você sentir necessidade frequente de uma categoria intermediária, isso é um
achado útil para documentar separadamente (um possível argumento para um judge
mais granular no futuro), mas não deve mudar silenciosamente a escala desta
comparação.

**O que observar:** as colunas `query` e `snippet` (as colunas `title` e `url`
são mostradas apenas como contexto — o LLM judge em si só vê o texto da query
+ snippet, então rotule com base na mesma informação para uma comparação
justa).

## Limitações (documentadas explicitamente, não omitidas)

- **Rotulador único, sem verificação de confiabilidade inter-avaliadores.**
  Estes rótulos refletem o julgamento de uma única pessoa. Uma segunda rodada
  de rotulagem para calcular a concordância entre avaliadores (ex.: Cohen's
  kappa) é trabalho futuro, não feito neste sprint.
- **Redução a uma escala binária.** Como observado acima, o comportamento real
  do judge é binário apesar de um schema nominalmente de 3 valores — o número
  de concordância deste dataset não diz nada sobre se uma escala mais
  granular teria um desempenho melhor.
- **Composição da amostra.** Os pares são amostrados de
  `runtime/search_logs/`, que tende a refletir quaisquer queries executadas
  durante o desenvolvimento e testes (fortemente buscas
  `incremental_technical` do trabalho de A/B das Weeks 8-9) — isso não é uma
  amostra representativa de todas as queries possíveis que a aplicação
  poderia receber em produção.

## Como calcular o score de concordância

Depois de preencher `human_label` para pelo menos algumas linhas:

```bash
uv run python scripts/compute_relevance_agreement.py
```

Isso reporta uma taxa de concordância exata entre `human_label` e
`llm_judge_relevance_level` do judge para cada linha rotulada, e informa
quantas linhas ainda não foram rotuladas.
