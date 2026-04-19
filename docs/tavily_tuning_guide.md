# Guia de Ajuste do Tavily

> **Público:** Usuários e operadores que configuram o agente de busca
> **Funcionalidade:** Integração do Tavily AI Search em `src/revisao_agents/config.py`

Este guia explica todos os parâmetros de configuração do Tavily, seus compromissos e perfis recomendados para diferentes casos de uso.

---

## Sumário

1. [Como o Tavily é utilizado](#1-como-o-tavily-é-utilizado)
2. [Parâmetros de configuração](#2-parâmetros-de-configuração)
3. [Estimativa de custo e créditos](#3-estimativa-de-custo-e-créditos)
4. [Perfis recomendados](#4-perfis-recomendados)
5. [Alterando as configurações](#5-alterando-as-configurações)

---

## 1. Como o Tavily é utilizado

O agente usa o Tavily para busca em tempo real na web em três cenários:

| Tipo de busca | Quando é acionado | Ferramenta |
|---|---|---|
| Busca acadêmica | Planejamento de revisão bibliográfica | `search_tavily` |
| Busca técnica | Planejamento de capítulo técnico | `search_tavily_technical` |
| Busca de imagens | Escrita de figuras técnicas | `search_tavily_images` |
| Extração | Extração completa do conteúdo de URLs | `extract_tavily` |
| Incremental | Busca iterativa durante a escrita | `search_tavily_incremental` |

Todas as buscas utilizam o `TavilySearchConfig` centralizado, carregado das variáveis de ambiente na inicialização.

---

## 2. Parâmetros de configuração

### `TAVILY_SEARCH_DEPTH`

Controla a profundidade e a qualidade da busca. Cada nível usa uma quantidade diferente de créditos de API por consulta.

| Valor | Créditos/consulta | Velocidade | Qualidade dos resultados | Melhor uso |
|---|---|---|---|---|
| `ultra-fast` | 1 | Muito rápido (~0,5s) | Baixa | Rascunho / protótipo |
| `fast` | 1 | Rápido (~1s) | Média | Revisões rápidas |
| `basic` | 1 | Moderado (~2s) | Boa | **Padrão** — equilibrado |
| `advanced` | 2 | Lento (~4s) | Muito alta | Pesquisa aprofundada |

> **Padrão:** `basic`
> **Fonte:** [Documentação Tavily Search API — parâmetro depth](https://docs.tavily.com/docs/tavily-api/rest_api#depth)

**Quando usar `advanced`:**
- O tema exige fontes acadêmicas autoritativas
- Você precisa de alta precisão, não de volume
- Créditos não são uma restrição

**Quando usar `fast` ou `ultra-fast`:**
- Prototipagem de um novo fluxo
- Teste de conectividade / ambiente de desenvolvimento
- Orçamento de créditos limitado

---

### `TAVILY_NUM_RESULTS`

Número de resultados retornados por consulta de busca.

| Valor | Impacto nos créditos | Caso de uso |
|---|---|---|
| 1–3 | Mínimo | Verificações rápidas |
| 5 (padrão) | Baixo | Padrão equilibrado |
| 7–10 | Maior | Cobertura mais ampla (mais deduplicação necessária) |

> **Limite da API Tavily:** máximo 10 por consulta.
> **Fonte:** [Tavily Search — parâmetro max_results](https://docs.tavily.com/docs/tavily-api/rest_api#max_results)

**Atenção:** O agente executa múltiplas consultas por sessão de planejamento. Total de resultados = `TAVILY_NUM_RESULTS × número_de_consultas`. Definir esse valor como 10 pode aumentar significativamente a latência em fluxos com múltiplas consultas.

---

### `TAVILY_INCLUDE_ANSWER`

Quando `true`, o Tavily retorna um resumo gerado por IA junto com os resultados individuais.

| Valor | Efeito | Custo em créditos |
|---|---|---|
| `true` (padrão) | Adiciona resposta sintetizada pelo LLM na resposta | Nenhum (incluso) |
| `false` | Apenas resultados brutos | Nenhum |

**Recomendação:** Mantenha `true`. O campo de resposta ajuda o agente a produzir planos de maior qualidade ao fornecer uma visão sintetizada dos resultados da busca.

---

### `TAVILY_INCLUDE_USAGE`

Quando `true`, a resposta da API inclui metadados de uso de créditos para a requisição.

| Valor | Efeito |
|---|---|
| `true` (padrão) | Adiciona dict `usage` com `input_tokens`, `output_tokens`, `total_tokens` e `request_id` |
| `false` | Sem metadados de uso |

Usado para rastreamento de custos pelo sistema de logs do agente (`_save_search_md`). Mantenha `true` se quiser que os logs de busca incluam o uso de créditos.

---

## 3. Estimativa de custo e créditos

### Custo em créditos por consulta

| `TAVILY_SEARCH_DEPTH` | Créditos por consulta |
|---|---|
| `ultra-fast` | 1 |
| `fast` | 1 |
| `basic` | 1 |
| `advanced` | 2 |

### Consumo típico de créditos por sessão

Uma sessão de planejamento acadêmico com 3 rodadas de refinamento:

| Perfil | Profundidade | Consultas (aprox.) | Total de créditos |
|---|---|---|---|
| Revisão rápida | `fast` | 10–15 | 10–15 |
| Equilibrado (padrão) | `basic` | 10–15 | 10–15 |
| Pesquisa aprofundada | `advanced` | 10–15 | 20–30 |

Uma sessão completa de escrita técnica (10 seções, cada uma com busca):

| Perfil | Profundidade | Consultas (aprox.) | Total de créditos |
|---|---|---|---|
| Rápido | `fast` | 30–50 | 30–50 |
| Equilibrado | `basic` | 30–50 | 30–50 |
| Aprofundado | `advanced` | 30–50 | 60–100 |

> **Plano gratuito:** O plano gratuito do Tavily inclui 1.000 créditos/mês.
> Para uso em produção, recomenda-se um plano pago.
> Monitore seu uso em: https://app.tavily.com

---

## 4. Perfis recomendados

### Perfil A: Revisão rápida (rascunho / testes)

```env
TAVILY_SEARCH_DEPTH=fast
TAVILY_NUM_RESULTS=3
TAVILY_INCLUDE_ANSWER=true
TAVILY_INCLUDE_USAGE=true
```

**Custo em créditos:** ~1× linha de base
**Use quando:** Prototipagem, testes, varreduras bibliográficas rápidas onde precisão não é prioridade.

---

### Perfil B: Equilibrado (padrão, recomendado)

```env
TAVILY_SEARCH_DEPTH=basic
TAVILY_NUM_RESULTS=5
TAVILY_INCLUDE_ANSWER=true
TAVILY_INCLUDE_USAGE=true
```

**Custo em créditos:** ~1× linha de base
**Use quando:** A maioria das revisões acadêmicas e técnicas. Bom equilíbrio entre qualidade, velocidade e custo.

---

### Perfil C: Pesquisa aprofundada

```env
TAVILY_SEARCH_DEPTH=advanced
TAVILY_NUM_RESULTS=7
TAVILY_INCLUDE_ANSWER=true
TAVILY_INCLUDE_USAGE=true
```

**Custo em créditos:** ~2–3× linha de base
**Use quando:** Revisões de alto impacto que exigem fontes abrangentes e autoritativas. Créditos não são uma restrição.

---

## 5. Alterando as configurações

### Opção 1: Editar `.env` (persistente)

```env
TAVILY_SEARCH_DEPTH=advanced
TAVILY_NUM_RESULTS=7
```

Reinicie a aplicação para aplicar as mudanças. A configuração é carregada uma única vez na inicialização.

### Opção 2: Variável de ambiente inline (execução única)

```bash
TAVILY_SEARCH_DEPTH=fast TAVILY_NUM_RESULTS=3 uv run python run_ui.py
```

### Verificando suas configurações

Após a inicialização, a configuração pode ser visualizada no REPL Python:

```python
from revisao_agents.config import TAVILY_CONFIG
print(TAVILY_CONFIG)
# TavilySearchConfig(depth='basic', num_results=5, include_answer=True, include_usage=True)
```

### Validação

Se `TAVILY_SEARCH_DEPTH` for definido com um valor não suportado, a aplicação lança `ValueError` na inicialização:

```
ValueError: Invalid TAVILY_SEARCH_DEPTH: 'turbo'. Must be one of ('ultra-fast', 'fast', 'basic', 'advanced')
```

Correção: defina a variável com um dos valores aceitos listados acima.

---

*Veja também: [Guia de configuração](setup_guide.md) · [Resolução de problemas](troubleshooting.md)*
