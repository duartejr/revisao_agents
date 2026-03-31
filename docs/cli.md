# Documentação da CLI

O sistema oferece dois modos de uso pela linha de comando:

1. **`revisao-agents`** — CLI script direto (Typer), focada em planejamento automatizado
2. **`python -m revisao_agents`** — menu interativo com todas as funcionalidades

---

## `revisao-agents` — CLI Script

### Instalação / Ativação

Após executar `./scripts/bootstrap.sh` (ou `uv sync --extra dev` para ambiente de desenvolvimento), o comando fica disponível como:

```bash
uv run revisao-agents
```

> Se o ambiente estiver ativado (`source .venv/bin/activate`), pode-se usar `revisao-agents` diretamente.

Para uso somente em runtime (sem lint/test/typecheck), `uv sync` também é suficiente.

### Ajuda geral

```bash
uv run revisao-agents --help
```

### Sintaxe

```
revisao-agents [TEMA_OU_ARQUIVO] [OPÇÕES]
```

O primeiro argumento pode ser:
- **Texto do tema** direto: `"Previsão de vazões com LSTM"`
- **Caminho para arquivo** `.md` ou `.txt` contendo o tema ou plano

### Opções

| Opção | Atalho | Padrão | Descrição |
|-------|--------|--------|-----------|
| `--review-type` | `-t` | `academic` | Tipo de revisão: `academic` ou `technical` |
| `--rounds` | `-r` | `3` | Número de rodadas de refinamento HITL |
| `--output` | `-o` | — | Caminho para salvar o plano gerado (opcional) |
| `--model` | — | `.env` | Modelo LLM a usar (sobrescreve `LLM_MODEL` do `.env`) |
| `--auto-response` | — | `"Keep the current plan."` | Resposta automática para etapas HITL (modo não interativo) |
| `--debug` | — | `false` | Ativa saída detalhada dos eventos internos do grafo |

### Exemplos

#### Planejamento acadêmico (padrão)

```bash
uv run revisao-agents "Uso de LSTM para previsão de cheias"
```

#### Planejamento técnico com mais rodadas

```bash
uv run revisao-agents "Chronos-2 vs LSTM para previsão de streamflow" \
  --review-type technical \
  --rounds 5
```

#### Salvar o plano gerado em arquivo

```bash
uv run revisao-agents "Modelos de previsão de secas" \
  --review-type academic \
  --output plans/meu_plano.md
```

#### Usar tema a partir de arquivo

```bash
uv run revisao-agents plans/plano_revisao_tecnica_tema.md \
  --review-type technical
```

#### Usar modelo específico

```bash
uv run revisao-agents "Energia solar fotovoltaica no Nordeste" \
  --model gemini-1.5-flash
```

#### Modo debug (ver eventos internos do LangGraph)

```bash
uv run revisao-agents "Meu tema" --debug
```

#### Modo não interativo (resposta automática para HITL)

```bash
uv run revisao-agents "Meu tema" \
  --auto-response "Mantenha o plano atual." \
  --rounds 2
```

### Comportamento esperado

1. O agente inicia o workflow de planejamento e processa as rodadas automaticamente.
2. Em cada rodada HITL, usa o valor de `--auto-response` sem interromper.
3. Ao final, imprime o plano gerado no terminal.
4. Se `--output` for informado, salva o plano no arquivo.
5. O plano também é salvo automaticamente em `plans/`.

---

## `python -m revisao_agents` — Menu Interativo

### Como iniciar

```bash
uv run python -m revisao_agents
```

### Menu principal

```
Options:
  [1] Plan Academic Review (narrative)
  [2] Plan Technical Review (chapter)
  [3] Execute Writing from Existing Plan (Technical or Academic)
  [4] Index Local PDFs → vectorize and save to MongoDB
  [5] Format References (ABNT, APA, IEEE, etc.) from YAML/JSON file

Choose [1/2/3/4/5]:
```

### Opção 1 — Planejar revisão acadêmica

```
Choose [1/2/3/4/5]: 1
Review theme: Modelos de previsão de cheias com deep learning
```

O agente executa o workflow acadêmico com entrevista HITL. O plano é salvo em `plans/`.

### Opção 2 — Planejar revisão técnica

```
Choose [1/2/3/4/5]: 2
Review theme: Comparação entre Chronos-2 e LSTM
```

Idêntico à opção 1, mas usa o workflow técnico.

### Opção 3 — Escrever a partir de plano existente

```
Choose [1/2/3/4/5]: 3
WRITING STYLE:
  [a] Technical section — didactic chapter (web search + MongoDB)
  [b] Academic — narrative literature review (corpus-first)

Choose [a/b, default=a]: a

REVIEW LANGUAGE:
  [pt] Portuguese (standard)
  [en] English

Choose [pt/en, default=pt]: pt

MINIMUM NUMBER OF DISTINCT SOURCES PER SECTION:
(default = 0; 0 = no restriction)
Minimum sources per section [0]:

Do you want to enable web/image search via Tavily?
  [y] Yes (web and image search)
  [n] No (local corpus only)

Enable Tavily? [y/N]: y

Plans found:
  [1] plans/plano_revisao_tecnica_tema.md
  [2] plans/plano_revisao_academico_outro.md

Choose [1-2 or path]: 1
```

O agente escreve as seções do plano selecionado. O resultado é salvo em `reviews/`.

### Opção 4 — Indexar PDFs locais

```
Choose [1/2/3/4/5]: 4
Path to folder with PDFs: ~/artigos/hidrologia
```

Indexa todos os PDFs da pasta no MongoDB.

```
INDEXING RESULT
  ✅ New PDFs indexed : 12
  ⏭️  Already in DB     : 3
  ⚠️  Insufficient text : 1
  ❌ Reading errors     : 0
  📦 Chunks inserted    : 847
```

### Opção 5 — Formatar referências

```
Choose [1/2/3/4/5]: 5
```

O menu guia pela seleção de arquivo YAML/JSON e padrão de formatação.

---

## Fluxos recomendados por cenário

### Cenário A: Escrever uma revisão acadêmica do zero

```bash
# 1. Indexe seus PDFs (se ainda não fez)
uv run python -m revisao_agents
# Escolha [4], informe a pasta

# 2. Planeje a revisão
uv run revisao-agents "Seu tema aqui" --review-type academic --output plans/meu_plano.md

# 3. Escreva o documento
uv run python -m revisao_agents
# Escolha [3] → [b] Academic → [pt] → selecione o plano
```

### Cenário B: Escrever um capítulo técnico com busca web

```bash
# 1. Planeje
uv run revisao-agents "Tema técnico" --review-type technical --output plans/plano_tecnico.md

# 2. Escreva com Tavily ativado
uv run python -m revisao_agents
# Escolha [3] → [a] Technical → [pt] → Tavily [y] → selecione o plano
```

### Cenário C: Automatizar planejamento em pipeline

```bash
# Planejamento não interativo (sem interface, para scripts)
uv run revisao-agents "Tema" \
  --review-type academic \
  --rounds 2 \
  --auto-response "Aceitar plano atual." \
  --output plans/automatico.md
```

---

## Troubleshooting CLI

### `command not found: revisao-agents`
```bash
# Use sempre uv run:
uv run revisao-agents --help

# Ou ative o ambiente virtual primeiro:
source .venv/bin/activate
revisao-agents --help
```

### `ModuleNotFoundError`
```bash
# Certifique-se de estar no diretório do projeto:
cd revisao_agent
uv run revisao-agents --help
```

### Erro de validação de ambiente na inicialização
```
⚠️  Configuration warnings detected:
   - GOOGLE_API_KEY not set
```
- Estes avisos são informativos e não impedem o uso do provedor configurado.
- Apenas a chave do `LLM_PROVIDER` definido no `.env` é obrigatória.

### Plano gerado aparece truncado
- Aumente as rodadas com `--rounds 5` ou `--rounds 6`.
- Ou use o menu interativo (`python -m revisao_agents`) para maior controle.

### Debug para diagnóstico

```bash
uv run revisao-agents "Tema" --debug 2>&1 | tee debug.log
```
