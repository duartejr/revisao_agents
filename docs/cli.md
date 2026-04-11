# Documentação da CLI

O sistema oferece uma CLI unificada que pode ser usada tanto como menu interativo quanto por comandos diretos.

---

## `revisao-agents` — CLI Unificada

### Instalação / Ativação

Após executar `./scripts/bootstrap.sh` (ou `uv sync` para ambiente de desenvolvimento), o comando fica disponível via `uv run`:

```bash
uv run revisao-agents
```

Este comando inicia o **Menu Interativo** por padrão se nenhum argumento for passado.

### Ajuda geral

```bash
uv run revisao-agents --help
```

---

## Modo Interativo (Menu)

Execute sem argumentos para abrir o assistente:

```bash
uv run revisao-agents
```

### Opções do Menu:
1. **Plan Academic Review**: Inicia workflow de planejamento para revisões narrativas/acadêmicas.
2. **Plan Technical Review**: Inicia workflow para capítulos técnicos ou revisões de ferramentas.
3. **Execute Writing**: Permite escolher um plano gerado anteriormente (na pasta `plans/`) e iniciar a escrita automática das seções.
4. **Index Local PDFs**: Processa uma pasta de arquivos PDF, extrai texto e salva no banco vetorial MongoDB.
5. **Format References**: Lê um arquivo YAML/JSON e gera referências formatadas em ABNT/APA/IEEE via LLM.

---

## Modo Direto (Flags e Automação)

Você pode pular o menu e executar o planejamento diretamente informando um tema:

```bash
uv run revisao-agents [TEMA_OU_ARQUIVO] [OPÇÕES]
```

O primeiro argumento pode ser:
- **Texto do tema** direto: `"Previsão de vazões com LSTM"`
- **Caminho para arquivo** `.md` ou `.txt` contendo o tema ou plano. O sistema tentará extrair o tema automaticamente do conteúdo.

### Opções detalhadas

| Opção | Atalho | Padrão | Descrição |
|-------|--------|--------|-----------|
| `--review-type` | `-t` | `academic` | Tipo de revisão: `academic` ou `technical`. |
| `--rounds` | `-r` | `3` | Número de rodadas de refinamento HITL (interativas). |
| `--thread-id` | — | gerado | ID da sessão no SQLite para persistência e retorno posterior. |
| `--auto-response` | — | `None` | Resposta automática para etapas HITL (torna o planning não-interativo). |
| `--model` | — | `.env` | Modelo LLM a usar (sobrescreve o `LLM_MODEL` configurado). |
| `--debug` | — | `false` | Exibe logs de execução interna do LangGraph. |

### Exemplos de uso direto

#### Planejamento acadêmico padrão
```bash
uv run revisao-agents "Blockchain em cadeias de suprimentos"
```

#### Planejamento técnico com persistência em thread específica
```bash
uv run revisao-agents "Arquitetura Clean em Python" --review-type technical --thread-id "proj-001"
```

#### Automação total (sem interação humana)
```bash
uv run revisao-agents "Meu tema" --auto-response "Ok, prossiga com o plano padrão." --rounds 2
```

---

## Persitência (SQLite Checkpointer)

O sistema agora suporta persistência de estado via SQLite. Isso permite que você pare uma execução e retorne a ela usando o mesmo `--thread-id`.

Configuração no `.env`:
- `CHECKPOINT_TYPE=sqlite` (ativa o banco de dados)
- `CHECKPOINT_PATH=checkpoints/checkpoints.db` (local do arquivo)

Se `CHECKPOINT_TYPE=memory` (padrão), o estado é perdido após fechar o terminal.

---

## Observações de Troubleshooting

- **Caminhos de Arquivos**: Ao informar caminhos no Linux/macOS, use `~` ou caminhos relativos ao diretório raiz do projeto.
- **Tavily/MongoDB**: Certifique-se de que as chaves estão corretas no `.env` antes de rodar o Writing ou Indexing.
- **Ambiente**: O comando `uv run` garante que todas as dependências isoladas sejam carregadas corretamente.
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
