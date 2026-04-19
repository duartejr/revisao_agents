# Guia de Configuração do Ambiente de Desenvolvimento

> **Tempo até o primeiro uso:** ≤ 30 minutos
> **Público:** Novos desenvolvedores e colaboradores

---

## Sumário

1. [Requisitos do sistema](#1-requisitos-do-sistema)
2. [Instalação](#2-instalação)
3. [Configuração do ambiente](#3-configuração-do-ambiente)
4. [Executando a aplicação](#4-executando-a-aplicação)
5. [Checklist de verificação](#5-checklist-de-verificação)
6. [Executando os testes](#6-executando-os-testes)
7. [Resolução de problemas de configuração](#7-resolução-de-problemas-de-configuração)
8. [Contribuindo](#8-contribuindo)

---

## 1. Requisitos do sistema

| Requisito | Versão mínima | Observações |
|---|---|---|
| Python | 3.11 | 3.12 recomendado |
| git | qualquer | |
| uv | qualquer | Gerenciador de pacotes — instalado pelo bootstrap se ausente |
| MongoDB | 6.0+ | Local ou MongoDB Atlas |

**Suporte a sistemas operacionais:** Linux (Ubuntu/Fedora/Debian), macOS, Windows 10/11 (PowerShell).

---

## 2. Instalação

### Passo 1: Clone o repositório

```bash
git clone https://github.com/duartejr/revisao_agents
cd revisao_agents
```

### Passo 2: Execute o script de bootstrap

O script verifica os requisitos, instala as dependências e cria o arquivo `.env` de forma interativa.

**Linux / macOS:**
```bash
chmod +x scripts/bootstrap.sh
./scripts/bootstrap.sh
```

**Windows (PowerShell):**
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\scripts\bootstrap.ps1
```

O script irá:
1. Verificar Python, git e instalar `uv` se ausente
2. Executar `uv sync --extra dev` para instalar todas as dependências
3. Guiar você pelo preenchimento do arquivo `.env`
4. Validar que as variáveis obrigatórias estão definidas
5. Imprimir os comandos para iniciar a aplicação

> **Instalação manual (caso pule o bootstrap):**
> ```bash
> uv sync --extra dev
> cp .env.example .env
> # Edite .env com suas chaves de API
> ```

---

## 3. Configuração do ambiente

Toda a configuração é feita via arquivo `.env` na raiz do projeto.
Veja [`.env.example`](../.env.example) para o modelo completo com todas as variáveis documentadas.

### Variáveis obrigatórias (mínimo para iniciar)

| Variável | Descrição | Onde obter |
|---|---|---|
| `OPENAI_API_KEY` | Usada para embeddings (sempre obrigatória) | [platform.openai.com](https://platform.openai.com/api-keys) |
| `TAVILY_API_KEY` | Busca na web | [app.tavily.com](https://app.tavily.com) |
| `MONGODB_URI` | Backend do corpus vetorial | [MongoDB Atlas](https://cloud.mongodb.com) ou local |
| `LLM_PROVIDER` | Backend LLM: `openai`, `google`, `groq`, `openrouter` | — |

Para valores de `LLM_PROVIDER` diferentes de `openai`, adicione a chave correspondente:

| Provedor | Chave obrigatória | Onde obter |
|---|---|---|
| `google` | `GOOGLE_API_KEY` | [aistudio.google.com](https://aistudio.google.com/apikey) |
| `groq` | `GROQ_API_KEY` | [console.groq.com](https://console.groq.com/keys) |
| `openrouter` | `OPENROUTER_API_KEY` | [openrouter.ai](https://openrouter.ai/keys) |

### Exemplo mínimo de `.env`

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=revisao_agents
```

### Variáveis opcionais com valores padrão úteis

| Variável | Padrão | Descrição |
|---|---|---|
| `LLM_MODEL` | padrão do provedor | Substituir o modelo (ex: `gpt-4o`) |
| `TEMPERATURE` | `0.3` | Temperatura de amostragem do LLM |
| `TAVILY_SEARCH_DEPTH` | `basic` | `ultra-fast` / `fast` / `basic` / `advanced` |
| `TAVILY_NUM_RESULTS` | `5` | Resultados por consulta Tavily (1–10) |
| `CHECKPOINT_TYPE` | `memory` | `memory` (padrão) ou `sqlite` para sessões persistentes |

---

## 4. Executando a aplicação

### Interface gráfica (Gradio)

```bash
uv run python run_ui.py
```

Abra **http://localhost:7860** no navegador.

Porta já em uso?
```bash
uv run python run_ui.py --port 8080
```

### Interface de linha de comando (CLI)

Menu interativo:
```bash
uv run revisao-agents
```

Planejamento direto:
```bash
uv run revisao-agents "aprendizado de máquina supervisionado" --review-type academic
```

Documentação completa da CLI: [`docs/cli.md`](cli.md).

---

## 5. Checklist de verificação

Após a configuração, execute os itens abaixo para confirmar que tudo funciona:

- [ ] `uv run python -c "import revisao_agents; print('OK')"` imprime `OK`
- [ ] `uv run revisao-agents --help` exibe informações de uso
- [ ] `uv run python run_ui.py` inicia sem erros na porta 7860
- [ ] Na UI, a aba **Tools** lista os threads disponíveis sem erros
- [ ] `uv run pytest tests/unit/ -q --tb=short` exibe todos os testes passando

Saída esperada dos testes (linha de base aprovada):
```
233 passed, 1 skipped
```

---

## 6. Executando os testes

```bash
# Apenas testes unitários (rápidos, sem chamadas a APIs externas)
uv run pytest tests/unit/ -q

# Testes de integração (requer .env válido com chaves de API)
uv run pytest tests/integration/ -q

# Todos os testes
uv run pytest -q

# Verificação de tipos
make typecheck

# Linting
make lint
```

Todos os alvos `make`:
```bash
make help          # lista os alvos disponíveis
make typecheck     # executa mypy
make lint          # executa ruff
make test          # executa pytest
```

---

## 7. Resolução de problemas de configuração

### `ModuleNotFoundError: No module named 'revisao_agents'`

Você executou `python` em vez de `uv run python`. Sempre use `uv run`:
```bash
uv run python run_ui.py
```

### `uv: command not found`

Instale o `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Depois reinicie o terminal.

### `ValueError: MONGODB_URI missing` ou `TAVILY_API_KEY missing`

Seu `.env` está sem algumas chaves obrigatórias. Execute o bootstrap novamente:
```bash
./scripts/bootstrap.sh
```
Ou edite manualmente o `.env` e adicione as chaves ausentes.

### `ServerSelectionTimeoutError` (MongoDB)

- **Local:** inicie o MongoDB com `mongod` e verifique com `mongod --version`.
- **Atlas:** adicione seu IP à lista de IPs permitidos no painel do Atlas.

### `AuthenticationError` / `Invalid API key`

- Verifique que o `.env` não tem espaços extras ou aspas ao redor do valor da chave.
- Confirme que a chave está ativa no painel do provedor.
- Certifique-se de que `LLM_PROVIDER` corresponde à chave configurada (ex: `LLM_PROVIDER=google` precisa de `GOOGLE_API_KEY`).

### Porta 7860 já em uso

```bash
uv run python run_ui.py --port 8080
```

### `Invalid TAVILY_SEARCH_DEPTH: 'xyz'`

`TAVILY_SEARCH_DEPTH` deve ser um de: `ultra-fast`, `fast`, `basic`, `advanced`.

---

## 8. Contribuindo

### Fluxo de trabalho

1. Crie uma branch a partir de `dev`:
   ```bash
   git checkout dev
   git checkout -b feature/minha-mudanca
   ```

2. Faça as alterações e execute linting + testes:
   ```bash
   make lint
   make typecheck
   make test
   ```

3. Faça um commit com uma mensagem clara e abra um PR para `dev`.

### Padrões de código

- **Python 3.11+** — use tipagem moderna (`X | Y`, `list[str]` etc.)
- **Docstrings no estilo Google** para todas as funções e classes públicas
- **ruff** para linting; **mypy** para verificação de tipos
- Todas as novas funções públicas devem ter testes unitários em `tests/unit/`

### Executando um arquivo de teste específico

```bash
uv run pytest tests/unit/test_agents/test_identify_and_refine.py -v
```

---

*Veja também: [Guia de credenciais e contas](contas_e_credenciais.md) · [Arquitetura](architecture.md) · [Referência da CLI](cli.md)*
