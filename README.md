# Agente de Revisão da Literatura

Sistema agêntico para planejamento e escrita de revisões acadêmicas e capítulos técnicos, baseado em LangGraph com suporte a múltiplos provedores de LLM (OpenAI, Google, Groq, OpenRouter).

## O que este projeto faz

- **Valida e refina o tema** antes de iniciar qualquer busca — detecta idioma e avisa quando o tema é vago demais, pedindo ao usuário um tópico mais específico
- **Planeja revisões** com entrevista guiada por IA (Human-in-the-Loop)
- **Escreve seções** completas buscando evidências no corpus local (MongoDB) e na web (Tavily)
- **Revisa interativamente** textos gerados via chat com o agente
- **Indexa PDFs** locais em base vetorial para busca semântica
- **Formata referências** a partir de arquivos YAML/JSON nos padrões ABNT, APA, IEEE etc.

**Modos de uso:** interface gráfica (UI Gradio) ou linha de comando (CLI).

---

## Início rápido

### Pré-requisitos

| Requisito | Versão mínima | Link |
|-----------|--------------|------|
| Python    | 3.11+        | [python.org](https://www.python.org/downloads/) |
| git       | qualquer     | [git-scm.com](https://git-scm.com/) |
| uv        | qualquer     | [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/) |

> **uv** é o gerenciador de pacotes recomendado. Se não estiver instalado, o script de bootstrap o instala automaticamente.

### 1. Clone o repositório

```bash
git clone https://github.com/duartejr/revisao_agents
cd revisao_agents
```

### 2. Execute o bootstrap (configura ambiente e credenciais)

**Linux / macOS:**
```bash
chmod +x scripts/bootstrap.sh
./scripts/bootstrap.sh
```

**Windows (PowerShell):**
```powershell
# Se necessário, libere a execução de scripts:
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

.\scripts\bootstrap.ps1
```

O script irá:
1. Verificar Python, uv e git
2. Instalar dependências com `uv sync --extra dev`
3. Criar o arquivo `.env` com um assistente interativo
4. Validar as variáveis obrigatórias
5. Exibir os comandos de início

### 3. Inicie a interface gráfica (UI)

```bash
uv run python run_ui.py
```

Acesse em: **http://localhost:7860**

---

## Configuração do `.env`

O bootstrap cria o `.env` automaticamente. Abaixo estão as variáveis agrupadas por perfil.

### Perfil mínimo (obrigatório para funcionamento)

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...       # Sempre obrigatório (usado para embeddings)
TAVILY_API_KEY=tvly-...     # Sempre obrigatório (busca web)
MONGODB_URI=mongodb://localhost:27017  # Sempre obrigatório (corpus vetorial)
MONGODB_DB=revisao_agents
```

### Perfil completo (todos os provedores)

```env
LLM_PROVIDER=openai         # openai | google | groq | openrouter

# OpenAI
OPENAI_API_KEY=sk-...

# Google Gemini
GOOGLE_API_KEY=...           # https://aistudio.google.com/apikey

# Groq
GROQ_API_KEY=gsk_...         # https://console.groq.com/keys

# OpenRouter
OPENROUTER_API_KEY=sk-or-... # https://openrouter.ai/keys

# Tavily (busca web)
TAVILY_API_KEY=tvly-...

# MongoDB (corpus vetorial)
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=revisao_agents

# Opcional (Persistência e Checkpoints)
TEMPERATURE=0.3
LLM_MODEL=gpt-4o-mini
CHECKPOINT_TYPE=sqlite       # memory | sqlite — Tipo de persistência para workflows LangGraph
CHECKPOINT_PATH=checkpoints/checkpoints.db  # Caminho para o banco SQLite (se CHECKPOINT_TYPE=sqlite)
```

### Matriz de requisitos por funcionalidade

| Funcionalidade         | OpenAI (sempre) | Tavily (sempre) | MongoDB (sempre) | Google/Groq/OpenRouter |
|-----------------------|-----------------|-----------------|------------------|------------------------|
| Planejar revisão      | ✔ (embeddings)  | ✔               | ✔                | opcional (LLM)         |
| Escrever seção técnica| ✔               | ✔               | ✔                | opcional               |
| Escrever seção acadêmica| ✔             | ✔               | ✔                | opcional               |
| Revisão Interativa    | ✔               | ✔               | ✔                | opcional               |
| Indexar PDFs          | ✔ (embeddings)  | ✔               | ✔                | —                      |
| Formatar referências  | ✔ (embeddings)  | ✔               | ✔                | opcional               |

---

## Uso via CLI

### Menu unificado interativo

O aplicativo centraliza todas as funcionalidades em um menu interativo:

```bash
uv run revisao-agents
```

Este comando exibe opções para:
1. **Planejar revisão acadêmica** (narrativa)
2. **Planejar revisão técnica** (capítulo estruturado)
3. **Executar escrita** a partir de planos existentes (acadêmico ou técnico)
4. **Indexar PDFs** locais para o banco vetorial (MongoDB)
5. **Formatar referências** (ABNT, APA, IEEE) a partir de arquivos YAML/JSON

### CLI automatizada (Flags)

Para automação ou execuções diretas de planejamento:

```bash
# Ajuda geral
uv run revisao-agents --help

# Planejar revisão acadêmica direta
uv run revisao-agents "meu tema de pesquisa" --review-type academic

# Planejar com número específico de rodadas e thread ID manual (SQLite)
uv run revisao-agents "meu tema" --rounds 4 --thread-id "sessao-001"

# Modo não interativo (envia resposta automática em pausas HITL)
uv run revisao-agents "meu tema" --auto-response "Ok, prossiga com o plano."
```

Veja documentação detalhada em [`docs/cli.md`](docs/cli.md).

---

## Compatibilidade de sistemas operacionais

| SO | Bootstrap | UI | CLI |
|----|-----------|-----|-----|
| Linux (Ubuntu/Fedora/Debian) | `bootstrap.sh` | ✔ | ✔ |
| Windows 10/11 (PowerShell) | `bootstrap.ps1` | ✔ | ✔ |
| macOS | `bootstrap.sh` | ✔ | ✔ |

---

## Troubleshooting inicial

### `ModuleNotFoundError` ao iniciar
```bash
# Certifique-se de usar uv run, não python diretamente:
uv run python run_ui.py
```

### Erro de autenticação MongoDB
```
ServerSelectionTimeoutError
```
- **Atlas:** verifique se o IP do seu computador está na lista de permissões do cluster.
- **Local:** verifique se o serviço MongoDB está rodando (`mongod --version`).

### Erro de chave de API (`AuthenticationError`, `Invalid API key`)
- Confirme que o `.env` foi salvo com a chave correta (sem espaços extras ou aspas).
- Troque `LLM_PROVIDER` para o provedor cuja chave você configurou.

### Erro de configuração obrigatória (`ValueError` com lista de problemas)
- **Chaves sempre obrigatórias:** `MONGODB_URI`, `TAVILY_API_KEY`, `OPENAI_API_KEY` (usada para embeddings).
- **Erro típico:** "MONGODB_URI missing", "TAVILY_API_KEY missing", "Missing OPENAI_API_KEY (required for embeddings)".
- Execute o bootstrap novamente: `./scripts/bootstrap.sh` (Linux/macOS) ou `.\scripts\bootstrap.ps1` (Windows).

### Provedor LLM inválido (`ValueError: Invalid provider 'xyz'. Accepted providers: google, openai, groq, openrouter`)
- Configure `LLM_PROVIDER` no `.env` com um dos valores aceitos: `google`, `openai`, `groq`, `openrouter`.
- Para Google Gemini, use `LLM_PROVIDER=google` (não `gemini`).
- Lista completa de provedores aceitos está nesta seção do README.

### Tavily retorna resultados vazios
- Confirme que `TAVILY_API_KEY` está preenchida no `.env`.
- O plano mínimo gratuito do Tavily tem limite de requisições — verifique sua cota em https://app.tavily.com.

### Porta 7860 já em uso
```bash
uv run python run_ui.py --port 8080
```

---

## Estrutura do projeto

```
revisao_agents/
├── run_ui.py              ← Ponto de entrada da UI Gradio
├── scripts/
│   ├── bootstrap.sh       ← Bootstrap Linux/macOS
│   └── bootstrap.ps1      ← Bootstrap Windows PowerShell
├── src/
│   ├── gradio_app/        ← Interface gráfica (Gradio)
│   │   ├── app.py         ← Abas e componentes Gradio
│   │   └── handlers/      ← Lógica de negócio das abas (subpacote)
│   │       ├── base.py, planning.py, review.py, writing.py, tools.py
│   │       └── review_parts/  ← document, intent, images, references
│   └── revisao_agents/    ← Pacote principal
│       ├── agents/        ← Nós do LangGraph
│       ├── tools/         ← Ferramentas LangChain
│       ├── workflows/     ← Grafos de estado
│       ├── nodes/         ← Nós especializados de escrita
│       ├── utils/         ← Utilitários (LLM, vetores, prompts)
│       ├── config.py      ← Configuração via .env
│       ├── cli.py         ← CLI Typer (revisao-agents)
│       └── __main__.py    ← Menu interativo CLI
├── docs/                  ← Documentação completa
│   ├── README.md          ← Índice da documentação
│   ├── ui/                ← Docs por aba da UI
│   ├── cli.md             ← Documentação da CLI
│   ├── contas_e_credenciais.md ← Guia de credenciais
│   └── architecture.md    ← Arquitetura do sistema
├── plans/                 ← Planos gerados
├── reviews/               ← Revisões geradas
└── .env.example           ← Modelo de configuração
```

---

## Documentação completa

- [Índice da documentação](docs/README.md)
- [Guia de credenciais e contas](docs/contas_e_credenciais.md)
- [Documentação da UI por aba](docs/ui/)
- [Documentação da CLI](docs/cli.md)
- [Arquitetura do sistema](docs/architecture.md)
