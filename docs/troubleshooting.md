# Guia de Resolução de Problemas e FAQ

> **Público:** Usuários com problemas
> **Solução rápida:** Execute `./scripts/bootstrap.sh` (Linux/macOS) ou `.\scripts\bootstrap.ps1` (Windows) para revalidar sua configuração.

---

## Sumário

1. [Problemas com chaves de API](#1-problemas-com-chaves-de-api)
2. [Erros de busca Tavily](#2-erros-de-busca-tavily)
3. [Erros de MongoDB e corpus](#3-erros-de-mongodb-e-corpus)
4. [Erros de checkpoint e sessão](#4-erros-de-checkpoint-e-sessão)
5. [Problemas com provedores de LLM](#5-problemas-com-provedores-de-llm)
6. [Problemas na UI Gradio](#6-problemas-na-ui-gradio)
7. [Problemas de idioma e codificação](#7-problemas-de-idioma-e-codificação)
8. [Logs de depuração](#8-logs-de-depuração)
9. [Perguntas frequentes (FAQ)](#9-perguntas-frequentes-faq)

---

## 1. Problemas com chaves de API

### `ValueError: Missing required keys: OPENAI_API_KEY`

**Sintoma:** A aplicação encerra na inicialização com uma lista de chaves ausentes.

**Causa:** Variável de ambiente obrigatória não definida no `.env`.

**Correção:**
```bash
# Execute o bootstrap novamente para ser guiado
./scripts/bootstrap.sh

# Ou edite o .env manualmente
nano .env   # adicione OPENAI_API_KEY=sk-...
```

**Chaves sempre obrigatórias:** `OPENAI_API_KEY`, `TAVILY_API_KEY`, `MONGODB_URI`.

---

### `AuthenticationError` / `Invalid API key`

**Sintoma:** Requisição para OpenAI, Google ou Groq falha com erro de autenticação.

**Causa:** Chave incorreta, expirada ou com espaços/aspas ao redor.

**Correção:**
1. Abra o `.env` e verifique se não há espaços extras ou aspas:
   ```env
   OPENAI_API_KEY=sk-proj-abc...   # correto
   OPENAI_API_KEY="sk-proj-abc"    # errado — remova as aspas
   ```
2. Verifique se a chave está ativa no painel do provedor.
3. Certifique-se de que `LLM_PROVIDER` no `.env` corresponde à chave configurada:
   ```env
   LLM_PROVIDER=google
   GOOGLE_API_KEY=AI...    # deve configurar esta, não OPENAI_API_KEY para o LLM
   ```
   Nota: `OPENAI_API_KEY` é sempre obrigatória mesmo ao usar outro provedor LLM (usada para embeddings).

---

### `ValueError: Invalid provider 'xyz'. Accepted providers: google, openai, groq, openrouter`

**Correção:** Defina `LLM_PROVIDER` no `.env` como um de: `openai`, `google`, `groq`, `openrouter`.
Para Google Gemini, use `LLM_PROVIDER=google` (não `gemini`).

---

## 2. Erros de busca Tavily

### `TAVILY_API_KEY missing` / `TavilyClient: unauthorized`

**Causa:** `TAVILY_API_KEY` não está definida ou é inválida.

**Correção:**
1. Obtenha uma chave em [app.tavily.com](https://app.tavily.com).
2. Adicione ao `.env`: `TAVILY_API_KEY=tvly-...`
3. Reinicie a aplicação.

---

### Tavily retorna resultados vazios ou muito poucos URLs

**Possíveis causas:**
- Cota esgotada — verifique seu uso em [app.tavily.com](https://app.tavily.com).
- O tema da consulta é muito específico ou usa jargão de domínio que o Tavily não consegue correspondência.
- `TAVILY_SEARCH_DEPTH=ultra-fast` pode não encontrar fontes relevantes.

**Correção:**
- Aumente `TAVILY_NUM_RESULTS` para 7 ou 10.
- Mude para `TAVILY_SEARCH_DEPTH=basic` ou `advanced`.
- Tente reformular a consulta em inglês (o agente prioriza resultados em inglês).

---

### `ValueError: Invalid TAVILY_SEARCH_DEPTH: 'turbo'`

**Correção:** Defina `TAVILY_SEARCH_DEPTH` como um de: `ultra-fast`, `fast`, `basic`, `advanced`.

---

### Limite de taxa / `429 Too Many Requests`

**Causa:** Cota de API do Tavily excedida.

**Correção:**
- Verifique o uso em [app.tavily.com](https://app.tavily.com).
- Reduza `TAVILY_NUM_RESULTS` ou mude para profundidade `ultra-fast` para economizar créditos.
- Faça upgrade do seu plano Tavily.

---

## 3. Erros de MongoDB e corpus

### `ServerSelectionTimeoutError`

**Causa:** MongoDB está inacessível.

**Para MongoDB local:**
```bash
# Verifique se o MongoDB está rodando
mongod --version
# Inicie se necessário:
sudo systemctl start mongod   # Linux
brew services start mongodb-community  # macOS
```

**Para MongoDB Atlas:**
- Adicione seu IP atual ao **Network Access** (lista de IPs permitidos) no Atlas.
- Verifique se `MONGODB_URI` é a string de conexão completa do Atlas, incluindo credenciais.

---

### `OperationFailure: Authentication failed`

**Causa:** Usuário/senha incorretos na URI do MongoDB.

**Correção:**
```env
# Formato correto:
MONGODB_URI=mongodb+srv://usuario:senha@cluster.mongodb.net/
```
Verifique se a senha não tem caracteres especiais que precisam de codificação URL (ex: `@` → `%40`).

---

### Busca vetorial não retorna resultados

**Causa:** O índice do corpus está vazio (nenhum PDF foi indexado).

**Correção:**
1. Indexe seus PDFs pela CLI:
   ```bash
   uv run revisao-agents   # escolha a opção 4: Indexar PDFs
   ```
2. Verifique se a coleção existe no MongoDB Atlas / Compass.

---

## 4. Erros de checkpoint e sessão

### `ValueError: Failed to create SQLite checkpointer: ...`

**Causa:** O diretório de checkpoint não tem permissão de escrita ou o caminho é inválido.

**Correção:**
```env
# Use um caminho relativo com permissão de escrita
CHECKPOINT_PATH=checkpoints/checkpoints.db
```
Certifique-se de que o usuário atual tem permissão de escrita no diretório.

---

### `Database is locked` (SQLite)

**Causa:** Duas instâncias da aplicação estão rodando com o mesmo arquivo SQLite.

**Correção:** Encerre a instância duplicada e reinicie a aplicação.

---

### Estado da sessão não restaurado após reinicialização

**Causa:** Usando `CHECKPOINT_TYPE=memory` (padrão) — sessões em memória não são persistentes.

**Correção:** Mude para SQLite:
```env
CHECKPOINT_TYPE=sqlite
CHECKPOINT_PATH=checkpoints/checkpoints.db
```

---

### `Thread not found` ao carregar uma sessão na UI

**Causa:**
- O ID do thread está incorreto.
- O backend é `memory` e o processo foi reiniciado.
- O arquivo SQLite foi deletado.

**Correção:**
- Use o ID de thread correto (exibido na barra de status da UI durante a sessão).
- Mude para `CHECKPOINT_TYPE=sqlite` para sessões persistentes.

---

## 5. Problemas com provedores de LLM

### `ImportError: No module named 'langchain_google_genai'`

**Causa:** A integração LangChain do Google não está instalada.

**Correção:**
```bash
uv add langchain-google-genai
```

---

### `Model not found` / `404` do provedor LLM

**Causa:** `LLM_MODEL` está definido com um modelo que não existe para o provedor selecionado.

**Correção:**
- Remova `LLM_MODEL` do `.env` para usar o modelo padrão do provedor.
- Ou consulte a lista de modelos do provedor e use um nome válido.

---

### Timeout de resposta do LLM

**Causa:** Rede lenta ou provedor sobrecarregado.

**Correção:**
- Tente novamente após alguns segundos.
- Mude para um provedor mais rápido: `LLM_PROVIDER=groq` (Groq tem latência muito baixa).

---

## 6. Problemas na UI Gradio

### `ModuleNotFoundError: No module named 'gradio'`

**Correção:**
```bash
uv sync --extra dev
```
Sempre use `uv run` para iniciar:
```bash
uv run python run_ui.py
```

---

### Porta 7860 já em uso

**Sintoma:** `OSError: [Errno 98] Address already in use`

**Correção:**
```bash
uv run python run_ui.py --port 8080
```

---

### UI inicia mas fica em branco ou exibe erros

1. Abra o console de desenvolvimento do navegador (F12) e verifique erros de JavaScript.
2. Tente outro navegador (Chrome é recomendado).
3. Faça uma atualização forçada da página: Ctrl+Shift+R.
4. Verifique o terminal por exceções Python.

---

## 7. Problemas de idioma e codificação

### Tema detectado como idioma `UNKNOWN`

**Causa:** O tema não contém marcadores de idioma reconhecíveis, ou mistura português e inglês.

**Correção:** O agente pedirá esclarecimento. Digite o tema em português ou inglês.

---

### Caracteres especiais corrompidos na saída (Windows)

**Causa:** A codificação do terminal não é UTF-8.

**Correção:**
```powershell
# Defina a codificação do terminal como UTF-8
chcp 65001
```

---

## 8. Logs de depuração

### Ativar saída detalhada (verbose)

Defina a variável de ambiente antes de executar:

```bash
# Linux/macOS
LOG_LEVEL=DEBUG uv run python run_ui.py

# Windows PowerShell
$env:LOG_LEVEL = "DEBUG"
uv run python run_ui.py
```

### Verificar logs de busca Tavily

Todas as buscas Tavily são registradas em `./search_logs/` (configurável via `SEARCH_LOGS_DIR`). Cada busca cria um arquivo Markdown com:
- Consulta realizada
- Resultados (URLs, títulos, trechos)
- Uso de créditos (se `TAVILY_INCLUDE_USAGE=true`)

```bash
ls search_logs/
# academic_aprendizado_maquina_supervisionado_20260418_143201.md
```

### Inspecionar o estado do checkpoint

```python
from revisao_agents.graphs.checkpoints import get_checkpointer
from revisao_agents.workflows import build_academic_workflow

saver = get_checkpointer()
app = build_academic_workflow(checkpointer=saver)

config = {"configurable": {"thread_id": "seu-id-de-thread"}}
state = app.get_state(config)
print(state.values)
```

---

## 9. Perguntas frequentes (FAQ)

**P: Preciso das três chaves de API (OpenAI, Tavily, MongoDB)?**
R: Sim. As três são obrigatórias. OpenAI é usada para embeddings de texto; Tavily para busca na web; MongoDB para o corpus vetorial local.

**P: Posso usar um LLM diferente para o agente principal e manter o OpenAI para embeddings?**
R: Sim. Defina `LLM_PROVIDER=groq` (ou `google`) e ainda forneça `OPENAI_API_KEY`. O agente principal usará Groq/Google; os embeddings usarão OpenAI.

**P: Como salvo minha sessão de planejamento para retomar depois?**
R: Defina `CHECKPOINT_TYPE=sqlite` no seu `.env`. A sessão é persistida automaticamente. Anote o ID de thread exibido na UI para retomar mais tarde.

**P: Posso executar múltiplas sessões em paralelo?**
R: Sim, cada sessão usa um ID de thread diferente. Com checkpointing SQLite, todas as sessões são persistidas.

**P: Meu plano está errado — posso recomeçar sem criar uma nova sessão?**
R: Inicie uma nova sessão com um novo tema. Sessões antigas permanecem no arquivo SQLite e não afetam novas sessões.

**P: Como apagar todas as minhas sessões salvas?**
R: Delete o arquivo SQLite: `rm checkpoints/checkpoints.db`. Todo o histórico de sessões será perdido.

**P: As buscas Tavily estão lentas. Como posso acelerar?**
R: Defina `TAVILY_SEARCH_DEPTH=fast` e `TAVILY_NUM_RESULTS=3`. Veja o [Guia de ajuste do Tavily](tavily_tuning_guide.md).

**P: O agente continua pedindo para esclarecer meu tema. Como faço para parar?**
R: Forneça um tópico mais específico. Em vez de "Aprendizado de máquina", tente "Transfer learning para classificação de textos em NLP". O agente sinaliza temas vagos para melhorar a qualidade da revisão.

**P: Como executo apenas os testes unitários sem precisar de chaves de API?**
R: `uv run pytest tests/unit/ -q` — os testes unitários são totalmente mockados e não precisam de serviços externos.

**P: Onde os planos e revisões gerados são salvos?**
R: Em `./plans/` e `./reviews/` por padrão. Configurável via `PLANS_DIR` e `REVIEWS_DIR`.

---

*Veja também: [Guia de configuração](setup_guide.md) · [Guia de ajuste do Tavily](tavily_tuning_guide.md) · [Gerenciamento de sessões](session_management.md)*
