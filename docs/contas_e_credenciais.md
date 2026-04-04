# Guia de Contas e Credenciais

Este guia explica como criar conta e obter as chaves de API para cada serviço usado pelo sistema. Ao final há um checklist de validação.

---

## Índice

- [MongoDB](#mongodb)
- [OpenAI](#openai)
- [Google Gemini](#google-gemini)
- [Groq](#groq)
- [Tavily](#tavily)
- [OpenRouter](#openrouter)
- [Checklist final](#checklist-final)

---

## MongoDB

**Para que serve:** armazenamento do corpus vetorial (chunks de PDFs indexados) e persistência de estado do LangGraph.

**Obrigatório para:** indexação de PDFs, escrita de seções, planejamento acadêmico.

### Opção A: MongoDB Atlas (nuvem — recomendado)

1. Acesse [mongodb.com/atlas](https://www.mongodb.com/atlas) e clique em **"Try Free"**.
2. Crie uma conta (ou entre com Google/GitHub).
3. Crie um **cluster gratuito** (tier M0 — 512 MB, suficiente para uso normal).
   - Escolha a região mais próxima de você.
4. Na tela "Security Quickstart":
   - Crie um usuário com **senha** (anote — será usada na URI).
   - Em "Where would you like to connect from?", adicione seu IP ou use `0.0.0.0/0` para acesso de qualquer lugar (menos seguro, apenas para desenvolvimento).
5. Clique em **"Connect"** no cluster → **"Connect your application"**.
6. Copie a **Connection String** no formato:
   ```
   mongodb+srv://SEU_USUARIO:SUA_SENHA@cluster0.xxxxx.mongodb.net/
   ```
7. Configure no `.env`:
   ```env
   MONGODB_URI=mongodb+srv://SEU_USUARIO:SUA_SENHA@cluster0.xxxxx.mongodb.net/
   MONGODB_DB=revisao_agents
   ```

### Opção B: MongoDB local

1. Instale o MongoDB Community Edition:
   - [Guia de instalação Linux](https://www.mongodb.com/docs/manual/administration/install-on-linux/)
   - [Guia de instalação Windows](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-windows/)
2. Inicie o serviço:
   ```bash
   # Linux (systemd)
   sudo systemctl start mongod
   sudo systemctl enable mongod   # para iniciar automaticamente

   # Windows
   net start MongoDB
   ```
3. Configure no `.env`:
   ```env
   MONGODB_URI=mongodb://localhost:27017
   MONGODB_DB=revisao_agents
   ```

### Como validar

```bash
# Verificar conexão (requer mongosh instalado)
mongosh "SUA_MONGODB_URI" --eval "db.adminCommand({ ping: 1 })"
# Esperado: { ok: 1 }
```

### Falhas comuns

| Erro | Causa | Solução |
|------|-------|---------|
| `ServerSelectionTimeoutError` | IP não autorizado no Atlas | Adicionar IP à lista de permissões no Atlas |
| `Authentication failed` | Usuário/senha errados na URI | Verificar credenciais no Atlas → Database Access |
| `Connection refused` | MongoDB local não está rodando | `sudo systemctl start mongod` |

---

## OpenAI

**Para que serve:** geração de embeddings (`text-embedding-3-small`) para o corpus vetorial, e como provedor LLM padrão.

**Obrigatório para:** indexação de PDFs (embeddings) e como LLM quando `LLM_PROVIDER=openai`.

> Mesmo usando outro provedor LLM (Google, Groq), a chave OpenAI ainda é necessária para os embeddings.

### Passo a passo

1. Acesse [platform.openai.com](https://platform.openai.com) e crie uma conta.
2. Adicione um método de pagamento em **Billing** → **Add payment method**.
   - O uso de embeddings é muito barato (~$0.02 por 1M tokens).
   - Adicione um crédito mínimo (ex: $5 USD) para começar.
3. Acesse **API Keys** em [platform.openai.com/api-keys](https://platform.openai.com/api-keys).
4. Clique em **"Create new secret key"**, dê um nome e copie a chave (começa com `sk-`).
   - **Atenção:** a chave só é exibida uma vez. Guarde em local seguro.
5. Configure no `.env`:
   ```env
   OPENAI_API_KEY=sk-...
   LLM_MODEL=gpt-4o-mini   # opcional — modelo padrão
   ```

### Como validar

```bash
uv run python -c "
from openai import OpenAI
client = OpenAI()
r = client.models.list()
print('OK — modelos disponíveis:', len(list(r)))
"
```

### Falhas comuns

| Erro | Causa | Solução |
|------|-------|---------|
| `AuthenticationError: Incorrect API key` | Chave inválida ou expirada | Gerar nova chave no painel |
| `RateLimitError` | Limite de requisições ou crédito insuficiente | Verificar uso e saldo em platform.openai.com/usage |
| `InsufficientQuotaError` | Sem crédito | Adicionar crédito em Billing |

---

## Google Gemini

**Para que serve:** provedor LLM alternativo ao OpenAI.

**Obrigatório para:** quando `LLM_PROVIDER=google` no `.env`.

### Passo a passo

1. Acesse [aistudio.google.com](https://aistudio.google.com) com sua conta Google.
2. Clique em **"Get API key"** no menu lateral.
3. Clique em **"Create API key"** e selecione um projeto Google Cloud (ou crie um novo).
4. Copie a chave gerada.
5. Configure no `.env`:
   ```env
   GOOGLE_API_KEY=AIza...
   LLM_PROVIDER=google
   LLM_MODEL=gemini-2.5-flash   # ou gemini-2.5-pro, gemini-2.0-flash, etc.
   ```

### Como validar

```bash
uv run python -c "
import google.generativeai as genai
import os
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
m = genai.GenerativeModel('gemini-2.5-flash')
r = m.generate_content('Olá')
print('OK:', r.text[:50])
"
```

### Falhas comuns

| Erro | Causa | Solução |
|------|-------|---------|
| `API key not valid` | Chave incorreta ou projeto sem APIs habilitadas | Verificar chave no Google AI Studio |
| `RESOURCE_EXHAUSTED` | Cota gratuita excedida | Aguardar resetar ou configurar billing no Google Cloud |
| `Model not found` | Modelo inválido | Verificar modelos disponíveis em aistudio.google.com |

---

## Groq

**Para que serve:** provedor LLM de alta velocidade (inferência rápida com modelos open-source como Llama, Mixtral).

**Obrigatório para:** quando `LLM_PROVIDER=groq` no `.env`.

### Passo a passo

1. Acesse [console.groq.com](https://console.groq.com) e crie uma conta.
2. No painel, vá em **API Keys** → **Create API Key**.
3. Dê um nome e copie a chave (começa com `gsk_`).
4. Configure no `.env`:
   ```env
   GROQ_API_KEY=gsk_...
   LLM_PROVIDER=groq
   LLM_MODEL=llama-3.3-70b-versatile   # ou meta-llama/llama-3-8b-instruct, etc.
   ```

### Limites do plano gratuito Groq

O plano gratuito tem limites de tokens por minuto e por dia. Se o processamento for intenso, o sistema pode receber erros 429 (rate limit) ou 413 (contexto muito grande). Nesse caso:
- Reduza o número de rodadas de refinamento.
- Use o modelo menor (ex: `llama-3.1-8b-instant`).

### Como validar

```bash
uv run python -c "
from groq import Groq
import os
client = Groq(api_key=os.environ['GROQ_API_KEY'])
r = client.chat.completions.create(
    model='llama-3.1-8b-instant',
    messages=[{'role': 'user', 'content': 'Olá'}]
)
print('OK:', r.choices[0].message.content[:50])
"
```

### Falhas comuns

| Erro | Causa | Solução |
|------|-------|---------|
| `AuthenticationError` | Chave inválida | Gerar nova chave no console Groq |
| `RateLimitError (429)` | Limite de tokens por minuto | Reduzir rodadas ou aguardar cooldown |
| `413 Request Entity Too Large` | Contexto muito grande | Reduzir número de seções ou usar modelo menor |

---

## Tavily

**Para que serve:** busca web em tempo real para encontrar fontes online durante a escrita técnica e revisão interativa.

**Obrigatório para:** aba ✍️ Write com busca web ativada, aba 🤖 Revisão Interativa com busca web, aba 📚 References com resolução de metadados.

### Passo a passo

1. Acesse [app.tavily.com](https://app.tavily.com) e crie uma conta.
2. Após confirmar o e-mail, acesse o **Dashboard**.
3. Copie a **API Key** exibida (começa com `tvly-`).
4. Configure no `.env`:
   ```env
   TAVILY_API_KEY=tvly-...
   ```

### Limites do plano gratuito

O plano gratuito oferece 1.000 requisições/mês. Para uso intenso, considere o plano pago.

### Como validar

```bash
uv run python -c "
from tavily import TavilyClient
import os
client = TavilyClient(api_key=os.environ['TAVILY_API_KEY'])
r = client.search('teste de conexão')
print('OK — resultados:', len(r.get('results', [])))
"
```

### Falhas comuns

| Erro | Causa | Solução |
|------|-------|---------|
| `InvalidAPIKeyError` | Chave inválida | Copiar chave corretamente do dashboard |
| `UsageLimitExceeded` | Cota mensal esgotada | Verificar uso em app.tavily.com, aguardar virada do mês |
| Resultados sempre vazios | Chave válida mas problema na query | Testar com query simples ("Python programming") |

---

## OpenRouter

**Para que serves:** acesso a múltiplos modelos de vários provedores (OpenAI, Anthropic, Meta, Mistral etc.) através de uma única API e chave.

**Obrigatório para:** quando `LLM_PROVIDER=openrouter` no `.env`.

### Passo a passo

1. Acesse [openrouter.ai](https://openrouter.ai) e crie uma conta.
2. Vá em **Keys** → **Create Key**.
3. Dê um nome e copie a chave (começa com `sk-or-`).
4. Adicione crédito em **Credits** (necessário para modelos pagos).
5. Configure no `.env`:
   ```env
   OPENROUTER_API_KEY=sk-or-...
   LLM_PROVIDER=openrouter
   LLM_MODEL=openai/gpt-4o-mini   # ou anthropic/claude-3-haiku, meta-llama/llama-3-8b-instruct:free, etc.
   ```

> **Dica:** Alguns modelos no OpenRouter têm versões gratuitas (marcadas com `:free`). Exemplo: `meta-llama/llama-3-8b-instruct:free`.

### Como validar

```bash
uv run python -c "
from openai import OpenAI
import os
client = OpenAI(
    api_key=os.environ['OPENROUTER_API_KEY'],
    base_url='https://openrouter.ai/api/v1'
)
r = client.chat.completions.create(
    model='openai/gpt-4o-mini',
    messages=[{'role': 'user', 'content': 'Olá'}]
)
print('OK:', r.choices[0].message.content[:50])
"
```

### Falhas comuns

| Erro | Causa | Solução |
|------|-------|---------|
| `AuthenticationError` | Chave inválida | Gerar nova chave no painel |
| `InsufficientCredits` | Sem crédito | Adicionar crédito em openrouter.ai/credits |
| `Model not found` | Nome do modelo incorreto | Consultar lista em openrouter.ai/models |

---

## Checklist final

Após configurar todas as credenciais, use este checklist para confirmar que tudo está pronto:

### Perfil mínimo

- [ ] `MONGODB_URI` configurado e conexão testada (`ping: 1`)
- [ ] `MONGODB_DB` definido (padrão: `revisao_agents`)
- [ ] `OPENAI_API_KEY` configurada e testada (lista de modelos retorna OK)
- [ ] `TAVILY_API_KEY` configurada e testada (busca retorna resultados)
- [ ] `LLM_PROVIDER=openai` no `.env`
- [ ] UI inicia sem erros: `uv run python run_ui.py`

### Perfil completo (adicional ao mínimo)

- [ ] `GOOGLE_API_KEY` configurada e testada
- [ ] `GROQ_API_KEY` configurada e testada
- [ ] `OPENROUTER_API_KEY` configurada e testada
- [ ] Troca de provedor na UI funciona (dropdown no topo retorna `✅` para cada provedor)

### Verificação rápida de todo o ambiente

```bash
uv run python -c "
import os
vars = ['OPENAI_API_KEY', 'TAVILY_API_KEY', 'MONGODB_URI', 'LLM_PROVIDER']
for v in vars:
    val = os.environ.get(v, '')
    status = '✔' if val and '...' not in val else '✗'
    print(f'  {status} {v}')
"
```
