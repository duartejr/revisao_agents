# Documentação — Agente de Revisão da Literatura

Bem-vindo à documentação completa do sistema. Use os links abaixo para navegar.

---

## Início rápido

→ [README principal](../README.md) — instalação, bootstrap, primeiro uso, troubleshooting

---

## Interface Gráfica (UI)

Documentação detalhada de cada aba da interface Gradio:

| Aba | Descrição | Documento |
|-----|-----------|-----------|
| 📋 Plan | Planejamento interativo de revisão com HITL | [ui/planner.md](ui/planner.md) |
| ✍️ Write | Escrita de seções a partir de um plano | [ui/writer.md](ui/writer.md) |
| 🤖 Revisão Interativa | Edição e refinamento via chat | [ui/review_chat.md](ui/review_chat.md) |
| 📄 View (Visualizador) | Leitura de planos e revisões gerados | [ui/visualizer.md](ui/visualizer.md) |
| 📚 References | Formatação de referências bibliográficas | [ui/references.md](ui/references.md) |
| 📁 Index PDFs | Indexação de PDFs no corpus vetorial | [ui/pdf_indexer.md](ui/pdf_indexer.md) |

---

## Linha de Comando (CLI)

→ [cli.md](cli.md) — `revisao-agents` script, menu interativo `python -m revisao_agents`, exemplos e troubleshooting

---

## Contas e Credenciais

→ [contas_e_credenciais.md](contas_e_credenciais.md) — passo a passo para configurar MongoDB, OpenAI, Google, Groq, Tavily e OpenRouter

---

## Arquitetura

→ [architecture.md](architecture.md) — estrutura do projeto, fluxos de dados, provedores suportados

---

## Referência rápida de comandos

```bash
# Iniciar UI
uv run python run_ui.py

# Menu interativo CLI
uv run python -m revisao_agents

# CLI script — planejamento
uv run revisao-agents "Meu tema" --review-type academic
uv run revisao-agents "Meu tema" --review-type technical --rounds 4 --output plans/plano.md

# Ajuda
uv run revisao-agents --help
```

---

## Referência rápida de variáveis de ambiente

| Variável | Obrigatória | Descrição |
|----------|-------------|-----------|
| `LLM_PROVIDER` | ✔ | `openai` · `google` · `groq` · `openrouter` |
| `OPENAI_API_KEY` | ✔ | Chave OpenAI (embeddings + LLM se provider=openai) |
| `TAVILY_API_KEY` | ✔ | Chave Tavily para busca web |
| `MONGODB_URI` | ✔ | URI de conexão MongoDB |
| `MONGODB_DB` | ✔ | Nome do banco (padrão: `revisao_agents`) |
| `GOOGLE_API_KEY` | se provider=google | Chave Google Gemini |
| `GROQ_API_KEY` | se provider=groq | Chave Groq |
| `OPENROUTER_API_KEY` | se provider=openrouter | Chave OpenRouter |
| `LLM_MODEL` | opcional | Modelo LLM específico |
| `TEMPERATURE` | opcional | Temperatura do modelo (padrão: 0.3) |
