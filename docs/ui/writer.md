# ✍️ Aba Write — Escrita de Seções

## Objetivo

A aba **✍️ Write** executa a escrita de um documento completo (revisão acadêmica ou capítulo técnico) a partir de um plano gerado na aba **📋 Plan**.

O agente percorre cada seção do plano, busca evidências no corpus local (MongoDB) e opcionalmente na web (Tavily), e escreve o conteúdo com base nas fontes encontradas. O progresso é exibido em tempo real no chat.

---

## Campos e controles

| Campo | Tipo | Descrição |
|-------|------|-----------|
| **Modo** | Seleção (Radio) | `Technical` (capítulo técnico) ou `Academic` (revisão narrativa) |
| **Plano** | Dropdown | Seleciona o arquivo `.md` de plano da pasta `plans/` — atualiza automaticamente ao mudar o modo |
| **Idioma** | Seleção (Radio) | `Português (pt-BR)` ou `English` |
| **Mínimo de fontes por seção** | Slider (0–10) | Número mínimo de fontes distintas que cada seção deve citar (0 = sem restrição) |
| **Busca web (Tavily)** | Checkbox | Permite que o agente busque fontes na internet além do corpus local |
| **Botão Iniciar** | Botão | Inicia o processo de escrita |
| **Chat de progresso** | Chat | Exibe atualizações em tempo real: nós do grafo, status de cada seção, erros |
| **Documento renderizado** | Área de Markdown | Exibe o documento completo ao final da execução |

---

## Fluxo passo a passo

1. **Certifique-se** de ter um plano gerado na aba 📋 Plan (ou use um arquivo `.md` de plano existente em `plans/`).
2. Selecione o **modo** de escrita: `Technical` para capítulo técnico, `Academic` para revisão narrativa.
3. O dropdown de planos se atualiza automaticamente. **Selecione o plano** desejado.
4. Escolha o **idioma** do documento gerado.
5. Defina o **mínimo de fontes por seção** (recomendado: 4 para academic, 0 para technical).
6. Marque **"Allow web search (Tavily)"** se quiser que o agente complemente com fontes da internet.
7. Clique em **"✍️ Start Writing"**.
8. Acompanhe o progresso no chat. Cada mensagem indica de qual nó ou seção o agente está trabalhando.
9. Ao finalizar, o documento completo aparece na área de Markdown abaixo do chat.
10. O arquivo é salvo automaticamente em `reviews/revisao_<tema>_<data>.md`.

> **Atenção:** A escrita pode demorar vários minutos dependendo do número de seções, do provedor LLM escolhido e da disponibilidade de fontes. Não feche a aba durante a execução.

---

![Tela inicial do writer](imgs/writer_01_configuracao.png)

---

## Erros comuns e como resolver

### Nenhum plano aparece no dropdown
- Verifique se existe algum arquivo `.md` na pasta `plans/`.
- Mude o modo (Technical ↔ Academic) para recarregar a lista.
- Gere um plano na aba **📋 Plan** primeiro.

### Seção gerada com conteúdo vazio ou muito curto
- Pode indicar que o corpus MongoDB não tem documentos relevantes para o tema.
- Ative a **busca web (Tavily)** para complementar com fontes externas.
- Aumente o tema do plano para cobrir mais sub-tópicos.

### Erro de conexão com MongoDB
```
pymongo.errors.ServerSelectionTimeoutError
```
- Verifique se o MongoDB está rodando e acessível.
- Confirme `MONGODB_URI` e `MONGODB_DB` no `.env`.
- Indexe PDFs relevantes na aba **📁 Index PDFs** antes de escrever.

### Tavily não encontra resultados
- Verifique se `TAVILY_API_KEY` está configurada no `.env`.
- Certifique-se de que sua cota Tavily não foi excedida em https://app.tavily.com.

### Escrita interrompida no meio
- O agente salva progresso por seção — o arquivo parcial pode estar em `reviews/`.
- Reinicie a escrita com o mesmo plano; seções já escritas podem ser reutilizadas.
