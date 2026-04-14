# 📋 Aba Plan — Planejamento de Revisão

## Objetivo

A aba **📋 Plan** é o ponto de partida do projeto. Ela inicia uma sessão interativa de planejamento onde o agente faz perguntas de refinamento sobre o tema informado e, ao final, gera um plano estruturado de revisão.

Dois tipos de revisão são suportados:
- **Acadêmica** — revisão narrativa de literatura (artigos, teses, periódicos)
- **Técnica** — capítulo técnico didático com base em fontes da web e corpus local

O plano gerado é salvo automaticamente na pasta `plans/` e pode ser usado como entrada na aba **✍️ Write**.

---

## Campos e controles

| Campo | Tipo | Descrição |
|-------|------|-----------|
| **Tema** | Caixa de texto | O tema ou pergunta central da revisão (ex: "Previsão de vazão com modelos de deep learning") |
| **Tipo de revisão** | Seleção (Radio) | `Academic (literature narrative)` · `Technical (didactic chapter)` · `Both` |
| **Rodadas de refinamento** | Slider (1–6) | Quantas rodadas de perguntas o agente fará antes de finalizar o plano |
| **Thread ID (Sessões existentes)** | Dropdown (vazio por padrão) | Lista as sessões salvas anteriormente; selecione uma para restaurar seu estado e histórico de conversa |
| **Botão 🔄 Atualizar sessões** | Botão | Recarrega a lista de sessões disponíveis no dropdown |
| **Botão Iniciar** | Botão | Inicia uma nova sessão de planejamento |
| **Campo de resposta** | Caixa de texto (3 linhas) | Aparece durante a sessão para o usuário responder às perguntas do agente |
| **Botão Responder** | Botão | Envia a resposta do usuário para o agente continuar |
| **Seletor de provedor LLM** | Dropdown (topo da tela) | Escolha entre google, groq, openai, openrouter |

---

## Fluxo passo a passo

### Fase 0 — Detecção de idioma e refinamento do tema (automática)

Antes de qualquer busca, o agente avalia o tema submetido:

- **Detecta o idioma** (Português · Inglês · Desconhecido).
- **Classifica o tema**: específico o suficiente ou vago demais (e.g. "Machine Learning", "IA", "Redes Neurais" são considerados vagos).

**Se o tema for específico e o idioma for detectado**, o agente confirma e avança direto para a busca. Nenhuma ação do usuário é necessária nessa fase.

**Se o tema for vago ou o idioma for desconhecido**, o agente exibe uma mensagem pedindo ao usuário que:
1. Refine o tema — forneça um tópico mais específico ou uma pergunta de pesquisa.
2. Confirme o idioma — responda `PT` (Português) ou `EN` (English) se não for possível detectá-lo automaticamente.

Após a resposta do usuário, a avaliação é refeita com o tema atualizado. O ciclo se repete até que o tema seja aprovado.

> **Exemplo de tema vago:** "Machine Learning" → o agente pedirá refinamento.
>
> **Exemplo de tema aprovado:** "Impacto do ENSO na variabilidade de chuvas na Amazônia" → o agente confirma e inicia a busca imediatamente.

---

### Fase 1 — Iniciar uma nova sessão

1. **Selecione o provedor LLM** no dropdown no topo da tela e aguarde o status mostrar `✅`.
2. **Digite o tema** da revisão no campo "Tema". Seja específico — quanto mais detalhado, melhor o plano gerado.
3. **Escolha o tipo** de revisão: Acadêmica (corpus MongoDB) ou Técnica (web + corpus).
4. **Ajuste as rodadas** de refinamento (padrão: 3). Mais rodadas = plano mais detalhado, mas demora mais.
5. **Clique em "🚀 Start Planning"**.
6. O agente avalia o tema (Fase 0). Se necessário, responda à mensagem de refinamento no campo abaixo e clique em **"💬 Reply"**.
7. Uma vez aprovado o tema, o agente exibirá a primeira pergunta de planejamento. **Leia** e **responda** no campo que aparece abaixo.
8. Clique em **"💬 Reply"** para enviar sua resposta.
9. Repita até o agente indicar que o plano está finalizado.
10. O **plano renderizado** aparece abaixo do chat assim que a sessão termina.
11. O arquivo é salvo automaticamente em `plans/plano_revisao_<tema>_<data>.md`.

> **Dica:** Se não souber responder a uma pergunta do agente, escreva "Não tenho preferência" ou "Pode decidir" — o agente continuará com escolhas padrão razoáveis.

### Retomar uma sessão existente

1. Clique em **"🔄 Atualizar sessões"** para carregar as sessões salvas no checkpoint.
2. No dropdown **Thread ID**, selecione a sessão desejada — o tema, tipo, rodadas e histórico de conversa serão restaurados automaticamente.
3. Continue enviando respostas normalmente a partir de onde parou.

> **Formato do Thread ID:** cada sessão recebe um identificador único no formato `revisao_<tipo>_<tema>_<YYYY-MM-DD_HH-MM>`, garantindo unicidade mesmo para o mesmo tema aberto em horários diferentes.

---


![Tela inicial do planner](imgs/planner_01_tela_inicial.png)

![Tela inicial da aba Plan](../../docs/assets/planner_theme_refinement_01_initial.png)

---

### Fase 0 — Exemplo: tema vago detectado

O agente identificou que "Machine Learning" é muito genérico e solicitou refinamento antes de iniciar qualquer busca. O campo **Your answer** aparece para o usuário fornecer um tema mais específico.

![Refinamento de tema vago](../../docs/assets/planner_theme_refinement_02_vague_theme.png)

---

![Chat do planner](imgs/planner_02_chat_em_andamento.png)


---

## Erros comuns e como resolver

### O botão "Start Planning" não responde
- Verifique se o provedor LLM está selecionado e com status `✅` no topo da tela.
- Verifique se o campo "Tema" não está vazio.

### Erro de autenticação do provedor LLM
```
AuthenticationError: Invalid API key
```
- Confirme que a chave do provedor selecionado está configurada no `.env`.
- Troque para outro provedor no dropdown e tente novamente.

### O chat trava e não exibe resposta
- Pode ser um timeout de rede ou limite de tokens. Tente com um tema mais curto.
- Reduza o número de rodadas de refinamento para 1 ou 2.

### O plano não é salvo em `plans/`
- A pasta `plans/` é criada automaticamente. Se não existir, rode `mkdir plans` no diretório do projeto e tente novamente.

### Sessão expirada após inatividade
- Clique em "🚀 Start Planning" novamente para reiniciar uma nova sessão.
- Ou use o dropdown **Thread ID** para retomar a sessão anterior (se o checkpoint estiver em SQLite).

### O dropdown de sessões está vazio
- Certifique-se de que `CHECKPOINT_TYPE=sqlite` no `.env` (sessões em memória não são persistidas).
- Clique em **"🔄 Atualizar sessões"** para recarregar a lista.
