# 🤖 Aba Revisão Interativa — Edição Via Chat

## Objetivo

A aba **🤖 Revisão Interativa** permite revisar e editar documentos gerados (planos ou revisões) através de um chat com o agente. Você faz perguntas ou dá comandos em linguagem natural e o agente sugere ou aplica edições diretamente no documento.

O agente pode:
- Resumir seções, listar citações ou descrever o que o documento aborda
- Sugerir adições, reescritas ou melhorias em trechos específicos
- Buscar novas fontes na web (quando a busca web está ativada) e incorporá-las
- Aplicar edições propostas com sua aprovação explícita

---

## Campos e controles

| Campo | Tipo | Descrição |
|-------|------|-----------|
| **Arquivo de revisão** | Dropdown | Seleciona o arquivo `.md` da pasta `reviews/` para revisar |
| **Botão Atualizar** | Botão (pequeno) | Recarrega a lista de arquivos |
| **Botão Iniciar sessão** | Botão | Carrega o arquivo e inicia a sessão de revisão |
| **Busca web** | Checkbox | Ativa o Tavily para o agente buscar fontes online durante a revisão |
| **Área de preview** | Caixa de texto (editável) | Exibe o documento em formato editável — você pode editar diretamente aqui |
| **Campo de pergunta/comando** | Caixa de texto (3 linhas) | Digite sua pergunta ou instrução para o agente |
| **Botão Enviar** | Botão | Envia a mensagem para o agente |
| **Botão Confirmar edição** | Botão | Aplica a edição proposta pelo agente ao documento |
| **Botão Cancelar edição** | Botão | Descarta a edição proposta e mantém o documento atual |
| **Botão Salvar edição manual** | Botão | Salva a versão do documento que você editou manualmente no preview |
| **Status** | Campo de texto (somente leitura) | Indica o estado da sessão: `🟡 Pending edit` ou `✅ Session active` |

---

## Fluxo passo a passo

1. No dropdown **Arquivo de revisão**, selecione o `.md` que deseja revisar (da pasta `reviews/`).
   - Se não encontrar o arquivo, clique em **"🔄 Refresh files"**.
2. Opcionalmente ative **"🌐 Allow web search"** se quiser que o agente busque novas referências online.
3. Clique em **"▶ Start session"**.
   - O agente carrega o arquivo e cria uma cópia de trabalho com carimbo de data/hora (ex: `...__review_edit_20260321_1530.md`).
   - O conteúdo aparece na área de preview à direita.
4. **Digite sua pergunta ou comando** no campo de texto (exemplos abaixo) e clique em **"💬 Send"**.
5. O agente responde no chat com análise, resposta ou proposta de edição.
6. Se houver uma **edição proposta**:
   - O status muda para `🟡 Pending edit`
   - Clique em **"✅ Confirm Edit"** para aplicar ao documento
   - Ou clique em **"🗑️ Cancel Edit"** para descartar
7. Você também pode **editar o texto diretamente** no preview e clicar em **"💾 Save manual edit"** para salvar.
8. Continue enviando perguntas e confirmando edições até ficar satisfeito.

### Exemplos de comandos para o agente

```
Quais são as principais referências citadas nesta seção?
Reescreva o parágrafo de introdução deixando mais objetivo.
Adicione uma subseção sobre limitações do método.
Melhore a argumentação do terceiro parágrafo.
Pesquise artigos recentes sobre o tema e sugira como incorporá-los.
Verifique se há inconsistências entre as seções 2 e 4.
```

---

![Tela inicial do review chat](imgs/review_chat_01_inicio_sessao.png)

---

## Erros comuns e como resolver

### Dropdown de arquivos vazio
- Gere uma revisão na aba **✍️ Write** antes de usar esta aba.
- Clique em **"🔄 Refresh files"**.

### Agente não propõe edições, apenas responde no chat
- Comandos de consulta (ex: "quais são as referências?") retornam respostas sem edição — isso é esperado.
- Para edições, use comandos explícitos: "reescreva", "adicione", "melhore", "corrija".

### Status permanece `🟡 Pending edit` após confirmar
- Clique novamente em **"✅ Confirm Edit"**.
- Se persistir, use **"💾 Save manual edit"** para salvar o estado atual do preview manualmente.

### Busca web não encontra resultados
- Verifique se `TAVILY_API_KEY` está configurada no `.env`.
- Desative a busca web e tente apenas com o conteúdo local do documento.

### Edição manual não salva
- Certifique-se de clicar em **"💾 Save manual edit"** após editar o preview.
- O botão salva o conteúdo atual da área de preview, sobrescrevendo o arquivo de trabalho.
