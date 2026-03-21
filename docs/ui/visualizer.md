# 📄 Aba View (Visualizador) — Leitura de Arquivos Gerados

## Objetivo

A aba **📄 View** (também chamada de **Visualizador**) permite navegar e visualizar qualquer arquivo `.md` gerado pelo sistema — planos da pasta `plans/` ou revisões da pasta `reviews/` — renderizado em Markdown formatado.

Use esta aba para ler o resultado de um plano ou revisão sem precisar abrir um editor de texto.

---

## Campos e controles

| Campo | Tipo | Descrição |
|-------|------|-----------|
| **Pasta** | Seleção (Radio) | `📋 Plans` — exibe arquivos da pasta `plans/` · `📝 Reviews` — exibe arquivos da pasta `reviews/` |
| **Arquivo** | Dropdown | Lista os arquivos `.md` disponíveis na pasta selecionada |
| **Botão Atualizar lista** | Botão (pequeno) | Recarrega a lista de arquivos (útil após gerar um novo plano/revisão) |
| **Botão View** | Botão | Carrega e renderiza o arquivo selecionado |
| **Área de visualização** | Markdown (altura 620px) | Exibe o conteúdo do arquivo formatado em Markdown |

---

## Fluxo passo a passo

1. Selecione a **pasta** onde o arquivo está: `📋 Plans` ou `📝 Reviews`.
2. O **dropdown** é preenchido automaticamente com os arquivos disponíveis.
3. Se não encontrar o arquivo esperado, clique em **"🔄 Refresh list"** para atualizar.
4. **Selecione o arquivo** no dropdown.
5. Clique em **"👁️ View"** para carregar e renderizar o conteúdo.
6. O arquivo é exibido formatado na área de visualização abaixo.

> **Dica:** Após gerar um plano ou revisão em outra aba, venha até o Visualizador e clique em "Refresh list" para ver o arquivo recém-gerado.

---

![Tela inicial do visualizador de md](imgs/visualizer_01_selecao_pasta.png)

---

## Erros comuns e como resolver

### Dropdown aparece vazio
- Não há arquivos `.md` na pasta selecionada.
- Gere um plano na aba **📋 Plan** ou uma revisão na aba **✍️ Write** primeiro.
- Clique em **"🔄 Refresh list"** após gerar o arquivo.

### Conteúdo não aparece ao clicar em View
- Certifique-se de que um arquivo está selecionado no dropdown.
- Tente clicar em "Refresh list" e selecionar o arquivo novamente.

### Arquivo aparece com caracteres estranhos
- O arquivo foi salvo em uma codificação diferente de UTF-8.
- Abra o arquivo diretamente em um editor de texto para verificar o conteúdo.
