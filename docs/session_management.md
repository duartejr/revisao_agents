# Gerenciamento de Sessões e Checkpoints

> **Público:** Usuários e operadores
> **Funcionalidade:** Persistência de checkpoints LangGraph para sessões de planejamento

---

## Sumário

1. [Ciclo de vida de uma sessão](#1-ciclo-de-vida-de-uma-sessão)
2. [Mecânica dos checkpoints](#2-mecânica-dos-checkpoints)
3. [Backends de checkpoint](#3-backends-de-checkpoint)
4. [Configurando checkpoints](#4-configurando-checkpoints)
5. [Retomando sessões](#5-retomando-sessões)
6. [Limpeza de sessões](#6-limpeza-de-sessões)

---

## 1. Ciclo de vida de uma sessão

Uma sessão de planejamento percorre os seguintes estágios:

```
Início → Identificar e Refinar Tema
       ↓
   [Pausa HITL] ← Usuário fornece o tópico ou esclarece tema vago
       ↓
   Busca inicial (acadêmica ou técnica)
       ↓
   Geração do plano inicial
       ↓
   Entrevista → [Pausa HITL] ← Usuário refina o plano (N rodadas)
       ↓
   Finalizar Plano → FIM
```

A cada **Pausa HITL** (`human_pause_node`), o estado completo do grafo é salvo no backend de checkpoint. Isso permite retomar a sessão após uma interrupção.

### O que é um thread?

Cada sessão é identificada por um **ID de thread** único (UUID). O ID de thread é a chave estável usada para retomar a sessão correta.

- Na **UI**: um ID de thread é gerado automaticamente no início da sessão. Você pode copiá-lo na barra de status.
- Na **CLI**: passe `--thread-id seu-id` para reutilizar uma sessão existente, ou deixe um ser gerado automaticamente.

---

## 2. Mecânica dos checkpoints

### Quando os checkpoints são salvos?

Os checkpoints são salvos pelo LangGraph a cada **transição de nó do grafo**. O ponto de salvamento mais importante é imediatamente **antes** do `human_pause_node`, configurado via:

```python
builder.compile(checkpointer=checkpointer, interrupt_before=["human_pause"])
```

Isso significa: se a aplicação encerrar inesperadamente durante uma etapa de busca ou geração de plano, o checkpoint terá capturado o último nó concluído — mas os resultados do nó em andamento podem ser perdidos.

### O que é persistido?

O dict completo de `ReviewState` é persistido a cada checkpoint. Isso inclui:

| Campo | Descrição |
|---|---|
| `theme` | O tema da revisão |
| `detected_language` | `PT`, `EN` ou `UNKNOWN` |
| `review_type` | `academico` ou `tecnico` |
| `current_plan` | O plano gerado mais recente (Markdown) |
| `interview_history` | Todas as rodadas da conversa até o momento |
| `questions_asked` | Contador de rodadas |
| `relevant_chunks` / `technical_snippets` | Resultados das buscas |
| `is_theme_refined` | Se o tema foi aceito |

### Como funciona a retomada?

Na retomada, o LangGraph recarrega o último checkpoint para o ID de thread e continua a partir do próximo nó no grafo. O grafo **não** reexecuta nós já concluídos.

---

## 3. Backends de checkpoint

### Em memória (padrão)

```env
CHECKPOINT_TYPE=memory
```

| Propriedade | Valor |
|---|---|
| Persistência | ❌ Nenhuma — perdida quando o processo encerra |
| Configuração | Zero configuração |
| Múltiplas sessões | ❌ (apenas o estado mais recente por thread em RAM) |
| Melhor para | Desenvolvimento, demonstrações, fluxos de sessão única |

Com o backend em memória, **as sessões não podem ser retomadas** após reinicialização. Este é o padrão para evitar acúmulo acidental de dados.

### SQLite

```env
CHECKPOINT_TYPE=sqlite
CHECKPOINT_PATH=checkpoints/checkpoints.db
```

| Propriedade | Valor |
|---|---|
| Persistência | ✔ Sobrevive a reinicializações |
| Configuração | Mínima (caminho do arquivo) |
| Múltiplas sessões | ✔ |
| Melhor para | **Produção**, sessões longas, fluxos de vários dias |

O arquivo SQLite é criado automaticamente quando a aplicação inicia. O diretório é criado caso não exista.

> **Dependência:** `langgraph-checkpoint-sqlite` deve estar instalado.
> O script de bootstrap já o inclui. Instalação manual: `uv add langgraph-checkpoint-sqlite`

---

## 4. Configurando checkpoints

### Configuração no `.env`

```env
# Em memória (padrão — sessões não sobrevivem à reinicialização)
CHECKPOINT_TYPE=memory

# SQLite (recomendado para produção)
CHECKPOINT_TYPE=sqlite
CHECKPOINT_PATH=checkpoints/checkpoints.db
```

### Verificando o backend

```python
from revisao_agents.graphs.checkpoints import get_checkpointer
cp = get_checkpointer()
print(type(cp).__name__)
# MemorySaver   ← para CHECKPOINT_TYPE=memory
# SqliteSaver   ← para CHECKPOINT_TYPE=sqlite
```

### Listando IDs de threads armazenados

```python
from revisao_agents.graphs.checkpoints import list_thread_ids
print(list_thread_ids())
# ['abc123', 'def456', ...]
```

Disponível apenas para o backend `sqlite`. Retorna `[]` para `memory`.

---

## 5. Retomando sessões

### Pela UI

1. Na aba **Planejamento**, selecione o tipo de revisão.
2. No campo **Thread ID**, insira o ID do thread da sessão anterior.
3. Clique em **Carregar Thread** — o histórico de chat e o plano anteriores serão restaurados.
4. Continue a sessão normalmente.

> Se o thread não for encontrado (ID errado ou backend de memória após reinicialização), a UI mostrará uma sessão vazia.

### Pela CLI

```bash
# Retomar uma sessão existente
uv run revisao-agents "mesmo tema" --thread-id "seu-id-de-thread"
```

O agente detectará o checkpoint existente e continuará a partir da última pausa HITL.

### O que acontece na retomada?

- O conteúdo do plano é restaurado exatamente como estava quando a sessão foi interrompida
- O histórico da entrevista é reapresentado no chat
- O contador de rodadas (`questions_asked`) é preservado
- Uma nova busca **não** é acionada — apenas a retomada da etapa atual de refinamento do plano

---

## 6. Limpeza de sessões

### Limpeza manual (SQLite)

As sessões se acumulam no arquivo SQLite ao longo do tempo. Para limpar todas as sessões:

```bash
rm checkpoints/checkpoints.db
```

> ⚠️ Essa operação é irreversível. Todo o histórico de sessões será perdido.

### Alterando o caminho do checkpoint

Para isolar sessões por projeto, defina um caminho diferente:

```env
CHECKPOINT_PATH=checkpoints/projeto_abc.db
```

---

*Veja também: [Guia de configuração](setup_guide.md) · [Resolução de problemas](troubleshooting.md)*
