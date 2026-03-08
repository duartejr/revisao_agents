# src/revisao_agents/state.py
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator


class ReviewState(TypedDict):
    """Estado compartilhado entre os nós do grafo de revisão."""

    # Texto original e versões em progresso
    original_text: str
    current_text: str                  # texto sendo revisado a cada iteração
    final_text: str | None             # saída final após todas as revisões

    # Mensagens / histórico (para memória e debug)
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # Feedback humano (quando HITL ativado)
    human_feedback: str | None

    # Metadados / controle de fluxo
    revision_count: int = 0
    needs_human_review: bool = False
    error: str | None = None