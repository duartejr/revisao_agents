from .academic_workflow import build_academic_workflow
from .technical_workflow import build_technical_workflow
from .technical_writing_workflow import build_technical_writing_workflow

__all__ = [
    "build_academic_workflow",
    "build_technical_workflow",
    "build_technical_writing_workflow",
    "build_review_graph",
]


def _normalize_review_type(review_type: str | None) -> str:
    """Normalize review type string to one of the three canonical values.

    Recognized aliases and their outputs:

    - ``"technical"`` or ``"tecnico"`` → ``"tecnico"``
    - ``"writing"`` , ``"redacao"`` , or ``"redação"`` → ``"redacao"``
    - Any other value (including ``None`` or ``"academico"``) → ``"academico"``

    Args:
        review_type: Input string indicating the desired review type.  Case and
            leading/trailing whitespace are ignored.

    Returns:
        One of ``"academico"``, ``"tecnico"``, or ``"redacao"``.
    """
    value = (review_type or "academico").strip().lower()
    if value in {"technical", "tecnico"}:
        return "tecnico"
    if value in {"writing", "redacao", "redação"}:
        return "redacao"
    return "academico"


def build_review_graph(
    review_type: str = "academico",
    checkpointer=None,
    tipo: str | None = None,
):
    """Factory that returns the appropriate compiled graph.

    Args:
        review_type: "academico" | "tecnico" | "redacao"
        checkpointer: optional LangGraph checkpointer (defaults to MemorySaver)
        tipo: legacy alias for review_type ("academico" | "tecnico")

    Returns:
        Compiled LangGraph graph instance for the specified review type.
    """
    normalized = _normalize_review_type(tipo if tipo is not None else review_type)
    if normalized == "tecnico":
        return build_technical_workflow(checkpointer=checkpointer)
    if normalized == "redacao":
        return build_technical_writing_workflow()
    return build_academic_workflow(checkpointer=checkpointer)
