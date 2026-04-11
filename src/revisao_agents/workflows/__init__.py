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
    """Normalize review type string to canonical values.

    Args:
        review_type: Input string indicating review type (e.g., "academic", "technical")

    Returns:
        Normalized review type: "academic", "technical" or "writing"
    """
    value = (review_type or "academic").strip().lower()
    if value in {"technical", "tecnico"}:
        return "technical"
    if value in {"writing", "redacao", "redação"}:
        return "writing"
    return "academic"


def build_review_graph(
    review_type: str = "academic",
    checkpointer=None,
    tipo: str | None = None,
):
    """Factory that returns the appropriate compiled graph.

    Args:
        review_type: "academic" | "technical" | "writing"
        checkpointer: optional LangGraph checkpointer (defaults to MemorySaver)
        tipo: legacy alias for review_type ("academico" | "tecnico")

    Returns:
        Compiled LangGraph graph instance for the specified review type.
    """
    normalized = _normalize_review_type(tipo if tipo is not None else review_type)
    if normalized == "technical":
        return build_technical_workflow(checkpointer=checkpointer)
    if normalized == "writing":
        return build_technical_writing_workflow()
    return build_academic_workflow(checkpointer=checkpointer)
