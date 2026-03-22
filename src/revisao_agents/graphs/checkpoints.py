"""
checkpoints.py - Persistence / checkpointer factories for LangGraph.

Supports in-memory (default) and can be extended to SQLite / Postgres.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver


def make_checkpointer(backend: str = "memory", **kwargs):
    """
    Return a LangGraph checkpointer.

    Args:
        backend: "memory" (default) | "sqlite" | "postgres" (future)
        **kwargs: backend-specific connection arguments

    Returns:
        A LangGraph-compatible checkpointer instance.
    """
    if backend == "memory":
        return MemorySaver()

    if backend == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            db_path = kwargs.get("db_path", "checkpoints.db")
            return SqliteSaver.from_conn_string(db_path)
        except ImportError as e:
            raise ImportError(
                "langgraph-checkpoint-sqlite is not installed. "
                "Run: pip install langgraph-checkpoint-sqlite"
            ) from e

    if backend == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver

            conn_string = kwargs["conn_string"]  # required
            return PostgresSaver.from_conn_string(conn_string)
        except ImportError as e:
            raise ImportError(
                "langgraph-checkpoint-postgres is not installed. "
                "Run: pip install langgraph-checkpoint-postgres"
            ) from e

    raise ValueError(f"Unknown checkpointer backend: {backend!r}")
