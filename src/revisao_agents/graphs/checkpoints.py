"""
checkpoints.py - Persistence / checkpointer factories for LangGraph.

Supports in-memory (default) and can be extended to SQLite / Postgres.
"""

from __future__ import annotations

import os
from typing import Any

from ..config import get_checkpointer_vars


def get_checkpointer() -> Any:
    """Factory function to create a checkpointer instance based on environment variables.

    Supported backends:
        - memory (default): In-memory, non-persistent checkpointer.
        - sqlite: Persistent checkpointer using SQLite database.

    Environment Variables:
        - CHECKPOINT_TYPE: 'memory' (default) | 'sqlite'. Specifies the type of checkpointer to use.
        - CHECKPOINT_PATH: File path for SQLite database (default: 'checkpoints/checkpoints.db'). Must be a valid path.

    Returns:
        A checkpointer instance compatible with LangGraph's checkpointing system.

    Raises:
        ValueError: If CHECKPOINT_TYPE is unsupported or CHECKPOINT_PATH is invalid/empty.
        ImportError: If required packages for the selected backend are not installed.
    """
    vars = get_checkpointer_vars()
    checkpoint_type = vars.get("CHECKPOINT_TYPE", "memory")
    checkpoint_path = vars.get("CHECKPOINT_PATH", "checkpoints/checkpoints.db")

    if not checkpoint_path or not isinstance(checkpoint_path, str):
        raise ValueError("CHECKPOINT_PATH must be a non-empty string")

    if checkpoint_type == "sqlite":
        try:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        except OSError as e:
            raise ValueError(f"Failed to create directory for SQLite checkpoint: {e}") from e

        # Validate that the directory is writable
        dir_path = os.path.dirname(checkpoint_path)
        if not os.access(dir_path, os.W_OK):
            raise ValueError(f"Directory '{dir_path}' is not writable")

        try:
            import sqlite3

            from langgraph.checkpoint.sqlite import SqliteSaver

            # check_same_thread=False allows the connection to be used across threads,
            # which is necessary for LangGraph's async operations
            return SqliteSaver(sqlite3.connect(checkpoint_path, check_same_thread=False))
        except ImportError as e:
            raise ImportError(
                "langgraph-checkpoint-sqlite is not installed. "
                "Run: pip install langgraph-checkpoint-sqlite"
            ) from e
        except Exception as e:
            raise ValueError(f"Failed to create SQLite checkpointer: {e}") from e
    elif checkpoint_type == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    else:
        raise ValueError(f"Unsupported CHECKPOINT_TYPE: {checkpoint_type}")
