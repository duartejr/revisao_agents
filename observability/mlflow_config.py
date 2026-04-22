"""MLflow configuration constants — isolated from the main execution config.

All MLflow-specific settings are read from environment variables here.
This module has no dependency on src/revisao_agents so that observability
concerns remain decoupled from agent execution logic.
"""

import os

# ---------------------------------------------------------------------------
# Tracking backend
# ---------------------------------------------------------------------------


#: URI for the MLflow tracking server.
#: Defaults to a local SQLite file at ``./mlruns/mlflow.db``.
#: Override with the ``MLFLOW_TRACKING_URI`` environment variable.
#:
#: .. note::
#:     This is evaluated lazily at call time so that loading order of ``.env`` does not matter.
def get_tracking_uri() -> str:
    """Return the MLflow tracking URI, reading the environment variable at call time."""
    return os.getenv("MLFLOW_TRACKING_URI", "sqlite:///./mlruns/mlflow.db")


# Backwards-compatible alias — prefer ``get_tracking_uri()`` for new code.
MLFLOW_TRACKING_URI: str = ""  # populated at first call via get_tracking_uri()

# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Experiment name constants — use these instead of bare strings
# ---------------------------------------------------------------------------

EXP_PLANNING_ACADEMIC: str = "planning_academic"
EXP_PLANNING_TECHNICAL: str = "planning_technical"
EXP_WRITING_ACADEMIC: str = "writing_academic"
EXP_WRITING_TECHNICAL: str = "writing_technical"
EXP_REVIEW_CHAT: str = "review_chat"

#: Mapping of canonical experiment names to human-readable descriptions.
#: Keys are the MLflow experiment names; values are display descriptions.
EXPERIMENTS: dict[str, str] = {
    EXP_PLANNING_ACADEMIC: "Academic Planning — Tavily depth and relevance comparison",
    EXP_PLANNING_TECHNICAL: "Technical Planning — Tavily depth and relevance comparison",
    EXP_WRITING_ACADEMIC: "Academic Writing — section generation quality metrics",
    EXP_WRITING_TECHNICAL: "Technical Writing — section generation quality metrics",
    EXP_REVIEW_CHAT: "Review Chat — interactive review interaction metrics",
}
