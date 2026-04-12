"""Public API for Gradio UI event handlers.

Re-exports every handler function and class needed by the Gradio app
layout so that ``app.py`` can do a single ``from gradio_app.handlers import
*`` without importing from individual sub-modules.
"""

from .base import (
    get_current_llm_provider,
    get_llm_provider_status,
    list_llm_providers,
    set_llm_provider,
)
from .planning import (
    continue_planning,
    list_available_threads,
    load_thread_state,
    start_planning,
)
from .review import (
    cancel_review_edit,
    confirm_review_edit,
    review_chat_turn,
    save_review_manual_edit,
    start_review_session,
)
from .review_parts.document import list_review_files
from .tools import format_references, index_pdfs
from .writing import list_plan_files, start_writing

__all__ = [
    "get_current_llm_provider",
    "get_llm_provider_status",
    "list_llm_providers",
    "set_llm_provider",
    "continue_planning",
    "list_available_threads",
    "load_thread_state",
    "start_planning",
    "cancel_review_edit",
    "confirm_review_edit",
    "list_review_files",
    "review_chat_turn",
    "save_review_manual_edit",
    "start_review_session",
    "format_references",
    "index_pdfs",
    "list_plan_files",
    "start_writing",
]
