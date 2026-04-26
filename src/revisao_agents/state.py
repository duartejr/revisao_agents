import operator
from typing import Annotated, TypedDict


class ReviewState(TypedDict):
    """State shared across academic and technical review workflows."""

    theme: str
    review_type: str
    relevant_chunks: list[str]
    technical_snippets: list[dict]
    technical_urls: list[str]
    current_plan: str
    interview_history: Annotated[list[tuple], operator.add]
    questions_asked: int
    max_questions: int
    final_plan: str
    final_plan_path: str
    status: str
    detected_language: str
    user_language_choice: str
    is_theme_vague: bool
    is_theme_refined: bool
    confidence_score: float
    refinement_feedback: list[str]
    urls_search_history: dict[str, int]  # Maps URL → number of times it appeared across searches
    total_credits_used: float  # Cumulative Tavily credits consumed in this session
    total_search_queries: int  # Total number of search queries executed in this session


class TechnicalWriterState(TypedDict):
    """State specific to the technical and academic writing workflow."""

    theme: str
    plan_summary: str
    sections: list[dict]
    plan_path: str
    written_sections: list[dict]
    refs_urls: list[str]
    refs_images: list[dict]
    cumulative_summary: str
    react_log: list[str]
    verification_stats: list[dict]
    status: str
    writer_config: dict  # WriterConfig.to_dict() — empty dict means technical defaults
    tavily_enabled: bool  # If False, disables all Tavily web/image search and extraction


class ReviewChatState(TypedDict):
    """State specific to the interactive review chatbot session."""

    original_file_path: str
    working_copy_path: str
    chat_history: list[dict]
    pending_edit: dict
    last_target_resolution: dict
    retrieval_trace: list[dict]
    status: str
