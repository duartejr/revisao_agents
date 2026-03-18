from typing import TypedDict, Annotated, List
import operator


class ReviewState(TypedDict):
    """State shared across academic and technical review workflows."""
    theme: str
    review_type: str
    relevant_chunks: List[str]
    technical_snippets: List[dict]
    technical_urls: List[str]
    current_plan: str
    interview_history: Annotated[List[tuple], operator.add]
    questions_asked: int
    max_questions: int
    final_plan: str
    final_plan_path: str
    status: str


class TechnicalWriterState(TypedDict):
    """State specific to the technical and academic writing workflow."""
    theme: str
    plan_summary: str
    sections: List[dict]
    plan_path: str
    written_sections: List[dict]
    refs_urls: List[str]
    refs_images: List[dict]
    cumulative_summary: str
    react_log: List[str]
    verification_stats: List[dict]
    status: str
    writer_config: dict  # WriterConfig.to_dict() — empty dict means technical defaults
    tavily_enabled: bool  # If False, disables all Tavily web/image search and extraction


class ReviewChatState(TypedDict):
    """State specific to the interactive review chatbot session."""
    original_file_path: str
    working_copy_path: str
    chat_history: List[dict]
    pending_edit: dict
    last_target_resolution: dict
    retrieval_trace: List[dict]
    status: str
