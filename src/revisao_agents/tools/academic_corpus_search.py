# src/revisao_agents/tools/academic_corpus_search.py
"""
Official search tool for the MongoDB corpus.
Simple and high-performance wrapper for direct use by agents.
"""

from langchain_core.tools import tool

# Import the original class (still in the utils folder for now)
from ..utils.vector_utils.mongodb_corpus import CorpusMongoDB


@tool
def search_academic_corpus(
    query: str,
    limit: int = 5,
    section_title: str = "",
) -> str:
    """
    Searches the MongoDB corpus for relevant academic content using vector search.

    Args:
        query: Text or anchor to search for (e.g., "stable diffusion model").
        limit: Maximum number of sources/chunks to return (default 5).
        section_title: Title of the current section (used only for internal logging).

    Returns:
        Formatted string ready to be pasted into the agent's prompt (with source headers).
        Includes the most relevant sources + full context.
    """
    try:
        corpus = CorpusMongoDB()
        # Use render_prompt but honour the caller's limit by restricting top_k
        # before the call so only `limit` chunks are retrieved.
        corpus._top_k_override = limit  # stored temporarily
        _original_query = corpus.query

        def _limited_query(texto_query: str, top_k: int = limit) -> list:
            return _original_query(texto_query, top_k=limit)

        corpus.query = _limited_query
        context, used_urls, _ = corpus.render_prompt(
            query=query,
            max_chars=8000,  # safe limit for LLM context
        )
        corpus.query = _original_query  # restore

        if not context.strip():
            return f"No relevant sources found for: '{query}'"

        header = (
            f"=== SOURCES FOUND FOR: '{query}' ===\n"
            f"Section: {section_title or 'Not provided'}\n"
            f"Total chunks used: {len(used_urls)}\n"
            f"{'=' * 60}\n\n"
        )

        return header + context

    except Exception as e:
        return f"Error searching the corpus: {str(e)}"
