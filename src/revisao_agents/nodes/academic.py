"""
Academic review agents - LangGraph nodes for literature review planning.

Nodes for the academic review workflow:
- Vector search for relevant papers
- Initial academic plan generation
- Plan refinement based on user feedback
- Final academic review plan

Prompts are loaded from YAML files in prompts/academic/.
"""

from ..config import PLANS_DIR
from ..state import ReviewState
from ..utils.file_utils.helpers import fmt_chunks, save_md, truncate
from ..utils.llm_utils.llm_providers import get_llm
from ..utils.llm_utils.prompt_loader import load_prompt
from ..utils.vector_utils.vector_store import accumulate_chunks, search_chunks
from .common import build_search_query

# Constants (may need to be moved to config)
CHUNKS_PER_QUERY = 10  # TODO: Move to config if it should be configurable


def vector_search_node(state: ReviewState) -> dict:
    """Search for initial chunks about the theme. Uses the theme as the query and retrieves relevant chunks from the vector store.

    Args:
        state (ReviewState): The current state of the review, expected to contain:
            - "theme": str, the review topic/theme to search for.

    Returns:
        dict: Updated state with retrieved relevant chunks and status.
    """
    theme = state["theme"]
    print("\n[MONGODB] query:", repr(theme))
    chunks = search_chunks(theme)
    print("   ", len(chunks), "recovered chunks")
    return {"relevant_chunks": chunks, "status": "chunks_ok"}


def initial_academic_plan_node(state: ReviewState) -> dict:
    """Generates the initial draft of the academic plan.

    Args:
        state (ReviewState): The current state of the review, expected to contain:
            - "theme": str, the review topic/theme to search for.

    Returns:
        dict: Updated state with the initial academic plan and status.
    """
    theme = state["theme"]
    ctx = fmt_chunks(state["relevant_chunks"], 900)
    prompt = load_prompt("academic/initial_plan", theme=repr(theme), ctx=ctx)
    resp = get_llm(temperature=prompt.temperature).invoke(prompt.text)
    plan = resp.content if hasattr(resp, "content") else str(resp)
    print("\nInitial academic plan generated.")
    return {"current_plan": plan, "status": "initial_plan_ready"}


def refine_academic_search_node(state: ReviewState) -> dict:
    """Refines the vector search based on the user's last question.

    Uses an LLM to translate the latest interview question/answer pair into a
    focused query string, then re-searches the vector store and accumulates the
    results with previously retrieved chunks.

    Args:
        state (ReviewState): The current state of the review, expected to contain:
            - "theme": str, the review topic/theme to search for.
            - "interview_history": list of tuples, the history of the interview.
            - "current_plan": str, the current draft academic plan.
            - "relevant_chunks": list, chunks accumulated from previous searches.

    Returns:
        dict: Updated state with refined relevant chunks and status.

    Raises:
        None: LLM and prompt errors are handled internally by ``build_search_query``;
            the node always returns a valid state dict.
    """
    query = build_search_query(state)
    print("\n[MONGODB re-query] interpreted query:", repr(query[:70]))
    novos = search_chunks(query)
    acum = accumulate_chunks(state["relevant_chunks"], novos)
    print("   ", len(novos), "retrieved | total:", len(acum))
    return {"relevant_chunks": acum, "status": "chunks_refined"}


def refine_academic_plan_node(state: ReviewState) -> dict:
    """Updates the academic plan with new chunks and feedback.

    Args:
        state (ReviewState): The current state of the review, expected to contain:
            - "theme": str, the review topic/theme to search for.
            - "current_plan": str, the current academic plan.
            - "interview_history": list of tuples, the history of the interview.

    Returns:
        dict: Updated state with the refined academic plan and status.
    """
    theme = state["theme"]
    current_plan = truncate(state["current_plan"], 700)
    last_feedback = ""
    for role, c in reversed(state["interview_history"]):
        if role == "user":
            last_feedback = c[:300]
            break
    new_sources = fmt_chunks(state["relevant_chunks"][-CHUNKS_PER_QUERY:], 600)
    prompt = load_prompt(
        "academic/refine_plan",
        theme=repr(theme),
        current_plan=current_plan,
        last_feedback=last_feedback,
        new_sources=new_sources,
    )
    ans = get_llm(temperature=prompt.temperature).invoke(prompt.text)
    plan = ans.content if hasattr(ans, "content") else str(ans)
    print("   Academic plan updated.")
    return {"current_plan": plan, "status": "plan_refined"}


def finalize_academic_plan_node(state: ReviewState) -> dict:
    """Generates the final academic plan and saves it in Markdown.

    Args:
        state (ReviewState): The current state of the review, expected to contain:
            - "theme": str, the review topic/theme to search for.
            - "current_plan": str, the current academic plan.
            - "relevant_chunks": list, the relevant chunks for the review.

    Returns:
        dict: Updated state with the final academic plan, its path, and status.
    """
    theme = state["theme"]
    current_plan = truncate(state["current_plan"], 1000)
    ctx = fmt_chunks(state["relevant_chunks"], 800)
    prompt = load_prompt(
        "academic/finalize_plan",
        theme=repr(theme),
        current_plan=current_plan,
        ctx=ctx,
    )
    ans = get_llm(temperature=prompt.temperature).invoke(prompt.text)
    final_plan = ans.content if hasattr(ans, "content") else str(ans)
    print("\n" + "=" * 70)
    print("FINAL PLAN — ACADEMIC REVIEW")
    print("=" * 70)
    print(final_plan)
    print("=" * 70)
    md = "# Plano de Revisao da Literatura\n\n**Tema:** " + theme + "\n\n" + final_plan
    path = save_md(md, f"{PLANS_DIR}/review_plan", theme)
    return {"final_plan": final_plan, "final_plan_path": path, "status": "completed"}
