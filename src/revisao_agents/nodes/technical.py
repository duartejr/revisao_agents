"""
Technical review agents - LangGraph nodes for technical chapter planning.

Nodes for the technical review workflow:
- Web search for technical sources
- Initial technical plan generation
- Plan refinement based on user feedback
- Final technical review plan

Prompts are loaded from YAML files in prompts/technical/.
"""

import mlflow
from observability.search_metrics import SearchQualityMetrics

from ..config import PLANS_DIR
from ..state import ReviewState
from ..utils.file_utils.helpers import fmt_snippets, save_md, truncate
from ..utils.llm_utils.llm_providers import get_llm
from ..utils.llm_utils.prompt_loader import load_prompt
from ..utils.search_utils.tavily_client import search_technical_content
from .common import build_search_query


@mlflow.trace(name="initial_technical_search", span_type="RETRIEVER")
def initial_technical_search_node(state: ReviewState) -> dict:
    """Initial search for technical content via Tavily.

    Args:
        state (ReviewState): The current state of the review, expected to contain:
            - "theme": str, the review topic/theme to search for.

    Returns:
        dict: Updated state keys:
            - "technical_urls": list[str], accumulated URLs from this search.
            - "technical_snippets": list[dict], search result snippets.
            - "status": str, set to ``"initial_technical_search_ok"``.
            - "urls_search_history": dict[str, int], URL appearance counts
              initialised from this first search.
            - "total_credits_used": float, Tavily credits consumed so far.
            - "total_search_queries": int, set to ``1`` after this node.
    """
    theme = state["theme"]
    print("\n[Initial technical search] theme:", repr(theme))
    ans = search_technical_content(theme, [])
    urls = ans.get("total_accumulated", [])
    new_urls = ans.get("new_urls", [])
    snippets = ans.get("results", [])
    usage = ans.get("usage", {})
    credits_used = usage.get("credits", 0.0)

    urls_search_history = SearchQualityMetrics.update_urls_search_history({}, new_urls)

    metrics_to_log = SearchQualityMetrics.calculate_all_search_metrics(
        new_urls=new_urls,
        total_accumulated=urls,
        urls_search_history=urls_search_history,
        credits_used=credits_used,
        total_credits_used=credits_used,
        total_search_queries=1,
    )
    SearchQualityMetrics.log_all_metrics_to_mlflow(metrics_to_log)

    return {
        "technical_urls": urls,
        "technical_snippets": snippets,
        "status": "initial_technical_search_ok",
        "urls_search_history": urls_search_history,
        "total_credits_used": credits_used,
        "total_search_queries": 1,
    }


@mlflow.trace(name="initial_technical_plan", span_type="AGENT")
def initial_technical_plan_node(state: ReviewState) -> dict:
    """Generates the initial draft of the technical plan.

    Args:
        state (ReviewState): The current state of the review, expected to contain:
            - "theme": str, the review topic/theme to search for.

    Returns:
        dict: Updated state with the initial technical plan and status.
    """
    theme = state["theme"]
    snippets = fmt_snippets(state.get("technical_snippets", []), 1200)
    prompt = load_prompt("technical/initial_plan", theme=theme, snippets=snippets)
    resp = get_llm(temperature=prompt.temperature).invoke(prompt.text)
    plano = resp.content if hasattr(resp, "content") else str(resp)
    print("\nInitial technical plan generated.")
    return {"current_plan": plano, "status": "initial_technical_plan_ready"}


@mlflow.trace(name="refine_technical_search", span_type="RETRIEVER")
def refine_technical_search_node(state: ReviewState) -> dict:
    """Refines the web search for technical content via Tavily.

    Uses an LLM to translate the latest interview question/answer pair into a
    focused query string, then performs a new Tavily web search, deduplicating
    against previously visited URLs and accumulating results.

    Args:
        state (ReviewState): The current state of the review, expected to contain:
            - "theme": str, the review topic/theme to search for.
            - "interview_history": list, the history of the interview.
            - "current_plan": str, the current draft technical plan.
            - "technical_urls": list[str], URLs already visited in previous searches.
            - "technical_snippets": list[dict], snippets accumulated from previous searches.
            - "urls_search_history": dict[str, int], URL appearance counts from prior
              searches (defaults to ``{}`` if absent).
            - "total_credits_used": float, cumulative Tavily credits so far (defaults to
              ``0.0`` if absent).
            - "total_search_queries": int, total queries executed so far (defaults to
              ``0`` if absent).

    Returns:
        dict: Updated state keys:
            - "technical_urls": list[str], expanded set of accumulated URLs.
            - "technical_snippets": list[dict], merged snippets (capped at 20).
            - "status": str, set to ``"refined_technical_search"``.
            - "urls_search_history": dict[str, int], updated URL appearance counts.
            - "total_credits_used": float, cumulative Tavily credits after this search.
            - "total_search_queries": int, incremented query counter.

    Raises:
        None: LLM and prompt errors are handled internally by ``build_search_query``;
            the node always returns a valid state dict.
    """
    query = build_search_query(state)
    print("\n[Refined technical search] interpreted query:", repr(query[:70]))
    urls_ant = state.get("technical_urls", [])
    ans = search_technical_content(query, urls_ant)
    urls = ans.get("total_accumulated", [])
    new_urls = ans.get("new_urls", [])
    total = ans.get("total_accumulated", urls_ant)
    snippets = ans.get("results", [])
    snippets_acum = (snippets + state.get("technical_snippets", []))[:20]
    credits_used = ans.get("usage", {}).get("credits", 0.0)

    old_history = state.get("urls_search_history", {})
    old_credits = state.get("total_credits_used", 0.0)
    old_queries = state.get("total_search_queries", 0)

    # Reconstruct the full set of URLs returned by this search:
    # new_urls (not in previous) + any previous URLs that appeared again
    # in the current Tavily response (detected via the raw urls_found list).
    urls_found_this_search = set(ans.get("urls_found", []))
    current_search_urls = new_urls + [u for u in urls_ant if u in urls_found_this_search]
    urls_history_updated = SearchQualityMetrics.update_urls_search_history(
        old_history, current_search_urls
    )

    metrics_to_log = SearchQualityMetrics.calculate_all_search_metrics(
        new_urls=new_urls,
        total_accumulated=urls,
        urls_search_history=urls_history_updated,
        credits_used=credits_used,
        total_credits_used=old_credits + credits_used,
        total_search_queries=old_queries + 1,
    )
    SearchQualityMetrics.log_all_metrics_to_mlflow(metrics_to_log)

    return {
        "technical_urls": total,
        "technical_snippets": snippets_acum,
        "status": "refined_technical_search",
        "urls_search_history": urls_history_updated,
        "total_credits_used": old_credits + credits_used,
        "total_search_queries": old_queries + 1,
    }


@mlflow.trace(name="refine_technical_plan", span_type="AGENT")
def refine_technical_plan_node(state: ReviewState) -> dict:
    """Updates the technical plan with new sources and feedback.

    Args:
        state (ReviewState): The current state of the review, expected to contain:
            - "theme": str, the review topic/theme to search for.
            - "current_plan": str, the current version of the technical plan.
            - "interview_history": list, the history of the interview.
            - "technical_snippets": list, the current technical snippets collected.

    Returns:
        dict: Updated state with the refined technical plan and status.
    """
    theme = state["theme"]
    current_plan = truncate(state["current_plan"], 700)
    last_msg = ""
    for role, c in reversed(state["interview_history"]):
        if role == "user":
            last_msg = c[:300]
            break
    snips = fmt_snippets(state.get("technical_snippets", [])[:5], 800)
    prompt = load_prompt(
        "technical/refine_plan",
        theme=theme,
        current_plan=current_plan,
        last_msg=last_msg,
        snips=snips,
    )
    ans = get_llm(temperature=prompt.temperature).invoke(prompt.text)
    plan = ans.content if hasattr(ans, "content") else str(ans)
    print("   Updated technical plan.")
    return {"current_plan": plan, "status": "refined_technical_plan"}


@mlflow.trace(name="finalize_technical_plan", span_type="AGENT")
def finalize_technical_plan_node(state: ReviewState) -> dict:
    """Generates the final technical plan and saves it in Markdown.

    Args:
        state (ReviewState): The current state of the review, expected to contain:
            - "theme": str, the review topic/theme to search for.
            - "current_plan": str, the current version of the technical plan.
            - "interview_history": list, the history of the interview.
            - "technical_snippets": list, the current technical snippets collected.
            - "technical_urls": list, the current technical URLs collected.

    Returns:
        dict: Updated state with the final technical plan, its path, and status.
    """
    theme = state["theme"]
    snips = fmt_snippets(state.get("technical_snippets", [])[:8], 800)
    urls = state.get("technical_urls", [])
    prompt = load_prompt("technical/finalize_plan", snips=snips)
    ans = get_llm(temperature=prompt.temperature).invoke(prompt.text)
    final_plan = ans.content if hasattr(ans, "content") else str(ans)
    print("\n" + "=" * 70)
    print("FINAL PLAN — TECHNICAL REVIEW")
    print("=" * 70)
    print(final_plan[:2000])
    print("=" * 70)
    urls_md = "\n".join("- " + u for u in urls[:30])
    md = (
        "# Technical Review Plan\n\n"
        "**Theme:** "
        + theme
        + "\n\n"
        + final_plan
        + "\n\n## Identified Technical URLs\n\n"
        + urls_md
        + "\n"
    )
    path = save_md(md, f"{PLANS_DIR}/technical_review_plan", theme)
    return {"final_plan": final_plan, "final_plan_path": path, "status": "completed"}
