"""
Technical review agents - LangGraph nodes for technical chapter planning.

Nodes for the technical review workflow:
- Web search for technical sources
- Initial technical plan generation
- Plan refinement based on user feedback
- Final technical review plan

Prompts are loaded from YAML files in prompts/technical/.
"""

from ..state import ReviewState
from ..utils.file_utils.helpers import fmt_snippets, save_md, truncate
from ..utils.llm_utils.llm_providers import get_llm
from ..utils.llm_utils.prompt_loader import load_prompt
from ..utils.search_utils.tavily_client import search_technical_content
from .common import build_search_query


def initial_technical_search_node(state: ReviewState) -> dict:
    """Initial search for technical content via Tavily.

    Args:
        state (ReviewState): The current state of the review, expected to contain:
            - "theme": str, the review topic/theme to search for.

    Returns:
        dict: Updated state with retrieved technical URLs and snippets, and status.
    """
    theme = state["theme"]
    print("\n[Initial technical search] theme:", repr(theme))
    ans = search_technical_content(theme, [])
    urls = ans.get("total_accumulated", [])
    snippets = ans.get("results", [])
    print("\n" + "=" * 70)
    print("TECHNICAL SOURCES FOUND")
    print("=" * 70)
    for i, r in enumerate(snippets, 1):
        print("\n  [" + str(i).rjust(2) + "] " + r.get("title", "")[:70])
        print("       " + r.get("url", "")[:80])
        print("       " + r.get("snippet", "")[:120] + "...")
    print("=" * 70)
    return {
        "technical_urls": urls,
        "technical_snippets": snippets,
        "status": "initial_technical_search_ok",
    }


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
            - "technical_urls": list, URLs already visited in previous searches.
            - "technical_snippets": list, snippets accumulated from previous searches.

    Returns:
        dict: Updated state with refined technical URLs and snippets, and status.

    Raises:
        None: LLM and prompt errors are handled internally by ``build_search_query``;
            the node always returns a valid state dict.
    """
    query = build_search_query(state)
    print("\n[Refined technical search] interpreted query:", repr(query[:70]))
    urls_ant = state.get("technical_urls", [])
    ans = search_technical_content(query, urls_ant)
    news = ans.get("new_urls", [])
    total = ans.get("total_accumulated", urls_ant)
    snips_n = ans.get("results", [])
    if news:
        print("\nNew sources (" + str(len(news)) + "):")
        for r in snips_n[:4]:
            print("  * " + r.get("title", "")[:60])
            print("    " + r.get("url", "")[:70])
    snips_acum = (snips_n + state.get("technical_snippets", []))[:20]
    return {
        "technical_urls": total,
        "technical_snippets": snips_acum,
        "status": "refined_technical_search",
    }


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
    path = save_md(md, "plans/technical_review_plan", theme)
    return {"final_plan": final_plan, "final_plan_path": path, "status": "completed"}
