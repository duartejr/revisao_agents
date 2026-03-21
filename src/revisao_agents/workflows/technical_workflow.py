from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..state import ReviewState
from ..nodes import (
    initial_technical_search_node,
    initial_technical_plan_node,
    interview_node,
    human_pause_node,
    refine_technical_search_node,
    refine_technical_plan_node,
    finalize_technical_plan_node,
    interview_router,
)


def build_technical_workflow():
    """Build the technical review workflow graph."""
    builder = StateGraph(ReviewState)
    builder.add_node("initial_technical_search", initial_technical_search_node)
    builder.add_node("initial_technical_plan", initial_technical_plan_node)
    builder.add_node("interview", interview_node)
    builder.add_node("human_pause", human_pause_node)
    builder.add_node("refine_technical_search", refine_technical_search_node)
    builder.add_node("refine_technical_plan", refine_technical_plan_node)
    builder.add_node("finalize_technical_plan", finalize_technical_plan_node)

    builder.set_entry_point("initial_technical_search")
    builder.add_edge("initial_technical_search", "initial_technical_plan")
    builder.add_edge("initial_technical_plan", "interview")
    builder.add_edge("interview", "human_pause")
    builder.add_edge("human_pause", "refine_technical_search")
    builder.add_edge("refine_technical_search", "refine_technical_plan")
    builder.add_conditional_edges(
        "refine_technical_plan",
        interview_router,
        {"continue": "interview", "finalize": "finalize_technical_plan"},
    )
    builder.add_edge("finalize_technical_plan", END)

    return builder.compile(checkpointer=MemorySaver(), interrupt_before=["human_pause"])

