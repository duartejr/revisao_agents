from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from ..nodes import (
    finalize_academic_plan_node,
    human_pause_node,
    initial_academic_plan_node,
    interview_node,
    interview_router,
    refine_academic_plan_node,
    refine_academic_search_node,
    vector_search_node,
)
from ..state import ReviewState


def build_academic_workflow():
    """Build the academic review workflow graph."""
    builder = StateGraph(ReviewState)
    builder.add_node("vector_search", vector_search_node)
    builder.add_node("initial_plan", initial_academic_plan_node)
    builder.add_node("interview", interview_node)
    builder.add_node("human_pause", human_pause_node)
    builder.add_node("refine_search", refine_academic_search_node)
    builder.add_node("refine_plan", refine_academic_plan_node)
    builder.add_node("finalize_plan", finalize_academic_plan_node)

    builder.set_entry_point("vector_search")
    builder.add_edge("vector_search", "initial_plan")
    builder.add_edge("initial_plan", "interview")
    builder.add_edge("interview", "human_pause")
    builder.add_edge("human_pause", "refine_search")
    builder.add_edge("refine_search", "refine_plan")
    builder.add_conditional_edges(
        "refine_plan",
        interview_router,
        {"continue": "interview", "finish": "finalize_plan"},
    )
    builder.add_edge("finalize_plan", END)

    return builder.compile(checkpointer=MemorySaver(), interrupt_before=["human_pause"])
