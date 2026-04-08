from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from ..nodes import (
    finalize_technical_plan_node,
    human_pause_node,
    initial_technical_plan_node,
    initial_technical_search_node,
    interview_node,
    interview_router,
    refine_technical_plan_node,
    refine_technical_search_node,
)
from ..state import ReviewState


def build_technical_workflow(
    checkpointer: BaseCheckpointSaver | None = None,
) -> StateGraph[ReviewState]:
    """Build the technical review workflow graph.

    The workflow consists of the following steps:
    1. Initial Technical Search: Perform an initial search for relevant technical materials.
    2. Initial Technical Plan: Create an initial technical plan based on the search results.
    3. Interview: Conduct an interview to gather more information and feedback.
    4. Human Pause: Pause the workflow to allow for human review and input.
    5. Refine Technical Search: Refine the search based on feedback from the interview.
    6. Refine Technical Plan: Refine the technical plan based on the refined search results.
    7. Finalize Technical Plan: Finalize the technical plan based on the refined plan and
       feedback from the interview.

    Args:
        checkpointer (BaseCheckpointSaver | None): An optional BaseCheckpointSaver instance for checkpointing the workflow state.
            Enables persistence of workflow execution across sessions, allowing resumption from interruptions.
            If None, defaults to MemorySaver (in-memory, non-persistent).

    Returns:
        StateGraph[ReviewState]: The compiled state graph representing the technical review workflow.
    """

    if checkpointer is None:
        checkpointer = MemorySaver()

    if not isinstance(checkpointer, BaseCheckpointSaver):
        raise ValueError(
            "checkpointer must be an instance of BaseCheckpointSaver (e.g., MemorySaver, SqliteSaver, RedisSaver, etc.)\n"
            + f"Received type: {type(checkpointer)}"
        )

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
        {"continue": "interview", "finish": "finalize_technical_plan"},
    )
    builder.add_edge("finalize_technical_plan", END)

    return builder.compile(checkpointer=checkpointer, interrupt_before=["human_pause"])
