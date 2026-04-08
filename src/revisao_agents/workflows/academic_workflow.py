from langgraph.checkpoint.base import BaseCheckpointSaver
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


def build_academic_workflow(
    checkpointer: BaseCheckpointSaver | None = None,
) -> StateGraph[ReviewState]:
    """Build the academic review workflow graph.

    The workflow consists of the following steps:
    1. Vector Search: Perform an initial search for relevant academic materials.
    2. Initial Plan: Create an initial academic plan based on the search results.
    3. Interview: Conduct an interview to gather more information and feedback.
    4. Human Pause: Pause the workflow to allow for human review and input.
    5. Refine Search: Refine the search based on feedback from the interview.
    6. Refine Plan: Refine the academic plan based on the refined search results.
    7. Finalize Plan: Finalize the academic plan based on the refined plan and
       feedback from the interview.

    Args:
        checkpointer (BaseCheckpointSaver | None): An optional BaseCheckpointSaver instance for checkpointing the workflow state.
            Enables persistence of workflow execution across sessions, allowing resumption from interruptions.
            If None, defaults to MemorySaver (in-memory, non-persistent).

    Returns:
        StateGraph[ReviewState]: The compiled state graph representing the academic review workflow.
    """

    if checkpointer is None:
        checkpointer = MemorySaver()

    if not isinstance(checkpointer, BaseCheckpointSaver):
        raise ValueError(
            "checkpointer must be an instance of BaseCheckpointSaver (e.g., MemorySaver, SqliteSaver, RedisSaver, etc.)\n"
            + f"Received type: {type(checkpointer)}"
        )

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

    return builder.compile(checkpointer=checkpointer, interrupt_before=["human_pause"])
