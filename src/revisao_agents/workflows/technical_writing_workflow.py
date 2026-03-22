from langgraph.graph import StateGraph, END

from ..state import TechnicalWriterState
from ..nodes.technical_writing import (
    parse_plan_node,
    write_sections_node,
    consolidate_node,
)


def build_technical_writing_workflow():
    """Build the technical writing workflow graph."""
    builder = StateGraph(TechnicalWriterState)
    builder.add_node("parse_plan", parse_plan_node)
    builder.add_node("write_sections", write_sections_node)
    builder.add_node("consolidate", consolidate_node)
    builder.set_entry_point("parse_plan")
    builder.add_edge("parse_plan", "write_sections")
    builder.add_edge("write_sections", "consolidate")
    builder.add_edge("consolidate", END)
    return builder.compile()


# Backward compatibility alias
def build_workflow():
    """Backward-compatible alias for build_technical_writing_workflow."""
    return build_technical_writing_workflow()
