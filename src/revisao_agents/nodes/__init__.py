"""
Agents module - LangGraph node definitions for different review types.

This module contains all the node functions that make up the various
review workflows (academic, technical, technical writing, etc.)
"""

from .academic import (
    finalize_academic_plan_node,
    initial_academic_plan_node,
    refine_academic_plan_node,
    refine_academic_search_node,
    vector_search_node,
)
from .common import (
    human_pause_node,
    identify_and_refine_node,
    interview_node,
    interview_router,
    post_pause_router,
    refinement_router,
)
from .technical import (
    finalize_technical_plan_node,
    initial_technical_plan_node,
    initial_technical_search_node,
    refine_technical_plan_node,
    refine_technical_search_node,
)
from .technical_writing import consolidate_node, parse_plan_node, write_sections_node

__all__ = [
    # Academic agents
    "vector_search_node",
    "initial_academic_plan_node",
    "refine_academic_search_node",
    "finalize_academic_plan_node",
    "vector_search_node",
    "refine_academic_plan_node",
    # Technical agents
    "initial_technical_search_node",
    "initial_technical_plan_node",
    "refine_technical_search_node",
    "refine_technical_plan_node",
    "finalize_technical_plan_node",
    # Common agents
    "human_pause_node",
    "interview_node",
    "interview_router",
    "identify_and_refine_node",
    "refinement_router",
    "post_pause_router",
    # Technical writing agents
    "parse_plan_node",
    "write_sections_node",
    "consolidate_node",
]
