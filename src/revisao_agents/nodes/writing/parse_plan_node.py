"""
parse_plan_node — parses a plan file and extracts sections
Part of the nodes/writing subpackage.
"""

import logging

from ...core.schemas.writer_config import WriterConfig
from ...state import TechnicalWriterState
from ...utils.file_utils.helpers import parse_academic_plan, parse_technical_plan

logger = logging.getLogger(__name__)


def parse_plan_node(state: TechnicalWriterState) -> dict:
    """Parses a plan file and extracts sections. Supports both technical and academic modes.

    Args:
        state (TechnicalWriterState): The current state of the technical writer, expected to contain:
            - "plan_path": str, path to the plan file (markdown or text).
            - "writer_config": dict, optional configuration for parsing (e.g., mode: "technical" or "academic").

    Returns:
        dict: Updated state with extracted theme, plan summary, sections, and initialized fields for writing.
    """
    config = WriterConfig.from_dict(state.get("writer_config", {}))
    plan_path = state["plan_path"]
    print(f"\n📖 Reading plan: {plan_path} (mode: {config.mode})")
    with open(plan_path, encoding="utf-8") as f:
        text = f.read()
    if config.mode == "academic":
        theme, plan_summary, sections = parse_academic_plan(text)
    else:
        theme, plan_summary, sections = parse_technical_plan(text)
    print(f"   ✅ Theme: {theme} | {len(sections)} sections")
    for s in sections:
        print(f"      [{s['index']+1}] {s['title']}")
    return {
        "theme": theme,
        "plan_summary": plan_summary,
        "sections": sections,
        "written_sections": [],
        "refs_urls": [],
        "refs_images": [],
        "cumulative_summary": "",
        "react_log": [],
        "verification_stats": [],
        "status": "plan_parsed",
        "plan_path": plan_path,
        "writer_config": state.get("writer_config", {}),
    }
