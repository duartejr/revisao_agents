"""
technical_writing.py — LangGraph graph nodes for technical/academic chapter authoring.

All logic has been extracted into the `nodes/writing/` subpackage for maintainability:
    text_filters.py   : regex patterns and LLM output cleanup.
    anchor_helpers.py : anchor extraction utilities.
    phase_runners.py  : phases 1-6 (plan, observe, draft, extract).
    verification.py   : adaptive judge (REACT verification loop).
    node_parsear.py   : parsear_plano_node implementation.
    node_escrever.py  : escrever_secoes_node implementation.
    node_consolidar.py: consolidar_node implementation.
"""

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graph nodes (implementations in parse_plan_node, write_sections_node, consolidate_node)
# ---------------------------------------------------------------------------
from .writing.parse_plan_node import parse_plan_node
from .writing.write_sections_node import write_sections_node
from .writing.consolidate_node import consolidate_node

__all__ = [
    "parse_plan_node",
    "write_sections_node",
    "consolidate_node",
]
