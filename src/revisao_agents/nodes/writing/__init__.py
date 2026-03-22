"""
nodes.writing — internal helpers and graph-node implementations for section writing.

Submodules
----------
text_filters     : regex patterns and strip helpers for LLM output cleanup.
anchor_helpers   : anchor extraction utilities.
phase_runners    : the individual writing phases (phases 1-6).
verification     : adaptive paragraph verification (judge + REACT loop).
parser_node      : parse_plan_node — parses a plan file and extracts sections.
writer_node      : sections_writer_node — writes sections with search and verification.
consolidate_node : consolidate_node — consolidates written sections into a final document.
"""

from ...helpers.anchor_helpers import (
    _extract_all_anchors_with_citations,
    _extract_citation_anchor,
    _extract_main_anchor,
)
from .phase_runners import (
    _draft_phase,
    _extract_with_fallback,
    _observation_phase,
    _thought_phase,
)
from .text_filters import (
    _ANCHORS_PATTERN,
    _strip_figure_table_refs,
    _strip_justification_blocks,
    _strip_meta_sentences,
)
from .verification import (
    _count_verifiable_claims,
    _judge_paragraph_improved,
    _monitor_verification_rate,
    _search_for_additional_content,
    _verify_and_correct_adaptative_section,
    _verify_and_correct_section_with_anchor,
    _verify_paragraph_with_anchor,
)

__all__ = [
    # text_filters
    "_ANCHORS_PATTERN",
    "_strip_justification_blocks",
    "_strip_meta_sentences",
    "_strip_figure_table_refs",
    # anchor_helpers
    "_extract_main_anchor",
    "_extract_citation_anchor",
    "_extract_all_anchors_with_citations",
    # phase_runners
    "_thought_phase",
    "_observation_phase",
    "_draft_phase",
    "_extract_with_fallback",
    # verification
    "_count_verifiable_claims",
    "_judge_paragraph_improved",
    "_monitor_verification_rate",
    "_search_for_additional_content",
    "_verify_and_correct_adaptative_section",
    "_verify_paragraph_with_anchor",
    "_verify_and_correct_section_with_anchor",
]
