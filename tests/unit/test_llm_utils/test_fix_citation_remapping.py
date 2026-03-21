"""Unit tests for citation remapping utilities."""

from revisao_agents.utils.llm_utils.fix_citation_remapping import (
    extract_numbered_citations,
    synchronize_text_with_references,
)


def test_complete_tracking() -> None:
    """End-to-end remap should keep citations and references aligned."""
    written_text = (
        "As shown in [13], a recent study [14] demonstrates that [13] is valid. "
        "Furthermore, [15] refutes the previous theory [14]."
    )

    original_source_map = {
        13: "https://paper-chronos-1.pdf",
        14: "https://paper-chronos-2.pdf",
        15: "https://paper-lstm.pdf",
    }

    synchronized_text, ordered_urls = synchronize_text_with_references(
        written_text,
        original_source_map,
    )

    assert "[1]" in synchronized_text
    assert "[2]" in synchronized_text
    assert "[3]" in synchronized_text
    assert "[13]" not in synchronized_text
    assert "[14]" not in synchronized_text
    assert "[15]" not in synchronized_text

    assert ordered_urls == [
        "https://paper-chronos-1.pdf",
        "https://paper-chronos-2.pdf",
        "https://paper-lstm.pdf",
    ]

    citations_in_text = extract_numbered_citations(synchronized_text)
    assert citations_in_text
    assert max(citations_in_text) <= len(ordered_urls)
