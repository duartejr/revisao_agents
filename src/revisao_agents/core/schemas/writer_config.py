"""
WriterConfig — strategy object controlling writing mode and prompt routing.

Passed through LangGraph state as a plain dict (via to_dict / from_dict)
to avoid Pydantic overhead in state transitions.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Literal


WritingMode = Literal["technical", "academic"]
CorpusStrategy = Literal["web_first", "corpus_first"]


@dataclass
class WriterConfig:
    """
    Immutable strategy object that drives writing style across all graph nodes.

    Attributes
    ----------
    mode:
        "technical" — didactic chapter authoring (default)
        "academic"  — narrative systematic review authoring
    prompt_dir:
        Subdirectory under `prompts/` that contains all phase YAML files.
        Defaults match the mode: "technical_writing" or "academic_writing".
    corpus_strategy:
        "web_first"    — always search the web before using MongoDB (technical default)
        "corpus_first" — query existing MongoDB first; run web search only when
                         corpus is insufficient (academic default)
    output_prefix:
        Prefix used for the output filename in reviews/.
    review_type_label:
        Human-readable label used in the document header.
    """
    mode: WritingMode = "technical"
    prompt_dir: str = "technical_writing"
    corpus_strategy: CorpusStrategy = "web_first"
    output_prefix: str = "revisao_tecnica"
    review_type_label: str = "Revisão Técnica"

    # --------------------------------------------------------------------------
    # Factory helpers
    # --------------------------------------------------------------------------

    @classmethod
    def technical(cls) -> "WriterConfig":
        """Default technical writing configuration."""
        return cls(
            mode="technical",
            prompt_dir="technical_writing",
            corpus_strategy="web_first",
            output_prefix="revisao_tecnica",
            review_type_label="Revisão Técnica",
        )

    @classmethod
    def academic(cls) -> "WriterConfig":
        """Academic systematic-review writing configuration."""
        return cls(
            mode="academic",
            prompt_dir="academic_writing",
            corpus_strategy="corpus_first",
            output_prefix="revisao_academica",
            review_type_label="Revisão Acadêmica da Literatura",
        )

    # --------------------------------------------------------------------------
    # LangGraph state compatibility (TypedDict stores plain dicts)
    # --------------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "WriterConfig":
        """Reconstruct from a plain dict stored in LangGraph state.

        Falls back to technical defaults when data is empty or missing keys.
        """
        if not data:
            return cls.technical()
        return cls(
            mode=data.get("mode", "technical"),
            prompt_dir=data.get("prompt_dir", "technical_writing"),
            corpus_strategy=data.get("corpus_strategy", "web_first"),
            output_prefix=data.get("output_prefix", "revisao_tecnica"),
            review_type_label=data.get("review_type_label", "Revisão Técnica"),
        )

    @property
    def is_corpus_first(self) -> bool:
        return self.corpus_strategy == "corpus_first"
