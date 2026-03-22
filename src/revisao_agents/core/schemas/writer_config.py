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
ReviewLanguage = Literal["pt", "en"]

# Human-readable language labels used to enforce language in prompts
_LANGUAGE_LABELS: dict = {
    "pt": "Brazilian Portuguese (pt-BR)",
    "en": "English",
}

_REVIEW_TYPE_LABELS: dict = {
    "technical": {
        "pt": "Revisão Técnica",
        "en": "Technical Review",
    },
    "academic": {
        "pt": "Revisão Acadêmica da Literatura",
        "en": "Academic Literature Review",
    },
}


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
    language:
        Output language for the review. "pt" = Brazilian Portuguese, "en" = English.
        All generated text (sections, intro, conclusion) must be in this language.
    """

    mode: WritingMode = "technical"
    prompt_dir: str = "technical_writing"
    corpus_strategy: CorpusStrategy = "web_first"
    output_prefix: str = "revisao_tecnica"
    review_type_label: str = "Revisão Técnica"
    language: ReviewLanguage = "pt"
    min_sources_per_section: int = 0  # 0 = no constraint; set via CLI

    @property
    def language_label(self) -> str:
        """Full language name for use in prompts."""
        return _LANGUAGE_LABELS.get(self.language, "Brazilian Portuguese (pt-BR)")

    @staticmethod
    def default_review_type_label(mode: str, language: str) -> str:
        """Return the localized default review label for a mode/language pair."""
        return _REVIEW_TYPE_LABELS.get(mode, _REVIEW_TYPE_LABELS["technical"]).get(
            language,
            _REVIEW_TYPE_LABELS["technical"]["pt"],
        )

    # --------------------------------------------------------------------------
    # Factory helpers
    # --------------------------------------------------------------------------

    @classmethod
    def technical(cls, language: str = "pt", min_sources: int = 0) -> "WriterConfig":
        """Default technical writing configuration.

        Args:
            language: Output language for the review. "pt" = Brazilian Portuguese, "en" = English.
            min_sources: Minimum number of sources per section.

        Returns:
            WriterConfig: Configured writer settings for technical writing.
        """
        return cls(
            mode="technical",
            prompt_dir="technical_writing",
            corpus_strategy="web_first",
            output_prefix="revisao_tecnica",
            review_type_label=cls.default_review_type_label("technical", language),
            language=language,
            min_sources_per_section=min_sources,
        )

    @classmethod
    def academic(cls, language: str = "pt", min_sources: int = 4) -> "WriterConfig":
        """Academic systematic-review writing configuration.

        Args:
            language: Output language for the review. "pt" = Brazilian Portuguese, "en" = English.
            min_sources: Minimum number of sources per section.

        Returns:
            WriterConfig: Configured writer settings for academic writing.
        """
        return cls(
            mode="academic",
            prompt_dir="academic_writing",
            corpus_strategy="corpus_first",
            output_prefix="revisao_academica",
            review_type_label=cls.default_review_type_label("academic", language),
            language=language,
            min_sources_per_section=min_sources,
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

        Args:
            data: A dict with keys matching WriterConfig fields, typically loaded from LangGraph state.

        Returns:
            WriterConfig: Reconstructed writer configuration.
        """
        if not data:
            return cls.technical()
        mode = data.get("mode", "technical")
        language = data.get("language", "pt")
        return cls(
            mode=mode,
            prompt_dir=data.get("prompt_dir", "technical_writing"),
            corpus_strategy=data.get("corpus_strategy", "web_first"),
            output_prefix=data.get("output_prefix", "revisao_tecnica"),
            review_type_label=data.get(
                "review_type_label",
                cls.default_review_type_label(mode, language),
            ),
            language=language,
            min_sources_per_section=data.get("min_sources_per_section", 0),
        )

    @property
    def is_corpus_first(self) -> bool:
        """Check if the corpus strategy is 'corpus_first'."""
        return self.corpus_strategy == "corpus_first"
