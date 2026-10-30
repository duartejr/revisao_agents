# src/revisao_agents/core/schemas/journal_review.py
"""Structured output schema for Journal Reviewer Mode.

A *referee report* on a generated **literature review** (not an empirical
paper). The four criteria are therefore interpreted for a review document:
methodology means how the literature was selected and synthesised, not how
an experiment was run.
"""

from typing import Literal

from pydantic import BaseModel, Field


class CriterionAssessment(BaseModel):
    """Assessment of one rubric criterion.

    Attributes:
        score: Integer judgment from 1 (poor) to 10 (excellent). Scores are
            relative LLM judgments, not calibrated measurements.
        strengths: Concrete strengths, each tied to evidence in the document.
        weaknesses: Concrete weaknesses, each tied to evidence in the document.
        suggestions: Actionable changes the author could make to raise the score.
    """

    score: int = Field(
        ...,
        ge=1,
        le=10,
        description="Integer score from 1 (poor) to 10 (excellent) for this criterion",
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="Concrete strengths, each referring to evidence in the document",
    )
    weaknesses: list[str] = Field(
        default_factory=list,
        description="Concrete weaknesses, each referring to evidence in the document",
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Actionable suggestions the author can apply to improve this criterion",
    )


class JournalReviewReport(BaseModel):
    """Referee report on a generated literature review.

    Criterion semantics (for a *review* document):

    * ``methodology``: transparency and soundness of how sources were chosen,
      grouped, compared and synthesised; coverage and balance of the evidence.
    * ``contribution``: how much the review helps a reader understand the
      field: structure, synthesis, identified gaps and open questions.
    * ``novelty``: new perspective, framing or connections beyond a summary of
      prior work; avoids merely restating well-known surveys.
    * ``writing_quality``: clarity, organisation, consistency, correct and
      traceable citations.

    Attributes:
        methodology: Assessment of the review methodology.
        contribution: Assessment of the review's contribution.
        novelty: Assessment of the review's novelty.
        writing_quality: Assessment of writing and citation quality.
        overall_recommendation: Editorial decision.
        summary: Short prose summary of the overall verdict.
    """

    methodology: CriterionAssessment
    contribution: CriterionAssessment
    novelty: CriterionAssessment
    writing_quality: CriterionAssessment
    overall_recommendation: Literal["accept", "minor_revision", "major_revision", "reject"] = Field(
        ..., description="Editorial decision for the document"
    )
    summary: str = Field(..., description="Short prose summary of the overall verdict")


#: Rubric criteria in rendering order.
CRITERIA: tuple[str, ...] = ("methodology", "contribution", "novelty", "writing_quality")
