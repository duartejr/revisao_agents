"""
evaluators.py - Evaluation logic for search snippet assessment in the academic review agent.

This module defines the evaluation process for assessing the relevance, academic quality,
and citation potential of search snippets extracted during the academic review workflow.
It delegates scoring to specialized MLflow judges and aggregates their outputs into
structured ``SnippetEvaluation`` objects.

Public functions:
    extract_domain: Parse a URL and return its network location (domain).
    evaluate_search_snippets: Run all three judges against a batch of snippets and return
        evaluation results as a list of ``SnippetEvaluation`` objects.
    log_snippet_evaluations_to_mlflow: Persist evaluation results and aggregate metrics to
        the active MLflow run for later analysis.
"""

import logging
from typing import Literal

import mlflow
from mlflow.entities.assessment import Feedback

from .snippet_evaluators import (
    get_or_create_academic_quality_judge,
    get_or_create_citation_potential_judge,
    get_or_create_relevance_judge,
)
from .types import SnippetEvaluation

logger = logging.getLogger(__name__)


def extract_domain(url: str) -> str:
    """Extract the domain from a given URL.

    Args:
        url: The URL string to extract the domain from.

    Returns:
        The domain part of the URL, or 'unknown' if extraction fails.
    """
    from urllib.parse import urlparse

    try:
        parsed_url = urlparse(url)
        return parsed_url.netloc
    except Exception as e:
        logger.warning(f"Failed to extract domain from URL '{url}': {e}")
        return "unknown"


@mlflow.trace
async def evaluate_search_snippets(
    query: str,
    snippets: list[str],
    urls: list[str],
    interview_metadata: dict,
) -> list[SnippetEvaluation]:
    """Evaluate all snippets for relevance, academic quality, and citation potential.

    Args:
        query: The original search query.
        snippets: List of text snippets extracted from search results.
        urls: Corresponding list of URLs for each snippet.
        interview_metadata: Additional context about the user's interview goals.

    Returns:
        A list of SnippetEvaluation objects containing the evaluation results for each snippet.

    Example:
        >>> evaluations = evaluate_search_snippets(
        ...     query="What are the latest advancements in quantum computing?",
        ...     snippets=["Snippet 1 text", "Snippet 2 text"],
        ...     urls=["http://example.com/snippet1", "http://example.com/snippet2"],
        ...     interview_metadata={
        ...         "interview_id": "i-001",
        ...         "user_goals": "Understand recent research in quantum computing"
        ...         "depth_setting": 3
        ...     }
        ... )
    """
    if not snippets:
        logger.warning("No snippets to evaluate for query: '%s'", query)
        return []

    logger.info(
        f"Evaluating {len(snippets)} snippets for query: '{query}' with interview metadata: {interview_metadata}"
    )

    relevance_judge = get_or_create_relevance_judge()
    academic_judge = get_or_create_academic_quality_judge()
    citation_judge = get_or_create_citation_potential_judge()

    evaluations: list[SnippetEvaluation] = []

    for idx, (snippet, url) in enumerate(zip(snippets, urls, strict=False)):
        logger.debug(
            f"Evaluating snippet {idx + 1}/{len(snippets)}: URL='{url}' Snippet='{snippet[:100]}...'"
        )

        try:
            domain = extract_domain(url)

            # Judge 1: Relevance
            relevance_feedback = relevance_judge(
                inputs={"query": query},
                outputs={"snippet": snippet},
            )
            # RelevanceToQuery returns "yes" or "no" as the feedback value
            relevance_value = str(relevance_feedback.value).lower()
            is_relevant = relevance_value == "yes"

            # Map the relevance level
            relevance_level: Literal["Perfectly relevant", "Not relevant"]
            if is_relevant:
                relevance_level = "Perfectly relevant"
                relevance_score = 1.0
            else:
                relevance_level = "Not relevant"
                relevance_score = 0.0

            relevance_rationale = f"RelevanceToQuery judge: {relevance_value}"

            # Judge 2: Academic Quality
            logger.debug(
                f"Evaluating academic quality for snippet {idx + 1}/{len(snippets)}: Domain='{domain}'"
            )
            academic_feedback = academic_judge(
                outputs={
                    "snippet": snippet,
                    "domain": domain,
                    "user_goals": interview_metadata.get("user_goals", ""),
                }
            )
            academic_quality = (
                str(
                    academic_feedback.value
                    if isinstance(academic_feedback, Feedback)
                    else academic_feedback
                ).lower()
                == "yes"
            )
            academic_quality_rationale = (
                academic_feedback.rationale
                if isinstance(academic_feedback, Feedback) and academic_feedback.rationale
                else f"Source: {domain}"
            )

            # Judge 3: Citation Potential
            logger.debug(
                f"Evaluating citation potential for snippet {idx + 1}/{len(snippets)}: Domain='{domain}'"
            )
            citation_feedback = citation_judge(
                outputs={
                    "snippet": snippet,
                    "user_goals": interview_metadata.get("user_goals", ""),
                }
            )
            citation_potential = (
                str(
                    citation_feedback.value
                    if isinstance(citation_feedback, Feedback)
                    else citation_feedback
                ).lower()
                == "yes"
            )
            citation_potential_rationale = (
                citation_feedback.rationale
                if isinstance(citation_feedback, Feedback) and citation_feedback.rationale
                else "Citation potential assessment"
            )

            # Combine into SnippetEvaluation
            evaluation = SnippetEvaluation(
                snippet=snippet,
                relevance_level=relevance_level,
                relevance_score=relevance_score,
                relevance_rationale=relevance_rationale,
                academic_quality=academic_quality,
                academic_quality_rationale=academic_quality_rationale,
                citation_potential=citation_potential,
                citation_potential_rationale=citation_potential_rationale,
                url=url,
                domain=domain,
            )

            evaluations.append(evaluation)

            logger.debug(
                f"Snippet {idx + 1}: "
                f"Relevance='{relevance_level}' "
                f"AcademicQuality='{academic_quality}' "
                f"CitationPotential='{citation_potential}' "
                f"URL='{url}' "
                f"Domain='{domain}'"
            )
        except Exception as e:
            logger.error(f"Error evaluating snippet {idx + 1} (URL='{url}'): {e}", exc_info=True)
            continue

    logger.info(f"Completed evaluation of {len(evaluations)} snippets for query: '{query}'")

    return evaluations


def log_snippet_evaluations_to_mlflow(
    evaluations: list[SnippetEvaluation],
    query: str,
    depth_setting: int | None = None,
) -> None:
    """Log the snippet evaluations to MLflow for analysis and tracking.

    Log:
    - Each evaluation as a separate MLflow metric with a structured naming convention.
    - Overall statistics such as average relevance score, percentage of academically qualified snippets, and percentage of snippets with citation potential.

    Args:
        evaluations: List of SnippetEvaluation objects to log.
        query: The original search query associated with these evaluations.
        depth_setting: Optional depth setting from the interview metadata for additional context in logging.
    """

    if not evaluations:
        logger.warning("No evaluations to log for query: '%s'", query)
        return

    active_run = mlflow.active_run()
    if not active_run:
        logger.warning("No active MLflow run to log evaluations for query: '%s'", query)
        return

    logger.debug(
        f"Logging {len(evaluations)} snippet evaluations to MLflow for query: '{query}' with depth setting: {depth_setting}"
    )

    for idx, eval in enumerate(evaluations):
        eval_dict = {
            "snippet": eval.snippet[:200],
            "relevance_level": eval.relevance_level,
            "relevance_score": eval.relevance_score,
            "academic_quality": eval.academic_quality,
            "citation_potential": eval.citation_potential,
            "domain": eval.domain,
            "url": eval.url,
            "query": query,
            "depth_setting": depth_setting,
        }

        mlflow.log_dict(eval_dict, artifact_file=f"snippet_evaluation_{idx:03d}.json")

    # Counters
    total_evals = len(evaluations)
    relevant_count = sum(
        1 for e in evaluations if e.relevance_level.lower() == "perfectly relevant"
    )
    partially_count = sum(
        1 for e in evaluations if e.relevance_level.lower() == "partially relevant"
    )
    academic_count = sum(1 for e in evaluations if e.academic_quality)
    citation_count = sum(1 for e in evaluations if e.citation_potential)

    # Percentages
    relevant_pct = (relevant_count / total_evals) * 100 if total_evals > 0 else 0.0
    partially_pct = (partially_count / total_evals) * 100 if total_evals > 0 else 0.0
    academic_pct = (academic_count / total_evals) * 100 if total_evals > 0 else 0.0
    citation_pct = (citation_count / total_evals) * 100 if total_evals > 0 else 0.0

    metrics = {
        "eval_snippets_total": total_evals,
        "eval_perfectly_relevant_count": relevant_count,
        "eval_partially_relevant_count": partially_count,
        "eval_academic_quality_count": academic_count,
        "eval_citation_potential_count": citation_count,
        "eval_perfectly_relevant_pct": relevant_pct,
        "eval_partially_relevant_pct": partially_pct,
        "eval_academic_quality_pct": academic_pct,
        "eval_citation_potential_pct": citation_pct,
        "eval_relevance_avg_score": sum(e.relevance_score for e in evaluations) / total_evals
        if total_evals > 0
        else 0.0,
    }

    for metric_name, metric_value in metrics.items():
        try:
            mlflow.log_metric(metric_name, metric_value)
            logger.debug(f"Logged metric '{metric_name}': {metric_value}")
        except Exception as e:
            logger.warning(f"Failed to log metric '{metric_name}': {e}", exc_info=True)

    mlflow.log_params(
        {
            "eval_query": query[:100],
            "eval_depth_setting": depth_setting,
        }
    )

    logger.info(
        f"Logged snippet evaluation metrics to MLflow for query: '{query}' with depth setting: {depth_setting}"
    )
