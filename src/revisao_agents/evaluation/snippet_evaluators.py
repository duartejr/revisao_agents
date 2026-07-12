"""
snippet_evaluators.py - MLflow judges for evaluating search snippets in the academic review workflow.

Implements three judges:
1. Relevance Judge: Evaluates how relevant a search snippet is to the user's query.
2. Academic Quality Judge: Assesses whether the snippet contains technically solid information from reputable sources.
3. Citation Potential Judge: Determines if the snippet is suitable for use as a citation in an academic paper.
"""

import logging

from mlflow.genai.judges import make_judge
from mlflow.genai.judges.base import Judge
from mlflow.genai.scorers import RelevanceToQuery

logger = logging.getLogger(__name__)


def get_relevance_judge() -> RelevanceToQuery:
    """Return judge built-in RelevanceToQuery for evaluating snippet relevance.

    Returns:
        An instance of RelevanceToQuery judge to evaluate the relevance of search snippets to user queries.

    Note:
        - Returns "yes" or "no"
        - Does not require a customized model or instructions, as it's a built-in judge with predefined behavior.
        - Optimized for search and retrieval evaluation, providing consistent relevance assessments based on the query and snippet content."""
    return RelevanceToQuery(name="search_relevance")


def get_academic_quality_judge() -> Judge:
    """Create a judge to evaluate the academic quality of a search snippet based on technical soundness, source credibility, and alignment with user goals.

    Criteria:
        Content + Source = Solid technical information from reputable sources.

    Returns:
        An instance of a judge to evaluate the academic quality of search snippets.
        Returns "yes" if the snippet meets academic quality standards, "no" otherwise.
    """
    return make_judge(
        name="academic_quality",
        instructions="""Evaluate if the snippet contains technically solid information from a reputable source aligned with the user's research goals.

        Outputs (contains snippet, domain, user_goals): {{ outputs }}

        Answer "yes" only if ALL three criteria are met:
        - Technical Soundness: the information is accurate, well-explained, and follows best practices
        - Source Credibility: the source is reputable (academic papers, official documentation, well-known experts)
        - Alignment with User Goals: the information helps the user achieve their research objectives

        Answer "no" if any criterion is not met.

        Reply with only "yes" or "no".
        """,
        model="openai:/gpt-4o-mini",
        description="Judge to evaluate the academic quality of search snippets based on technical soundness, source credibility, and alignment with user goals.",
    )


def get_citation_potential_judge() -> Judge:
    """Create a judge to evaluate if a snippet can be used as a basis for a claim.

    Main question: Can I use this specific snippet to support/sustain an academic claim in a research paper?

    Returns:
        An instance of a judge to evaluate the citation potential of search snippets based on criteria such as
        specificity, source attribution, authority, and context preservation.
    """
    return make_judge(
        name="citation_potential",
        instructions="""Evaluate if this snippet can be used to support an academic claim in a research paper.

        Outputs (contains snippet, user_goals): {{ outputs }}

        Answer "yes" only if ALL criteria are met:
        - Specificity: contains concrete, quotable information
        - Source Attribution: author or publication is identifiable
        - Authority: the source is authoritative on this topic
        - Context Preservation: the snippet will make sense when cited out of context

        Answer "no" if any criterion is not met.

        Reply with only "yes" or "no".
        """,
        model="openai:/gpt-4o-mini",
        description="Judge to evaluate if a search snippet is suitable for use as a citation in an academic paper based on criteria such as specificity, source attribution, authority, and context preservation.",
    )


# Singleton instances of judges to be reused across evaluations
_relevance_judge: RelevanceToQuery | None = None
_academic_quality_judge: Judge | None = None
_citation_potential_judge: Judge | None = None


def get_or_create_relevance_judge() -> RelevanceToQuery:
    """Get or create a singleton instance of the relevance judge.

    This function ensures that only one instance of the relevance judge is created and reused across evaluations,
    optimizing resource usage and maintaining consistency in relevance assessments.

    Returns:
        An instance of the RelevanceToQuery judge, either newly created or reused from a previous
        instantiation.
    """
    global _relevance_judge
    if _relevance_judge is None:
        _relevance_judge = get_relevance_judge()
        logger.debug("Created new relevance judge instance.")
    return _relevance_judge


def get_or_create_academic_quality_judge() -> Judge:
    """Get or create a singleton instance of the academic quality judge.

    This function ensures that only one instance of the academic quality judge is created
    and reused across evaluations, optimizing resource usage and maintaining consistency
    in academic quality assessments.

    Returns:
        An instance of the academic quality Judge, either newly created or reused from a
        previous instantiation.
    """
    global _academic_quality_judge
    if _academic_quality_judge is None:
        _academic_quality_judge = get_academic_quality_judge()
        logger.debug("Created new academic quality judge instance.")
    return _academic_quality_judge


def get_or_create_citation_potential_judge() -> Judge:
    """Get or create a singleton instance of the citation potential judge.

    This function ensures that only one instance of the citation potential judge is created
    and reused across evaluations, optimizing resource usage and maintaining consistency
    in citation potential assessments.

    Returns:
        An instance of the citation potential Judge, either newly created or reused from a
        previous instantiation.
    """
    global _citation_potential_judge
    if _citation_potential_judge is None:
        _citation_potential_judge = get_citation_potential_judge()
        logger.debug("Created new citation potential judge instance.")
    return _citation_potential_judge
