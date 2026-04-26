"""
Integration test: single technical search logs all 4 quality metrics to MLflow.

Requires real Tavily and MLflow credentials. Mark with -m integration to skip
in CI unless explicitly enabled.
"""

from __future__ import annotations

import mlflow
import pytest
from dotenv import load_dotenv
from observability import workflow_run
from observability.mlflow_config import EXP_PLANNING_TECHNICAL, get_tracking_uri

from revisao_agents.nodes.technical import initial_technical_search_node
from revisao_agents.state import ReviewState

load_dotenv()

REQUIRED_METRICS = {
    "search_coverage",
    "result_reuse_percent",
    "credit_efficiency_individual",
    "credit_efficiency_aggregated",
}


@pytest.fixture()
def technical_state() -> ReviewState:
    return ReviewState(
        theme="Kubernetes architecture fundamentals",
        review_type="technical",
        relevant_chunks=[],
        technical_snippets=[],
        technical_urls=[],
        current_plan="",
        interview_history=[],
        questions_asked=0,
        max_questions=5,
        final_plan="",
        final_plan_path="",
        status="initial",
        detected_language="en",
        user_language_choice="en",
        is_theme_vague=False,
        is_theme_refined=True,
        confidence_score=1.0,
        refinement_feedback=[],
        urls_search_history={},
        total_credits_used=0.0,
        total_search_queries=0,
    )


@pytest.mark.integration
def test_initial_technical_search_logs_all_metrics(technical_state: ReviewState) -> None:
    """initial_technical_search_node logs all 4 quality metrics to the parent workflow run."""
    mlflow.set_tracking_uri(get_tracking_uri())
    client = mlflow.tracking.MlflowClient()

    with workflow_run(
        EXP_PLANNING_TECHNICAL,
        "integration/test-search-metrics",
        params={"theme": technical_state["theme"]},
    ) as active_run:
        run_id = active_run.info.run_id
        result = initial_technical_search_node(technical_state)

    assert result["status"] == "initial_technical_search_ok"
    assert result["total_search_queries"] == 1

    logged = client.get_run(run_id).data.metrics
    assert REQUIRED_METRICS.issubset(
        logged.keys()
    ), f"Missing metrics: {REQUIRED_METRICS - logged.keys()}"
    assert logged["search_coverage"] >= 0
    assert logged["result_reuse_percent"] == 0.0, "No reuse on first turn"
    assert logged["credit_efficiency_individual"] >= 0
    assert logged["credit_efficiency_aggregated"] >= 0
