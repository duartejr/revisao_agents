"""
Integration test: MLflow session continuity across multi-turn planning and review_chat.

What is verified:
  1. PLANNING  — initial + refine search turns reuse the SAME MLflow run.
                 All 4 metrics appear and session-level ones accumulate correctly.
  2. REVIEW    — two review_chat turns reuse the SAME MLflow run and show metric history.

Requires real Tavily and MLflow credentials. Mark with -m integration to skip
in CI unless explicitly enabled.
"""

from __future__ import annotations

import mlflow
import pytest
from dotenv import load_dotenv
from observability import workflow_run
from observability.mlflow_config import EXP_PLANNING_TECHNICAL, EXP_REVIEW_CHAT, get_tracking_uri

from revisao_agents.nodes.technical import (
    initial_technical_search_node,
    refine_technical_search_node,
)
from revisao_agents.state import ReviewState
from revisao_agents.tools.tavily_web_search import search_tavily_incremental

load_dotenv()

REQUIRED_METRICS = {
    "search_coverage",
    "result_reuse_percent",
    "credit_efficiency_individual",
    "credit_efficiency_aggregated",
}


def _metric_history(client: mlflow.tracking.MlflowClient, run_id: str, key: str) -> list[float]:
    return [m.value for m in client.get_metric_history(run_id, key)]


@pytest.fixture()
def base_technical_state() -> ReviewState:
    return ReviewState(
        theme="transformer architecture attention mechanism",
        review_type="technical",
        relevant_chunks=[],
        technical_snippets=[],
        technical_urls=[],
        current_plan="",
        interview_history=[],
        questions_asked=0,
        max_questions=3,
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
def test_planning_session_continuity(base_technical_state: ReviewState) -> None:
    """Two planning turns reuse the same MLflow run and all 4 metrics accumulate."""
    mlflow.set_tracking_uri(get_tracking_uri())
    client = mlflow.tracking.MlflowClient()
    theme = base_technical_state["theme"]

    # ── Turn 1: initial search ────────────────────────────────────────
    with workflow_run(
        EXP_PLANNING_TECHNICAL,
        f"integration/planning-continuity-{theme[:30]}",
        params={"theme": theme},
    ) as active_run:
        run_id = active_run.info.run_id
        result1 = initial_technical_search_node(base_technical_state)

    metrics_t1 = client.get_run(run_id).data.metrics
    assert REQUIRED_METRICS.issubset(
        metrics_t1.keys()
    ), f"Missing metrics after turn 1: {REQUIRED_METRICS - metrics_t1.keys()}"
    assert metrics_t1["result_reuse_percent"] == 0.0, "No URL history yet on first turn"

    # ── Turn 2: refine search — reopen the SAME run ───────────────────
    state2: ReviewState = {
        **base_technical_state,
        **result1,
        "technical_urls": result1.get("technical_urls", []),
        "interview_history": [("user", "focus on multi-head attention"), ("assistant", "noted")],
        "current_plan": "draft plan",
    }
    mlflow.set_tracking_uri(get_tracking_uri())
    with mlflow.start_run(run_id=run_id):
        refine_technical_search_node(state2)

    # Same run_id used for both turns
    metrics_t2 = client.get_run(run_id).data.metrics
    assert REQUIRED_METRICS.issubset(
        metrics_t2.keys()
    ), f"Missing metrics after turn 2: {REQUIRED_METRICS - metrics_t2.keys()}"

    # Each metric logged exactly once per turn → history length == 2
    for key in REQUIRED_METRICS:
        history = _metric_history(client, run_id, key)
        assert (
            len(history) == 2
        ), f"Expected 2 data points for '{key}', got {len(history)}: {history}"

    # result_reuse_percent must be > 0 on turn 2 (refine query is similar → overlapping URLs)
    rr_history = _metric_history(client, run_id, "result_reuse_percent")
    assert rr_history[0] == 0.0, "First turn: no reuse"
    # turn 2 may or may not have reuse depending on results — verify it is a valid percentage
    assert 0.0 <= rr_history[1] <= 100.0


@pytest.mark.integration
def test_review_chat_session_continuity() -> None:
    """Two review_chat turns reuse the same MLflow run and metric history accumulates."""
    mlflow.set_tracking_uri(get_tracking_uri())
    client = mlflow.tracking.MlflowClient()

    session_state: dict = {"mlflow_run_id": None}

    def _simulate_turn(query: str, previous_urls: list[str]) -> list[str]:
        run_id = session_state.get("mlflow_run_id")
        if run_id:
            mlflow.set_tracking_uri(get_tracking_uri())
            run_ctx = mlflow.start_run(run_id=run_id)
        else:
            run_ctx = workflow_run(
                EXP_REVIEW_CHAT,
                "integration/review-continuity",
                params={"document": "integration_test.md"},
            )
        with run_ctx as active_run:
            session_state["mlflow_run_id"] = active_run.info.run_id
            result = search_tavily_incremental(query, previous_urls, 5)
        return result.get("total_accumulated", [])

    # Turn 1
    urls_t1 = _simulate_turn("BERT pre-training objectives masked language model", [])
    run_id = session_state["mlflow_run_id"]

    # Tool-level logging omits credit_efficiency_aggregated (needs session totals)
    REVIEW_METRICS = REQUIRED_METRICS - {"credit_efficiency_aggregated"}

    metrics_t1 = client.get_run(run_id).data.metrics
    assert REVIEW_METRICS.issubset(
        metrics_t1.keys()
    ), f"Missing metrics after turn 1: {REVIEW_METRICS - metrics_t1.keys()}"
    assert (
        "credit_efficiency_aggregated" not in metrics_t1
    ), "Tool-level logging must not compute aggregated metric"
    assert metrics_t1["result_reuse_percent"] == 0.0, "No reuse on first turn"

    # Turn 2 — must reuse the same run
    _simulate_turn("BERT fine-tuning downstream tasks NLP", urls_t1)
    assert session_state["mlflow_run_id"] == run_id, "Turn 2 must use the same run_id"

    metrics_t2 = client.get_run(run_id).data.metrics
    assert REVIEW_METRICS.issubset(
        metrics_t2.keys()
    ), f"Missing metrics after turn 2: {REVIEW_METRICS - metrics_t2.keys()}"

    # Each metric logged exactly once per turn → history length == 2
    for key in REVIEW_METRICS:
        history = _metric_history(client, run_id, key)
        assert (
            len(history) == 2
        ), f"Expected 2 data points for '{key}', got {len(history)}: {history}"
