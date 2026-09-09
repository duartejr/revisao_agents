"""Unit tests for ``scripts/run_refinement_ab_experiment.py``.

``scripts/`` is not an installed package, so the module under test is loaded
via ``importlib`` from its file path rather than a normal import.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "run_refinement_ab_experiment.py"


def _load_module():
    """Import ``run_refinement_ab_experiment.py`` fresh, bypassing module caching."""
    spec = importlib.util.spec_from_file_location("run_refinement_ab_experiment", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_refinement_ab_experiment"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def refinement_script(mlflow_local_store):
    """Load the script module against the isolated MLflow tracking store."""
    return _load_module()


class _FakeStateSnapshot:
    """Minimal stand-in for LangGraph's ``StateSnapshot``."""

    def __init__(self, next_nodes: tuple, values: dict):
        self.next = next_nodes
        self.values = values


class _FakeGraph:
    """Scripted fake LangGraph compiled graph for ``run_single_arm`` tests.

    ``stream_sequences`` and ``get_state_sequence`` are consumed in order as
    the code under test calls ``.stream()`` / ``.get_state()``, so each test
    supplies exactly the sequence its scenario needs.
    """

    def __init__(self, stream_sequences: list[list[dict]], get_state_sequence: list):
        self._stream_sequences = list(stream_sequences)
        self._get_state_sequence = list(get_state_sequence)
        self.update_state_calls: list[tuple] = []

    def stream(self, _state_or_none, config=None):
        return iter(self._stream_sequences.pop(0))

    def get_state(self, _config):
        return self._get_state_sequence.pop(0)

    def update_state(self, _config, update: dict, as_node: str | None = None):
        self.update_state_calls.append((update, as_node))


# ── run_single_arm ──────────────────────────────────────────────────────────


async def test_bypassed_arm_never_uses_clarification(refinement_script):
    """Even if the (bypassed) node somehow reported is_theme_refined=False, the
    bypass arm must answer any pause with the generic response, never the
    clarification text — bypass means the layer never asked in the first place."""
    fake_graph = _FakeGraph(
        stream_sequences=[
            [{"identify_and_refine": {}}],  # initial stream to first pause
            [{"refine_search": {}}, {"finalize_plan": {}}],  # resume to completion
        ],
        get_state_sequence=[
            _FakeStateSnapshot(
                ("human_pause",), {"is_theme_refined": True, "interview_history": []}
            ),
            _FakeStateSnapshot((), {"final_plan": "## Intro\n## Body\n"}),
            _FakeStateSnapshot(
                (), {"final_plan": "## Intro\n## Body\n", "theme": "machine learning"}
            ),
        ],
    )

    with (
        patch.object(refinement_script, "build_review_graph", return_value=fake_graph),
        patch.object(refinement_script, "evaluate_plan_quality", AsyncMock(return_value=8.0)),
    ):
        result = await refinement_script.run_single_arm(
            theme="machine learning",
            clarification="A specific clarified theme.",
            bypass=True,
            run_id="test_bypassed",
        )

    assert result["arm"] == "bypassed"
    assert result["clarification_used"] is False
    update, as_node = fake_graph.update_state_calls[0]
    assert as_node == "human_pause"
    assert update["interview_history"][-1] == ("user", "Keep the current plan.")
    assert result["refinement_rounds"] == 1
    assert result["plan_section_count"] == 2


async def test_refined_arm_uses_clarification_when_theme_not_refined(refinement_script):
    """When the refinement layer flags the theme as not-yet-refined, the pause
    must be answered with the clarification text, not the generic response."""
    fake_graph = _FakeGraph(
        stream_sequences=[
            [{"identify_and_refine": {}}],
            [
                {"identify_and_refine": {}},
                {"initial_plan": {}},
                {"interview": {}},
            ],  # re-evaluation round after clarification, reaches next pause
        ],
        get_state_sequence=[
            _FakeStateSnapshot(
                ("human_pause",), {"is_theme_refined": False, "interview_history": []}
            ),
            _FakeStateSnapshot((), {"final_plan": "## Intro\n"}),
            _FakeStateSnapshot((), {"final_plan": "## Intro\n", "theme": "clarified theme"}),
        ],
    )

    fake_judge = AsyncMock(return_value=6.5)
    with (
        patch.object(refinement_script, "build_review_graph", return_value=fake_graph),
        patch.object(refinement_script, "evaluate_plan_quality", fake_judge),
    ):
        result = await refinement_script.run_single_arm(
            theme="AI",
            clarification="A specific clarified theme.",
            bypass=False,
            run_id="test_refined",
        )

    assert result["arm"] == "refined"
    assert result["clarification_used"] is True
    update, as_node = fake_graph.update_state_calls[0]
    assert update["interview_history"][-1] == ("user", "A specific clarified theme.")
    assert result["refinement_rounds"] == 2
    # Regression: the plan must be judged against the theme it was actually
    # written for (rewritten by identify_and_refine_node during the
    # clarification round), not the original, now-stale vague theme.
    assert result["final_theme"] == "clarified theme"
    fake_judge.assert_awaited_once_with("clarified theme", "## Intro\n")


async def test_no_final_plan_yields_zero_quality_without_calling_judge(refinement_script):
    """If the run produced no final plan, evaluate_plan_quality must not be called
    and the score must default to 0.0."""
    fake_graph = _FakeGraph(
        stream_sequences=[[{"identify_and_refine": {}}]],
        get_state_sequence=[
            _FakeStateSnapshot((), {"final_plan": ""}),
            _FakeStateSnapshot((), {"final_plan": ""}),
        ],
    )
    fake_judge = AsyncMock(return_value=9.9)

    with (
        patch.object(refinement_script, "build_review_graph", return_value=fake_graph),
        patch.object(refinement_script, "evaluate_plan_quality", fake_judge),
    ):
        result = await refinement_script.run_single_arm(
            theme="AI", clarification="x", bypass=True, run_id="test_no_plan"
        )

    fake_judge.assert_not_called()
    assert result["plan_quality_score"] == 0.0


async def test_judge_failure_scores_zero_and_does_not_abort_the_run(refinement_script):
    """A transient evaluate_plan_quality failure (e.g. LLM API error) must not
    propagate out of run_single_arm — it should degrade that run's score to
    0.0 so a batch of runs can continue rather than aborting entirely."""
    fake_graph = _FakeGraph(
        stream_sequences=[[{"identify_and_refine": {}}]],
        get_state_sequence=[
            _FakeStateSnapshot((), {"final_plan": "## Intro\n"}),
            _FakeStateSnapshot((), {"final_plan": "## Intro\n"}),
        ],
    )
    fake_judge = AsyncMock(side_effect=RuntimeError("LLM API unavailable"))

    with (
        patch.object(refinement_script, "build_review_graph", return_value=fake_graph),
        patch.object(refinement_script, "evaluate_plan_quality", fake_judge),
    ):
        result = await refinement_script.run_single_arm(
            theme="AI", clarification="x", bypass=True, run_id="test_judge_failure"
        )

    assert result["plan_quality_score"] == 0.0
    assert result["final_plan_generated"] is True  # a plan exists — only the judge call failed


# ── main_async ───────────────────────────────────────────────────────────────


async def test_main_async_logs_params_and_metrics_per_run(refinement_script):
    """main_async() must configure the dedicated experiment and log params +
    all four numeric metrics for every (theme, arm) pair. TEST_CASES is
    reduced to one entry so the assertions run against exactly two runs
    (refined, bypassed) without adding repetitive coverage."""
    single_case = [{"vague_theme": "AI", "clarification": "A specific clarified theme."}]

    fake_result = {
        "theme": "AI",
        "final_theme": "AI",
        "arm": "refined",
        "clarification_used": True,
        "language_detected": "EN",
        "session_duration": 1.23,
        "refinement_rounds": 2,
        "plan_section_count": 3,
        "plan_quality_score": 7.0,
        "final_plan_generated": True,
    }

    mock_mlflow = MagicMock()
    mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()
    mock_mlflow.start_run.return_value.__exit__.return_value = False

    with (
        patch.object(refinement_script, "TEST_CASES", single_case),
        patch.object(refinement_script, "mlflow", mock_mlflow),
        patch.object(refinement_script, "get_tracking_uri", return_value="file:///tmp/mlruns-test"),
        patch.object(refinement_script, "run_single_arm", AsyncMock(return_value=fake_result)),
    ):
        await refinement_script.main_async()

    mock_mlflow.set_experiment.assert_called_once_with("planning_refinement_ab")
    assert mock_mlflow.start_run.call_count == 2  # one per arm

    logged_metrics: dict = mock_mlflow.log_metrics.call_args.args[0]
    assert set(logged_metrics) == {
        "session_duration",
        "refinement_rounds",
        "plan_section_count",
        "plan_quality_score",
    }

    logged_params: dict = mock_mlflow.log_params.call_args.args[0]
    assert "language_detected" in logged_params
    assert "clarification_used" in logged_params
