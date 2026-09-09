"""Unit tests for ``revisao_agents.evaluation.evaluators.evaluate_plan_quality`` (W9-STORY-03)."""

from unittest.mock import MagicMock, patch

import pytest
from mlflow.entities.assessment import Feedback

from revisao_agents.evaluation.evaluators import evaluate_plan_quality

# The judge call (see mlflow.genai.judges.base.Judge.__call__) can return
# either a bare value or that value wrapped in a Feedback — evaluate_plan_quality
# must handle both shapes identically, so every scenario below is exercised
# against both.
_WRAPPERS = [
    pytest.param(lambda v: v, id="bare-value"),
    pytest.param(lambda v: Feedback(value=v), id="feedback-wrapped"),
]


def _fake_judge(value, wrap):
    """Build a fake MLflow judge callable returning `value`, optionally Feedback-wrapped."""
    return MagicMock(return_value=wrap(value))


async def test_empty_plan_returns_zero_without_calling_judge():
    fake_judge = _fake_judge(8, wrap=lambda v: v)
    with patch(
        "revisao_agents.evaluation.evaluators.get_or_create_plan_quality_judge",
        return_value=fake_judge,
    ):
        score = await evaluate_plan_quality("some theme", "   ")

    assert score == 0.0
    fake_judge.assert_not_called()


@pytest.mark.parametrize("wrap", _WRAPPERS)
async def test_in_range_integer_returns_float_score(wrap):
    fake_judge = _fake_judge(7, wrap)
    with patch(
        "revisao_agents.evaluation.evaluators.get_or_create_plan_quality_judge",
        return_value=fake_judge,
    ):
        score = await evaluate_plan_quality("theme", "## Section\ncontent")

    assert score == 7.0
    assert isinstance(score, float)


@pytest.mark.parametrize("wrap", _WRAPPERS)
@pytest.mark.parametrize("value", [0, 11, -3, 100])
async def test_out_of_range_integer_returns_zero(value, wrap):
    fake_judge = _fake_judge(value, wrap)
    with patch(
        "revisao_agents.evaluation.evaluators.get_or_create_plan_quality_judge",
        return_value=fake_judge,
    ):
        score = await evaluate_plan_quality("theme", "## Section\ncontent")

    assert score == 0.0


@pytest.mark.parametrize("wrap", _WRAPPERS)
async def test_non_integer_value_returns_zero(wrap):
    """Defends against a judge that (despite feedback_value_type=int) returns
    something that can't be coerced to int, e.g. a free-text string."""
    fake_judge = _fake_judge("not a number", wrap)
    with patch(
        "revisao_agents.evaluation.evaluators.get_or_create_plan_quality_judge",
        return_value=fake_judge,
    ):
        score = await evaluate_plan_quality("theme", "## Section\ncontent")

    assert score == 0.0


async def test_judge_call_exception_propagates():
    """A transient LLM/API failure inside the judge call must propagate, not
    be silently swallowed into a 0.0 score — callers running this unattended
    are expected to catch around the call (see docstring)."""
    fake_judge = MagicMock(side_effect=RuntimeError("LLM API unavailable"))
    with (
        patch(
            "revisao_agents.evaluation.evaluators.get_or_create_plan_quality_judge",
            return_value=fake_judge,
        ),
        pytest.raises(RuntimeError, match="LLM API unavailable"),
    ):
        await evaluate_plan_quality("theme", "## Section\ncontent")


async def test_judge_called_with_theme_and_truncated_plan():
    fake_judge = _fake_judge(5, wrap=lambda v: v)
    long_plan = "## Section\n" + ("x" * 5000)
    with patch(
        "revisao_agents.evaluation.evaluators.get_or_create_plan_quality_judge",
        return_value=fake_judge,
    ):
        await evaluate_plan_quality("my theme", long_plan)

    call_kwargs = fake_judge.call_args.kwargs
    assert call_kwargs["inputs"] == {"theme": "my theme"}
    assert len(call_kwargs["outputs"]["plan"]) == 4000
