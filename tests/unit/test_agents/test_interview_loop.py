"""
tests/unit/test_agents/test_interview_loop.py

Unit tests for interview_router in revisao_agents.nodes.common.
"""

import pytest

from revisao_agents.nodes.common import interview_router
from revisao_agents.state import ReviewState


def _make_state(**overrides) -> ReviewState:
    """Create a base ReviewState with optional overrides for testing.

    Args:
        **overrides: Key-value pairs to override default state values.

    Returns:
        ReviewState with the specified overrides applied.
    """
    base: ReviewState = {
        "theme": "Test topic",
        "review_type": "academico",
        "relevant_chunks": [],
        "technical_snippets": [],
        "technical_urls": [],
        "current_plan": "",
        "interview_history": [],
        "questions_asked": 0,
        "max_questions": 3,
        "final_plan": "",
        "final_plan_path": "",
        "status": "starting",
    }
    base.update(overrides)
    return base


class TestInterviewRouter:
    """Unit tests for ``interview_router`` routing logic."""

    def test_returns_finish_when_questions_at_max(self):
        state = _make_state(questions_asked=3, max_questions=3)
        assert interview_router(state) == "finish"

    def test_returns_finish_when_questions_exceed_max(self):
        state = _make_state(questions_asked=5, max_questions=3)
        assert interview_router(state) == "finish"

    def test_returns_continue_when_below_max(self):
        state = _make_state(questions_asked=1, max_questions=3)
        assert interview_router(state) == "continue"

    def test_returns_continue_when_zero_questions(self):
        state = _make_state(questions_asked=0, max_questions=3)
        assert interview_router(state) == "continue"

    def test_quota_check_takes_priority_over_keyword(self):
        """When questions_asked >= max_questions, quota short-circuits before keyword check."""
        state = _make_state(
            questions_asked=3,
            max_questions=3,
            interview_history=[("user", "machine learning")],  # no keyword
        )
        assert interview_router(state) == "finish"

    @pytest.mark.parametrize(
        "keyword",
        [
            "fim",
            "terminar",
            "sair",
            "encerrar",
            "pronto",
            "acabar",
            "end",
            "finish",
            "exit",
            "done",
            "stop",
            "quit",
        ],
    )
    def test_returns_finish_on_termination_keyword(self, keyword: str):
        state = _make_state(
            questions_asked=1,
            max_questions=5,
            interview_history=[("assistant", "What is your topic?"), ("user", keyword)],
        )
        assert interview_router(state) == "finish"

    def test_returns_continue_when_last_user_message_is_not_keyword(self):
        state = _make_state(
            questions_asked=1,
            max_questions=5,
            interview_history=[("assistant", "Tell me more."), ("user", "machine learning")],
        )
        assert interview_router(state) == "continue"

    def test_ignores_assistant_keyword_message(self):
        state = _make_state(
            questions_asked=1,
            max_questions=5,
            interview_history=[("assistant", "done"), ("user", "tell me more")],
        )
        assert interview_router(state) == "continue"

    def test_empty_history_returns_continue(self):
        state = _make_state(questions_asked=0, max_questions=3, interview_history=[])
        assert interview_router(state) == "continue"
