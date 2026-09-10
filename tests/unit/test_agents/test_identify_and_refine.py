"""
tests/unit/test_agents/test_identify_and_refine.py

Unit tests for identify_and_refine_node in revisao_agents.nodes.common.
"""

from unittest.mock import MagicMock, patch

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
        "is_theme_vague": False,
        "is_theme_refined": False,
        "detected_language": "",
        "refinement_feedback": [],
    }
    base.update(overrides)
    return base


def _make_llm_response(content: str) -> MagicMock:
    """Create a mock LLM response with the given content string.

    Args:
        content: The text to set as the mock's ``.content`` attribute.

    Returns:
        A ``MagicMock`` whose ``.content`` attribute equals ``content``.
    """
    mock = MagicMock()
    mock.content = content
    return mock


class TestIdentifyAndRefineNode:
    """Unit tests for ``identify_and_refine_node`` state transitions."""

    def test_pt_non_vague_sets_refined_true(self):
        from revisao_agents.nodes.common import identify_and_refine_node

        llm_content = "DETECTED LANGUAGE: PT\nIS THE THEME VAGUE? NO\n"
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = _make_llm_response(llm_content)

        with patch("revisao_agents.nodes.common.get_llm", return_value=fake_llm):
            result = identify_and_refine_node(
                _make_state(theme="Aprendizado de máquina supervisionado")
            )

        assert result["detected_language"] == "PT"
        assert result["is_theme_vague"] is False
        assert result["is_theme_refined"] is True
        assert result["theme"] == "Aprendizado de máquina supervisionado"

    def test_en_vague_sets_vague_true_and_refined_false(self):
        from revisao_agents.nodes.common import identify_and_refine_node

        llm_content = "DETECTED LANGUAGE: EN\nIS THE THEME VAGUE? YES\n"
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = _make_llm_response(llm_content)

        with patch("revisao_agents.nodes.common.get_llm", return_value=fake_llm):
            result = identify_and_refine_node(_make_state(theme="AI"))

        assert result["detected_language"] == "EN"
        assert result["is_theme_vague"] is True
        assert result["is_theme_refined"] is False

    def test_unknown_language_sets_refined_false(self):
        from revisao_agents.nodes.common import identify_and_refine_node

        llm_content = "DETECTED LANGUAGE: UNKNOWN\nIS THE THEME VAGUE? NO\n"
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = _make_llm_response(llm_content)

        with patch("revisao_agents.nodes.common.get_llm", return_value=fake_llm):
            result = identify_and_refine_node(_make_state(theme="Some topic"))

        assert result["detected_language"] == "UNKNOWN"
        assert result["is_theme_refined"] is False

    def test_user_reply_used_as_candidate_when_was_vague(self):
        """When the previous round detected vagueness, the last user message becomes the new theme."""
        from revisao_agents.nodes.common import identify_and_refine_node

        llm_content = "DETECTED LANGUAGE: EN\nIS THE THEME VAGUE? NO\n"
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = _make_llm_response(llm_content)

        state = _make_state(
            theme="AI",
            is_theme_vague=True,
            interview_history=[
                ("assistant", "Could you narrow down your topic?"),
                ("user", "Transfer learning for NLP tasks"),
            ],
        )

        with patch("revisao_agents.nodes.common.get_llm", return_value=fake_llm):
            result = identify_and_refine_node(state)

        assert result["theme"] == "Transfer learning for NLP tasks"
        assert result["is_theme_refined"] is True

    def test_refinement_feedback_contains_llm_output(self):
        from revisao_agents.nodes.common import identify_and_refine_node

        llm_content = "DETECTED LANGUAGE: EN\nIS THE THEME VAGUE? NO\n"
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = _make_llm_response(llm_content)

        with patch("revisao_agents.nodes.common.get_llm", return_value=fake_llm):
            result = identify_and_refine_node(_make_state(theme="Neural networks"))

        assert result["refinement_feedback"] == [llm_content]

    def test_interview_history_appended_with_assistant_message(self):
        from revisao_agents.nodes.common import identify_and_refine_node

        llm_content = "DETECTED LANGUAGE: PT\nIS THE THEME VAGUE? NO\n"
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = _make_llm_response(llm_content)

        with patch("revisao_agents.nodes.common.get_llm", return_value=fake_llm):
            result = identify_and_refine_node(_make_state(theme="Redes neurais profundas"))

        assert len(result["interview_history"]) == 1
        role, _msg = result["interview_history"][0]
        assert role == "assistant"


class TestRefinementBypass:
    """Unit tests for the W9-STORY-03 ``BYPASS_REFINEMENT_LAYER`` bypass flag.

    Confirms the flag defaults to off (production path unaffected) and that
    enabling it skips the LLM call entirely rather than merely changing its
    outcome.
    """

    def test_bypass_disabled_by_default_calls_llm(self, monkeypatch):
        from revisao_agents.nodes.common import identify_and_refine_node

        monkeypatch.delenv("BYPASS_REFINEMENT_LAYER", raising=False)
        llm_content = "DETECTED LANGUAGE: EN\nIS THE THEME VAGUE? NO\n"
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = _make_llm_response(llm_content)

        with patch("revisao_agents.nodes.common.get_llm", return_value=fake_llm):
            identify_and_refine_node(_make_state(theme="Reinforcement learning"))

        fake_llm.invoke.assert_called_once()

    def test_bypass_enabled_skips_llm_and_refines_immediately(self, monkeypatch):
        from revisao_agents.nodes.common import identify_and_refine_node

        monkeypatch.setenv("BYPASS_REFINEMENT_LAYER", "true")
        fake_llm = MagicMock()

        with patch("revisao_agents.nodes.common.get_llm", return_value=fake_llm):
            result = identify_and_refine_node(_make_state(theme="machine learning"))

        fake_llm.invoke.assert_not_called()
        assert result["is_theme_refined"] is True
        assert result["is_theme_vague"] is False
        assert result["detected_language"] == "EN"
        assert result["interview_history"] == []
        assert result["theme"] == "machine learning"

    def test_bypass_is_case_insensitive_and_rejects_truthy_junk(self, monkeypatch):
        from revisao_agents.nodes.common import identify_and_refine_node

        monkeypatch.setenv("BYPASS_REFINEMENT_LAYER", "TRUE")
        fake_llm = MagicMock()
        with patch("revisao_agents.nodes.common.get_llm", return_value=fake_llm):
            identify_and_refine_node(_make_state(theme="AI"))
        fake_llm.invoke.assert_not_called()

        monkeypatch.setenv("BYPASS_REFINEMENT_LAYER", "1")
        fake_llm2 = MagicMock()
        fake_llm2.invoke.return_value = _make_llm_response(
            "DETECTED LANGUAGE: EN\nIS THE THEME VAGUE? NO\n"
        )
        with patch("revisao_agents.nodes.common.get_llm", return_value=fake_llm2):
            identify_and_refine_node(_make_state(theme="AI"))
        fake_llm2.invoke.assert_called_once()
