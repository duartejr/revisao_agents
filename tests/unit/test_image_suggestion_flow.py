"""Unit tests for image suggestion state machine in review_chat_turn."""

from unittest.mock import patch

import pytest

from gradio_app.handlers import (
    _build_image_confirmation_prompt,
    _build_image_scope_description,
    _is_image_request,
    review_chat_turn,
)

# ---------------------------------------------------------------------------
# _is_image_request
# ---------------------------------------------------------------------------

_SAMPLE_MD = """\
# My Paper

## Introduction

This paper presents results.

## Methodology

We used a transformer model.
"""

_SECTIONS = [
    {
        "title": "Introduction",
        "paragraphs": [
            {"text": "This paper presents results."},
            {"text": "The method is described here."},
        ],
    },
    {
        "title": "Methodology",
        "paragraphs": [
            {"text": "We used a transformer model."},
        ],
    },
]


class TestIsImageRequest:
    def test_pt_imagem(self):
        assert _is_image_request("adicione uma imagem aqui") is True

    def test_pt_figura(self):
        assert _is_image_request("precisa de uma figura neste trecho") is True

    def test_pt_diagrama(self):
        assert _is_image_request("insira um diagrama") is True

    def test_en_image(self):
        assert _is_image_request("find an image for this section") is True

    def test_en_diagram(self):
        assert _is_image_request("suggest a diagram here") is True

    def test_not_image_request(self):
        assert _is_image_request("resolva a referência [3]") is False

    def test_empty_string(self):
        assert _is_image_request("") is False


# ---------------------------------------------------------------------------
# _build_image_scope_description
# ---------------------------------------------------------------------------


class TestBuildImageScopeDescription:
    def test_all_sections_default_includes_all_titles(self):
        _, excerpt = _build_image_scope_description("add images", _SECTIONS)
        assert "Introduction" in excerpt
        assert "Methodology" in excerpt

    def test_all_sections_scope_en(self):
        scope, _ = _build_image_scope_description("add images", _SECTIONS, language="en")
        assert "section" in scope.lower() or "all" in scope.lower()

    def test_all_sections_scope_pt(self):
        scope, _ = _build_image_scope_description("adicionar imagens", _SECTIONS, language="pt")
        assert "seç" in scope.lower() or "todas" in scope.lower()

    def test_paragraph_request_includes_marker(self):
        _, excerpt = _build_image_scope_description(
            "imagem para o parágrafo 1", _SECTIONS, language="pt"
        )
        assert "[PARAGRAPH 1]" in excerpt

    def test_scope_pt_paragraph_label(self):
        scope, _ = _build_image_scope_description(
            "imagem para o parágrafo 2", _SECTIONS, language="pt"
        )
        assert "parágrafo" in scope.lower()

    def test_scope_en_paragraph_label(self):
        scope, _ = _build_image_scope_description("image for paragraph 1", _SECTIONS, language="en")
        assert "paragraph" in scope.lower()

    def test_scope_en_section_label(self):
        scope, _ = _build_image_scope_description(
            "add image to section 1", _SECTIONS, language="en"
        )
        assert "section" in scope.lower()

    def test_scope_pt_section_label(self):
        scope, _ = _build_image_scope_description(
            "adicionar imagem na seção 1", _SECTIONS, language="pt"
        )
        assert "seç" in scope.lower()


# ---------------------------------------------------------------------------
# _build_image_confirmation_prompt
# ---------------------------------------------------------------------------


class TestBuildImageConfirmationPrompt:
    def test_pt_prompt_contains_scope(self):
        prompt = _build_image_confirmation_prompt("seção 1 — Introduction", "pt")
        assert "Introduction" in prompt
        assert any(kw in prompt.lower() for kw in ["sim", "seção", "escopo"])

    def test_en_prompt_contains_scope(self):
        prompt = _build_image_confirmation_prompt("section 1 — Introduction", "en")
        assert "Introduction" in prompt
        assert any(kw in prompt.lower() for kw in ["yes", "section", "scope"])


# ---------------------------------------------------------------------------
# review_chat_turn — image state machine
# ---------------------------------------------------------------------------


@pytest.fixture()
def working_copy(tmp_path):
    f = tmp_path / "paper.md"
    f.write_text(_SAMPLE_MD, encoding="utf-8")
    return str(f)


@pytest.fixture()
def base_state(working_copy):
    return {
        "working_copy_path": working_copy,
        "current_markdown": _SAMPLE_MD,
        "chat_history": [],
        "pending_edit": {},
        "pending_image_action": {},
        "awaiting_image_confirmation": False,
        "retrieval_trace": [],
    }


class TestImageStateMachine:
    def test_first_image_request_sets_awaiting_confirmation(self, base_state):
        _, state, _, _ = review_chat_turn("adicione uma imagem nesta seção", [], base_state)
        assert state.get("awaiting_image_confirmation") is True
        assert state.get("pending_image_action")

    def test_first_image_request_returns_confirmation_in_history(self, base_state):
        history, _, _, _ = review_chat_turn("adicione uma imagem nesta seção", [], base_state)
        last_reply = history[-1]["content"]
        assert any(kw in last_reply.lower() for kw in ["sim", "yes", "confirme", "confirm"])

    def test_cancel_clears_awaiting_flag(self, base_state):
        base_state["awaiting_image_confirmation"] = True
        base_state["pending_image_action"] = {
            "scope": "section 1",
            "excerpt": "some text",
            "original_request": "find image",
        }
        _, state, _, _ = review_chat_turn("não", [], base_state)
        assert not state.get("awaiting_image_confirmation")

    def test_cancel_clears_pending_action(self, base_state):
        base_state["awaiting_image_confirmation"] = True
        base_state["pending_image_action"] = {
            "scope": "section 1",
            "excerpt": "some text",
            "original_request": "find image",
        }
        _, state, _, _ = review_chat_turn("cancelar", [], base_state)
        assert not state.get("pending_image_action")

    def test_affirmative_calls_image_agent(self, base_state):
        base_state["awaiting_image_confirmation"] = True
        base_state["pending_image_action"] = {
            "scope": "section 1 — Introduction",
            "excerpt": "some text",
            "original_request": "find image",
        }
        with patch(
            "gradio_app.handlers.run_image_suggestion_agent",
            return_value="![img](https://example.com/img.png)",
        ) as mock_agent:
            history, state, _, _ = review_chat_turn("sim", [], base_state)
            mock_agent.assert_called_once()

        assert not state.get("awaiting_image_confirmation")
        assert history[-1]["content"] == "![img](https://example.com/img.png)"

    def test_affirmative_clears_pending_action(self, base_state):
        base_state["awaiting_image_confirmation"] = True
        base_state["pending_image_action"] = {
            "scope": "section 1",
            "excerpt": "some text",
            "original_request": "find image",
        }
        with patch("gradio_app.handlers.run_image_suggestion_agent", return_value="result"):
            _, state, _, _ = review_chat_turn("yes", [], base_state)
        assert state.get("pending_image_action") == {}

    def test_scope_override_during_confirmation_updates_scope(self, base_state):
        base_state["awaiting_image_confirmation"] = True
        base_state["pending_image_action"] = {
            "scope": "all sections of the document",
            "excerpt": "original excerpt",
            "original_request": "find image",
        }
        with patch(
            "gradio_app.handlers.run_image_suggestion_agent", return_value="result"
        ) as mock_agent:
            _, state, _, _ = review_chat_turn(
                "busque imagens apenas para o parágrafo 1", [], base_state
            )
            # Agent should have been called with the overridden scope, not "all sections"
            call_kwargs = mock_agent.call_args[1]
            assert "all sections" not in call_kwargs.get("scope_description", "")

        assert not state.get("awaiting_image_confirmation")
