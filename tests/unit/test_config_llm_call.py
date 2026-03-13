"""
Unit tests for config.llm_call behavior and failure semantics.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from revisao_agents.config import LLMInvocationError, llm_call


def test_llm_call_returns_text_content():
    fake_llm = SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content="ok"))

    with patch("revisao_agents.utils.llm_utils.llm_providers.get_llm", return_value=fake_llm):
        result = llm_call("hello", temperature=0.1)

    assert result == "ok"


def test_llm_call_structured_output_path():
    class _Structured:
        def invoke(self, prompt):
            return {"status": "ok"}

    class _LLM:
        def with_structured_output(self, schema):
            return _Structured()

    with patch("revisao_agents.utils.llm_utils.llm_providers.get_llm", return_value=_LLM()):
        result = llm_call("hello", response_schema=dict)

    assert result == {"status": "ok"}


def test_llm_call_raises_typed_error_on_provider_failure():
    with patch(
        "revisao_agents.utils.llm_utils.llm_providers.get_llm",
        side_effect=RuntimeError("provider unavailable"),
    ):
        with pytest.raises(LLMInvocationError):
            llm_call("hello")
