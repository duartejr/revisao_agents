"""
Unit tests for LLM provider handling in gradio_app.handlers, including normalization, validation, and runtime configuration summary reporting.
Tests cover:
- Setting valid and invalid LLM providers and checking normalization and error handling.
- Getting the current LLM provider and ensuring it defaults to 'openai' when not set.
- Validating runtime configuration and checking that missing required keys are reported as issues.
- Ensuring the runtime configuration summary includes expected keys and handles invalid providers gracefully.
"""

import pytest

from gradio_app.handlers import get_current_llm_provider, set_llm_provider


@pytest.mark.parametrize("provider", ["openai", "google", "groq", "openrouter"])
def test_set_valid_llm_provider(monkeypatch, provider):
    """Test that set_llm_provider correctly normalizes valid provider names and returns appropriate status.
    This ensures that the set_llm_provider function accepts valid provider names, normalizes them,
    and returns a status message indicating the provider is set.

    Args:
        monkeypatch: pytest fixture for modifying environment variables
        provider: the LLM provider to test

    Asserts:
        The returned normalized provider matches the input and the status message indicates the provider is set.
    """
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("OPENAI_API_KEY", "openai_key-value")
    monkeypatch.setenv("MONGODB_URI", "mongodb_uri-value")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily_key-value")

    normalized, status = set_llm_provider(provider)

    assert normalized == provider
    assert "Provedor" in status


@pytest.mark.parametrize("provider", ["gemini", "azure", "aws", "custom"])
def test_set_invalid_llm_provider(monkeypatch, provider):
    """Test that set_llm_provider raises a ValueError for invalid provider names and reports the issue in the status message.
    This ensures that the set_llm_provider function correctly identifies unsupported provider names, raises an appropriate
    error, and that the error is reflected in the status message.

    Args:
        monkeypatch: pytest fixture for modifying environment variables
        provider: the invalid LLM provider to test

    Asserts:
        A ValueError is raised for invalid provider names, and the status message includes information about the unsupported provider.
    """
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("OPENAI_API_KEY", "openai_key-value")
    monkeypatch.setenv("MONGODB_URI", "mongodb_uri-value")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily_key-value")

    with pytest.raises(ValueError, match="is not supported. Accepted:"):
        set_llm_provider(provider)


def test_get_llm_provider_defaults_to_openai(monkeypatch):
    """Test that get_current_llm_provider defaults to 'openai' when LLM_PROVIDER env var is not set.
    This ensures that get_current_llm_provider correctly defaults to 'openai' when no provider
    is specified in the environment variables.

    Args:
        monkeypatch: pytest fixture for modifying environment variables

    Asserts:
        The get_current_llm_provider function returns 'openai' when LLM_PROVIDER is not set.
    """
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    provider = get_current_llm_provider()

    assert provider == "openai"


@pytest.mark.parametrize("provider", ["gemini", "azure", "aws", "custom"])
def test_get_llm_provider_with_invalid_provider(monkeypatch, provider):
    """Test that get_current_llm_provider returns 'openai' when LLM_PROVIDER env var is set to an invalid provider.
    This ensures that get_current_llm_provider correctly handles invalid provider names in the environment variables
    and defaults to 'openai' instead.

    Args:
        monkeypatch: pytest fixture for modifying environment variables
        provider: the invalid LLM provider to test

    Asserts:
        The get_current_llm_provider function returns 'openai' when LLM_PROVIDER is set to an unsupported provider name.
    """
    monkeypatch.setenv("LLM_PROVIDER", provider)

    result = get_current_llm_provider()

    assert result == "openai"


def test_get_llm_provider_status_with_missing_api_key(monkeypatch):
    """Test that get_llm_provider_status reports missing API key correctly.

    Args:
        monkeypatch: pytest fixture for modifying environment variables

    Asserts:
        Status message includes warning about missing API key.
    """
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai_key-value")
    monkeypatch.setenv("MONGODB_URI", "mongodb_uri-value")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily_key-value")

    from gradio_app.handlers import get_llm_provider_status

    status = get_llm_provider_status()

    assert "⚠️" in status
    assert "missing GROQ_API_KEY" in status


def test_get_llm_provider_status_with_valid_config(monkeypatch):
    """Test that get_llm_provider_status shows success status for valid configuration.

    Args:
        monkeypatch: pytest fixture for modifying environment variables

    Asserts:
        Status message includes success indicator and correct provider/model info.
    """
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "openai_key-value")
    monkeypatch.setenv("MONGODB_URI", "mongodb_uri-value")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily_key-value")

    from gradio_app.handlers import get_llm_provider_status

    status = get_llm_provider_status()

    assert "✅" in status
    assert "Provedor: Openai" in status
    assert "key ok" in status
