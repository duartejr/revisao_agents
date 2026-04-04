"""
Unit tests for config.validate_provider behavior and failure semantics.
"""

import pytest

from revisao_agents.config import validate_provider


@pytest.mark.parametrize("provider", ["openai", "google", "groq", "openrouter"])
def test_validate_provider_is_valid(provider):
    """Test that valid provider names are normalized correctly.
    This ensures that the validate_provider function accepts valid provider names and returns them in a consistent format.

    Args:
        provider: the LLM provider name to test

    Asserts:
        The validate_provider function returns the expected normalized provider name for valid inputs.
    """
    assert validate_provider(provider) == provider


@pytest.mark.parametrize("provider", ["", None])
def test_validate_provider_defaults_to_openai(provider):
    """Test that empty or None provider defaults to 'openai'.
    This ensures that the validate_provider function correctly defaults to 'openai' when no provider is specified.

    Args:
        provider: the LLM provider name to test

    Asserts:
        The validate_provider function returns 'openai' for empty or None inputs.
    """
    assert validate_provider(provider) == "openai"


@pytest.mark.parametrize("provider", ["gemini", "Gemini", "GEMINI", "azure", "custom"])
def test_validate_provider_invalid(provider):
    """Test that invalid provider names raise a ValueError.
    This ensures that the validate_provider function correctly identifies unsupported provider names and raises an appropriate error.
    Args:
        provider: the LLM provider name to test
    Asserts:
        A ValueError is raised for invalid provider names.
    """
    with pytest.raises(ValueError, match="is not supported. Accepted:"):
        validate_provider(provider)
