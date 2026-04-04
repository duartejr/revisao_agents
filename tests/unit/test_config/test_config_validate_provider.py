"""
Unit tests for config.validate_provider behavior and failure semantics.
"""

import pytest

from revisao_agents.config import validate_provider


@pytest.mark.parametrize("provider", ["openai", "google", "groq", "openrouter"])
def test_validate_provider_is_valid(provider):
    """Test that valid provider names are normalized correctly."""
    assert validate_provider(provider) == provider


@pytest.mark.parametrize("provider", ["", None])
def test_validate_provider_defaults_to_openai(provider):
    """Test that empty or None provider defaults to 'openai'."""
    assert validate_provider(provider) == "openai"


@pytest.mark.parametrize("provider", ["gemini", "Gemini", "GEMINI", "azure", "custom"])
def test_validate_provider_invalid(provider):
    """Test that invalid provider names raise a ValueError."""
    with pytest.raises(ValueError, match="is not supported. Accepted:"):
        validate_provider(provider)
