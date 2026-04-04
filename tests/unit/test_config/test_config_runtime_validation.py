"""
Unit tests for runtime configuration validation helpers.
"""

import pytest

from revisao_agents.config import get_runtime_config_summary, validate_runtime_config


@pytest.mark.parametrize("provider", ["openai", "google", "groq", "openrouter"])
def test_validate_runtime_config_valid_providers(monkeypatch, provider):
    """Test that valid providers do not produce LLM_PROVIDER errors.
    This ensures that the validate_provider function is correctly integrated into validate_runtime_config.

    Args:
        monkeypatch: pytest fixture for modifying environment variables
        provider: the LLM provider to test

    Asserts:
        No issues related to LLM_PROVIDER are present in the validation results for valid providers.
    """
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("OPENAI_API_KEY", "openai_key-value")
    monkeypatch.setenv("MONGODB_URI", "mongodb_uri-value")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily_key-value")

    issues = validate_runtime_config(strict=False)

    assert not any("LLM_PROVIDER error" in issue for issue in issues)


def test_validate_config_summary_when_llm_provider_env_not_set_defaults_to_openai(monkeypatch):
    """Test that when LLM_PROVIDER env var is missing, the runtime summary defaults to 'openai' and no errors are raised.
    This verifies that the defaulting behavior in validate_provider is correctly reflected in the runtime configuration summary.

    Args:
        monkeypatch: pytest fixture for modifying environment variables

    Asserts:
        The runtime config summary reports 'openai' as the llm_provider when LLM_PROVIDER env var is not set.
    """
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai_key-value")
    monkeypatch.setenv("MONGODB_URI", "mongodb_uri-value")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily_key-value")

    summary = get_runtime_config_summary()

    assert summary["llm_provider"] == "openai"


def test_validate_runtime_config_reports_missing_provider_key(monkeypatch):
    """Test that if a valid provider is set but its required API key is missing, the validation reports the missing key.
    This ensures that validate_runtime_config correctly checks for provider-specific requirements and reports them as issues.

    Args:
        monkeypatch: pytest fixture for modifying environment variables

    Asserts:
        The validation issues include a missing key for the specified provider.
    """
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai_key-value")
    monkeypatch.setenv("MONGODB_URI", "mongodb_uri-value")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily_key-value")

    issues = validate_runtime_config(strict=False)

    assert any("GROQ_API_KEY" in issue for issue in issues)


def test_validate_runtime_config_missing_global_mandatory_requirements_reports(monkeypatch):
    """Test that when specific requirements are set (e.g. require_mongodb, require_tavily, require_openai_embeddings), the validation reports all missing configurations.
    This verifies that validate_runtime_config correctly accumulates issues for all missing requirements when strict mode is
    disabled.

    Args:
        monkeypatch: pytest fixture for modifying environment variables

    Asserts:
        The validation issues include all missing configurations for the specified requirements.
    """
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("MONGODB_URI", "")
    monkeypatch.setenv("TAVILY_API_KEY", "")

    issues = validate_runtime_config(strict=False)

    assert any("MONGODB_URI" in issue for issue in issues)
    assert any("TAVILY_API_KEY" in issue for issue in issues)
    assert any("OPENAI_API_KEY" in issue for issue in issues)


@pytest.mark.parametrize("provider", ["gemini", "azure", "aws", "custom"])
def test_validate_runtime_config_accumulates_issues_when_provider_invalid(monkeypatch, provider):
    """Test that when an invalid provider is set, the validation accumulates issues for all missing configurations.
    This ensures that validate_runtime_config correctly reports issues for invalid providers in combination with other missing requirements.

    Args:
        monkeypatch: pytest fixture for modifying environment variables
        provider: the invalid LLM provider to test

    Asserts:
        The validation issues include an LLM_PROVIDER error and all missing configurations for the specified requirements.
    """
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("OPENAI_API_KEY", "openai_key-value")
    monkeypatch.setenv("MONGODB_URI", "mongodb_uri-value")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily_key-value")

    issues = validate_runtime_config(strict=False)

    assert any("LLM_PROVIDER error" in issue for issue in issues)


def test_runtime_summary_has_expected_keys():
    """Test that the runtime config summary includes all expected keys.

    Asserts:
        The summary contains keys for LLM provider, model, API keys, and other required configurations.
    """

    summary = get_runtime_config_summary()

    expected = {
        "llm_provider",
        "llm_model",
        "llm_provider_key",
        "llm_provider_key_present",
        "mongodb_uri_present",
        "tavily_key_present",
        "openai_key_present",
    }
    assert expected.issubset(set(summary.keys()))
