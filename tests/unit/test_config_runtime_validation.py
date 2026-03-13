"""
Unit tests for runtime configuration validation helpers.
"""

import os

from revisao_agents.config import get_runtime_config_summary, validate_runtime_config


def test_validate_runtime_config_reports_missing_provider_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    issues = validate_runtime_config(strict=False)

    assert any("GROQ_API_KEY" in issue for issue in issues)


def test_validate_runtime_config_reports_mode_requirements(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("MONGODB_URI", "")
    monkeypatch.setenv("TAVILY_API_KEY", "")

    issues = validate_runtime_config(
        require_mongodb=True,
        require_tavily=True,
        require_openai_embeddings=True,
        strict=False,
    )

    assert any("MONGODB_URI" in issue for issue in issues)
    assert any("TAVILY_API_KEY" in issue for issue in issues)
    assert any("OPENAI_API_KEY" in issue for issue in issues)


def test_runtime_summary_has_expected_keys():
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
