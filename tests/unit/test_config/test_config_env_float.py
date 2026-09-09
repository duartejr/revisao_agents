"""Unit tests for ``revisao_agents.config._env_float`` (W9-STORY-05)."""

import pytest

from revisao_agents.config import _env_float


def test_returns_default_when_unset(monkeypatch):
    monkeypatch.delenv("SOME_FLOAT_VAR", raising=False)
    assert _env_float("SOME_FLOAT_VAR", 0.7) == 0.7


def test_returns_default_when_empty_string(monkeypatch):
    monkeypatch.setenv("SOME_FLOAT_VAR", "")
    assert _env_float("SOME_FLOAT_VAR", 0.7) == 0.7


def test_parses_explicit_value(monkeypatch):
    monkeypatch.setenv("SOME_FLOAT_VAR", "0.5")
    assert _env_float("SOME_FLOAT_VAR", 0.7) == 0.5


def test_strips_quotes_and_whitespace(monkeypatch):
    monkeypatch.setenv("SOME_FLOAT_VAR", ' "0.9" ')
    assert _env_float("SOME_FLOAT_VAR", 0.7) == 0.9


def test_raises_on_non_numeric_value(monkeypatch):
    monkeypatch.setenv("SOME_FLOAT_VAR", "not-a-number")
    with pytest.raises(ValueError, match="SOME_FLOAT_VAR"):
        _env_float("SOME_FLOAT_VAR", 0.7)
