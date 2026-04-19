"""
tests/unit/test_cli.py

Unit tests for resolve_topic in revisao_agents.cli.
"""

from pathlib import Path


def test_resolve_topic_plain_string():
    from revisao_agents.cli import resolve_topic

    result = resolve_topic("  machine learning  ")
    assert result == "machine learning"


def test_resolve_topic_nonexistent_path_returns_as_is(tmp_path):
    from revisao_agents.cli import resolve_topic

    nonexistent = str(tmp_path / "no_such_file.txt")
    result = resolve_topic(nonexistent)
    assert result == nonexistent


def test_resolve_topic_file_with_topic_header(tmp_path: Path):
    from revisao_agents.cli import resolve_topic

    f = tmp_path / "plan.md"
    f.write_text(
        "# Review Plan\n\n**Topic:** Transfer learning for NLP\n\nMore content here.",
        encoding="utf-8",
    )
    result = resolve_topic(str(f))
    assert result == "Transfer learning for NLP"


def test_resolve_topic_file_with_theme_header(tmp_path: Path):
    from revisao_agents.cli import resolve_topic

    f = tmp_path / "plan.md"
    f.write_text("**Theme:** Deep reinforcement learning\n\nIntroduction.", encoding="utf-8")
    result = resolve_topic(str(f))
    assert result == "Deep reinforcement learning"


def test_resolve_topic_file_falls_back_to_first_line(tmp_path: Path):
    from revisao_agents.cli import resolve_topic

    f = tmp_path / "topic.txt"
    f.write_text("\n\nConvolutional neural networks\nSome other content.", encoding="utf-8")
    result = resolve_topic(str(f))
    assert result == "Convolutional neural networks"


def test_resolve_topic_file_empty_falls_back_to_input(tmp_path: Path):
    from revisao_agents.cli import resolve_topic

    f = tmp_path / "empty.txt"
    f.write_text("   \n\n   ", encoding="utf-8")
    input_val = str(f)
    result = resolve_topic(input_val)
    assert result == input_val.strip()
