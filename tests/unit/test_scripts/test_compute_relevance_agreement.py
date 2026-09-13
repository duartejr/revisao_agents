"""Unit tests for ``scripts/compute_relevance_agreement.py``.

``scripts/`` is not an installed package, so the module under test is loaded
via ``importlib`` from its file path rather than a normal import.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "compute_relevance_agreement.py"


def _load_module():
    """Import ``compute_relevance_agreement.py`` fresh, bypassing module caching."""
    spec = importlib.util.spec_from_file_location("compute_relevance_agreement", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["compute_relevance_agreement"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def agreement():
    """Load the script module (no external dependencies, so no fixture needed)."""
    return _load_module()


def _row(human_label: str, judge_level: str) -> dict:
    return {"human_label": human_label, "llm_judge_relevance_level": judge_level}


# ── compute_agreement ────────────────────────────────────────────────────────


def test_all_agree(agreement):
    rows = [
        _row("relevant", "Perfectly relevant"),
        _row("not_relevant", "Not relevant"),
    ]
    result = agreement.compute_agreement(rows)
    assert result["labeled_rows"] == 2
    assert result["unlabeled_rows"] == 0
    assert result["agreements"] == 2
    assert result["disagreements"] == 0
    assert result["agreement_rate"] == pytest.approx(1.0)


def test_all_disagree(agreement):
    rows = [
        _row("relevant", "Not relevant"),
        _row("not_relevant", "Perfectly relevant"),
    ]
    result = agreement.compute_agreement(rows)
    assert result["agreements"] == 0
    assert result["disagreements"] == 2
    assert result["agreement_rate"] == pytest.approx(0.0)


def test_unlabeled_rows_excluded_from_agreement(agreement):
    rows = [
        _row("relevant", "Perfectly relevant"),
        _row("", "Not relevant"),  # not yet labeled — must be excluded
        _row("   ", "Not relevant"),  # whitespace-only — also excluded
    ]
    result = agreement.compute_agreement(rows)
    assert result["total_rows"] == 3
    assert result["labeled_rows"] == 1
    assert result["unlabeled_rows"] == 2
    assert result["agreement_rate"] == pytest.approx(1.0)


def test_unrecognized_label_value_is_reported_and_excluded(agreement):
    rows = [_row("maybe", "Perfectly relevant")]
    result = agreement.compute_agreement(rows)
    assert result["unrecognized_labels"] == ["maybe"]
    assert result["agreement_rate"] is None


def test_no_labeled_rows_gives_none_rate(agreement):
    rows = [_row("", "Perfectly relevant")]
    result = agreement.compute_agreement(rows)
    assert result["agreement_rate"] is None


def test_human_label_case_insensitive(agreement):
    rows = [_row("RELEVANT", "Perfectly relevant")]
    result = agreement.compute_agreement(rows)
    assert result["agreements"] == 1


# ── format_report ────────────────────────────────────────────────────────────


def test_format_report_mentions_unlabeled_note_when_no_scoreable_rows(agreement):
    result = agreement.compute_agreement([_row("", "Perfectly relevant")])
    report = agreement.format_report(result)
    assert "No scoreable labeled rows yet" in report


def test_format_report_includes_rate_and_caveat(agreement):
    result = agreement.compute_agreement([_row("relevant", "Perfectly relevant")])
    report = agreement.format_report(result)
    assert "100.0%" in report
    assert "single-labeler" in report


# ── main() error handling ────────────────────────────────────────────────────


def test_main_exits_cleanly_when_csv_missing(agreement, tmp_path, monkeypatch, capsys):
    missing_path = tmp_path / "does_not_exist.csv"
    monkeypatch.setattr(sys, "argv", ["compute_relevance_agreement.py", str(missing_path)])

    with pytest.raises(SystemExit):
        agreement.main()

    captured = capsys.readouterr()
    assert "not found" in captured.out
