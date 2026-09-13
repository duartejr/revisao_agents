"""
compute_relevance_agreement.py - W10-STORY-05 Human/LLM-Judge Agreement Check

Reads the labeled CSV produced by ``scripts/sample_relevance_labels.py``
(after a human has filled in its ``human_label`` column) and computes a
simple exact-match agreement rate between the human labels and the existing
LLM-as-judge relevance scores, as a first calibration check.

Expected ``human_label`` values (matching the LLM judge's actual binary
scale — see ``data/eval/README.md``):
    - ``relevant``     (maps to the judge's "Perfectly relevant" / 1.0)
    - ``not_relevant`` (maps to the judge's "Not relevant" / 0.0)

Rows with a blank ``human_label`` (not yet labeled) are excluded from the
agreement computation and reported separately, so a partially-labeled CSV
still produces a meaningful (if partial) result rather than an error.

Usage
-----
    uv run python scripts/compute_relevance_agreement.py [path/to/labels.csv]

    Defaults to ``data/eval/relevance_labels_template.csv`` if no path is given.
"""

import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELS_PATH = _ROOT / "data" / "eval" / "relevance_labels_template.csv"

_HUMAN_LABEL_TO_JUDGE_LEVEL = {
    "relevant": "Perfectly relevant",
    "not_relevant": "Not relevant",
}


def load_rows(csv_path: Path) -> list[dict]:
    """Load labeled rows from the CSV template.

    Args:
        csv_path: Path to a CSV produced by ``sample_relevance_labels.py``.

    Returns:
        A list of row dicts, as parsed by ``csv.DictReader``.
    """
    with csv_path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_agreement(rows: list[dict]) -> dict:
    """Compute exact-match agreement between human labels and the LLM judge.

    Args:
        rows: Output of :func:`load_rows`.

    Returns:
        A dict with ``total_rows``, ``labeled_rows``, ``unlabeled_rows``,
        ``agreements``, ``disagreements``, and ``agreement_rate`` (``None``
        if there are zero labeled rows).
    """
    labeled = [r for r in rows if (r.get("human_label") or "").strip()]
    unlabeled = len(rows) - len(labeled)

    agreements = 0
    disagreements = 0
    unrecognized_labels: list[str] = []
    for row in labeled:
        human_label = row["human_label"].strip().lower()
        expected_level = _HUMAN_LABEL_TO_JUDGE_LEVEL.get(human_label)
        if expected_level is None:
            unrecognized_labels.append(human_label)
            continue
        if row.get("llm_judge_relevance_level") == expected_level:
            agreements += 1
        else:
            disagreements += 1

    scored = agreements + disagreements
    agreement_rate = (agreements / scored) if scored else None

    return {
        "total_rows": len(rows),
        "labeled_rows": len(labeled),
        "unlabeled_rows": unlabeled,
        "unrecognized_labels": unrecognized_labels,
        "agreements": agreements,
        "disagreements": disagreements,
        "agreement_rate": agreement_rate,
    }


def format_report(result: dict) -> str:
    """Render the agreement result as a short human-readable report.

    Args:
        result: Output of :func:`compute_agreement`.

    Returns:
        Multi-line report text.
    """
    lines = [
        f"Total rows: {result['total_rows']}",
        f"Labeled: {result['labeled_rows']} | Unlabeled (skipped): {result['unlabeled_rows']}",
    ]
    if result["unrecognized_labels"]:
        lines.append(
            f"Unrecognized human_label values (expected 'relevant'/'not_relevant'): "
            f"{sorted(set(result['unrecognized_labels']))}"
        )
    if result["agreement_rate"] is None:
        lines.append(
            "No scoreable labeled rows yet — fill in the 'human_label' column "
            "(see data/eval/README.md) before re-running this script."
        )
    else:
        lines.append(
            f"Agreement: {result['agreements']}/{result['agreements'] + result['disagreements']} "
            f"({result['agreement_rate']:.1%})"
        )
        lines.append(
            "Caveat: single-labeler dataset, no inter-rater reliability check yet "
            "(see data/eval/README.md Limitations)."
        )
    return "\n".join(lines)


def main() -> None:
    """Load the labels CSV (from argv[1] or the default path) and print the agreement report."""
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LABELS_PATH
    if not csv_path.exists():
        print(f"Labels file not found: {csv_path}")
        print("Run scripts/sample_relevance_labels.py first.")
        raise SystemExit(1)

    rows = load_rows(csv_path)
    result = compute_agreement(rows)
    print(format_report(result))


if __name__ == "__main__":
    main()
