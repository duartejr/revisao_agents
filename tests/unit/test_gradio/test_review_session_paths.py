"""Path validation in ``start_review_session``.

The review dropdown lists files under ``REVIEWS_DIR`` (default
``./runtime/reviews``), so session start must accept those paths.
"""

from gradio_app.handlers.review import start_review_session
from gradio_app.handlers.review_parts import document

DOC = "# Title\n\n## Intro\n\nFirst paragraph.\n"


def test_accepts_file_inside_configured_reviews_dir(tmp_path, monkeypatch):
    reviews = tmp_path / "runtime" / "reviews"
    reviews.mkdir(parents=True)
    (reviews / "doc.md").write_text(DOC, encoding="utf-8")
    monkeypatch.setattr(document, "REVIEWS_DIR", str(reviews))

    _, state, status, _ = start_review_session(str(reviews / "doc.md"), [], {})

    assert state["original_file_path"].endswith("doc.md") and "✅" in status


def test_accepts_legacy_relative_reviews_prefix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reviews").mkdir()
    (tmp_path / "reviews" / "doc.md").write_text(DOC, encoding="utf-8")

    _, state, status, _ = start_review_session("reviews/doc.md", [], {})

    assert state["original_file_path"] == "reviews/doc.md" and "✅" in status


def test_rejects_file_outside_reviews_dir(tmp_path, monkeypatch):
    reviews = tmp_path / "reviews_dir"
    reviews.mkdir()
    outside = tmp_path / "secret.md"
    outside.write_text("x", encoding="utf-8")
    monkeypatch.setattr(document, "REVIEWS_DIR", str(reviews))

    _, state, status, _ = start_review_session(str(outside), [], {})

    assert state == {} and "❌" in status


def test_rejects_directories(tmp_path, monkeypatch):
    reviews = tmp_path / "reviews_dir"
    reviews.mkdir()
    monkeypatch.setattr(document, "REVIEWS_DIR", str(reviews))

    assert document._is_allowed_review_path(str(reviews)) is False
