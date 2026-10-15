"""Unit tests for Journal Reviewer Mode: schema, prompt, context builder, agent, renderer."""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from revisao_agents.agents import journal_reviewer_agent as agent
from revisao_agents.config import LLMInvocationError
from revisao_agents.core.schemas.journal_review import (
    CriterionAssessment,
    JournalReviewReport,
)
from revisao_agents.utils.llm_utils.prompt_loader import load_prompt

LLM_CALL = "revisao_agents.agents.journal_reviewer_agent.llm_call"


def _criterion(score: int = 6) -> CriterionAssessment:
    return CriterionAssessment(
        score=score,
        strengths=["Clear structure in Introduction"],
        weaknesses=["No selection criteria"],
        suggestions=["State the search protocol"],
    )


def make_report(recommendation: str = "major_revision") -> JournalReviewReport:
    return JournalReviewReport(
        methodology=_criterion(5),
        contribution=_criterion(6),
        novelty=_criterion(4),
        writing_quality=_criterion(8),
        overall_recommendation=recommendation,  # type: ignore[arg-type]
        summary="Solid draft that needs a clearer method.",
    )


def _doc(*sections: tuple[str, str], preamble: str = "") -> tuple[str, list[dict]]:
    """Build a markdown doc and sections shaped like ``_split_sections`` output."""
    parts = [preamble] if preamble else []
    secs = []
    for title, body in sections:
        text = f"## {title}\n\n{body}\n\n"
        parts.append(text)
        secs.append({"title": title, "text": text})
    return "".join(parts), secs


class TestSchema:
    @pytest.mark.parametrize("bad", [0, 11, -3])
    def test_rejects_scores_out_of_range(self, bad):
        with pytest.raises(ValidationError):
            CriterionAssessment(score=bad)

    @pytest.mark.parametrize("ok", [1, 10])
    def test_accepts_boundary_scores(self, ok):
        assert CriterionAssessment(score=ok).score == ok

    def test_rejects_unknown_recommendation(self):
        data = make_report().model_dump()
        data["overall_recommendation"] = "maybe"
        with pytest.raises(ValidationError):
            JournalReviewReport.model_validate(data)

    def test_round_trips_through_model_dump(self):
        report = make_report()
        assert JournalReviewReport.model_validate(report.model_dump()) == report


class TestPrompt:
    def test_journal_reviewer_prompt_loads_and_renders(self):
        prompt = load_prompt("common/journal_reviewer", document="DOC-BODY", language="en")
        assert "DOC-BODY" in prompt.text
        for criterion in ("methodology", "contribution", "novelty", "writing_quality"):
            assert criterion in prompt.text
        assert "{" not in prompt.text.replace("{{", "")  # no unrendered placeholders


class TestBuildReviewContext:
    def test_within_budget_is_untouched(self):
        md, secs = _doc(("A", "x" * 50))
        ctx = agent.build_review_context(md, secs, max_chars=10_000)
        assert ctx.text == md and not ctx.truncated and ctx.omitted_titles == []

    def test_clips_at_section_boundary_and_names_omitted_sections(self):
        md, secs = _doc(("Intro", "a" * 300), ("Methods", "b" * 300), ("Conclusion", "c" * 300))
        ctx = agent.build_review_context(md, secs, max_chars=330)
        assert ctx.truncated
        assert "## Intro" in ctx.text and "b" * 10 not in ctx.text
        assert ctx.omitted_titles == ["Methods", "Conclusion"]
        assert "TRUNCATED" in ctx.text and "Methods, Conclusion" in ctx.text

    def test_keeps_preamble_before_first_section(self):
        md, secs = _doc(
            ("Intro", "a" * 200), ("Methods", "b" * 400), preamble="# Title\n\nAbstract.\n\n"
        )
        ctx = agent.build_review_context(md, secs, max_chars=250)
        assert ctx.text.startswith("# Title") and "## Intro" in ctx.text
        assert ctx.omitted_titles == ["Methods"]

    def test_first_section_larger_than_budget_is_clipped_inside(self):
        md, secs = _doc(("Intro", "word " * 400), ("Methods", "b" * 100))
        ctx = agent.build_review_context(md, secs, max_chars=300)
        assert ctx.truncated and "TRUNCATED" in ctx.text
        assert len(ctx.text) < 300 + 200  # clipped text plus marker
        assert ctx.omitted_titles == ["Methods"]

    def test_document_without_sections_is_clipped_as_plain_text(self):
        md = "paragraph one.\n\n" + "z" * 500
        ctx = agent.build_review_context(md, [], max_chars=100)
        assert ctx.truncated and "TRUNCATED" in ctx.text and len(ctx.text) < 300


class TestRunJournalReview:
    def test_empty_markdown_does_not_call_llm(self):
        with patch(LLM_CALL) as llm:
            result = agent.run_journal_review("   \n ", "en")
        assert isinstance(result, str) and "empty" in result
        llm.assert_not_called()

    def test_llm_invocation_error_becomes_string(self):
        with patch(LLM_CALL, side_effect=LLMInvocationError("boom")):
            result = agent.run_journal_review("# Doc\n\n## A\n\ntext", "pt")
        assert isinstance(result, str) and "boom" in result

    def test_none_result_becomes_string(self):
        with patch(LLM_CALL, return_value=None):
            result = agent.run_journal_review("## A\n\ntext", "en")
        assert isinstance(result, str) and "valid report" in result

    def test_dict_result_is_validated(self):
        with patch(LLM_CALL, return_value=make_report().model_dump()):
            result = agent.run_journal_review("## A\n\ntext", "en")
        assert isinstance(result, JournalReviewReport)

    def test_invalid_dict_becomes_string(self):
        bad = make_report().model_dump()
        bad["methodology"]["score"] = 99
        with patch(LLM_CALL, return_value=bad):
            result = agent.run_journal_review("## A\n\ntext", "en")
        assert isinstance(result, str)

    def test_prompt_contains_truncation_marker_for_long_documents(self):
        md, secs = _doc(("Intro", "a" * 300), ("Methods", "b" * 300))
        with (
            patch.object(agent, "JOURNAL_REVIEW_MAX_CHARS", 330),
            patch(LLM_CALL, return_value=make_report()) as llm,
        ):
            result = agent.run_journal_review(md, "en", sections=secs)
        assert isinstance(result, JournalReviewReport)
        sent_prompt = llm.call_args.args[0]
        assert "TRUNCATED" in sent_prompt and "omitted sections: Methods" in sent_prompt
        assert llm.call_args.kwargs["response_schema"] is JournalReviewReport

    def test_mlflow_failure_does_not_break_report(self):
        with (
            patch(LLM_CALL, return_value=make_report()),
            patch("mlflow.start_run", side_effect=RuntimeError("mlflow down")),
        ):
            result = agent.run_journal_review("## A\n\ntext", "en", review_file="r.md")
        assert isinstance(result, JournalReviewReport)

    def test_mlflow_run_exit_failure_does_not_break_report(self):
        with (
            patch(LLM_CALL, return_value=make_report()),
            patch("mlflow.log_params", side_effect=RuntimeError("log failed")),
        ):
            result = agent.run_journal_review("## A\n\ntext", "en", review_file="r.md")
        assert isinstance(result, JournalReviewReport)

    def test_logs_four_scores_and_params_to_journal_review_experiment(self):
        import mlflow

        with patch(LLM_CALL, return_value=make_report()):
            agent.run_journal_review("## A\n\ntext", "en", review_file="runtime/reviews/x.md")
        experiment = mlflow.get_experiment_by_name("journal_review")
        assert experiment is not None
        runs = mlflow.search_runs([experiment.experiment_id], output_format="list")
        run = next(r for r in runs if r.data.params.get("review_file") == "x.md")
        assert run.data.metrics == {
            "methodology_score": 5.0,
            "contribution_score": 6.0,
            "novelty_score": 4.0,
            "writing_quality_score": 8.0,
        }
        assert run.data.params["overall_recommendation"] == "major_revision"
        assert run.data.params["truncated"] == "False"


class TestReportToMarkdown:
    @pytest.mark.parametrize(
        ("language", "labels", "recommendation", "disclaimer"),
        [
            (
                "pt",
                ["Metodologia", "Contribuição", "Novidade", "Qualidade da escrita"],
                "Revisão maior",
                "não calibrados",
            ),
            (
                "en",
                ["Methodology", "Contribution", "Novelty", "Writing quality"],
                "Major revision",
                "not calibrated",
            ),
        ],
    )
    def test_contains_criteria_scores_recommendation_and_disclaimer(
        self, language, labels, recommendation, disclaimer
    ):
        md = agent.report_to_markdown(make_report(), language)
        for label, score in zip(labels, (5, 6, 4, 8), strict=True):
            assert f"{label} — {score}/10" in md
        assert recommendation in md and disclaimer in md
        assert "No selection criteria" in md and "State the search protocol" in md


class TestBuildReviewContextPartialSection:
    def test_oversized_middle_section_is_kept_in_part(self):
        md, secs = _doc(("Intro", "short"), ("Big", "word " * 400), ("Concl", "end"))
        ctx = agent.build_review_context(md, secs, max_chars=1000)
        assert ctx.partial_title == "Big"
        assert ctx.omitted_titles == ["Concl"]
        assert "word word" in ctx.text and "partially kept: Big" in ctx.text

    def test_oversized_first_section_is_reported_as_partial(self):
        md, secs = _doc(("Big", "word " * 1000), ("Small", "x" * 50))
        ctx = agent.build_review_context(md, secs, max_chars=1000)
        assert ctx.partial_title == "Big" and ctx.omitted_titles == ["Small"]
        assert "partially kept: Big" in ctx.text

    def test_tiny_remaining_budget_omits_instead_of_clipping(self):
        md, secs = _doc(("Intro", "a" * 280), ("Big", "b" * 500))
        ctx = agent.build_review_context(md, secs, max_chars=320)
        assert ctx.partial_title == "" and ctx.omitted_titles == ["Big"]

    @pytest.mark.parametrize("bad", [0, 1, 3, -5])
    def test_non_positive_budget_never_crashes(self, bad):
        md, secs = _doc(("Intro", "a" * 50))
        ctx = agent.build_review_context(md, secs, max_chars=bad)
        assert ctx.truncated and "TRUNCATED" in ctx.text


class TestEnvInt:
    def test_unset_returns_default(self, monkeypatch):
        from revisao_agents.config import _env_int

        monkeypatch.delenv("X_INT", raising=False)
        assert _env_int("X_INT", 7) == 7

    def test_parses_value(self, monkeypatch):
        from revisao_agents.config import _env_int

        monkeypatch.setenv("X_INT", " 42 ")
        assert _env_int("X_INT", 7) == 42

    def test_rejects_non_integer(self, monkeypatch):
        from revisao_agents.config import _env_int

        monkeypatch.setenv("X_INT", "abc")
        with pytest.raises(ValueError, match="not an integer"):
            _env_int("X_INT", 7)

    def test_rejects_value_below_minimum(self, monkeypatch):
        from revisao_agents.config import _env_int

        monkeypatch.setenv("X_INT", "10")
        with pytest.raises(ValueError, match="minimum"):
            _env_int("X_INT", 7, minimum=1000)
