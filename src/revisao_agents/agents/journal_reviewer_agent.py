# src/revisao_agents/agents/journal_reviewer_agent.py
"""Journal Reviewer Agent.

Produces a structured, rubric-scored referee report (:class:`JournalReviewReport`)
for a generated literature review. Report generation is *toolless*: a single
structured-output ``llm_call`` over the document, with no retrieval.

Like the sibling agents, :func:`run_journal_review` never raises: failures are
returned as human-readable error strings.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import mlflow
from pydantic import ValidationError

from ..config import JOURNAL_REVIEW_MAX_CHARS, LLMInvocationError, llm_call
from ..core.schemas.journal_review import CRITERIA, JournalReviewReport
from ..observability.mlflow_config import EXP_JOURNAL_REVIEW, get_tracking_uri
from ..utils.llm_utils.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

PROMPT_PATH = "common/journal_reviewer"

#: Every rendered report starts with this prefix, so callers can recognise report bubbles.
REPORT_HEADING_PREFIX = "## 🧑‍⚖️"

_CRITERION_LABELS = {
    "pt": {
        "methodology": "Metodologia",
        "contribution": "Contribuição",
        "novelty": "Novidade",
        "writing_quality": "Qualidade da escrita",
    },
    "en": {
        "methodology": "Methodology",
        "contribution": "Contribution",
        "novelty": "Novelty",
        "writing_quality": "Writing quality",
    },
}

_RECOMMENDATION_LABELS = {
    "pt": {
        "accept": "Aceitar",
        "minor_revision": "Revisão menor",
        "major_revision": "Revisão maior",
        "reject": "Rejeitar",
    },
    "en": {
        "accept": "Accept",
        "minor_revision": "Minor revision",
        "major_revision": "Major revision",
        "reject": "Reject",
    },
}


@dataclass
class ReviewContext:
    """Document text prepared for the referee prompt.

    Attributes:
        text: The (possibly clipped) document, including a truncation marker
            when content was omitted.
        truncated: ``True`` when part of the document was left out.
        omitted_titles: Titles of the sections that were left out entirely.
        partial_title: Title of the section that was kept only in part, if any.
    """

    text: str
    truncated: bool = False
    omitted_titles: list[str] = field(default_factory=list)
    partial_title: str = ""


def _clip_at_paragraph(text: str, limit: int) -> str:
    """Clip ``text`` to ``limit`` characters, preferring a paragraph boundary.

    Args:
        text: Text to clip.
        limit: Maximum number of characters to keep.

    Returns:
        ``text`` unchanged when it fits, otherwise a prefix of at most ``limit``
        characters cut at the last paragraph break in the second half, if any.
    """
    if len(text) <= limit:
        return text
    clipped = text[:limit]
    cut = clipped.rfind("\n\n")
    return clipped[:cut] if cut > limit // 2 else clipped


def build_review_context(
    markdown: str,
    sections: list[dict] | None = None,
    max_chars: int | None = None,
) -> ReviewContext:
    """Build the document context for the referee prompt within a character budget.

    Documents within budget are passed through untouched. Longer documents are
    clipped: the preamble (everything before the first ``##`` heading) and as
    many leading sections as fit are kept. The first section that does not fit
    is kept in part (clipped at a paragraph boundary) when at least 20% of the
    budget is still free, so one large section cannot erase the whole body; the
    remaining sections are omitted. An explicit marker names the partially kept
    and omitted sections so the reviewer knows it is seeing a partial document.
    A first piece larger than the budget is clipped inside; a document with no
    sections is clipped as plain text.

    Args:
        markdown: Full markdown of the review document.
        sections: Pre-split sections as produced by the handler's
            ``_split_sections`` (dicts with ``title`` and ``text``). Passed in
            to avoid a ``revisao_agents`` → ``gradio_app`` import.
        max_chars: Character budget. Defaults to ``JOURNAL_REVIEW_MAX_CHARS``;
            values below 1 are treated as 1. The truncation marker is appended
            on top of the budget.

    Returns:
        A :class:`ReviewContext` with the prepared text and truncation info.
    """
    limit = max(max_chars if max_chars is not None else JOURNAL_REVIEW_MAX_CHARS, 1)
    if len(markdown) <= limit:
        return ReviewContext(text=markdown)

    pieces: list[tuple[str, str]] = []  # (title, text)
    if sections:
        first_offset = markdown.find(sections[0].get("text", ""))
        preamble = markdown[:first_offset] if first_offset > 0 else ""
        if preamble.strip():
            pieces.append(("(preamble)", preamble))
        pieces.extend((str(s.get("title", "")), str(s.get("text", ""))) for s in sections)

    kept: list[str] = []
    omitted: list[str] = []
    partial = ""
    used = 0
    if not pieces:
        kept.append(_clip_at_paragraph(markdown, limit))
    else:
        for idx, (title, text) in enumerate(pieces):
            if idx == 0 and len(text) > limit:
                kept.append(_clip_at_paragraph(text, limit))
                partial = title
                omitted.extend(t for t, _ in pieces[1:])
                break
            if not omitted and not partial and used + len(text) <= limit:
                kept.append(text)
                used += len(text)
            elif not omitted and not partial and limit - used >= max(limit // 5, 1):
                kept.append(_clip_at_paragraph(text, limit - used))
                partial = title
            else:
                omitted.append(title)

    marker = "[... DOCUMENT TRUNCATED to fit the context budget"
    if partial:
        marker += f"; partially kept: {partial}"
    if omitted:
        marker += "; omitted sections: " + ", ".join(omitted)
    marker += " ...]"
    return ReviewContext(
        text="".join(kept).rstrip() + "\n\n" + marker,
        truncated=True,
        omitted_titles=omitted,
        partial_title=partial,
    )


def report_to_markdown(report: JournalReviewReport, language: str = "pt") -> str:
    """Render a referee report as markdown for the chat bubble or an export.

    This is the single rendering path for the report.

    Args:
        report: The structured report to render.
        language: ``"pt"`` or ``"en"``; anything else is rendered as English.

    Returns:
        Markdown with the recommendation, the four criteria (score, strengths,
        weaknesses, suggestions), the summary and an "LLM-generated, not
        calibrated" disclaimer.
    """
    lang = "pt" if language == "pt" else "en"
    labels = _CRITERION_LABELS[lang]
    pt = lang == "pt"
    heading = "Relatório de revisão por pares" if pt else "Referee report"
    rec_label = "Recomendação geral" if pt else "Overall recommendation"
    sections_titles = (
        ("Pontos fortes", "Pontos fracos", "Sugestões")
        if pt
        else ("Strengths", "Weaknesses", "Suggestions")
    )
    summary_label = "Resumo" if pt else "Summary"

    lines = [
        f"{REPORT_HEADING_PREFIX} {heading}",
        "",
        f"**{rec_label}:** {_RECOMMENDATION_LABELS[lang][report.overall_recommendation]}",
        "",
    ]
    for key in CRITERIA:
        item = getattr(report, key)
        lines.append(f"### {labels[key]} — {item.score}/10")
        for title, values in zip(
            sections_titles, (item.strengths, item.weaknesses, item.suggestions), strict=True
        ):
            if values:
                lines.append(f"**{title}**")
                lines.extend(f"- {value}" for value in values)
        lines.append("")
    lines += [f"### {summary_label}", report.summary, ""]
    lines.append(
        "_⚠️ Avaliação gerada por LLM; as notas são julgamentos relativos, não calibrados._"
        if pt
        else "_⚠️ LLM-generated assessment; scores are relative judgments, not calibrated._"
    )
    return "\n".join(lines)


def _coerce_report(result: Any) -> JournalReviewReport | None:
    """Return ``result`` as a :class:`JournalReviewReport`, or ``None`` if invalid.

    Args:
        result: Raw ``llm_call`` output: a report instance, a dict to validate, or anything else.

    Returns:
        The report, or ``None`` when ``result`` is neither a report nor a valid dict.
    """
    if isinstance(result, JournalReviewReport):
        return result
    if isinstance(result, dict):
        try:
            return JournalReviewReport.model_validate(result)
        except ValidationError:
            return None
    return None


def activate_journal_review_experiment() -> None:
    """Select the ``journal_review`` experiment so the trace lands there.

    MLflow attaches a trace to the experiment active when its span opens, so
    this must run *before* the traced function is entered. Failures are logged,
    never raised.
    """
    try:
        mlflow.set_tracking_uri(get_tracking_uri())
        mlflow.set_experiment(EXP_JOURNAL_REVIEW)
    except Exception as exc:
        logger.warning("Could not activate MLflow experiment %s: %s", EXP_JOURNAL_REVIEW, exc)


def _log_report_to_mlflow(
    report: JournalReviewReport,
    review_file: str,
    truncated: bool,
    prompt_version: str,
) -> None:
    """Log the rubric scores and run parameters of one generation to MLflow.

    The whole block, including run entry and exit, is guarded so an MLflow
    failure can never prevent the report from being returned.

    Args:
        report: The generated report; its four scores are logged as metrics.
        review_file: Path of the reviewed file (only its basename is logged).
        truncated: Whether the document was clipped to fit the budget.
        prompt_version: Version string of the referee prompt template.
    """
    try:
        with mlflow.start_run(
            run_name=f"journal_review/{os.path.basename(review_file)[:40] or 'document'}",
            nested=mlflow.active_run() is not None,
        ):
            mlflow.log_metrics({f"{key}_score": getattr(report, key).score for key in CRITERIA})
            mlflow.log_params(
                {
                    "overall_recommendation": report.overall_recommendation,
                    "review_file": os.path.basename(review_file) or "unknown",
                    "provider": os.getenv("LLM_PROVIDER", "openai"),
                    "model": os.getenv("LLM_MODEL", "") or "<default>",
                    "prompt_version": prompt_version,
                    "truncated": str(truncated),
                }
            )
    except Exception as exc:
        logger.warning("Could not log journal review to MLflow: %s", exc)


@mlflow.trace(name="journal_review", span_type="AGENT")
def _generate_report(
    markdown: str,
    language: str,
    sections: list[dict] | None,
    review_file: str,
) -> JournalReviewReport | str:
    """Traced body of :func:`run_journal_review`.

    Args:
        markdown: Full markdown of the review document.
        language: ``"pt"`` or ``"en"``.
        sections: Optional pre-split sections for context budgeting.
        review_file: Path of the reviewed file, for MLflow logging only.

    Returns:
        The validated report, or a localized error string when the prompt
        cannot be loaded, the LLM call fails or the output is invalid.
    """
    pt = language == "pt"

    context = build_review_context(markdown, sections)
    try:
        prompt = load_prompt(PROMPT_PATH, document=context.text, language=language)
    except Exception as exc:
        logger.error("Failed to load journal_reviewer prompt: %s", exc)
        return (
            f"⚠️ Erro ao carregar o prompt do revisor: {exc}"
            if pt
            else f"⚠️ Could not load the reviewer prompt: {exc}"
        )

    try:
        result = llm_call(
            prompt.text, temperature=prompt.temperature, response_schema=JournalReviewReport
        )
    except LLMInvocationError as exc:
        logger.error("Journal review LLM call failed: %s", exc)
        return (
            f"⚠️ Não foi possível gerar o relatório do revisor: {exc}"
            if pt
            else f"⚠️ Could not generate the referee report: {exc}"
        )

    report = _coerce_report(result)
    if report is None:
        return (
            "⚠️ O modelo não retornou um relatório válido. Tente novamente."
            if pt
            else "⚠️ The model did not return a valid report. Please try again."
        )

    _log_report_to_mlflow(
        report, review_file, context.truncated, str(prompt.metadata.get("version", ""))
    )
    return report


def run_journal_review(
    markdown: str,
    language: str = "pt",
    *,
    sections: list[dict] | None = None,
    review_file: str = "",
) -> JournalReviewReport | str:
    """Generate a structured referee report for a review document.

    The call is traced to the ``journal_review`` MLflow experiment and the four
    rubric scores are logged as metrics. MLflow problems never affect the result.

    Args:
        markdown: Full markdown of the review document.
        language: ``"pt"`` or ``"en"``; language of the report and of error messages.
        sections: Optional pre-split sections (see :func:`build_review_context`).
        review_file: Path of the reviewed file, used only as an MLflow parameter.

    Returns:
        A :class:`JournalReviewReport` on success, or a human-readable error
        string on failure (empty document, prompt/LLM failure, invalid output).
        This function does not raise.

    Example:
        >>> result = run_journal_review(markdown_text, "en")  # doctest: +SKIP
        >>> if isinstance(result, str):  # doctest: +SKIP
        ...     print(result)  # human-readable error
        ... else:
        ...     print(report_to_markdown(result, "en"))
    """
    if not markdown or not markdown.strip():
        return (
            "⚠️ O documento está vazio; não há o que revisar."
            if language == "pt"
            else "⚠️ The document is empty; there is nothing to review."
        )

    activate_journal_review_experiment()
    try:
        return _generate_report(markdown, language, sections, review_file)
    except Exception as exc:  # defensive: honour the no-raise contract
        logger.error("Journal review failed unexpectedly: %s", exc)
        return (
            f"⚠️ Erro inesperado no revisor: {exc}"
            if language == "pt"
            else f"⚠️ Unexpected reviewer error: {exc}"
        )
