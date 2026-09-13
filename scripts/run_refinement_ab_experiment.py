"""
run_refinement_ab_experiment.py - Planning Efficiency A/B Experiment
(Language & Ambiguity Refinement Layer, Phase D / Week 5)

Design:
    N intentionally vague test themes (min 3), each run through the full
    technical planning workflow (``build_review_graph(review_type="technical")``,
    in-process MemorySaver, ``max_questions=1``) under two arms:

    Uses the *technical* workflow rather than academic: both share the exact
    same entry point under test (``identify_and_refine_node`` /
    ``refinement_router``), but academic planning sources from the MongoDB
    corpus (``vector_search_node``) while technical planning sources from
    Tavily (``initial_technical_search_node``). At the time this script was
    authored, the project's MongoDB Atlas cluster was unreachable (DNS SRV
    lookup for the cluster hostname returned NXDOMAIN — an infrastructure
    state outside this script's control, not a code issue), so the technical
    workflow was used to keep the experiment runnable against live services.
    The choice of workflow does not change what is being measured: the
    refinement layer itself is workflow-agnostic.

    - "refined": the refinement layer (``identify_and_refine_node``) runs
      normally. If it flags the theme as vague/ambiguous and pauses for
      clarification, the pause is answered with a theme-specific
      clarification string that supplies the missing specificity/language —
      mirroring how a real user would respond.
    - "bypassed": the ``BYPASS_REFINEMENT_LAYER`` environment variable is set
      for the duration of the run, which makes ``identify_and_refine_node``
      short-circuit to a cheap heuristic language guess
      (:func:`revisao_agents.core.utils.detect_language`) and skip
      clarification entirely — the vague theme flows straight into the
      interview loop unchanged.

    Any *other* HITL pause encountered (the normal plan-refinement interview,
    unrelated to the language/ambiguity layer) is answered with a generic
    "Keep the current plan." response, matching the batch-mode pattern
    already used by ``revisao_agents.cli.run_planning``
    (``--auto-response``, see CLAUDE.md "HITL pauses").

    Results are logged to a dedicated MLflow experiment,
    ``planning_refinement_ab``, one run per (theme, arm) pair.

Metrics logged per run:
    - session_duration (float, seconds)
    - refinement_rounds (float, count of ``identify_and_refine`` node executions)
    - plan_section_count (float, count of markdown ``##`` headers in final_plan)
    - plan_quality_score (float, 1-10 LLM-as-judge score; 0 if no plan was produced)

Params logged per run (categorical, not numeric — see Note below):
    - theme, arm, clarification_used, language_detected

Note on ``language_detected``:
    The story text lists ``language_detected`` alongside the numeric metrics,
    but MLflow's tracking API only accepts numeric values for
    ``mlflow.log_metric`` (see https://mlflow.org/docs/latest/tracking/ —
    "Metrics" are float/int time series). Since PT/EN/UNKNOWN is categorical,
    it is logged via ``mlflow.log_param`` instead so the value is preserved
    and visible in the MLflow UI without violating the tracking API contract.

Limitations (documented per W9-STORY-03 acceptance criteria):
    - "bypassed" is a synthetic short-circuit, not a real user choosing to
      skip clarification: the node never even inspects the theme's
      vagueness, so this measures the layer's *presence vs. absence*, not a
      genuine user opt-out.
    - Runs against the technical (Tavily-backed) workflow, not academic
      (MongoDB-backed) — see the Design note above. Tavily search result
      variance is not controlled for by this script.
    - Small sample (>= 3 themes x 2 arms = 6 runs): directional signal only.
      Statistical significance testing is deferred to Week 10 per the roadmap.
    - The LLM in the "refined" arm may not always flag a theme as vague
      (model judgment varies run to run); when it doesn't, both arms take
      the same path and ``refinement_rounds`` is 1 for both.
"""

import asyncio
import os
import re
import sys
import time
from pathlib import Path

import mlflow
import typer

# Ensure repo root is on sys.path when running directly with `uv run python`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from revisao_agents.evaluation.evaluators import evaluate_plan_quality  # noqa: E402
from revisao_agents.nodes.common import BYPASS_REFINEMENT_ENV_VAR  # noqa: E402
from revisao_agents.observability.mlflow_config import (  # noqa: E402
    EXP_PLANNING_REFINEMENT_AB,
    get_tracking_uri,
)
from revisao_agents.workflows import build_review_graph  # noqa: E402

MAX_HITL_ROUNDS = 6  # Safety cap: abort a run rather than loop forever on unexpected routing.

TEST_CASES: list[dict] = [
    {
        "vague_theme": "inteligência artificial na educação",
        "clarification": (
            "O impacto de tutores baseados em LLM no desempenho de estudantes "
            "do ensino médio em matemática no Brasil."
        ),
    },
    {
        "vague_theme": "machine learning in healthcare",
        "clarification": (
            "The use of gradient-boosted tree models for early sepsis prediction in ICU patients."
        ),
    },
    {
        "vague_theme": "mudanças climáticas",
        "clarification": (
            "Modelos de aprendizado de máquina para previsão de secas no semiárido brasileiro."
        ),
    },
    {
        "vague_theme": "segurança cibernética",
        "clarification": (
            "Técnicas de detecção de intrusão baseadas em aprendizado de máquina para "
            "redes IoT industriais."
        ),
    },
    {
        "vague_theme": "renewable energy storage",
        "clarification": (
            "Lithium-ion battery degradation models for grid-scale solar energy storage systems."
        ),
    },
    {
        "vague_theme": "processamento de linguagem natural",
        "clarification": (
            "Modelos de linguagem de grande porte aplicados à sumarização automática "
            "de laudos médicos em português."
        ),
    },
    {
        "vague_theme": "autonomous vehicles",
        "clarification": (
            "Sensor fusion approaches combining LiDAR and camera data for pedestrian "
            "detection in urban autonomous driving."
        ),
    },
    {
        "vague_theme": "biotecnologia agrícola",
        "clarification": (
            "Edição genética CRISPR aplicada ao desenvolvimento de variedades de soja "
            "resistentes à seca no Cerrado brasileiro."
        ),
    },
]


def _build_initial_state(theme: str, workflow_type: str = "technical") -> dict:
    """Build the minimal ReviewState dict expected by the planning workflow entry point.

    Args:
        theme: The review theme to plan for.
        workflow_type: ``"technical"`` (Tavily-backed) or ``"academic"``
            (MongoDB-backed). Both graphs share the same generic state fields;
            the technical-only fields are harmless no-ops for the academic
            graph since LangGraph state updates aren't strictly validated
            against unused keys.

    Returns:
        A state dict matching the fields ``revisao_agents.cli.run_planning`` initializes.
    """
    return {
        "theme": theme,
        "review_type": workflow_type,
        "relevant_chunks": [],
        "technical_snippets": [],
        "technical_urls": [],
        "current_plan": "",
        "interview_history": [],
        "questions_asked": 0,
        "max_questions": 1,
        "final_plan": "",
        "final_plan_path": "",
        "status": "starting",
    }


async def run_single_arm(
    theme: str,
    clarification: str | None,
    bypass: bool,
    run_id: str,
    workflow_type: str = "technical",
) -> dict:
    """Run one (theme, arm) pair through the planning workflow to completion.

    Args:
        theme: The (intentionally vague) review theme.
        clarification: Answer to supply if the refinement layer pauses for
            clarification. Ignored when ``bypass`` is True.
        bypass: If True, sets ``BYPASS_REFINEMENT_LAYER`` for the duration of
            this run so the refinement layer is skipped.
        run_id: Unique thread id for this graph execution.
        workflow_type: ``"technical"`` (Tavily-backed, default) or
            ``"academic"`` (MongoDB-backed) — see the module docstring for why
            technical is the default while MongoDB Atlas is unreachable.

    Returns:
        A dict with the measured metrics and metadata for this run.
    """
    if bypass:
        os.environ[BYPASS_REFINEMENT_ENV_VAR] = "true"
    else:
        os.environ.pop(BYPASS_REFINEMENT_ENV_VAR, None)

    clarification_used = False
    try:
        graph = build_review_graph(review_type=workflow_type)
        config = {"configurable": {"thread_id": run_id}}
        state_init = _build_initial_state(theme, workflow_type=workflow_type)

        t0 = time.perf_counter()
        refinement_rounds = 0

        for chunk in graph.stream(state_init, config=config):
            refinement_rounds += "identify_and_refine" in chunk

        rounds_of_hitl = 0
        while True:
            current = graph.get_state(config)
            if not current.next:
                break
            if "human_pause" not in current.next:
                print(f"[{run_id}] Unexpected routing: waiting for {current.next}. Stopping run.")
                break

            rounds_of_hitl += 1
            if rounds_of_hitl > MAX_HITL_ROUNDS:
                print(f"[{run_id}] Exceeded {MAX_HITL_ROUNDS} HITL rounds. Aborting run.")
                break

            is_theme_refined = current.values.get("is_theme_refined", False)
            if not bypass and not is_theme_refined and clarification:
                response = clarification
                clarification_used = True
            else:
                response = "Keep the current plan."

            history = current.values.get("interview_history", [])
            graph.update_state(
                config,
                {"interview_history": history + [("user", response)]},
                as_node="human_pause",
            )
            for chunk in graph.stream(None, config=config):
                refinement_rounds += "identify_and_refine" in chunk

        session_duration = time.perf_counter() - t0
        final_state = graph.get_state(config).values
        final_plan = final_state.get("final_plan", "")
        final_theme = final_state.get("theme", theme)

        # Judge against `final_theme`, not the original (possibly stale) `theme`:
        # in the "refined" arm, identify_and_refine_node may have rewritten the
        # theme via the clarification round, and the plan was written for that
        # rewritten theme, not the original vague one.
        if final_plan:
            try:
                plan_quality_score = await evaluate_plan_quality(final_theme, final_plan)
            except Exception as exc:
                print(f"[{run_id}] evaluate_plan_quality failed, scoring as 0.0: {exc}")
                plan_quality_score = 0.0
        else:
            plan_quality_score = 0.0
        plan_section_count = len(re.findall(r"^##\s", final_plan, re.MULTILINE))

        return {
            "theme": theme,
            "final_theme": final_theme,
            "arm": "bypassed" if bypass else "refined",
            "workflow_type": workflow_type,
            "clarification_used": clarification_used,
            "language_detected": final_state.get("detected_language", "UNKNOWN"),
            "session_duration": session_duration,
            "refinement_rounds": refinement_rounds,
            "plan_section_count": plan_section_count,
            "plan_quality_score": plan_quality_score,
            "final_plan_generated": bool(final_plan),
        }
    finally:
        os.environ.pop(BYPASS_REFINEMENT_ENV_VAR, None)


async def main_async(workflow_type: str = "technical") -> None:
    """Run the refinement A/B experiment for all test cases and both arms, logging to MLflow.

    Args:
        workflow_type: ``"technical"`` (default) or ``"academic"``, matched
            case-insensitively. The academic arm requires a reachable
            MongoDB Atlas cluster — see the module docstring.

    Raises:
        ValueError: If ``workflow_type`` (after stripping/lowercasing) is not
            exactly ``"technical"`` or ``"academic"``. Rejecting anything
            else here — rather than letting an unrecognized value silently
            reach ``build_review_graph``'s own lenient normalization — matters
            because ``nodes.common.interview_node`` does its own *case-sensitive*
            check against ``{"tecnico", "technical"}`` on the raw
            ``state["review_type"]`` value; a value that ``build_review_graph``
            would still resolve to the technical graph (e.g. ``"Technical"``)
            could otherwise silently route the interview step down the
            academic branch instead, corrupting the experiment's own metrics.
    """
    workflow_type = workflow_type.strip().lower()
    if workflow_type not in {"technical", "academic"}:
        raise ValueError(f"workflow_type must be 'technical' or 'academic', got {workflow_type!r}")

    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment(EXP_PLANNING_REFINEMENT_AB)

    for idx, case in enumerate(TEST_CASES):
        for bypass in (False, True):
            arm = "bypassed" if bypass else "refined"
            run_id = f"refinement_ab_{arm}_{idx}"

            with mlflow.start_run(run_name=run_id):
                print(f"Running arm='{arm}' theme={case['vague_theme']!r}")
                mlflow.log_param("theme", case["vague_theme"])
                mlflow.log_param("arm", arm)
                mlflow.log_param("workflow_type", workflow_type)

                result = await run_single_arm(
                    theme=case["vague_theme"],
                    clarification=case["clarification"],
                    bypass=bypass,
                    run_id=run_id,
                    workflow_type=workflow_type,
                )

                mlflow.log_params(
                    {
                        "final_theme": result["final_theme"][:250],
                        "clarification_used": result["clarification_used"],
                        "language_detected": result["language_detected"],
                        "final_plan_generated": result["final_plan_generated"],
                    }
                )
                mlflow.log_metrics(
                    {
                        "session_duration": result["session_duration"],
                        "refinement_rounds": float(result["refinement_rounds"]),
                        "plan_section_count": float(result["plan_section_count"]),
                        "plan_quality_score": result["plan_quality_score"],
                    }
                )
                mlflow.log_dict(result, f"refinement_ab_{arm}_{idx}.json")
                print(
                    f"Completed arm='{arm}' theme={case['vague_theme']!r}: "
                    f"rounds={result['refinement_rounds']} "
                    f"quality={result['plan_quality_score']}"
                )


app = typer.Typer(add_completion=False)


@app.command()
def run(
    workflow_type: str = typer.Option(
        "technical",
        "--workflow-type",
        "-w",
        help=(
            "Which planning workflow to exercise: 'technical' (Tavily-backed, "
            "default — works today) or 'academic' (MongoDB-backed — requires "
            "a reachable MongoDB Atlas cluster)."
        ),
    ),
) -> None:
    """Run the refinement A/B experiment for all test cases and both arms."""
    asyncio.run(main_async(workflow_type=workflow_type))


def main() -> None:
    """Entry point for the script."""
    app()


if __name__ == "__main__":
    main()
