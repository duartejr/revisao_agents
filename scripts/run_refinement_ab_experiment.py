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
]


def _build_initial_state(theme: str) -> dict:
    """Build the minimal ReviewState dict expected by the academic workflow entry point.

    Args:
        theme: The review theme to plan for.

    Returns:
        A state dict matching the fields ``revisao_agents.cli.run_planning`` initializes.
    """
    return {
        "theme": theme,
        "review_type": "technical",
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


async def run_single_arm(theme: str, clarification: str | None, bypass: bool, run_id: str) -> dict:
    """Run one (theme, arm) pair through the academic planning workflow to completion.

    Args:
        theme: The (intentionally vague) review theme.
        clarification: Answer to supply if the refinement layer pauses for
            clarification. Ignored when ``bypass`` is True.
        bypass: If True, sets ``BYPASS_REFINEMENT_LAYER`` for the duration of
            this run so the refinement layer is skipped.
        run_id: Unique thread id for this graph execution.

    Returns:
        A dict with the measured metrics and metadata for this run.
    """
    if bypass:
        os.environ[BYPASS_REFINEMENT_ENV_VAR] = "true"
    else:
        os.environ.pop(BYPASS_REFINEMENT_ENV_VAR, None)

    clarification_used = False
    try:
        graph = build_review_graph(review_type="technical")
        config = {"configurable": {"thread_id": run_id}}
        state_init = _build_initial_state(theme)

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


async def main_async() -> None:
    """Run the refinement A/B experiment for all test cases and both arms, logging to MLflow."""
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

                result = await run_single_arm(
                    theme=case["vague_theme"],
                    clarification=case["clarification"],
                    bypass=bypass,
                    run_id=run_id,
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


def main() -> None:
    """Entry point for the script."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
