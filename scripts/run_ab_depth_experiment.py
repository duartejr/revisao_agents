"""
run_ab_depth_experiment.py - Tavily Depth A/B Experiment

Design:
    3 MLflow runs (one per depth: fast / basic / advanced)
    Each run aggregates results from all TEST QUERIES
    LLM-as-judge scores computed via evaluation/evaluators.py
    Summary artifact logged as JSON per depth variant
"""

import asyncio
import sys
import time
from pathlib import Path

import mlflow
import typer
from tavily import TavilyClient

# Ensure repo root is on sys.path when running directly with `uv run python`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from revisao_agents.evaluation.evaluators import evaluate_search_snippets  # noqa: E402
from revisao_agents.observability.mlflow_config import (  # noqa: E402
    EXP_AB_DEPTH_EXPERIMENTS,
    get_tracking_uri,
)
from revisao_agents.utils.core.commons import get_clean_key  # noqa: E402

DEPTHS = ["fast", "basic", "advanced"]
MAX_CREDITS_PER_DEPTH = 30

TEST_QUERIES: list[dict] = [
    {"query": "transformer attention mechaninsm self-attention", "type": "academic"},
    {"query": "RAG retrieval augmented evaluation RAGAS metrics", "type": "academic"},
    {"query": "LangGraph state management checkpointing production", "type": "technical"},
    {"query": "MLflow experiment tracking A/B testing machine learning", "type": "technical"},
    {"query": "LLM automated academic review quality evaluation", "type": "mixed"},
]


async def run_single_depth(depth: str, queries: list[dict]) -> dict:
    """
    Run a single depth variant of the A/B experiment.

    Args:
        depth (str): The depth variant to run (fast, basic, advanced).
        queries (list[dict]): List of test queries.

    Returns:
        dict: Aggregated results for the depth variant.
    """
    client = TavilyClient(api_key=get_clean_key("TAVILY_API_KEY"))

    results = []
    total_credits_used = 0

    for query in queries:
        if total_credits_used >= MAX_CREDITS_PER_DEPTH:
            print(f"Reached max credits for depth '{depth}'. Stopping further queries.")
            break

        t0 = time.perf_counter()
        response = client.search(
            query["query"], search_depth=depth, max_results=5, include_usage=True
        )
        latency = time.perf_counter() - t0

        credits = response.get("usage", {}).get("credits", 0)
        total_credits_used += credits
        snippets = [r.get("content", "")[:500] for r in response.get("results", [])]
        urls = [r.get("url", "") for r in response.get("results", [])]

        ## LLM judges
        evaluations = await evaluate_search_snippets(
            query=query["query"],
            snippets=snippets,
            urls=urls,
            interview_metadata={
                "user_goals": query["query"],
                "depth_setting": depth,
            },
        )

        results.append(
            {
                "query": query["query"],
                "type": query["type"],
                "latency": latency,
                "credits": credits,
                "urls_found": len(urls),
                "relevance_score": sum(e.relevance_score for e in evaluations)
                / max(len(evaluations), 1),
                "academic_quality_pct": sum(1 for e in evaluations if e.academic_quality)
                / max(len(evaluations), 1)
                * 100,
                "citation_potential_pct": sum(1 for e in evaluations if e.citation_potential)
                / max(len(evaluations), 1)
                * 100,
            }
        )

    return {
        "depth": depth,
        "queries": results,
        "total_credits": total_credits_used,
    }


async def main_async(runs_per_depth: int = 1, depths: list[str] | None = None):
    """
    Main function to run the A/B depth experiment across the given depth variants.

    Args:
        runs_per_depth (int): Number of repeated executions per depth variant.
            Defaults to 1, preserving the original single-execution behavior
            (in which case the run name has no index suffix).
        depths (list[str] | None): Which depth variants to execute, matched
            case-insensitively (mirroring ``run_refinement_ab_experiment.py``'s
            ``workflow_type`` normalization). Defaults to all of ``DEPTHS``
            (``fast``, ``basic``, ``advanced``). Duplicate entries are
            deduplicated (order-preserving) rather than re-executed, so a
            copy-paste or shell-quoting mistake doesn't silently double the
            credit spend for a depth. Useful to resume a partial run (e.g.
            after an interrupted process) by re-targeting only the depth(s)
            that didn't finish, without re-spending credits on depths that
            already reached their target.

    Raises:
        ValueError: If ``depths`` is an empty list (an empty selection would
            otherwise silently run zero iterations and exit 0 with no
            signal that nothing happened — pass ``None`` or omit the
            argument to run all depths instead), or if it contains a value
            that isn't (case-insensitively) one of ``DEPTHS``.
    """
    raw_depths = depths if depths is not None else DEPTHS
    if not raw_depths:
        raise ValueError(f"depths must not be empty; pass None to run all of {DEPTHS}")

    normalized = [d.strip().lower() for d in raw_depths]
    unknown = [d for d in normalized if d not in DEPTHS]
    if unknown:
        raise ValueError(f"Unknown depth(s) {unknown}; must be a subset of {DEPTHS}")

    selected_depths = list(dict.fromkeys(normalized))  # order-preserving de-duplication

    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment(EXP_AB_DEPTH_EXPERIMENTS)

    for depth in selected_depths:
        for run_index in range(runs_per_depth):
            run_name = f"depth_{depth}" if runs_per_depth == 1 else f"depth_{depth}_run_{run_index}"
            with mlflow.start_run(run_name=run_name):
                print(f"Running depth variant: {depth} (run {run_index + 1}/{runs_per_depth})")
                mlflow.log_param("depth", depth)
                mlflow.log_param("num_queries", len(TEST_QUERIES))
                if runs_per_depth > 1:
                    mlflow.log_param("run_index", run_index)

                aggregated_results = await run_single_depth(depth, TEST_QUERIES)

                # Aggregate metrics for logging
                metrics_list = aggregated_results["queries"]
                n = max(len(metrics_list), 1)
                mlflow.log_metrics(
                    {
                        "average_latency": sum(m["latency"] for m in metrics_list) / n,
                        "total_credits": aggregated_results["total_credits"],
                        "average_urls_found": sum(m["urls_found"] for m in metrics_list) / n,
                        "average_relevance_score": sum(m["relevance_score"] for m in metrics_list)
                        / n,
                        "average_academic_quality_pct": sum(
                            m["academic_quality_pct"] for m in metrics_list
                        )
                        / n,
                        "average_citation_potential_pct": sum(
                            m["citation_potential_pct"] for m in metrics_list
                        )
                        / n,
                    }
                )

                # Log aggregated results as JSON artifact
                mlflow.log_dict(aggregated_results, f"ab_results_{depth}.json")
                print(
                    f"Completed depth variant: {depth}, "
                    f"total credits used: {aggregated_results['total_credits']}"
                )


app = typer.Typer(add_completion=False)


@app.command()
def run(
    runs_per_depth: int = typer.Option(
        1,
        "--runs-per-depth",
        "-n",
        min=1,
        help="Number of repeated executions per depth variant (fast/basic/advanced).",
    ),
    depths: str = typer.Option(
        ",".join(DEPTHS),
        "--depths",
        "-d",
        help=(
            "Comma-separated subset of depth variants to run (default: all of "
            f"{DEPTHS}). Useful to resume a partial/interrupted run without "
            "re-spending credits on depths that already finished."
        ),
    ),
) -> None:
    """Run the Tavily depth A/B experiment across the given depth variants."""
    selected_depths = [d.strip() for d in depths.split(",") if d.strip()]
    asyncio.run(main_async(runs_per_depth=runs_per_depth, depths=selected_depths))


def main():
    """
    Entry point for the script.
    """
    app()


if __name__ == "__main__":
    main()
