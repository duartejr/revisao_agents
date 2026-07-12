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
from tavily import TavilyClient

# Ensure repo root is on sys.path when running directly with `uv run python`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from revisao_agents.evaluation.evaluators import evaluate_search_snippets  # noqa: E402
from revisao_agents.observability.mlflow_config import get_tracking_uri  # noqa: E402
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


async def main_async():
    """
    Main function to run the A/B depth experiment across all depth variants.
    """
    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment("ab_depth_experiments")

    for depth in DEPTHS:
        with mlflow.start_run(
            run_name=f"depth_{depth}",
        ):
            print(f"Running depth variant: {depth}")
            mlflow.log_param("depth", depth)
            mlflow.log_param("num_queries", len(TEST_QUERIES))

            aggregated_results = await run_single_depth(depth, TEST_QUERIES)

            # Aggregate metrics for logging
            metrics_list = aggregated_results["queries"]
            n = max(len(metrics_list), 1)
            mlflow.log_metrics(
                {
                    "average_latency": sum(m["latency"] for m in metrics_list) / n,
                    "total_credits": aggregated_results["total_credits"],
                    "average_urls_found": sum(m["urls_found"] for m in metrics_list) / n,
                    "average_relevance_score": sum(m["relevance_score"] for m in metrics_list) / n,
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
                f"Completed depth variant: {depth}, total credits used: {aggregated_results['total_credits']}"
            )


def main():
    """
    Entry point for the script.
    """
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
