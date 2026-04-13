"""
examples/review_paper.py

Minimal example: run the academic review workflow from a script.

Usage:
    python examples/review_paper.py "Cronos-2 streamflow forecasting Amazon"
"""

import sys

from revisao_agents.config import ensure_runtime_dirs
from revisao_agents.graphs import build_review_graph


def main(theme: str) -> None:
    """Run the academic review workflow for a given theme.

    Args:
        theme (str): The review topic/theme to search for and plan around.

    Returns:
        None
    """
    ensure_runtime_dirs()
    graph = build_review_graph(tipo="academico")
    config = {"configurable": {"thread_id": "example-run"}}
    state = {
        "theme": theme,
        "review_type": "academico",
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

    print(f"Starting academic review for: {theme!r}\n")

    for step in graph.stream(state, config=config):
        node_name = list(step.keys())[0]
        print(f"[{node_name}] done")

    print("\nReview complete.")


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) or "Transformer models for time series forecasting"
    main(topic)
