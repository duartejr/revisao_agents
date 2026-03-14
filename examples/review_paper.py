"""
examples/review_paper.py

Minimal example: run the academic review workflow from a script.

Usage:
    python examples/review_paper.py "Cronos-2 streamflow forecasting Amazon"
"""

import sys
from revisao_agents.graphs.review_graph import build_review_graph


def main(tema: str):
    graph = build_review_graph(tipo="academico")
    config = {"configurable": {"thread_id": "example-run"}}
    state = {
        "theme": tema,
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
        "status": "iniciando",
    }

    print(f"Starting academic review for: {tema!r}\n")

    for step in graph.stream(state, config=config):
        node_name = list(step.keys())[0]
        print(f"[{node_name}] done")

    print("\nReview complete.")


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) or "Transformer models for time series forecasting"
    main(topic)
