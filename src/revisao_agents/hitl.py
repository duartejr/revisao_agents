from typing import Any


def run_hitl_loop(app: Any, config: dict, state_init: dict) -> None:
    """Loop of human-in-the-loop interaction for refining plans until completion.
    
    Args:
        app: compiled LangGraph graph to run
        config: graph execution config dict
        state_init: initial state dict to start the graph with
    Returns:
        final state dict after graph completion
    """
    for _ in app.stream(state_init, config):
        pass
    while True:
        state = app.get_state(config)
        if not state.next:
            print("\nPlanning complete!")
            break
        if "human_pause" not in state.next:
            print("\nUnexpected state:", str(state.next))
            break
        # Show the agent's last question or prompt
        for role, c in reversed(state.values.get("interview_history", [])):
            if role == "assistant":
                print("\n🤖", c)
                break
        p  = state.values.get("questions_asked", 0)
        mp = state.values.get("max_questions", 3)
        tp = state.values.get("review_type", "academic")
        print(f"\n   [Round {p}/{mp} — {tp} — ok to finish]")
        resp = input("👤 ").strip() or "Keep the current plan."
        hist = state.values.get("interview_history", [])
        app.update_state(config,
            {"interview_history": hist + [("user", resp)]},
            as_node="human_pause")
        print("\n[Refining plan...]")
        for _ in app.stream(None, config):
            pass
        if not app.get_state(config).next:
            print("\nPlanning complete!")
            break