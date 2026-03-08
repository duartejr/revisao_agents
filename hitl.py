# src/revisao_agents/hitl.py
from typing import Dict, Any
from rich.console import Console
from rich.prompt import Prompt

console = Console()


def human_feedback_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Nó de Human-in-the-Loop: pede feedback ao usuário quando necessário."""
    current = state.get("current_text", "")
    console.print("\n[bold yellow]=== AGUARDANDO FEEDBACK HUMANO ===[/bold yellow]")
    console.print(f"Texto atual:\n{current[:800]}...\n")  # preview

    feedback = Prompt.ask(
        "[bold]Seu feedback / instruções de correção[/bold] (ou ENTER para aprovar)",
        default="",
    )

    if feedback.strip():
        state["human_feedback"] = feedback
        state["needs_human_review"] = False  # continua após feedback
    else:
        state["human_feedback"] = None
        state["needs_human_review"] = False  # aprova e segue

    return state