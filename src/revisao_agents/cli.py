import os
import re
from pathlib import Path

import typer
from rich.console import Console

from .graphs.review_graph import build_review_graph

console = Console()

def resolve_topic(input_value: str) -> str:
    """
    Resolves the input as either raw topic text or a file path containing a topic/plan.
    
    If the input is a valid file path, this function attempts to extract the 
    topic from a structured header (e.g., "**Topic:** ..."). If no header is 
    found, it falls back to the first non-empty line of the file.

    Args:
        input_value (str): A string that is either the topic itself or a path 
            to a text file.

    Returns:
        str: The extracted topic or the original input string if no file exists.
    """
    path = Path(input_value)
    
    # If it's not a file, treat the input as the raw topic text
    if not path.exists() or not path.is_file():
        return input_value.strip()

    try:
        content = path.read_text(encoding="utf-8")
        
        # Look for a common header pattern (Case-insensitive 'Topic' or 'Tema')
        match = re.search(r"\*\*(?:Topic|Theme|Tema|T[óo]pico):\*\*\s*(.+)", content, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Fallback: Extract the first non-empty line
        first_line = next((line.strip() for line in content.splitlines() if line.strip()), "")
        return first_line or input_value.strip()
        
    except Exception:
        # If file reading fails for any reason, return the original input
        return input_value.strip()


def _run_planning_until_complete(
    graph,
    theme: str,
    review_type: str,
    rounds: int,
    auto_response: str,
    debug: bool,
) -> dict:
    """Execute planning graph with automatic HITL responses until completion.
    
    Args:
        graph: the compiled review graph to execute
        theme: the review topic/theme
        review_type: "academic" or "technical"
        rounds: number of refinement rounds for HITL steps
        auto_response: the response to use for all HITL prompts
        debug: whether to print intermediate events
    
    Returns:
        the final state dict after graph execution completes
    """
    state_init = {
        "theme": theme,
        "review_type": review_type,
        "relevant_chunks": [],
        "technical_snippets": [],
        "technical_urls": [],
        "current_plan": "",
        "interview_history": [],
        "questions_asked": 0,
        "max_questions": max(1, int(rounds)),
        "final_plan": "",
        "final_plan_path": "",
        "status": "starting",
    }
    config = {"configurable": {"thread_id": f"cli_{review_type}_{theme[:20]}"}}

    for event in graph.stream(state_init, config=config):
        if debug:
            console.print(event)

    while True:
        current = graph.get_state(config)
        if not current.next:
            return current.values

        if "human_pause" not in current.next:
            raise RuntimeError(f"Unexpected flow: waiting for nodes {current.next}")

        history = current.values.get("interview_history", [])
        graph.update_state(
            config,
            {"interview_history": history + [("user", auto_response)]},
            as_node="human_pause",
        )
        for event in graph.stream(None, config=config):
            if debug:
                console.print(event)


def main(
    input_value: str = typer.Argument(..., help="Review theme or path to file containing theme/plan"),
    review_type: str = typer.Option("academic", "--review-type", "-t", help="Type: academic or technical"),  # noqa: B008
    rounds: int = typer.Option(3, "--rounds", "-r", help="Number of refinement rounds"),  # noqa: B008
    output_file: Path = typer.Option(None, "--output", "-o", help="Save final plan to file (optional)"),  # noqa: B008
    model: str = typer.Option("", "--model", help="LLM model to use (optional)"),  # noqa: B008
    auto_response: str = typer.Option("Keep the current plan.", "--auto-response", help="Automatic response for HITL steps"),  # noqa: B008
    debug: bool = typer.Option(False, "--debug", help="Verbose mode"),  # noqa: B008
):
    """Execute academic/technical planning until final plan is generated.
    
    Args:
        input_value: either the review theme text or a path to a text file containing the theme/plan
        review_type: "academic" or "technical"
        rounds: number of refinement rounds for human-in-the-loop steps
        output_file: optional path to save the final plan
        model: optional LLM model name to set via environment variable
        auto_response: response to use for all HITL prompts (default: "Keep the current plan.")
        debug: whether to print intermediate events during execution
    
    Returns:
        None (prints final plan and optionally saves to file)
    """
    if model:
        os.environ["LLM_MODEL"] = model

    review_type_norm = review_type.strip().lower()
    if review_type_norm not in {"academic", "technical"}:
        console.print("[bold red]Error:[/bold red] --review-type must be 'academic' or 'technical'.")
        raise typer.Exit(2)

    theme = resolve_topic(input_value)
    if not theme:
        console.print("[bold red]Error:[/bold red] theme is empty after reading the argument/file.")
        raise typer.Exit(2)

    console.print(f"[bold green]Starting planning:[/bold green] {review_type_norm} | theme={theme!r}")

    try:
        graph = build_review_graph(review_type=review_type_norm)
        result = _run_planning_until_complete(
            graph=graph,
            theme=theme,
            review_type=review_type_norm,
            rounds=rounds,
            auto_response=auto_response,
            debug=debug,
        )

        final_plan = result.get("final_plan", "")
        plan_path = result.get("final_plan_path", "")

        console.print("\n[bold]Final planning result:[/bold]")
        console.print(final_plan or result.get("current_plan", "No final plan generated."))
        if plan_path:
            console.print(f"\n[green]Plan automatically saved at:[/green] {plan_path}")

        if output_file:
            payload = final_plan or result.get("current_plan", "")
            output_file.write_text(payload, encoding="utf-8")
            console.print(f"[green]Saved at:[/green] {output_file}")

    except Exception as e:
        console.print(f"[bold red]Error during review:[/bold red] {e}")
        raise typer.Exit(1) from e


# Use typer.run() for direct CLI invocation (no subcommands)
app = typer.Typer()
app.command()(main)


if __name__ == "__main__":
    app()
