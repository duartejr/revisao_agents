import os
import re
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from .graphs.checkpoints import get_checkpointer
from .graphs.review_graph import build_review_graph
from .hitl import run_hitl_loop

console = Console()


def resolve_topic(input_value: str) -> str:
    """
    Resolves the input as either raw topic text or a file path containing a topic/plan.

    Args:
        input_value: A string that is either the review theme/topic or a path to a file
                        containing the theme/plan. If it's a file, the function will attempt
                        to extract the theme from it.

    Returns:
        A string representing the review theme/topic, either directly from the input or extracted from the file.
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


def run_planning(
    theme: str,
    review_type: str,
    rounds: int,
    thread_id: str | None = None,
    interactive: bool = True,
    auto_response: str = "Keep the current plan.",
) -> None:
    """Execute planning with optional interactivity and SQLite checkpointer.

    Args:
        theme: The review theme or topic to plan for.
        review_type: "academic" or "technical" to determine the workflow.
        rounds: Number of refinement rounds for the planning process.
        thread_id: Optional custom thread ID for the graph execution (defaults to auto-generated).
        interactive: If True, runs in human-in-the-loop mode, pausing for user input at designated steps.
        auto_response: If provided and interactive is False, this response will be automatically sent at human pause steps.

    Returns:
        None. The function runs the planning workflow and prints the final plan or status to the console.
    """
    review_type_norm = review_type.strip().lower()
    if review_type_norm not in {"academic", "technical"}:
        console.print("[bold red]Error:[/bold red] review-type must be 'academic' or 'technical'.")
        return

    console.print(
        f"[bold green]Starting planning:[/bold green] {review_type_norm} | theme={theme!r}"
    )

    checkpointer = get_checkpointer()
    graph = build_review_graph(review_type=review_type_norm, checkpointer=checkpointer)

    state_init = {
        "theme": theme,
        "review_type": review_type_norm,
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

    if not thread_id:
        safe_theme = re.sub(r"[^a-zA-Z0-9_\-]", "_", theme[:20])
        thread_id = f"cli_{review_type_norm}_{safe_theme}"

    config = {"configurable": {"thread_id": thread_id}}

    if interactive:
        run_hitl_loop(graph, config, state_init)
    else:
        for _ in graph.stream(state_init, config=config):
            pass

        while True:
            current = graph.get_state(config)
            if not current.next:
                break

            if "human_pause" not in current.next:
                console.print(f"[yellow]Unexpected flow: waiting for nodes {current.next}[/yellow]")
                break

            history = current.values.get("interview_history", [])
            graph.update_state(
                config,
                {"interview_history": history + [("user", auto_response)]},
                as_node="human_pause",
            )
            for _ in graph.stream(None, config=config):
                pass

    final_state = graph.get_state(config).values
    final_plan = final_state.get("final_plan", "")
    plan_path = final_state.get("final_plan_path", "")

    if final_plan:
        console.print("\n[bold green]Final Plan Generated![/bold green]")
        if plan_path:
            console.print(f"Saved to: [blue]{plan_path}[/blue]")
    else:
        console.print("\n[yellow]No final plan was generated.[/yellow]")


def show_menu():
    """Display an interactive menu for the user to choose between different review and writing options.
    The menu allows users to select between planning academic or technical reviews, executing writing from existing plans
    (both technical and academic), indexing local PDFs into MongoDB, or formatting references from a file.

    Depending on the user's choice, the appropriate workflow or agent will be executed.

    Args:
        None. The function interacts with the user via console input and output.

    Returns:
        None. The function executes the selected workflow or agent and prints results to the console.
    """
    import glob

    from .agents.reference_formatter_agent import run_reference_formatter_agent
    from .config import print_runtime_config_summary, validate_runtime_config
    from .core.schemas.writer_config import WriterConfig
    from .utils.vector_utils.pdf_ingestor import ingest_pdf_folder
    from .workflows.technical_writing_workflow import build_technical_writing_workflow

    print_runtime_config_summary()
    startup_issues = validate_runtime_config(strict=False)
    if startup_issues:
        print("⚠️  Configuration warnings detected:")
        for issue in startup_issues:
            print(f"   - {issue}")
        print("   (The flow may fail in options that require these integrations.)\n")

    print("\n" + "=" * 70)
    print("REVISAO AGENTS - UNIFIED CLI")
    print("=" * 70)
    print("\nOptions:")
    print("  [1] Plan Academic Review (narrative)")
    print("  [2] Plan Technical Review (chapter)")
    print("  [3] Execute Writing from Existing Plan (Technical or Academic)")
    print("  [4] Index Local PDFs → vectorize and save to MongoDB")
    print("  [5] Format References (ABNT, APA, IEEE, etc.) from YAML/JSON file")
    choice = input("\nChoose [1/2/3/4/5]: ").strip()

    if choice == "4":
        print("\n" + "=" * 70)
        print("INDEX LOCAL PDFs")
        print("=" * 70)
        folder = input("\nPath to folder with PDFs: ").strip()
        if not folder:
            print("❌ Empty path.")
            return
        folder = os.path.expanduser(folder)
        if not os.path.isdir(folder):
            print(f"❌ Folder not found: {folder}")
            return
        result = ingest_pdf_folder(folder)
        print("\n" + "=" * 70)
        print("INDEXING RESULT")
        print("=" * 70)
        print(f"  ✅ New PDFs indexed : {result['indexed']}")
        print(f"  ⏭️  Already in DB     : {result['already']}")
        print(f"  ⚠️  Insufficient text : {result['skipped']}")
        print(f"  ❌ Reading errors     : {result['errors']}")
        print(f"  📦 Chunks inserted    : {result['total_chunks']}")
        print("=" * 70)
        return

    if choice == "5":
        run_reference_formatter_agent()
        return

    if choice == "3":
        print("\n" + "-" * 70)
        print("WRITING STYLE:")
        print("  [a] Technical section — didactic chapter (web search + MongoDB)")
        print("  [b] Academic — narrative literature review (corpus-first)")
        select_mode = input("\nChoose [a/b, default=a]: ").strip().lower() or "a"

        lang_opt = input("\nReview Language [pt/en, default=pt]: ").strip().lower() or "pt"
        if select_mode == "b":
            writer_config = WriterConfig.academic(language=lang_opt)
            glob_pattern = "plans/plano_revisao_*.md"
        else:
            writer_config = WriterConfig.technical(language=lang_opt)
            glob_pattern = "plans/plano_revisao_tecnica_*.md"

        print("\nEnable web search via Tavily?")
        tavily_opt = input("Enable Tavily? [y/N]: ").strip().lower() or "n"

        planos = sorted(glob.glob(glob_pattern)) + sorted(glob.glob("plans/review_plan*.md"))
        if not planos:
            print("❌ No plans found in plans/")
            return

        print("\nPlans found:")
        for i, p in enumerate(planos, 1):
            print(f"  [{i}] {p}")
        idx = input(f"\nChoose [1-{len(planos)}]: ").strip()
        plan_path = planos[int(idx) - 1] if idx.isdigit() and 1 <= int(idx) <= len(planos) else idx

        if not os.path.exists(plan_path):
            print(f"❌ File not found: {plan_path}")
            return

        state_init = {
            "theme": "",
            "plan_summary": "",
            "sections": [],
            "plan_path": plan_path,
            "written_sections": [],
            "refs_urls": [],
            "refs_images": [],
            "cumulative_summary": "",
            "react_log": [],
            "verification_stats": [],
            "status": "starting",
            "writer_config": writer_config.to_dict(),
            "tavily_enabled": (tavily_opt == "y"),
        }
        app = build_technical_writing_workflow()
        try:
            for event in app.stream(state_init):
                node = list(event.keys())[0] if event else "?"
                if node != "__end__":
                    st = event.get(node, {}).get("status", "")
                    if st:
                        print(f"\n   ▶ [{node}] → {st}")
        except KeyboardInterrupt:
            print("\nCancelled.")
        return

    if choice in ("1", "2"):
        theme = input("\nReview theme: ").strip()
        if not theme:
            return
        review_type = "academic" if choice == "1" else "technical"
        rounds_input = input("\nNumber of refinement rounds [3]: ").strip()
        rounds = int(rounds_input) if rounds_input.isdigit() else 3
        run_planning(theme=theme, review_type=review_type, rounds=rounds, interactive=True)


app = typer.Typer(help="Revisao Agents - AI Powered Literature Review Tool")


@app.command()
def main(
    input_value: Annotated[
        str | None,
        typer.Argument(help="Review theme or path to file containing theme/plan"),
    ] = None,
    review_type: Annotated[
        str, typer.Option("--review-type", "-t", help="Type: academic or technical")
    ] = "academic",
    rounds: Annotated[int, typer.Option("--rounds", "-r", help="Number of refinement rounds")] = 3,
    model: Annotated[str, typer.Option("--model", help="LLM model to use (optional)")] = "",
    auto_response: Annotated[
        str | None, typer.Option("--auto-response", help="Automatic response for HITL steps")
    ] = None,
    debug: Annotated[bool, typer.Option("--debug", help="Verbose mode")] = False,
    thread_id: Annotated[str | None, typer.Option("--thread-id", help="Custom thread ID")] = None,
) -> None:
    """Execute academic/technical planning or show interactive menu.
    If no input_value is provided, an interactive menu will be shown to choose between different review and writing options.
    If input_value is provided, it will be resolved as either a review theme or a file
    containing the theme/plan, and the planning workflow will be executed directly.

    Args:
        input_value: Optional string that is either the review theme/topic or a path to a file containing the theme/plan. If it's a file, the function will attempt to extract the theme from
                        it. If not provided, the user will be shown an interactive menu.
        review_type: The type of review to plan for ("academic" or "technical"). Ignored if input_value is not provided and the menu is shown.
        rounds: Number of refinement rounds for the planning process. Ignored if input_value is not provided and the menu is shown.
        model: Optional LLM model name to set via environment variable (e.g., "gpt-4"). If not provided, defaults to existing environment configuration.
        auto_response: If provided, this response will be automatically sent at human-in-the-loop steps during planning. If not provided, the system will wait for user input at those steps.
        debug: If True, enables verbose mode with additional logging.
        thread_id: Optional custom thread ID for the graph execution. If not provided, a default ID will be generated based on the review type and theme.

    Returns:
        None. The function executes the selected workflow or agent and prints results to the console.
    """
    if model:
        os.environ["LLM_MODEL"] = model

    if input_value is None:
        show_menu()
    else:
        theme = resolve_topic(input_value)
        interactive = auto_response is None
        run_planning(
            theme=theme,
            review_type=review_type,
            rounds=rounds,
            thread_id=thread_id,
            interactive=interactive,
            auto_response=auto_response or "Keep the current plan.",
        )


if __name__ == "__main__":
    app()
