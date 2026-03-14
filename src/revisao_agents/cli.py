import os
import re
from pathlib import Path

import typer
from rich.console import Console

from .graphs.review_graph import build_review_graph

console = Console()


def _resolve_tema(input_value: str) -> str:
    """Resolve input as either raw theme text or a file containing a theme/plan."""
    path = Path(input_value)
    if not path.exists() or not path.is_file():
        return input_value.strip()

    raw = path.read_text(encoding="utf-8")
    match = re.search(r"\*\*Tema:\*\*\s*(.+)", raw)
    if match:
        return match.group(1).strip()

    first_non_empty = next((line.strip() for line in raw.splitlines() if line.strip()), "")
    return first_non_empty or input_value.strip()


def _run_planning_until_complete(
    graph,
    theme: str,
    review_type: str,
    rodadas: int,
    auto_response: str,
    debug: bool,
) -> dict:
    """Execute planning graph with automatic HITL responses until completion."""
    state_init = {
        "theme": theme,
        "review_type": review_type,
        "relevant_chunks": [],
        "technical_snippets": [],
        "technical_urls": [],
        "current_plan": "",
        "interview_history": [],
        "questions_asked": 0,
        "max_questions": max(1, int(rodadas)),
        "final_plan": "",
        "final_plan_path": "",
        "status": "iniciando",
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
            raise RuntimeError(f"Fluxo inesperado: aguardando nós {current.next}")

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
    input_value: str = typer.Argument(..., help="Tema da revisão ou caminho para arquivo com tema/plano"),
    tipo: str = typer.Option("academico", "--tipo", "-t", help="Tipo: academico ou tecnico"),
    rodadas: int = typer.Option(3, "--rodadas", "-r", help="Rodadas de refinamento"),
    output_file: Path = typer.Option(None, "--output", "-o", help="Salvar plano final em arquivo (opcional)"),
    model: str = typer.Option("", "--model", help="Modelo LLM a usar (opcional)"),
    auto_response: str = typer.Option("Manter o plano atual.", "--auto-response", help="Resposta automática para etapas HITL"),
    debug: bool = typer.Option(False, "--debug", help="Modo verbose"),
):
    """Executa planejamento acadêmico/técnico até gerar plano final."""
    if model:
        os.environ["LLM_MODEL"] = model

    tipo_norm = tipo.strip().lower()
    if tipo_norm not in {"academico", "tecnico"}:
        console.print("[bold red]Erro:[/bold red] --tipo deve ser 'academico' ou 'tecnico'.")
        raise typer.Exit(2)

    theme = _resolve_tema(input_value)
    if not theme:
        console.print("[bold red]Erro:[/bold red] tema vazio após leitura do argumento/arquivo.")
        raise typer.Exit(2)

    console.print(f"[bold green]Iniciando planejamento:[/bold green] {tipo_norm} | tema={theme!r}")

    try:
        graph = build_review_graph(tipo=tipo_norm)
        result = _run_planning_until_complete(
            graph=graph,
            theme=theme,
            review_type=tipo_norm,
            rodadas=rodadas,
            auto_response=auto_response,
            debug=debug,
        )

        final_plan = result.get("final_plan", "")
        plan_path = result.get("final_plan_path", "")

        console.print("\n[bold]Resultado final do planejamento:[/bold]")
        console.print(final_plan or result.get("current_plan", "Sem plano final gerado."))
        if plan_path:
            console.print(f"\n[green]Plano salvo automaticamente em:[/green] {plan_path}")

        if output_file:
            payload = final_plan or result.get("current_plan", "")
            output_file.write_text(payload, encoding="utf-8")
            console.print(f"[green]Salvo em:[/green] {output_file}")

    except Exception as e:
        console.print(f"[bold red]Erro durante a revisão:[/bold red] {e}")
        raise typer.Exit(1)


# Use typer.run() for direct CLI invocation (no subcommands)
app = typer.Typer()
app.command()(main)


if __name__ == "__main__":
    app()
