# src/revisao_agents/cli.py
import typer
from rich.console import Console
from pathlib import Path
from .graphs.review_graph import build_review_graph, run_review_graph
from .config import get_settings

console = Console()


def main(
    input_file: Path = typer.Argument(..., help="Arquivo .md ou .txt com o texto a revisar"),
    output_file: Path = typer.Option(None, "--output", "-o", help="Salvar saída revisada (opcional)"),
    model: str = typer.Option("gpt-4o-mini", "--model", help="Modelo LLM a usar"),
    debug: bool = typer.Option(False, "--debug", help="Modo verbose"),
):
    """Executa o fluxo completo de revisão acadêmica no texto fornecido."""
    settings = get_settings()
    if model:
        settings.llm_model = model  # override via CLI

    console.print(f"[bold green]Iniciando revisão de:[/bold green] {input_file}")

    try:
        graph = build_review_graph()
        result = run_review_graph(
            graph=graph,
            input_text=input_file.read_text(encoding="utf-8"),
            debug=debug,
        )

        console.print("\n[bold]Resultado final da revisão:[/bold]")
        console.print(result.get("final_text", "Sem texto final gerado."))

        if output_file:
            output_file.write_text(result["final_text"], encoding="utf-8")
            console.print(f"[green]Salvo em:[/green] {output_file}")

    except Exception as e:
        console.print(f"[bold red]Erro durante a revisão:[/bold red] {e}")
        raise typer.Exit(1)


# Use typer.run() for direct CLI invocation (no subcommands)
app = typer.Typer()
app.command()(main)


if __name__ == "__main__":
    app()
