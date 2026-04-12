from __future__ import annotations

import os
from typing import Any

from revisao_agents.config import validate_runtime_config
from revisao_agents.tools.reference_formatter import format_references_from_file
from revisao_agents.utils.vector_utils.pdf_ingestor import ingest_pdf_folder


def index_pdfs(folder_path: str) -> str:
    """Index PDFs from the specified folder into the vector database.

    Args:
        folder_path: The path to the folder containing PDF files to be indexed.

    Returns:
        A status message indicating the result of the indexing operation.
    """
    cfg_issues = validate_runtime_config(strict=False)
    if cfg_issues:
        return "❌ Configuração incompleta:\n- " + "\n- ".join(cfg_issues)

    if not folder_path.strip():
        return "❌ Informe o caminho da pasta."
    folder_path = os.path.expanduser(folder_path.strip())
    if not os.path.isdir(folder_path):
        return f"❌ Pasta não encontrada: {folder_path}"
    try:
        result = ingest_pdf_folder(folder_path)
    except Exception as exc:
        return f"❌ Erro durante indexação: {exc}"
    return (
        "✅ Indexação concluída!\n\n"
        f"- Novos PDFs indexados : **{result['indexed']}**\n"
        f"- Já no banco          : **{result['already']}**\n"
        f"- Texto insuficiente   : **{result['skipped']}**\n"
        f"- Erros de leitura     : **{result['errors']}**\n"
        f"- Chunks inseridos     : **{result['total_chunks']}**"
    )


def format_references(
    yaml_file_obj: Any,
    tavily_enabled: bool,
    output_dir: str,
) -> tuple[str, str]:
    """Format references from a YAML file into markdown.

    Args:
        yaml_file_obj: A file-like object or string path to the YAML file containing references to format.
        tavily_enabled: A boolean indicating whether Tavily integration is enabled for enhanced formatting.
        output_dir: A string path to the directory where the formatted markdown file should be saved.

    Returns:
        A tuple containing the formatted markdown string and a status message.
    """
    if yaml_file_obj is None:
        return "", "❌ Nenhum arquivo selecionado."

    input_path = yaml_file_obj if isinstance(yaml_file_obj, str) else yaml_file_obj.name

    output_path = None
    if output_dir.strip():
        os.makedirs(output_dir.strip(), exist_ok=True)
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir.strip(), f"{base}_formatted.md")

    try:
        result_md = format_references_from_file(
            input_path=input_path,
            tavily_enabled=tavily_enabled,
            output_path=output_path,
        )
    except Exception as exc:
        return "", f"❌ Erro ao formatar referências: {exc}"

    status = "✅ Referências formatadas com sucesso!"
    if output_path:
        status += f"\n\nArquivo salvo em: `{output_path}`"
    return result_md, status
