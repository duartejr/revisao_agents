# src/revisao_agents/prompts/__init__.py
"""
Central de prompts do projeto.
Todos os prompts são carregados como YAML para facilitar edição e manutenção.
"""

from pathlib import Path
from typing import Dict
import yaml


def load_prompt(name: str, version: str = "latest") -> Dict:
    """
    Carrega um prompt YAML.
    Exemplo: load_prompt("technical_writing.writer_judge")
    """
    # Caminho relativo ao pacote
    base = Path(__file__).parent
    # Suporta subpastas com "."
    parts = name.split(".")
    file_path = (
        base / "/".join(parts) / f"{parts[-1]}.yaml"
    )  # technical_writing/writer_judge.yaml

    if not file_path.exists():
        raise FileNotFoundError(f"Prompt não encontrado: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
