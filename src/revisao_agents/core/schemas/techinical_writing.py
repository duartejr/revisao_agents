# src/revisao_agents/core/schemas/techinical_writing.py
from pydantic import BaseModel, Field
from typing import List


# ── Schema de saída ───────────────────────────────────────────────────────────
class Fonte(BaseModel):
    """Representa uma fonte citada no corpus MongoDB."""

    id: int = Field(
        description="Índice numérico da fonte conforme aparece no corpus, ex: 1, 2, 3"
    )
    url: str = Field(description="URL completa da fonte extraída do corpus")
    titulo: str = Field(description="Título do documento ou página da fonte")


class RespostaSecao(BaseModel):
    """Modelo de saída esperado pelo agent de escrita técnica."""

    rascunho: str = Field(
        description=(
            "Texto completo da seção com todas as âncoras [ÂNCORA: '...'] "
            "e citações [N] embutidas inline, em Markdown."
        )
    )
    fontes_usadas: List[Fonte] = Field(
        description="Apenas as fontes efetivamente citadas no rascunho, sem repetição."
    )