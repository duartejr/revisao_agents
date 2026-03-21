# src/revisao_agents/core/schemas/techinical_writing.py
from pydantic import BaseModel, Field
from typing import List


# ── Schema de saída ───────────────────────────────────────────────────────────
class Source(BaseModel):
    """Represent a source cited in the MongoDB corpus."""

    id: int = Field(
        description="Numeric index of the source as it appears in the corpus, e.g., 1, 2, 3"
    )
    url: str = Field(description="Full URL of the source extracted from the corpus")
    title: str = Field(description="Title of the document or page of the source")


class SectionAnswer(BaseModel):
    """Expected output model for the technical writing agent."""

    draft: str = Field(
        description=(
            "Full text of the section with all anchors [ANCHOR: '...'] "
            "and citations [N] embedded inline, in Markdown."
        )
    )
    used_sources: List[Source] = Field(
        description="Only the sources actually cited in the draft, without repetition."
    )