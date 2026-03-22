# src/revisao_agents/core/schemas/__init__.py
"""
Centralized schemas for the project.
Import everything from here for clean access anywhere in the project.
"""

# Pydantic models — technical writing output
# NamedTuple schemas — corpus / retrieval
from .corpus import Chunk
from .techinical_writing import SectionAnswer, Source

# Writer strategy configuration
from .writer_config import WriterConfig

__all__ = [
    # technical_writing
    "Source",
    "SectionAnswer",
    # corpus
    "Chunk",
    # writer config
    "WriterConfig",
]
