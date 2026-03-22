# src/revisao_agents/core/schemas/__init__.py
"""
Centralized schemas for the project.
Import everything from here for clean access anywhere in the project.
"""

# Pydantic models — technical writing output
from .techinical_writing import Source, SectionAnswer

# NamedTuple schemas — corpus / retrieval
from .corpus import Chunk

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
