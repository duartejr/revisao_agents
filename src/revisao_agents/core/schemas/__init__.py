# src/revisao_agents/core/schemas/__init__.py
"""
Schemas centralizados do projeto.
Importe tudo daqui para ter acesso limpo em qualquer lugar do projeto.
"""

# Pydantic models — technical writing output
from .techinical_writing import Fonte, RespostaSecao

# NamedTuple schemas — corpus / retrieval
from .corpus import Chunk

# Writer strategy configuration
from .writer_config import WriterConfig

__all__ = [
    # technical_writing
    "Fonte",
    "RespostaSecao",
    # corpus
    "Chunk",
    # writer config
    "WriterConfig",
]