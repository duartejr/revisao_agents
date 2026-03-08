# src/revisao_agents/utils/constants.py
"""
Configuration constants for the revision agents system.

This module defines all configuration constants that utilities need,
reading from environment variables with sensible defaults.
"""

import os
from typing import List

# ============================================================================
# MongoDB Configuration
# ============================================================================
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "revisao_agents")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "chunks")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "chunks_vector_index")

# ============================================================================
# OpenAI Configuration
# ============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# ============================================================================
# Chunk & Context Configuration
# ============================================================================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "5000"))
MAX_CHUNKS_TOTAL = int(os.getenv("MAX_CHUNKS_TOTAL", "100"))
MAX_CORPUS_PROMPT = int(os.getenv("MAX_CORPUS_PROMPT", "10000"))
PLANO_MAX_CHARS = int(os.getenv("PLANO_MAX_CHARS", "8000"))

# ============================================================================
# Search Parameters
# ============================================================================
TOP_K_ESCRITA = int(os.getenv("TOP_K_ESCRITA", "5"))
TOP_K_VERIFICACAO = int(os.getenv("TOP_K_VERIFICACAO", "3"))
SNIPPET_MIN_SCORE = float(os.getenv("SNIPPET_MIN_SCORE", "0.3"))

# ============================================================================
# Technical Search Configuration
# ============================================================================
TECNICO_MAX_RESULTS = int(os.getenv("TECNICO_MAX_RESULTS", "10"))
DOMINIOS_PRIORITARIOS = os.getenv(
    "DOMINIOS_PRIORITARIOS",
    "arxiv.org,researchgate.net,scholar.google.com,github.com,sciencedirect.com"
).split(",")
DOMINIOS_BLOQUEADOS_EXTRACT = os.getenv(
    "DOMINIOS_BLOQUEADOS_EXTRACT",
    ""
).split(",") if os.getenv("DOMINIOS_BLOQUEADOS_EXTRACT") else []
MAX_IMAGENS_SECAO = int(os.getenv("MAX_IMAGENS_SECAO", "2"))

# ============================================================================
# Anchor Matching Configuration
# ============================================================================
ANCORA_MIN_SIM_FAISS = float(os.getenv("ANCORA_MIN_SIM_FAISS", "0.5"))
ANCORA_MIN_SIM_FUZZY = float(os.getenv("ANCORA_MIN_SIM_FUZZY", "0.6"))
EXTRACT_MIN_CHARS = int(os.getenv("EXTRACT_MIN_CHARS", "50"))

# ============================================================================
# Caching Configuration
# ============================================================================
CHUNKS_CACHE_DIR = os.getenv("CHUNKS_CACHE_DIR", ".chunks_cache")
HIST_MAX_TURNS = int(os.getenv("HIST_MAX_TURNS", "20"))

# ============================================================================
# Dialog Configuration
# ============================================================================
DIALOG_MAX_TURNS = int(os.getenv("DIALOG_MAX_TURNS", "5"))
DIALOG_TURN_CHAR_LIMIT = int(os.getenv("DIALOG_TURN_CHAR_LIMIT", "3000"))

# ============================================================================
# Performance Configuration
# ============================================================================
DEFAULT_CHECKPOINT_TYPE = os.getenv("DEFAULT_CHECKPOINT_TYPE", "memory")
DEFAULT_CHECKPOINT_DB = os.getenv("DEFAULT_CHECKPOINT_DB", "sqlite:///checkpoints.db")
DEFAULT_CHECKPOINT_POSTGRES_URL = os.getenv(
    "DEFAULT_CHECKPOINT_POSTGRES_URL",
    "postgresql://user:password@localhost/checkpoints"
)

# ============================================================================
# Defaults
# ============================================================================
__all__ = [
    # MongoDB
    "MONGODB_URI",
    "MONGODB_DB",
    "MONGODB_COLLECTION",
    "VECTOR_INDEX_NAME",
    # OpenAI
    "OPENAI_API_KEY",
    "OPENAI_EMBEDDING_MODEL",
    # Chunking
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "CHUNK_MAX_CHARS",
    "MAX_CHUNKS_TOTAL",
    "MAX_CORPUS_PROMPT",
    "PLANO_MAX_CHARS",
    # Search
    "TOP_K_ESCRITA",
    "TOP_K_VERIFICACAO",
    "SNIPPET_MIN_SCORE",
    # Technical
    "TECNICO_MAX_RESULTS",
    "DOMINIOS_PRIORITARIOS",
    "DOMINIOS_BLOQUEADOS_EXTRACT",
    "MAX_IMAGENS_SECAO",
    # Anchor
    "ANCORA_MIN_SIM_FAISS",
    "ANCORA_MIN_SIM_FUZZY",
    "EXTRACT_MIN_CHARS",
    # Caching
    "CHUNKS_CACHE_DIR",
    "HIST_MAX_TURNS",
    # Dialog
    "DIALOG_MAX_TURNS",
    "DIALOG_TURN_CHAR_LIMIT",
    # Performance
    "DEFAULT_CHECKPOINT_TYPE",
    "DEFAULT_CHECKPOINT_DB",
    "DEFAULT_CHECKPOINT_POSTGRES_URL",
]
