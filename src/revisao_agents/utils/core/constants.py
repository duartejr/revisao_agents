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
PLAN_MAX_CHARS = int(os.getenv("PLAN_MAX_CHARS", "8000"))

# ============================================================================
# Search Parameters
# ============================================================================
TOP_K_WRITER = int(os.getenv("TOP_K_WRITER", "5"))
TOP_K_VERIFICATION = int(os.getenv("TOP_K_VERIFICATION", "3"))
SNIPPET_MIN_SCORE = float(os.getenv("SNIPPET_MIN_SCORE", "0.3"))

# ============================================================================
# Technical Search Configuration
# ============================================================================
TECHNICAL_MAX_RESULTS = int(os.getenv("TECHNICAL_MAX_RESULTS", "10"))
PRIORITY_DOMAINS = os.getenv(
    "PRIORITY_DOMAINS",
    "arxiv.org,researchgate.net,scholar.google.com,github.com,sciencedirect.com"
).split(",")
BLOCKED_DOMAINS_EXTRACT = os.getenv(
    "BLOCKED_DOMAINS_EXTRACT",
    ""
).split(",") if os.getenv("BLOCKED_DOMAINS_EXTRACT") else []
MAX_IMAGES_SECTION = int(os.getenv("MAX_IMAGES_SECTION", "2"))

# ============================================================================
# Anchor Matching Configuration
# ============================================================================
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
    "PLAN_MAX_CHARS",
    # Search
    "TOP_K_WRITER",
    "TOP_K_VERIFICATION",
    "SNIPPET_MIN_SCORE",
    # Technical
    "TECHNICAL_MAX_RESULTS",
    "PRIORITY_DOMAINS",
    "BLOCKED_DOMAINS_EXTRACT",
    "MAX_IMAGES_SECTION",
    # Anchor
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
