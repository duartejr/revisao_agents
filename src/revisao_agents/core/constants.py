"""
constants.py - Default configuration constants for the revisao_agents package.

These are fallback values used by utility modules. Override via environment
variables where needed (e.g., in config.py with pydantic_settings).
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# MongoDB & Vector Store
# ─────────────────────────────────────────────────────────────────────────────
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "revisao_agents")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "corpus")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "vector_index")

# ─────────────────────────────────────────────────────────────────────────────
# OpenAI / LLM
# ─────────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# ─────────────────────────────────────────────────────────────────────────────
# Chunk / Context Limits
# ─────────────────────────────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "480"))
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "50000"))
MAX_CHUNKS_TOTAL = int(os.getenv("MAX_CHUNKS_TOTAL", "500"))
MAX_CORPUS_PROMPT = int(os.getenv("MAX_CORPUS_PROMPT", "3000"))
PLAN_MAX_CHARS = int(os.getenv("PLAN_MAX_CHARS", "2000"))

# ─────────────────────────────────────────────────────────────────────────────
# Search / Retrieval
# ─────────────────────────────────────────────────────────────────────────────
TOP_K_WRITER = int(os.getenv("TOP_K_WRITER", "10"))
TOP_K_VERIFICATION = int(os.getenv("TOP_K_VERIFICATION", "5"))
SNIPPET_MIN_SCORE = float(os.getenv("SNIPPET_MIN_SCORE", "0.3"))

# ─────────────────────────────────────────────────────────────────────────────
# Technical Search (Tavily)
# ─────────────────────────────────────────────────────────────────────────────
TECHNICAL_MAX_RESULTS = int(os.getenv("TECHNICAL_MAX_RESULTS", "5"))
PRIORITY_DOMAINS = [
    "github.com", "arxiv.org", "researchgate.net",
    "tensorflow.org", "pytorch.org", "huggingface.co"
]
BLOCKED_DOMAINS_EXTRACT = [
    "paywall", "paywalled", "subscription", "pdf"
]
MAX_IMAGES_SECTION = int(os.getenv("MAX_IMAGES_SECTION", "5"))

# ─────────────────────────────────────────────────────────────────────────────
# Anchor / Reference Matching
# ─────────────────────────────────────────────────────────────────────────────
EXTRACT_MIN_CHARS = int(os.getenv("EXTRACT_MIN_CHARS", "500"))

# ─────────────────────────────────────────────────────────────────────────────
# Caching
# ─────────────────────────────────────────────────────────────────────────────
CHUNKS_CACHE_DIR = os.getenv("CHUNKS_CACHE_DIR", ".chunks_cache")
HIST_MAX_TURNS = int(os.getenv("HIST_MAX_TURNS", "20"))
