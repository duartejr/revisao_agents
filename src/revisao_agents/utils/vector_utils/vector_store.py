import os
from typing import Any, List
import pymongo
from pymongo.collection import Collection
from openai import OpenAI

from ...config import (
    MONGODB_URI, MONGODB_DB, MONGODB_COLLECTION,
    VECTOR_INDEX_NAME, CHUNK_MAX_CHARS, MAX_CHUNKS_TOTAL, CHUNKS_CACHE_DIR
)

_client = None
_collection = None
_openai_client = None

# OpenAI embedding model
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"


def _project_root() -> str:
    """Returns the absolute path to the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _resolve_chunk_path(file_path: str) -> str:
    """Resolves the file path for a chunk's content, checking absolute path, project root, and cache directory.
    
    Args:
        file_path: The original file path stored in MongoDB for the chunk content.
    Returns:
        The resolved file path if it exists, or an empty string if not found.
    """
    if not file_path:
        return ""
    if os.path.isabs(file_path):
        return file_path

    root = _project_root()
    candidate = os.path.abspath(os.path.join(root, file_path))
    if os.path.exists(candidate):
        return candidate

    cache_dir = CHUNKS_CACHE_DIR if os.path.isabs(CHUNKS_CACHE_DIR) else os.path.abspath(os.path.join(root, CHUNKS_CACHE_DIR))
    by_basename = os.path.join(cache_dir, os.path.basename(file_path))
    if os.path.exists(by_basename):
        return by_basename

    return candidate


def _read_chunk_text(result: dict) -> str:
    """Reads the chunk text from the MongoDB result, either directly or from the file path.

    Args:
        result: A dictionary representing a MongoDB document for a chunk, which may contain 'text' or 'file_path'.
    Returns:
        The text content of the chunk, or an empty string if it cannot be read.
    """
    if result.get("text"):
        return str(result["text"])

    file_path = _resolve_chunk_path(str(result.get("file_path", "")))
    if not file_path:
        return ""
    try:
        with open(file_path, "r", encoding="utf-8") as file_handle:
            return file_handle.read()
    except Exception:
        return ""

def _get_mongo_collection() -> Collection:
    """Returns the MongoDB collection (connects if necessary).
    Uses global variables to cache the client and collection.
    
    Returns:
        pymongo Collection object for the configured MongoDB Atlas collection.
    Raises:
        RuntimeError if MONGODB_URI is not set or connection fails.
    """
    global _client, _collection
    if _collection is not None:
        return _collection
    if not MONGODB_URI:
        raise RuntimeError("MONGODB_URI not defined in the environment.")
    _client = pymongo.MongoClient(MONGODB_URI)
    db = _client[MONGODB_DB]
    _collection = db[MONGODB_COLLECTION]
    # Test connection
    _client.admin.command('ping')
    print("   Connected to MongoDB Atlas.")
    return _collection

def _get_openai_client():
    """Returns OpenAI client (initializes if necessary)."""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not defined in the environment.")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client

def _generate_embedding(text: str) -> List[float]:
    """
    Generates embedding for a single text using OpenAI.
    Truncates the text if necessary (model limit is generous, but we'll truncate beforehand).

    Args:
        text: input text to generate embedding for a single text (e.g., a query or chunk content)

    Returns:
        List of floats representing the embedding vector.
    
    Raises:
        RuntimeError if OpenAI client is not configured or API call fails.
    """
    client = _get_openai_client()
    # OpenAI recommends replacing newlines with spaces for better embedding quality
    text_clean = text.replace("\n", " ").strip()
    # Truncate if too long (around 8000 tokens, but we'll limit to 8000 characters as a heuristic)
    if len(text_clean) > 8000:
        text_clean = text_clean[:8000]
    try:
        response = client.embeddings.create(
            input=text_clean,
            model=OPENAI_EMBEDDING_MODEL
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"   Error generating embedding: {e}")
        # Return empty embedding? Better to propagate exception
        raise

def search_chunks(query: str, k: int = 16) -> List[str]:
    """
    Searches for chunks similar to the query using MongoDB Atlas Vector Search.
    Generates query embedding via OpenAI.

    Args:
        query: the search query text
        k: number of top similar chunks to return
    Returns:
        List of truncated strings (content of the chunks).
    """
    collection = _get_mongo_collection()
    
    # Generates embedding for the query using OpenAI
    try:
        query_embedding = _generate_embedding(query)
    except Exception as e:
        print(f"   Failure to generate query embedding.: {e}")
        return []
    
    # Aggregation pipeline with $vectorSearch
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "embedding",          # field where the embedding is stored
                "queryVector": query_embedding,
                "numCandidates": k * 10,      # number of candidates for search
                "limit": k,
            }
        },
        {
            "$project": {
                "text": 1,
                "file_path": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    
    try:
        results = list(collection.aggregate(pipeline))
    except Exception as e:
        print(f"   Error in MongoDB vector search: {e}")
        return []
    
    chunks = []
    for result in results:
        chunk_text = _read_chunk_text(result)
        if chunk_text:
            chunks.append(chunk_text[:CHUNK_MAX_CHARS])
    print(f"   {len(chunks)} chunks retrieved from MongoDB.")
    return chunks


def search_chunk_records(query: str, k: int = 16) -> List[dict[str, Any]]:
    """Search chunks and return text plus source metadata.

    Args:
        query: Search query text.
        k: Number of top similar chunks to return.

    Returns:
        List of records with chunk text, source metadata, and score.
    """
    collection = _get_mongo_collection()

    try:
        query_embedding = _generate_embedding(query)
    except Exception as e:
        print(f"   Failure to generate query embedding.: {e}")
        return []

    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": k * 10,
                "limit": k,
            }
        },
        {
            "$project": {
                "text": 1,
                "file_path": 1,
                "title": 1,
                "url": 1,
                "source_title": 1,
                "source_url": 1,
                "doi": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    try:
        results = list(collection.aggregate(pipeline))
    except Exception as e:
        print(f"   Error in MongoDB vector search: {e}")
        return []

    records: List[dict[str, Any]] = []
    for result in results:
        chunk_text = _read_chunk_text(result)
        if not chunk_text:
            continue

        file_path = str(result.get("file_path", ""))
        source_title = (
            str(result.get("title", "") or "").strip()
            or str(result.get("source_title", "") or "").strip()
            or (os.path.basename(file_path) if file_path else "(unknown source)")
        )
        source_url = (
            str(result.get("url", "") or "").strip()
            or str(result.get("source_url", "") or "").strip()
        )
        doi = str(result.get("doi", "") or "").strip()

        records.append(
            {
                "chunk": chunk_text[:CHUNK_MAX_CHARS],
                "file_path": file_path,
                "source_title": source_title,
                "source_url": source_url,
                "doi": doi,
                "score": float(result.get("score", 0.0) or 0.0),
            }
        )

    print(f"   {len(records)} chunk records retrieved from MongoDB.")
    return records

def accumulate_chunks(existing: List[str], new: List[str]) -> List[str]:
    """Accumulates new chunks without duplicates, respecting the maximum limit.
    
    Args:
        existing: List of existing chunks.
        new: List of new chunks to add.

    Returns:
        List of accumulated chunks, truncated to the maximum limit if necessary.
    """
    seen = set(existing)
    accumulated = existing + [c for c in new if c not in seen]
    if len(accumulated) > MAX_CHUNKS_TOTAL:
        accumulated = accumulated[-MAX_CHUNKS_TOTAL:]
    return accumulated

__all__ = [
    "search_chunks",
    "search_chunk_records",
    "accumulate_chunks",
]