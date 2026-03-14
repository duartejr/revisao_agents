import os
from typing import List
import pymongo
from pymongo.collection import Collection
from openai import OpenAI

from ...config import (
    MONGODB_URI, MONGODB_DB, MONGODB_COLLECTION,
    VECTOR_INDEX_NAME, CHUNK_MAX_CHARS, MAX_CHUNKS_TOTAL
)

_client = None
_collection = None
_openai_client = None

# OpenAI embedding model
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

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
    
    # Gera embedding para a query
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
                "text": 1,                     # field with the chunk content
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    
    try:
        results = list(collection.aggregate(pipeline))
    except Exception as e:
        print(f"   Error in MongoDB vector search: {e}")
        return []
    
    # Extract text, truncating if necessary
    chunks = [r["text"][:CHUNK_MAX_CHARS] for r in results if "text" in r]
    print(f"   {len(chunks)} chunks retrieved from MongoDB.")
    return chunks

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
    "accumulate_chunks",
]