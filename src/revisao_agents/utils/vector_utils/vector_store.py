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

# Modelo de embedding da OpenAI
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

def _get_mongo_collection() -> Collection:
    """Retorna a coleção MongoDB (conecta se necessário)."""
    global _client, _collection
    if _collection is not None:
        return _collection
    if not MONGODB_URI:
        raise RuntimeError("MONGODB_URI não definida no ambiente.")
    _client = pymongo.MongoClient(MONGODB_URI)
    db = _client[MONGODB_DB]
    _collection = db[MONGODB_COLLECTION]
    # Testa conexão
    _client.admin.command('ping')
    print("   Conectado ao MongoDB Atlas.")
    return _collection

def _get_openai_client():
    """Retorna cliente OpenAI (inicializa se necessário)."""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY não definida no ambiente.")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client

def _gerar_embedding(texto: str) -> List[float]:
    """
    Gera embedding para um único texto usando OpenAI.
    Trunca o texto se necessário (limite do modelo é generoso, mas vamos truncar antes).
    """
    client = _get_openai_client()
    # OpenAI recomenda substituir newlines por espaços
    texto_limpo = texto.replace("\n", " ").strip()
    # Truncar se for muito longo (cerca de 8000 tokens, mas vamos limitar a 8000 caracteres)
    if len(texto_limpo) > 8000:
        texto_limpo = texto_limpo[:8000]
    try:
        response = client.embeddings.create(
            input=texto_limpo,
            model=OPENAI_EMBEDDING_MODEL
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"   Erro ao gerar embedding: {e}")
        # Retorna embedding vazio? Melhor propagar exceção
        raise

def buscar_chunks(query: str, k: int = 16) -> List[str]:
    """
    Busca chunks similares à query usando MongoDB Atlas Vector Search.
    Gera embedding da query via OpenAI.
    Retorna lista de strings (conteúdo dos chunks) truncadas.
    """
    collection = _get_mongo_collection()
    
    # Gera embedding para a query
    try:
        query_embedding = _gerar_embedding(query)
    except Exception as e:
        print(f"   Falha ao gerar embedding da consulta: {e}")
        return []
    
    # Pipeline de agregação com $vectorSearch
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "embedding",          # campo onde está o embedding
                "queryVector": query_embedding,
                "numCandidates": k * 10,      # número de candidatos para busca
                "limit": k,
            }
        },
        {
            "$project": {
                "text": 1,                     # campo com o conteúdo do chunk
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    
    try:
        results = list(collection.aggregate(pipeline))
    except Exception as e:
        print(f"   Erro na busca vetorial MongoDB: {e}")
        return []
    
    # Extrai o texto, truncando se necessário
    chunks = [r["text"][:CHUNK_MAX_CHARS] for r in results if "text" in r]
    print(f"   {len(chunks)} chunks recuperados do MongoDB.")
    return chunks

def acumular_chunks(existentes: List[str], novos: List[str]) -> List[str]:
    """Acumula chunks novos sem duplicatas, respeitando o limite máximo."""
    vistos = set(existentes)
    acum = existentes + [c for c in novos if c not in vistos]
    if len(acum) > MAX_CHUNKS_TOTAL:
        acum = acum[-MAX_CHUNKS_TOTAL:]
    return acum