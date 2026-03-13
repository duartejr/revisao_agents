import os
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pymongo
from pymongo.collection import Collection
from openai import OpenAI
import tiktoken
import time

from ...config import (
    MONGODB_URI, MONGODB_DB, MONGODB_COLLECTION, VECTOR_INDEX_NAME,
    OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_WRITER, TOP_K_VERIFICATION,
    MAX_CORPUS_PROMPT, ANCHOR_MIN_SIM,
    EXTRACT_MIN_CHARS, CHUNKS_CACHE_DIR, SNIPPET_MIN_SCORE
)
from ..file_utils.helpers import normalizar, fuzzy_sim, fuzzy_search_in_text
from ..search_utils.tavily_client import score_url  # import local
from ...core.schemas.corpus import Chunk

class CorpusMongoDB:
    def __init__(self):
        self._client = None
        self._collection = None
        self._openai_client = None
        self._tokenizer = None
        self._urls_usadas: List[str] = []
        self._fonte_map: Dict[int, str] = {}
        self._n_docs = 0
        self._total_chunks = 0
        # Garantir que o diretório de cache existe
        os.makedirs(CHUNKS_CACHE_DIR, exist_ok=True)

    def _get_collection(self) -> Collection:
        if self._collection is not None:
            return self._collection
        if not MONGODB_URI:
            raise RuntimeError("MONGODB_URI não definida.")
        self._client = pymongo.MongoClient(MONGODB_URI)
        db = self._client[MONGODB_DB]
        self._collection = db[MONGODB_COLLECTION]
        self._client.admin.command('ping')
        print("   Conectado ao MongoDB Atlas.")
        return self._collection

    def connect(self) -> None:
        """
        Public method to establish MongoDB connection.
        Calls _get_collection internally to ensure connection is established.
        """
        self._get_collection()

    def close(self) -> None:
        """
        Close MongoDB connection and clean up resources.
        """
        if self._client is not None:
            self._client.close()
            self._client = None
            self._collection = None

    def _get_openai_client(self):
        if self._openai_client is None:
            if not OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY não definida.")
            self._openai_client = OpenAI(api_key=OPENAI_API_KEY)
        return self._openai_client

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = tiktoken.encoding_for_model(OPENAI_EMBEDDING_MODEL)
        return self._tokenizer

    @staticmethod
    def _chunkar(texto: str) -> List[str]:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
            )
            return splitter.split_text(texto)
        except ImportError:
            # fallback simples
            chunks, inicio = [], 0
            while inicio < len(texto):
                fim = min(inicio + CHUNK_SIZE, len(texto))
                if fim < len(texto):
                    for sep in ("\n\n", "\n", ". ", " "):
                        pos = texto.rfind(sep, inicio + CHUNK_SIZE // 2, fim)
                        if pos != -1:
                            fim = pos + len(sep)
                            break
                chunk = texto[inicio:fim].strip()
                if chunk:
                    chunks.append(chunk)
                inicio = fim - CHUNK_OVERLAP
            return chunks

    def _gerar_embeddings_batch(self, textos: List[str]) -> List[List[float]]:
        client = self._get_openai_client()
        tokenizer = self._get_tokenizer()
        MAX_TOKENS_PER_REQUEST = 300_000

        textos_limpos = [t.replace("\n", " ").strip()[:8000] for t in textos]

        batches = []
        current_batch = []
        current_tokens = 0

        for texto in textos_limpos:
            tokens = len(tokenizer.encode(texto))
            if current_tokens + tokens > MAX_TOKENS_PER_REQUEST and current_batch:
                batches.append(current_batch)
                current_batch = [texto]
                current_tokens = tokens
            else:
                current_batch.append(texto)
                current_tokens += tokens
        if current_batch:
            batches.append(current_batch)

        all_embeddings = []
        for batch in batches:
            try:
                response = client.embeddings.create(
                    input=batch,
                    model=OPENAI_EMBEDDING_MODEL
                )
                all_embeddings.extend([item.embedding for item in response.data])
            except Exception as e:
                print(f"   Erro ao gerar embeddings em batch: {e}")
                raise
        return all_embeddings

    def _save_chunk_to_file(self, text: str, url: str, chunk_index: int) -> str:
        """Salva o texto do chunk em arquivo e retorna o caminho."""
        # Cria um nome único baseado na URL e índice
        url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
        filename = f"{url_hash}_{chunk_index}.txt"
        file_path = os.path.join(CHUNKS_CACHE_DIR, filename)
        # Evita sobrescrever se já existir (pode ser chamado novamente para mesmo chunk)
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)
        return file_path

    def _read_chunk_from_file(self, file_path: str) -> str:
        """Lê o texto do chunk do arquivo."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"   Erro ao ler chunk de {file_path}: {e}")
            return ""

    def url_exists(self, url: str) -> bool:
        collection = self._get_collection()
        return collection.count_documents({"url": url}, limit=1) > 0

    def build(self, extraidos: List[dict], snippets: List[dict], prefixo: str = "secao") -> "CorpusMongoDB":
        collection = self._get_collection()
        self._urls_usadas = []
        self._fonte_map = {}
        self._n_docs = 0
        self._total_chunks = 0

        documentos_para_inserir = []
        fonte_idx = 1

        # Processa extraídos
        for item in extraidos:
            url = item.get("url", "")
            if self.url_exists(url):
                print(f"      ⏭️ URL já indexada: {url[:60]}")
                self._urls_usadas.append(url)
                self._fonte_map[fonte_idx] = url
                fonte_idx += 1
                continue

            titulo = item.get("titulo") or url[:80]
            conteudo = item.get("conteudo", "")
            if not conteudo or not url:
                continue

            self._urls_usadas.append(url)
            self._fonte_map[fonte_idx] = url

            chunks_txt = self._chunkar(conteudo)
            if not chunks_txt:
                fonte_idx += 1
                continue

            try:
                embeddings = self._gerar_embeddings_batch(chunks_txt)
            except Exception as e:
                print(f"   Falha ao gerar embeddings para {url[:60]}, pulando. Erro: {e}")
                fonte_idx += 1
                continue

            # Salva cada chunk em arquivo e prepara documento
            for i, (chunk_text, emb) in enumerate(zip(chunks_txt, embeddings)):
                file_path = self._save_chunk_to_file(chunk_text, url, i)
                documentos_para_inserir.append({
                    "file_path": file_path,
                    "embedding": emb,
                    "url": url,
                    "titulo": titulo,
                    "fonte_idx": fonte_idx,
                    "tipo": "extraido",
                    "chunk_id": f"{url}_{i}",
                })

            print(f"      📄 [{fonte_idx}] {url[:60]} ({len(conteudo):,}c -> {len(chunks_txt)} chunks)")
            fonte_idx += 1
            self._n_docs += 1
            self._total_chunks += len(chunks_txt)

        # Processa snippets
        for s in snippets:
            url = s.get("url", "")
            if self.url_exists(url):
                print(f"      ⏭️ snippet já indexado: {url[:55]}")
                self._urls_usadas.append(url)
                self._fonte_map[fonte_idx] = url
                fonte_idx += 1
                continue

            titulo = s.get("title", "")
            snippet = s.get("snippet", "")[:600]
            if not snippet or not url or url in self._urls_usadas:
                continue

            score = score_url(url, snippet, float(s.get("score", 0)))
            if score < SNIPPET_MIN_SCORE:
                print(f"      ⛔ snippet ignorado (score={score:.1f}): {url[:55]}")
                continue

            self._urls_usadas.append(url)
            self._fonte_map[fonte_idx] = url

            texto_snip = f"[SNIPPET | {url[:60]}]\n{snippet}"
            try:
                emb = self._gerar_embeddings_batch([texto_snip])[0]
            except Exception as e:
                print(f"   Falha ao gerar embedding para snippet {url[:55]}: {e}")
                fonte_idx += 1
                continue

            file_path = self._save_chunk_to_file(texto_snip, url, 0)
            documentos_para_inserir.append({
                "file_path": file_path,
                "embedding": emb,
                "url": url,
                "titulo": titulo,
                "fonte_idx": fonte_idx,
                "tipo": "snippet",
                "chunk_id": f"snippet_{url}_{fonte_idx}",
            })
            fonte_idx += 1
            self._total_chunks += 1

        if documentos_para_inserir:
            try:
                collection.insert_many(documentos_para_inserir, ordered=False)
                print(f"      💾 {len(documentos_para_inserir)} novos chunks inseridos no MongoDB.")
            except Exception as e:
                print(f"      Erro na inserção: {e}")

        self._n_docs = len(set(d["url"] for d in documentos_para_inserir + [{"url": u} for u in self._urls_usadas]))
        print(f"      {self._n_docs} documentos totais | {self._total_chunks} chunks nesta seção")
        return self

    def query(self, texto_query: str, top_k: int = TOP_K_WRITER) -> List[Chunk]:
        collection = self._get_collection()
        client = self._get_openai_client()

        # Gera embedding da consulta
        try:
            emb = self._gerar_embeddings_batch([texto_query])[0]
        except Exception as e:
            print(f"   ❌ Erro ao gerar embedding da query: {e}")
            return []

        # Pipeline de busca vetorial
        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": emb,
                    "numCandidates": top_k * 10,
                    "limit": top_k,
                }
            },
            {
                "$project": {
                    "file_path": 1,
                    "url": 1,
                    "titulo": 1,
                    "fonte_idx": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        try:
            results = list(collection.aggregate(pipeline))
            print(f"      🔍 Query retornou {len(results)} resultados do MongoDB")
        except Exception as e:
            print(f"   ❌ Erro na busca vetorial: {e}")
            return []

        if not results:
            print("      ⚠️ Nenhum resultado encontrado. Verifique se:")
            print("         - O índice vetorial '{}' existe e está ativo".format(VECTOR_INDEX_NAME))
            print("         - A coleção contém documentos com embeddings")
            return []

        chunks = []
        for r in results:
            file_path = r.get("file_path")
            if not file_path:
                print(f"      ⚠️ Documento sem file_path: {r.get('url', 'desconhecido')}")
                continue

            if not os.path.exists(file_path):
                print(f"      ⚠️ Arquivo não encontrado: {file_path}")
                texto = ""
            else:
                texto = self._read_chunk_from_file(file_path)

            chunks.append(Chunk(
                chunk_idx=r.get("_id", ""),
                texto=texto,
                url=r.get("url", ""),
                titulo=r.get("titulo", ""),
                fonte_idx=r.get("fonte_idx", 0),
                file_path=file_path
            ))

        return chunks

    def get_neighbors(
        self,
        chunk: Chunk,
        window: int = 1,
        include_self: bool = True,
    ) -> List[Chunk]:
        """
        Dado um chunk de referência, retorna seus vizinhos no mesmo documento.

        Args:
            chunk:        Chunk de referência (precisa ter url e chunk_idx).
            window:       Quantos chunks buscar para cada lado.
            include_self: Se True, inclui o próprio chunk no resultado.

        Returns:
            Lista ordenada por chunk_idx (anterior → referência → posterior).
        """
        collection = self._get_collection()

        idx = chunk.chunk_idx  # garanta que Chunk tem esse campo
        idx_min = idx - window
        idx_max = idx + window

        try:
            cursor = collection.find(
                {
                    "url": chunk.url,
                    "chunk_idx": {"$gte": idx_min, "$lte": idx_max},
                },
                {"file_path": 1, "url": 1, "titulo": 1, "fonte_idx": 1, "chunk_idx": 1},
            ).sort("chunk_idx", 1)

            docs = list(cursor)
            print(f"      📎 {len(docs)} chunks encontrados (janela ±{window} em '{chunk.url}')")
        except Exception as e:
            print(f"   ❌ Erro ao buscar vizinhos: {e}")
            return []

        results = []
        for doc in docs:
            if not include_self and doc.get("chunk_idx") == idx:
                continue

            fp = doc.get("file_path", "")
            if fp and os.path.exists(fp):
                texto = self._read_chunk_from_file(fp)
            else:
                print(f"      ⚠️ Arquivo não encontrado: {fp}")
                texto = ""

            results.append(Chunk(
                texto=texto,
                url=doc.get("url", ""),
                titulo=doc.get("titulo", ""),
                fonte_idx=doc.get("fonte_idx", 0),
                file_path=fp,
                chunk_idx=doc.get("chunk_idx", 0),
            ))

        return results

    def get_url_chunks(self, url: str, max_chunks: int = 12) -> List[Chunk]:
        """
        Retrieve all stored chunks for a URL, sorted by their chunk index.

        Used to provide surrounding document context during verification.
        Chunks are sorted by the integer suffix in their chunk_id (e.g. 'url_3').

        Args:
            url:        URL to fetch all chunks for.
            max_chunks: Maximum number of chunks to return.

        Returns:
            List of Chunk objects sorted by chunk position within the document.
        """
        collection = self._get_collection()
        try:
            cursor = collection.find(
                {"url": url},
                {"file_path": 1, "url": 1, "titulo": 1, "fonte_idx": 1, "chunk_id": 1},
            )
            docs = list(cursor)
        except Exception as e:
            print(f"   ❌ Erro ao buscar chunks por URL: {e}")
            return []

        def _sort_key(doc: dict) -> int:
            chunk_id = doc.get("chunk_id", "")
            try:
                return int(str(chunk_id).rsplit("_", 1)[-1])
            except (ValueError, IndexError):
                return 0

        docs.sort(key=_sort_key)
        docs = docs[:max_chunks]

        results = []
        for doc in docs:
            fp = doc.get("file_path", "")
            texto = self._read_chunk_from_file(fp) if fp and os.path.exists(fp) else ""
            results.append(Chunk(
                chunk_idx=str(doc.get("_id", "")),
                texto=texto,
                url=doc.get("url", ""),
                titulo=doc.get("titulo", ""),
                fonte_idx=doc.get("fonte_idx", 0),
                file_path=fp,
            ))
        return results

    def anchor_exists(self, ancora: str) -> tuple:
        if not ancora or len(ancora.strip()) < 15:
            return False, 0.0, ""

        ancora_norm = normalizar(ancora)
        candidatos = self.query(ancora, top_k=TOP_K_VERIFICATION)

        for c in candidatos:
            if ancora_norm in normalizar(c.texto):
                return True, 1.0, c.texto

        melhor_score, melhor_trecho = 0.0, ""
        for c in candidatos:
            score = fuzzy_sim(ancora_norm, normalizar(c.texto))
            if score > melhor_score:
                melhor_score = score
                melhor_trecho = c.texto

        encontrada = melhor_score >= ANCHOR_MIN_SIM
        return encontrada, melhor_score, melhor_trecho

    def render_prompt(self, query: str, max_chars: int = MAX_CORPUS_PROMPT) -> tuple:
        chunks = self.query(query, top_k=TOP_K_WRITER)

        if not chunks:
            return "", self._urls_usadas, self._fonte_map

        partes = []
        urls_render = []
        chars = 0
        fontes_vistas = set()

        for chunk in chunks:
            if chunk.fonte_idx not in fontes_vistas:
                fontes_vistas.add(chunk.fonte_idx)
                titulo = chunk.titulo or ""
                cab = f"{'━'*55}\nFONTE [{chunk.fonte_idx}] — {titulo}\nURL: {chunk.url}\n{'─'*55}\n"
            else:
                cab = f"[cont. FONTE {chunk.fonte_idx} URL: {chunk.url}]\n"

            bloco = cab + chunk.texto + "\n\n"
            if chars + len(bloco) > max_chars:
                break

            partes.append(bloco)
            if chunk.url not in urls_render:
                urls_render.append(chunk.url)
            chars += len(bloco)

        contexto = "".join(partes)
        print(f"      📨 {len(chunks)} chunks | {len(fontes_vistas)} fontes | {chars:,} chars")
        return contexto, urls_render, self._fonte_map
    
    def render_prompt_url(
        self,
        texto_ancora: str,
        url_citada: str,
        max_chars: int = 3000,
        top_k: int = 5,
        include_neighbors: bool = False,
        neighbor_window: int = 2,
    ) -> Tuple[str, List[str], int]:
        """
        Renderiza prompt para verificação baseado em âncora + URL específica.

        Args:
            texto_ancora:      texto literal da âncora (copiado do corpus)
            url_citada:        URL da fonte citada
            max_chars:         máximo de caracteres no prompt
            top_k:             número de chunks primários a buscar
            include_neighbors: se True, adiciona chunks vizinhos do mesmo documento
                               como [CONTEXTO VIZINHO] para detecção de anacronias
            neighbor_window:   número de chunks extras do documento a incluir

        Returns:
            (prompt_texto, [urls_usadas], total_chunks_usados)
        """
        # Busca vetorial pelo texto da âncora
        chunks = self.query(texto_ancora, top_k=top_k * 2)

        # Filtra apenas chunks da URL citada
        chunks_da_url = [
            chunk for chunk in chunks
            if chunk.url.strip() == url_citada.strip()
        ]

        # Fallback: retorna chunks da busca geral se URL não encontrada
        if not chunks_da_url:
            print(f"   ⚠️  Nenhum chunk encontrado para URL: {url_citada[:60]}")
            chunks_da_url = chunks[:top_k]

        chunks_da_url = chunks_da_url[:top_k]

        partes: List[str] = []
        chars_acumulados = 0
        urls_usadas: List[str] = []

        for chunk in chunks_da_url:
            bloco = (
                f"[FONTE {chunk.fonte_idx} | {chunk.url[:70]}]\n"
                f"{chunk.texto}\n\n"
            )
            if chars_acumulados + len(bloco) > max_chars:
                break
            partes.append(bloco)
            chars_acumulados += len(bloco)
            if chunk.url not in urls_usadas:
                urls_usadas.append(chunk.url)

        # Adiciona contexto vizinho do mesmo documento para verificação temporal
        if include_neighbors and chunks_da_url:
            primary_texts = {c.texto for c in chunks_da_url}
            all_url_chunks = self.get_url_chunks(url_citada, max_chunks=20)
            neighbor_chunks = [
                c for c in all_url_chunks if c.texto not in primary_texts
            ][:neighbor_window]
            for nc in neighbor_chunks:
                bloco = (
                    f"[CONTEXTO VIZINHO — {nc.url[:70]}]\n"
                    f"{nc.texto}\n\n"
                )
                if chars_acumulados + len(bloco) > max_chars:
                    break
                partes.append(bloco)
                chars_acumulados += len(bloco)

        if not partes:
            return "", [], 0

        return "".join(partes), urls_usadas, len(chunks_da_url)


    # ============================================================================
    # VERSÃO ALTERNATIVA: Busca por múltiplas âncoras
    # ============================================================================

    def render_prompt_ancoras(
        self,
        ancoras_com_urls: List[Tuple[str, str]],
        max_chars: int = 3000,
    ) -> Tuple[str, List[str], int]:
        """
        Renderiza prompt baseado em múltiplas âncoras com suas URLs.
        
        Útil quando um parágrafo tem várias citações.
        
        Args:
            ancoras_com_urls: lista de (texto_ancora, url_citada)
            max_chars: máximo de caracteres no prompt
        
        Returns:
            (prompt_texto, [urls_usadas], total_chunks_usados)
        
        Exemplo:
            >>> corpus.render_prompt_ancoras([
            ...     ("100 épocas de treinamento", "https://arxiv.org/..."),
            ...     ("MSE is used as loss", "https://papers.nips.cc/..."),
            ... ])
        """
        partes = []
        chars_acumulados = 0
        urls_usadas = []
        chunks_usados = 0
        
        for texto_ancora, url_citada in ancoras_com_urls:
            # Busca chunks para cada âncora
            chunks = self.query(texto_ancora, top_k=3)
            
            # Filtra por URL
            chunks_da_url = [
                chunk for chunk in chunks
                if chunk.url.strip() == url_citada.strip()
            ]
            
            # Se não encontrou da URL específica, usa os melhores matches
            if not chunks_da_url:
                chunks_da_url = chunks[:2]
            
            # Adiciona ao prompt
            for chunk in chunks_da_url[:2]:  # Máx 2 chunks por âncora
                bloco = (
                    f"[FONTE {chunk.fonte_idx} | {chunk.url[:70]}]\n"
                    f"[ÂNCORA: {texto_ancora[:50]}...]\n"
                    f"{chunk.texto}\n\n"
                )
                
                if chars_acumulados + len(bloco) > max_chars:
                    break
                
                partes.append(bloco)
                chars_acumulados += len(bloco)
                chunks_usados += 1
                
                if chunk.url not in urls_usadas:
                    urls_usadas.append(chunk.url)
        
        if not partes:
            return "", [], 0
        
        prompt_final = "".join(partes)
        
        return prompt_final, urls_usadas, chunks_usados