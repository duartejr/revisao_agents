import hashlib
import os

import pymongo
import tiktoken
from openai import OpenAI
from pymongo.collection import Collection

from ...config import (
    ANCHOR_MIN_SIM,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_CACHE_DIR,
    MAX_CORPUS_PROMPT,
    MONGODB_COLLECTION,
    MONGODB_DB,
    MONGODB_URI,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    SNIPPET_MIN_SCORE,
    TOP_K_VERIFICATION,
    TOP_K_WRITER,
    VECTOR_INDEX_NAME,
)
from ...core.schemas.corpus import Chunk
from ..file_utils.helpers import fuzzy_sim, normalize
from ..search_utils.tavily_client import score_url  # import local


class CorpusMongoDB:
    def __init__(self):
        self._client = None
        self._collection = None
        self._openai_client = None
        self._tokenizer = None
        self._used_urls: list[str] = []
        self._source_map: dict[int, str] = {}
        self._n_docs = 0
        self._total_chunks = 0
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
        )
        if os.path.isabs(CHUNKS_CACHE_DIR):
            self._chunks_cache_dir = CHUNKS_CACHE_DIR
        else:
            self._chunks_cache_dir = os.path.abspath(os.path.join(project_root, CHUNKS_CACHE_DIR))
        os.makedirs(self._chunks_cache_dir, exist_ok=True)

    def _get_collection(self) -> Collection:
        """Establishes and returns the MongoDB collection connection.

        Args:
            None

        Returns:
            pymongo Collection object for the configured MongoDB Atlas collection."""
        if self._collection is not None:
            return self._collection
        if not MONGODB_URI:
            raise RuntimeError("MONGODB_URI not defined.")
        self._client = pymongo.MongoClient(MONGODB_URI)
        db = self._client[MONGODB_DB]
        self._collection = db[MONGODB_COLLECTION]
        self._client.admin.command("ping")
        print("   Connected to MongoDB Atlas.")
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
        """Initializes and returns the OpenAI client for embedding generation.

        Args:
            None

        Returns:
            OpenAI client instance configured with the API key.

        Raises:
            RuntimeError: If OPENAI_API_KEY is not defined.
        """
        if self._openai_client is None:
            if not OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY not defined.")
            self._openai_client = OpenAI(api_key=OPENAI_API_KEY)
        return self._openai_client

    def _get_tokenizer(self):
        """Initializes and returns the tokenizer for the embedding model.
        Caches the tokenizer instance for reuse.

        Returns:
            Tokenizer instance compatible with the embedding model.
        """
        if self._tokenizer is None:
            self._tokenizer = tiktoken.encoding_for_model(OPENAI_EMBEDDING_MODEL)
        return self._tokenizer

    @staticmethod
    def _chunker(text: str) -> list[str]:
        """Splits the input text into smaller chunks for processing.

        Args:
            text (str): The input text to be chunked.

        Returns:
            List[str]: A list of text chunks.
        """
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
            )
            return splitter.split_text(text)
        except ImportError:
            # simple fallback if langchain is not available
            chunks, start = [], 0
            while start < len(text):
                end = min(start + CHUNK_SIZE, len(text))
                if end < len(text):
                    for sep in ("\n\n", "\n", ". ", " "):
                        pos = text.rfind(sep, start + CHUNK_SIZE // 2, end)
                        if pos != -1:
                            end = pos + len(sep)
                            break
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                start = end - CHUNK_OVERLAP
            return chunks

    def _generate_batch_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generates embeddings for a batch of texts using the OpenAI API.

        Args:
            texts (List[str]): A list of texts to generate embeddings for.

        Returns:
            List[List[float]]: A list of embeddings corresponding to the input texts.

        Raises:
            RuntimeError: If there is an error generating embeddings.
        """
        client = self._get_openai_client()
        tokenizer = self._get_tokenizer()
        MAX_TOKENS_PER_REQUEST = 300_000

        cleaned_texts = [t.replace("\n", " ").strip()[:8000] for t in texts]

        batches = []
        current_batch = []
        current_tokens = 0

        for text in cleaned_texts:
            tokens = len(tokenizer.encode(text))
            if current_tokens + tokens > MAX_TOKENS_PER_REQUEST and current_batch:
                batches.append(current_batch)
                current_batch = [text]
                current_tokens = tokens
            else:
                current_batch.append(text)
                current_tokens += tokens
        if current_batch:
            batches.append(current_batch)

        all_embeddings = []
        for batch in batches:
            try:
                response = client.embeddings.create(input=batch, model=OPENAI_EMBEDDING_MODEL)
                all_embeddings.extend([item.embedding for item in response.data])
            except Exception as e:
                print(f"   Error generating embeddings in batch.: {e}")
                raise
        return all_embeddings

    def _save_chunk_to_file(self, text: str, url: str, chunk_index: int) -> str:
        """Saves the chunk text to a file and returns the file path.

        Args:
            text (str): The chunk text to be saved.
            url (str): The source URL of the chunk, used for naming.
            chunk_index (int): The index of the chunk within the document.

        Returns:
            str: The file path where the chunk text is saved.
        """
        # Create a unique name based on the URL and index to avoid collisions
        url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
        filename = f"{url_hash}_{chunk_index}.txt"
        file_path = os.path.join(self._chunks_cache_dir, filename)
        # Avoid overwriting if it already exists (can be called again for the same chunk)
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)
        return file_path

    def _read_chunk_from_file(self, file_path: str) -> str:
        """Reads the chunk text from a file.

        Args:
            file_path (str): The path to the file containing the chunk text.

        Returns:
            str: The chunk text read from the file.
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"   Error reading chunk from {file_path}: {e}")
            return ""

    def url_exists(self, url: str) -> bool:
        """Checks if a URL already exists in the MongoDB collection.

        Args:
            url (str): The URL to check for existence in the collection.

        Returns:
            bool: True if the URL exists, False otherwise.
        """
        collection = self._get_collection()
        return collection.count_documents({"url": url}, limit=1) > 0

    def build(
        self,
        extracted_documents: list[dict],
        snippets: list[dict],
        prefix: str = "section",
    ) -> "CorpusMongoDB":
        """Builds the MongoDB corpus by processing extracted documents and snippets, generating embeddings, and storing them in the collection.

        Args:
            extracted_documents (List[dict]): A list of extracted documents to be processed.
            snippets (List[dict]): A list of snippets to be processed.
            prefix (str, optional): A prefix for the section names. Defaults to "section".

        Returns:
            CorpusMongoDB: The instance of the CorpusMongoDB with the built corpus.
        """
        collection = self._get_collection()
        self._used_urls = []
        self._source_map = {}
        self._n_docs = 0
        self._total_chunks = 0

        documents_to_insert = []
        source_idx = 1

        # Process extracted documents
        for item in extracted_documents:
            url = item.get("url", "")
            if self.url_exists(url):
                print(f"      ⏭️ URL already indexed: {url[:120]}")
                self._used_urls.append(url)
                self._source_map[source_idx] = url
                source_idx += 1
                continue

            title = item.get("title") or url[:160]
            content = item.get("content", "")
            if not content or not url:
                continue

            self._used_urls.append(url)
            self._source_map[source_idx] = url

            chunks_txt = self._chunker(content)
            if not chunks_txt:
                source_idx += 1
                continue

            try:
                embeddings = self._generate_batch_embeddings(chunks_txt)
            except Exception as e:
                print(f"   Failed to generate embeddings for {url[:100]}, skipping. Error: {e}")
                source_idx += 1
                continue

            # Save each chunk to a file and prepare the document for MongoDB insertion
            for i, (chunk_text, emb) in enumerate(zip(chunks_txt, embeddings, strict=False)):
                file_path = self._save_chunk_to_file(chunk_text, url, i)
                documents_to_insert.append(
                    {
                        "file_path": file_path,
                        "embedding": emb,
                        "url": url,
                        "title": title,
                        "source_idx": source_idx,
                        "type": "extracted",
                        "chunk_id": f"{url}_{i}",
                    }
                )

            print(
                f"      📄 [{source_idx}] {url[:120]} ({len(content):,}c -> {len(chunks_txt)} chunks)"
            )
            source_idx += 1
            self._n_docs += 1
            self._total_chunks += len(chunks_txt)

        # Process snippets
        for s in snippets:
            url = s.get("url", "")
            if self.url_exists(url):
                print(f"      ⏭️ snippet already indexed: {url[:110]}")
                self._used_urls.append(url)
                self._source_map[source_idx] = url
                source_idx += 1
                continue

            title = s.get("title", "") or url[:160]
            snippet = s.get("snippet", "")[:600]
            if not snippet or not url or url in self._used_urls:
                continue

            score = score_url(url, snippet, float(s.get("score", 0)))
            if score < SNIPPET_MIN_SCORE:
                print(f"      ⛔ snippet ignored (score={score:.1f}): {url[:110]}")
                continue

            self._used_urls.append(url)
            self._source_map[source_idx] = url

            texto_snip = f"[SNIPPET | {url[:120]}]\n{snippet}"
            try:
                emb = self._generate_batch_embeddings([texto_snip])[0]
            except Exception as e:
                print(f"   Failed to generate embedding for snippet {url[:110]}: {e}")
                source_idx += 1
                continue

            file_path = self._save_chunk_to_file(texto_snip, url, 0)
            documents_to_insert.append(
                {
                    "file_path": file_path,
                    "embedding": emb,
                    "url": url,
                    "title": title,
                    "source_idx": source_idx,
                    "type": "snippet",
                    "chunk_id": f"snippet_{url}_{source_idx}",
                }
            )
            source_idx += 1
            self._total_chunks += 1

        if documents_to_insert:
            try:
                collection.insert_many(documents_to_insert, ordered=False)
                print(f"      💾 {len(documents_to_insert)} new chunks inserted into MongoDB.")
            except Exception as e:
                print(f"      ❌ Error inserting documents: {e}")

        self._n_docs = len(
            set(d["url"] for d in documents_to_insert + [{"url": u} for u in self._used_urls])
        )
        print(f"      {self._n_docs} total documents | {self._total_chunks} chunks in this section")
        return self

    def query(self, texto_query: str, top_k: int = TOP_K_WRITER) -> list[Chunk]:
        """Searches for chunks similar to the query using MongoDB Atlas Vector Search.
        Generates query embedding via OpenAI and retrieves the most relevant chunks.

        Args:
            texto_query (str): The search query text.
            top_k (int, optional): The number of top similar chunks to return. Defaults to TOP_K_WRITER.

        Returns:
            List[Chunk]: A list of chunks similar to the query.
        """
        collection = self._get_collection()

        # Generate embedding for the query
        try:
            emb = self._generate_batch_embeddings([texto_query])[0]
        except Exception as e:
            print(f"   ❌ Error generating embedding for query: {e}")
            return []

        # Vector search pipeline
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
                    "title": 1,
                    "source_idx": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        try:
            results = list(collection.aggregate(pipeline))
            print(f"      🔍 Query returned {len(results)} results from MongoDB")
        except Exception as e:
            print(f"   ❌ Error in vector search: {e}")
            return []

        if not results:
            print("      ⚠️ No results found. Check if:")
            print(f"         - The vector index '{VECTOR_INDEX_NAME}' exists and is active")
            print("         - The collection contains documents with embeddings")
            return []

        chunks = []
        for r in results:
            file_path = r.get("file_path")
            if not file_path:
                print(f"      ⚠️ Document without file_path: {r.get('url', 'unknown url')}")
                continue

            if not os.path.exists(file_path):
                print(f"      ⚠️ File not found: {file_path}")
                text = ""
            else:
                text = self._read_chunk_from_file(file_path)

            chunks.append(
                Chunk(
                    chunk_idx=r.get("_id", ""),
                    text=text,
                    url=r.get("url", ""),
                    title=r.get("title", ""),
                    source_idx=r.get("source_idx", 0),
                    file_path=file_path,
                )
            )

        return chunks

    def get_neighbors(
        self,
        chunk: Chunk,
        window: int = 1,
        include_self: bool = True,
    ) -> list[Chunk]:
        """
        Given a reference chunk, returns its neighbors in the same document.

        Args:
            chunk: Reference chunk (must have URL and chunk_idx).
            window: How many chunks to search for on each side.
            include_self: If True, includes the chunk itself in the result.

        Returns:
            List ordered by chunk_idx (previous → reference → next).
        """
        collection = self._get_collection()

        idx = chunk.chunk_idx  # Make sure Chunk has this field.
        idx_min = idx - window
        idx_max = idx + window

        try:
            cursor = collection.find(
                {
                    "url": chunk.url,
                    "chunk_idx": {"$gte": idx_min, "$lte": idx_max},
                },
                {"file_path": 1, "url": 1, "title": 1, "source_idx": 1, "chunk_idx": 1},
            ).sort("chunk_idx", 1)

            docs = list(cursor)
            print(f"      📎 {len(docs)} chunks encontrados (janela ±{window} em '{chunk.url}')")
        except Exception as e:
            print(f"   ❌ Error searching for neighbors: {e}")
            return []

        results = []
        for doc in docs:
            if not include_self and doc.get("chunk_idx") == idx:
                continue

            fp = doc.get("file_path", "")
            if fp and os.path.exists(fp):
                text = self._read_chunk_from_file(fp)
            else:
                print(f"      ⚠️ File not found: {fp}")
                text = ""

            results.append(
                Chunk(
                    text=text,
                    url=doc.get("url", ""),
                    title=doc.get("title", ""),
                    source_idx=doc.get("source_idx", 0),
                    file_path=fp,
                    chunk_idx=doc.get("chunk_idx", 0),
                )
            )

        return results

    def get_url_chunks(self, url: str, max_chunks: int = 12) -> list[Chunk]:
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
                {"file_path": 1, "url": 1, "title": 1, "source_idx": 1, "chunk_id": 1},
            )
            docs = list(cursor)
        except Exception as e:
            print(f"   ❌ Error fetching chunks by URL: {e}")
            return []

        def _sort_key(doc: dict) -> int:
            """Extracts the integer suffix from the chunk_id for sorting.

            Args:
                doc: A document from MongoDB containing a 'chunk_id' field.
            Returns:
                The integer index extracted from the 'chunk_id', or 0 if parsing fails.
            """
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
            text = self._read_chunk_from_file(fp) if fp and os.path.exists(fp) else ""
            results.append(
                Chunk(
                    chunk_idx=str(doc.get("_id", "")),
                    text=text,
                    url=doc.get("url", ""),
                    title=doc.get("title", ""),
                    source_idx=doc.get("source_idx", 0),
                    file_path=fp,
                )
            )
        return results

    def anchor_exists(self, anchor: str) -> tuple:
        """
        Check if an anchor exists in the corpus.

        Args:
            anchor: The anchor text to search for.

        Returns:
            A tuple (found: bool, score: float, text: str) indicating whether the anchor was found,
            the similarity score, and the matching text.
        """
        if not anchor or len(anchor.strip()) < 15:
            return False, 0.0, ""

        anchor_norm = normalize(anchor)
        candidates = self.query(anchor, top_k=TOP_K_VERIFICATION)

        for c in candidates:
            if anchor_norm in normalize(c.text):
                return True, 1.0, c.text

        best_score, best_text = 0.0, ""
        for c in candidates:
            score = fuzzy_sim(anchor_norm, normalize(c.text))
            if score > best_score:
                best_score = score
                best_text = c.text

        found = best_score >= ANCHOR_MIN_SIM
        return found, best_score, best_text

    def render_prompt(self, query: str, max_chars: int = MAX_CORPUS_PROMPT) -> tuple:
        """Renders a prompt by retrieving relevant chunks from the corpus based on a query.
        Retrieves chunks using vector search and concatenates them until the max character limit is reached.

        Args:
            query (str): The input query to search for relevant chunks.
            max_chars (int, optional): The maximum number of characters for the rendered prompt. Defaults to MAX_CORPUS_PROMPT.

        Returns:
            A tuple containing the rendered prompt text, a list of URLs used in the prompt, and a mapping of source indices to URLs.
        """
        chunks = self.query(query, top_k=TOP_K_WRITER)

        if not chunks:
            return "", self._used_urls, self._source_map

        parts = []
        urls_render = []
        chars = 0
        sources_viewed = set()

        for chunk in chunks:
            if chunk.source_idx not in sources_viewed:
                sources_viewed.add(chunk.source_idx)
                title = chunk.title or ""
                cab = f"{'━' * 55}\nSOURCE [{chunk.source_idx}] — {title}\nURL: {chunk.url}\n{'─' * 55}\n"
            else:
                cab = f"[cont. SOURCE {chunk.source_idx} URL: {chunk.url}]\n"

            block = cab + chunk.text + "\n\n"
            if chars + len(block) > max_chars:
                break

            parts.append(block)
            if chunk.url not in urls_render:
                urls_render.append(chunk.url)
            chars += len(block)

        context = "".join(parts)
        print(f"      📨 {len(chunks)} chunks | {len(sources_viewed)} sources | {chars:,} chars")
        return context, urls_render, self._source_map

    def render_prompt_url(
        self,
        anchor_text: str,
        cited_urls: str,
        max_chars: int = 3000,
        top_k: int = 5,
        include_neighbors: bool = False,
        neighbor_window: int = 2,
    ) -> tuple[str, list[str], int]:
        """
        Renders prompt for verification based on anchor + specific URL.

        Args:
            anchor_text: literal text of the anchor (copied from the corpus)
            cited_urls: URL of the cited source
            max_chars: maximum number of characters in the prompt
            top_k: number of primary chunks to fetch
            include_neighbors: if True, adds neighboring chunks from the same document
            as [NEIGHBORING CONTEXT] for anachronism detection
            neighbor_window: number of extra chunks from the document to include

        Returns:
            (prompt_text, [urls_used], total_chunks_used)
        """
        # Vector search for anchor text to get candidate chunks
        chunks = self.query(anchor_text, top_k=top_k * 2)

        # Filter only chunks from the quoted URL.
        chunks_of_url = [chunk for chunk in chunks if chunk.url.strip() == cited_urls.strip()]

        # Fallback: returns chunks from the overall search if URL not found.
        if not chunks_of_url:
            print(f"   ⚠️  No chunk found for URL: {cited_urls[:120]}")
            chunks_of_url = chunks[:top_k]

        chunks_of_url = chunks_of_url[:top_k]

        parts: list[str] = []
        accumulated_chars = 0
        used_urls: list[str] = []

        for chunk in chunks_of_url:
            block = f"[SOURCE {chunk.source_idx} | {chunk.url[:140]}]\n{chunk.text}\n\n"
            if accumulated_chars + len(block) > max_chars:
                break
            parts.append(block)
            accumulated_chars += len(block)
            if chunk.url not in used_urls:
                used_urls.append(chunk.url)

        # Adds neighboring context from the same document for temporal verification.
        if include_neighbors and chunks_of_url:
            primary_texts = {c.text for c in chunks_of_url}
            all_url_chunks = self.get_url_chunks(cited_urls, max_chunks=20)
            neighbor_chunks = [c for c in all_url_chunks if c.text not in primary_texts][
                :neighbor_window
            ]
            for nc in neighbor_chunks:
                block = f"[NEIGHBORING CONTEXT — {nc.url[:140]}]\n{nc.text}\n\n"
                if accumulated_chars + len(block) > max_chars:
                    break
                parts.append(block)
                accumulated_chars += len(block)

        if not parts:
            return "", [], 0

        return "".join(parts), used_urls, len(chunks_of_url)

    # ============================================================================
    # ALTERNATIVE VERSION: Search for multiple anchors
    # ============================================================================

    def render_prompt_anchors(
        self,
        anchors_with_urls: list[tuple[str, str]],
        max_chars: int = 3000,
    ) -> tuple[str, list[str], int]:
        """
        Render prompt based on multiple anchors with their URLs.

        Useful when a paragraph has multiple citations.

        Args:
            anchors_with_urls: list of (anchor_text, cited_url) pairs
            max_chars: maximum number of characters in the prompt

        Returns:
            (prompt_text, [urls_used], total_chunks_used)

        Example usage:
            >>> corpus.render_prompt_anchors([
            ...     ("100 epochs of training", "https://arxiv.org/..."),
            ...     ("MSE is used as loss", "https://papers.nips.cc/..."),
            ... ])
        """
        parts = []
        accumulated_chars = 0
        used_urls = []
        chunks_used = 0

        for anchor_text, cited_url in anchors_with_urls:
            # Search for chunks for each anchor
            chunks = self.query(anchor_text, top_k=3)

            # Filter by URL
            chunks_of_url = [chunk for chunk in chunks if chunk.url.strip() == cited_url.strip()]

            # If not found for the specific URL, use the best matches
            if not chunks_of_url:
                chunks_of_url = chunks[:3]

            # Add to the prompt
            for chunk in chunks_of_url[:3]:  # Max 3 chunks per anchor
                block = (
                    f"[SOURCE {chunk.source_idx} | {chunk.url[:140]}]\n"
                    f"[ANCHOR: {anchor_text[:50]}...]\n"
                    f"{chunk.text}\n\n"
                )

                if accumulated_chars + len(block) > max_chars:
                    break

                parts.append(block)
                accumulated_chars += len(block)
                chunks_used += 1

                if chunk.url not in used_urls:
                    used_urls.append(chunk.url)

        if not parts:
            return "", [], 0

        prompt_final = "".join(parts)

        return prompt_final, used_urls, chunks_used
