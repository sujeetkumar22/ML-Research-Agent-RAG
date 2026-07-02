"""
src/retrieval/vector_store.py

Builds and searches a FAISS vector index using sentence-transformers embeddings.
Handles save/load so you don't re-embed every time you restart.
"""
import faiss
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from loguru import logger
from config import EMBED_MODEL, VECTOR_STORE_PATH


class VectorStore:
    """
    FAISS-backed vector store with sentence-transformer embeddings.

    Interview talking point:
    - We use IndexFlatIP (inner product) with normalized vectors
      which is equivalent to cosine similarity — best for semantic search
    - normalize_embeddings=True means cosine sim == dot product,
      letting us use the fast FAISS IndexFlatIP
    """

    def __init__(self, model_name: str = EMBED_MODEL):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[dict] = []   # parallel to index vectors
        self._dim: int | None = None

    # ── BUILD ────────────────────────────────────────────────────────────────

    def build(self, chunks: list[dict], batch_size: int = 64) -> None:
        """Embed all chunks and build FAISS index."""
        if not chunks:
            raise ValueError("No chunks to embed!")

        texts = [c["text"] for c in chunks]
        self.chunks = chunks

        logger.info(f"Embedding {len(texts)} chunks (batch_size={batch_size})…")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,   # required for cosine via inner product
            convert_to_numpy=True,
        )
        embeddings = embeddings.astype(np.float32)
        self._dim = embeddings.shape[1]

        # Flat index = exact search (no approximation)
        # For 50–100 papers this is perfectly fast
        self.index = faiss.IndexFlatIP(self._dim)
        self.index.add(embeddings)

        logger.success(
            f"✅ Built index: {self.index.ntotal} vectors, dim={self._dim}"
        )

    # ── SEARCH ───────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search for the top_k most similar chunks.

        Returns list of chunk dicts with added 'dense_score' field.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() or load() first.")

        q_emb = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        scores, indices = self.index.search(q_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])   # copy
            chunk["dense_score"] = float(score)
            chunk["retrieval_method"] = "dense"
            results.append(chunk)

        return results

    # ── SAVE / LOAD ──────────────────────────────────────────────────────────

    def save(self, path: str = VECTOR_STORE_PATH) -> None:
        """Save index and chunks to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        logger.success(f"Saved vector store to {path}.*")

    def load(self, path: str = VECTOR_STORE_PATH) -> bool:
        """
        Load index from disk. Returns True if successful, False if not found.
        Call this on app startup to avoid re-embedding.
        """
        faiss_path = f"{path}.faiss"
        chunks_path = f"{path}.chunks.pkl"

        if not (Path(faiss_path).exists() and Path(chunks_path).exists()):
            logger.warning("No saved vector store found — need to build first")
            return False

        self.index = faiss.read_index(faiss_path)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        logger.success(
            f"Loaded vector store: {self.index.ntotal} vectors "
            f"({len(self.chunks)} chunks)"
        )
        return True

    # ── UTILS ────────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self.index is not None and len(self.chunks) > 0

    @property
    def num_vectors(self) -> int:
        return self.index.ntotal if self.index else 0
