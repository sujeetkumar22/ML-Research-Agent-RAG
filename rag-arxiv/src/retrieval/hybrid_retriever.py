"""
src/retrieval/hybrid_retriever.py

Combines dense (semantic) + BM25 (keyword) retrieval using
Reciprocal Rank Fusion (RRF). This is the key technique that
separates a production RAG from a tutorial demo.

Interview talking point:
- Dense retrieval excels at semantic similarity ("tell me about attention")
- BM25 excels at exact keyword matches ("LoRA", "BERT", author names, IDs)
- RRF combines both without needing to tune score weights
"""
import re
from rank_bm25 import BM25Okapi
from loguru import logger
from sentence_transformers import CrossEncoder
from src.retrieval.vector_store import VectorStore


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25 — lowercased, alpha-only tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    """
    Hybrid retriever = Dense (FAISS) + Sparse (BM25) + RRF fusion.

    Usage:
        retriever = HybridRetriever(vector_store, chunks)
        results = retriever.retrieve("what is LoRA fine-tuning?", top_k=5)
    """

    def __init__(self, vector_store: VectorStore, chunks: list[dict]):
        self.vector_store = vector_store
        self.chunks = chunks

        logger.info("Building BM25 index…")
        tokenized_corpus = [_tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.success(f"BM25 index ready ({len(chunks)} documents)")

        logger.info("Loading Cross-Encoder reranker model (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.success("Reranker model loaded successfully!")

    # ── BM25 SEARCH ──────────────────────────────────────────────────────────

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        """Return top_k chunks by BM25 score."""
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # Get indices of top-k non-zero scores
        top_indices = scores.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk = dict(self.chunks[idx])
            chunk["bm25_score"] = float(scores[idx])
            chunk["retrieval_method"] = "sparse"
            results.append(chunk)

        return results

    # ── RRF FUSION ───────────────────────────────────────────────────────────

    @staticmethod
    def _reciprocal_rank_fusion(
        dense_results: list[dict],
        sparse_results: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion (RRF).

        Formula: score(d) = Σ 1 / (k + rank(d))
        k=60 is the standard constant (from the original RRF paper).

        Why RRF instead of score normalization?
        - Score scales differ between BM25 and cosine similarity
        - RRF only uses *rank position*, not raw scores → no tuning needed
        - Empirically outperforms linear combination in most benchmarks
        """
        rrf_scores: dict[str, float] = {}
        chunk_by_key: dict[str, dict] = {}

        def key(chunk: dict) -> str:
            """Unique key per chunk using text fingerprint."""
            return chunk["text"][:100]

        for rank, chunk in enumerate(dense_results):
            ck = key(chunk)
            rrf_scores[ck] = rrf_scores.get(ck, 0.0) + 1.0 / (k + rank + 1)
            chunk_by_key[ck] = chunk

        for rank, chunk in enumerate(sparse_results):
            ck = key(chunk)
            rrf_scores[ck] = rrf_scores.get(ck, 0.0) + 1.0 / (k + rank + 1)
            if ck not in chunk_by_key:
                chunk_by_key[ck] = chunk

        # Sort by descending RRF score
        sorted_keys = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)

        fused = []
        for ck in sorted_keys:
            chunk = dict(chunk_by_key[ck])
            chunk["rrf_score"] = rrf_scores[ck]
            chunk["retrieval_method"] = "hybrid"
            fused.append(chunk)

        return fused

    # ── PUBLIC API ───────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int = 25,   # fetch more candidates before RRF
        use_reranker: bool = True,
    ) -> list[dict]:
        """
        Retrieve top_k most relevant chunks using hybrid search and reranking.

        Steps:
        1. Dense search → top fetch_k candidates
        2. BM25 search  → top fetch_k candidates
        3. RRF fusion   → combined ranked list
        4. Rerank top 20 candidates using Cross-Encoder
        5. Return top_k
        """
        # 1. Dense retrieval
        dense = self.vector_store.search(query, top_k=fetch_k)

        # 2. Sparse retrieval
        sparse = self._bm25_search(query, top_k=fetch_k)

        # 3. Fuse
        fused = self._reciprocal_rank_fusion(dense, sparse)

        # 4. Rerank if enabled
        if use_reranker and hasattr(self, "reranker") and self.reranker is not None:
            candidates = fused[:20]
            if candidates:
                pairs = [[query, c["text"]] for c in candidates]
                scores = self.reranker.predict(pairs)
                for idx, score in enumerate(scores):
                    candidates[idx]["rerank_score"] = float(score)
                # Sort by rerank score descending
                candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
                results = candidates[:top_k]
                logger.debug(
                    f"Hybrid retrieve + Rerank: {len(dense)} dense + {len(sparse)} sparse "
                    f"→ {len(fused)} fused → reranked top 20 → top {top_k} returned"
                )
            else:
                results = []
        else:
            results = fused[:top_k]
            logger.debug(
                f"Hybrid retrieve: {len(dense)} dense + {len(sparse)} sparse "
                f"→ {len(fused)} fused → top {top_k} returned"
            )

        return results

    def retrieve_with_scores(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Same as retrieve() but also includes score breakdown.
        Useful for the evaluation tab and debugging.
        """
        return self.retrieve(query, top_k=top_k)
