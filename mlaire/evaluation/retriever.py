import hashlib
import logging
import os

import numpy as np
from tqdm import tqdm

from mlaire.models.base import BaseModel
from mlaire.types import RetrievalPool

logger = logging.getLogger(__name__)

RankedResult = tuple[str, str, str, float]  # (passage_id, group_id, language, score)
QueryResult = tuple[str, list[RankedResult]]  # (query_id, ranked_results)


class Retriever:
    def __init__(self, model: BaseModel, pool: RetrievalPool, cache_dir: str | None = None):
        self.model = model
        self.pool = pool
        self.cache_dir = cache_dir
        self._passage_embeddings: np.ndarray | None = None

    def _cache_key(self, prefix: str, ids: list[str]) -> str:
        """Generate cache key from model name and IDs."""
        content = f"{self.model.model_name}::{','.join(sorted(ids))}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _cache_path(self, prefix: str, ids: list[str]) -> str | None:
        if self.cache_dir is None:
            return None
        key = self._cache_key(prefix, ids)
        return os.path.join(self.cache_dir, f"{prefix}_{key}.npy")

    def _load_cache(self, cache_path: str | None) -> np.ndarray | None:
        if cache_path and os.path.exists(cache_path):
            logger.info(f"Loading cached embeddings from {cache_path}")
            return np.load(cache_path)
        return None

    def _save_cache(self, cache_path: str | None, embeddings: np.ndarray) -> None:
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, embeddings)
            logger.info(f"Saved embeddings cache to {cache_path}")

    def _encode_passages(self) -> np.ndarray:
        if self._passage_embeddings is None:
            passage_ids = [p.passage_id for p in self.pool.passages]
            cache_path = self._cache_path("passages", passage_ids)
            cached = self._load_cache(cache_path)
            if cached is not None:
                self._passage_embeddings = cached
            else:
                texts = [p.text for p in self.pool.passages]
                self._passage_embeddings = self.model.encode_passages(texts)
                self._save_cache(cache_path, self._passage_embeddings)
        return self._passage_embeddings

    def _encode_queries(self) -> np.ndarray:
        query_ids = [q.query_id for q in self.pool.queries]
        cache_path = self._cache_path("queries", query_ids)
        cached = self._load_cache(cache_path)
        if cached is not None:
            return cached
        query_texts = [q.text for q in self.pool.queries]
        query_embs = self.model.encode_queries(query_texts)
        self._save_cache(cache_path, query_embs)
        return query_embs

    def retrieve_all(self, query_chunk_size: int = 256, top_k: int = 200) -> list[QueryResult]:
        """Retrieve top-k passages for each query.

        Parameters
        ----------
        query_chunk_size:
            Number of queries to process at once (controls peak GPU/RAM for similarity matrix).
        top_k:
            Only keep the top-k ranked passages per query. This is critical for memory:
            without it, 67K queries × 56K passages = 376 GB of Python tuples.
            Default 200 is sufficient for metrics up to @100.
        """
        passage_embs = self._encode_passages()
        query_embs = self._encode_queries()

        # Ensure float32 for CPU matrix multiplication.
        # float16 numpy matmul falls back to a scalar loop (~2000x slower than float32 BLAS).
        if passage_embs.dtype != np.float32:
            passage_embs = passage_embs.astype(np.float32)
        if query_embs.dtype != np.float32:
            query_embs = query_embs.astype(np.float32)

        # Normalize for cosine similarity.
        # If the model already normalizes, this is a no-op (normalizing unit vectors is idempotent).
        passage_norms = passage_embs / (np.linalg.norm(passage_embs, axis=1, keepdims=True) + 1e-9)
        query_norms = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-9)

        results: list[QueryResult] = []
        num_queries = len(self.pool.queries)
        num_passages = len(self.pool.passages)
        effective_k = min(top_k, num_passages)

        for start in tqdm(range(0, num_queries, query_chunk_size), desc="Retrieving"):
            end = min(start + query_chunk_size, num_queries)
            chunk_sims = query_norms[start:end] @ passage_norms.T  # (chunk, num_passages)

            for i, query in enumerate(self.pool.queries[start:end]):
                scores = chunk_sims[i]
                # Use argpartition + argsort for O(n) partial sort instead of O(n log n) full sort
                # Fallback to full argsort when effective_k == num_passages (argpartition kth out of bounds)
                if effective_k >= num_passages:
                    top_indices = np.argsort(-scores)
                else:
                    top_indices = np.argpartition(-scores, effective_k)[:effective_k]
                    top_indices = top_indices[np.argsort(-scores[top_indices])]

                ranked: list[RankedResult] = []
                for idx in top_indices:
                    p = self.pool.passages[idx]
                    ranked.append((p.passage_id, p.group_id, p.language, float(scores[idx])))

                results.append((query.query_id, ranked))

        return results
