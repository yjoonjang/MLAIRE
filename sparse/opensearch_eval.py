"""OpenSearch neural sparse evaluation using sentence-transformers SparseEncoder.

Handles the opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1
model (SparseStaticEmbedding router for queries, MLMTransformer+SpladePooling
for documents).

Uses scipy CSR sparse matrices for query×document similarity to avoid the
CUDA sparse tensor backend limitations (aten::as_strided not implemented for
SparseCUDA) and to keep memory use manageable on large corpora.

Usage:
    .venv-colbert-sparse/bin/python SPARSE/opensearch_eval.py \\
        --config mlaire/configs/compact.yaml \\
        --output results_compact_sparse/opensearch_neural_sparse_compact.json
"""

import argparse

import numpy as np
import torch
from scipy.sparse import csr_matrix
from sentence_transformers import SparseEncoder

from _common import load_config, run_and_save
from mlaire.types import RetrievalPool


def _torch_sparse_coo_to_scipy_csr(t: torch.Tensor) -> csr_matrix:
    """Convert a 2-D torch sparse COO tensor to scipy CSR on CPU."""
    t = t.coalesce().cpu()
    indices = t.indices().numpy()
    values = t.values().numpy().astype(np.float32)
    shape = tuple(t.shape)
    return csr_matrix(
        (values, (indices[0], indices[1])),
        shape=shape,
    )


def make_sparse_retriever(
    model_name: str,
    device: str = "cuda",
    batch_size: int = 64,
    query_chunk: int = 256,
):
    model = SparseEncoder(model_name, device=device)

    def retrieve(pool: RetrievalPool, top_k: int) -> list:
        passage_ids = [p.passage_id for p in pool.passages]
        passage_langs = [p.language for p in pool.passages]
        passage_gids = [p.group_id for p in pool.passages]
        passage_texts = [p.text for p in pool.passages]
        query_texts = [q.text for q in pool.queries]
        query_ids = [q.query_id for q in pool.queries]

        print(f"  Encoding {len(passage_texts)} passages (sparse)...", flush=True)
        doc_embs_t = model.encode_document(
            passage_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_sparse_tensor=True,
            save_to_cpu=True,
        )
        print(f"  Encoding {len(query_texts)} queries (sparse)...", flush=True)
        query_embs_t = model.encode_query(
            query_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_sparse_tensor=True,
            save_to_cpu=True,
        )

        print("  Converting to scipy CSR...", flush=True)
        doc_csr = _torch_sparse_coo_to_scipy_csr(doc_embs_t)
        query_csr = _torch_sparse_coo_to_scipy_csr(query_embs_t)
        del doc_embs_t, query_embs_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(
            f"  doc CSR nnz={doc_csr.nnz:,}  query CSR nnz={query_csr.nnz:,}",
            flush=True,
        )
        # Transpose docs once for efficient query @ doc.T
        doc_csr_t = doc_csr.T.tocsr()

        print("  Computing similarities (scipy sparse)...", flush=True)
        results = []
        num_queries = len(query_texts)
        for start in range(0, num_queries, query_chunk):
            end = min(start + query_chunk, num_queries)
            q_chunk = query_csr[start:end]
            # sim is (chunk_size, num_docs) sparse; densify the slice.
            sim = (q_chunk @ doc_csr_t).toarray()
            for j in range(sim.shape[0]):
                scores = sim[j]
                k = min(top_k, len(scores))
                if k == len(scores):
                    order = np.argsort(-scores)
                else:
                    top_idx = np.argpartition(-scores, k)[:k]
                    order = top_idx[np.argsort(-scores[top_idx])]
                ranked = [
                    (
                        passage_ids[i],
                        passage_gids[i],
                        passage_langs[i],
                        float(scores[i]),
                    )
                    for i in order
                ]
                results.append((query_ids[start + j], ranked))
        return results

    return retrieve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model",
        default="opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--query-chunk", type=int, default=256)
    args = parser.parse_args()

    config = load_config(args.config)
    retrieve_fn = make_sparse_retriever(
        args.model, args.device, args.batch_size, args.query_chunk
    )
    run_and_save(config, args.model, retrieve_fn, args.output)


if __name__ == "__main__":
    main()
