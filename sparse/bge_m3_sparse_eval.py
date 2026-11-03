"""BGE-M3 sparse (lexical-weight) head evaluation.

BGE-M3 exposes three scoring heads on one backbone; the dense head is
evaluated through the main ``mlaire`` CLI, and this script evaluates its
sparse head. Each text is encoded with FlagEmbedding into a
``{token_id: weight}`` map and scored as the dot product over shared
tokens (FlagEmbedding's ``compute_lexical_matching_score``), computed
full-pool with scipy CSR matrices.

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv-colbert-sparse/bin/python sparse/bge_m3_sparse_eval.py \\
        --config mlaire/configs/xquad.yaml \\
        --output results/bge_m3_sparse_xquad.json
"""

import argparse

import numpy as np
from FlagEmbedding import BGEM3FlagModel
from scipy.sparse import csr_matrix

from _common import load_config, run_and_save
from mlaire.types import RetrievalPool


def _to_csr(weights: list[dict], vocab: dict) -> csr_matrix:
    """List of {token: weight} -> CSR over ``vocab`` (unknown tokens dropped)."""
    rows, cols, vals = [], [], []
    for i, lw in enumerate(weights):
        for tok, w in lw.items():
            j = vocab.get(tok)
            if j is not None:
                rows.append(i)
                cols.append(j)
                vals.append(float(w))
    return csr_matrix(
        (np.asarray(vals, dtype=np.float32), (rows, cols)),
        shape=(len(weights), len(vocab)),
    )


def make_bgem3_sparse_retriever(
    model_name: str = "BAAI/bge-m3",
    batch_size: int = 32,
    max_length: int = 1024,
    query_chunk: int = 256,
):
    model = BGEM3FlagModel(model_name, use_fp16=True)

    def encode(texts):
        out = model.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return out["lexical_weights"]

    def retrieve(pool: RetrievalPool, top_k: int) -> list:
        passage_ids = [p.passage_id for p in pool.passages]
        passage_langs = [p.language for p in pool.passages]
        passage_gids = [p.group_id for p in pool.passages]
        query_ids = [q.query_id for q in pool.queries]

        print(f"  Encoding {len(passage_ids)} passages (sparse)...", flush=True)
        doc_w = encode([p.text for p in pool.passages])
        print(f"  Encoding {len(query_ids)} queries (sparse)...", flush=True)
        query_w = encode([q.text for q in pool.queries])

        # Query tokens absent from every passage cannot contribute to a
        # dot product, so the passage vocabulary is sufficient.
        vocab: dict = {}
        for lw in doc_w:
            for tok in lw:
                vocab.setdefault(tok, len(vocab))
        doc_csr_t = _to_csr(doc_w, vocab).T.tocsr()
        query_csr = _to_csr(query_w, vocab)
        print(f"  vocab={len(vocab):,}  doc nnz={doc_csr_t.nnz:,}", flush=True)

        print("  Computing similarities (scipy sparse)...", flush=True)
        results = []
        for start in range(0, len(query_ids), query_chunk):
            sim = (query_csr[start:start + query_chunk] @ doc_csr_t).toarray()
            for j in range(sim.shape[0]):
                scores = sim[j]
                k = min(top_k, len(scores))
                if k == len(scores):
                    order = np.argsort(-scores)
                else:
                    top_idx = np.argpartition(-scores, k)[:k]
                    order = top_idx[np.argsort(-scores[top_idx])]
                ranked = [
                    (passage_ids[i], passage_gids[i], passage_langs[i], float(scores[i]))
                    for i in order
                ]
                results.append((query_ids[start + j], ranked))
        return results

    return retrieve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--query-chunk", type=int, default=256)
    args = parser.parse_args()

    config = load_config(args.config)
    retrieve_fn = make_bgem3_sparse_retriever(
        args.model, args.batch_size, args.max_length, args.query_chunk
    )
    run_and_save(config, "bge-m3-sparse", retrieve_fn, args.output)


if __name__ == "__main__":
    main()
