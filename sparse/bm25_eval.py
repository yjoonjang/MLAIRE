"""BM25 evaluation using bm25s (scipy-sparse BM25, ~500x faster than rank_bm25).

Tokenization uses the XLM-R SentencePiece tokenizer so all 12 languages
(including CJK/Thai) are split into meaningful subwords. This yields a
language-agnostic BM25 variant that naturally handles cross-lingual term
overlap (numbers, cognates, proper nouns) while still being a pure lexical
baseline.

Usage:
    .venv-colbert-sparse/bin/python SPARSE/bm25_eval.py \\
        --config mlaire/configs/compact.yaml \\
        --output results_compact_sparse/bm25_compact.json
"""

import argparse

import bm25s
import numpy as np
from transformers import AutoTokenizer

from _common import load_config, run_and_save
from mlaire.types import RetrievalPool


def make_bm25_retriever(tokenizer_name: str, k1: float, b: float):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    max_tokens = 512

    def tokenize_batch(texts: list[str]) -> list[list[str]]:
        out: list[list[str]] = []
        for t in texts:
            out.append(tokenizer.tokenize(t)[:max_tokens])
        return out

    def retrieve(pool: RetrievalPool, top_k: int) -> list:
        passage_ids = [p.passage_id for p in pool.passages]
        passage_langs = [p.language for p in pool.passages]
        passage_gids = [p.group_id for p in pool.passages]
        passage_texts = [p.text for p in pool.passages]

        print(f"  Tokenizing {len(passage_texts)} passages...", flush=True)
        corpus_tokens = tokenize_batch(passage_texts)

        print(f"  Building BM25 index (k1={k1}, b={b})...", flush=True)
        retriever = bm25s.BM25(k1=k1, b=b)
        retriever.index(corpus_tokens, show_progress=False)

        query_texts = [q.text for q in pool.queries]
        query_ids = [q.query_id for q in pool.queries]

        print(f"  Tokenizing {len(query_texts)} queries...", flush=True)
        query_tokens = tokenize_batch(query_texts)

        k = min(top_k, len(passage_ids))
        print(f"  Retrieving top-{k}...", flush=True)
        doc_indices, doc_scores = retriever.retrieve(
            query_tokens, k=k, show_progress=True
        )
        # doc_indices: (num_queries, k) np.ndarray of int
        # doc_scores:  (num_queries, k) np.ndarray of float

        results = []
        for qi in range(len(query_ids)):
            idxs = doc_indices[qi]
            scrs = doc_scores[qi]
            ranked = [
                (
                    passage_ids[int(idxs[j])],
                    passage_gids[int(idxs[j])],
                    passage_langs[int(idxs[j])],
                    float(scrs[j]),
                )
                for j in range(len(idxs))
            ]
            results.append((query_ids[qi], ranked))
        return results

    return retrieve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument(
        "--tokenizer",
        default="xlm-roberta-base",
        help="HF tokenizer (default: xlm-roberta-base for cross-lingual support)",
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=1.5,
        help="BM25 k1 parameter (default: 1.5, rank_bm25 convention)",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=0.75,
        help="BM25 b parameter (default: 0.75, rank_bm25 convention)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    retrieve_fn = make_bm25_retriever(args.tokenizer, args.k1, args.b)
    model_name = f"bm25-{args.tokenizer.replace('/', '_')}"
    run_and_save(config, model_name, retrieve_fn, args.output)


if __name__ == "__main__":
    main()
