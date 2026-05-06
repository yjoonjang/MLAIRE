"""Recompute LPR for BM25 (XLM-R tokenizer + bm25s).

Run from the colbert-sparse venv:

    .venv-colbert-sparse/bin/python measure_lpr/compute_lpr_bm25.py \\
        --dataset mlqa
"""

import argparse
import sys
import time
from pathlib import Path

import bm25s
import numpy as np
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from measure_lpr._common import (  # noqa: E402
    aggregate_lpr,
    load_pool,
    output_path,
    save_json,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["mlqa", "xquad", "belebele"])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--tokenizer", default="xlm-roberta-base")
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "measure_lpr" / "results"))
    args = ap.parse_args()

    model_name = "bm25"
    out_path = output_path(Path(args.output_dir), args.dataset, model_name)
    if out_path.exists():
        print(f"Already done: {out_path}", flush=True)
        return

    print(f"=== bm25 / {args.dataset} ===", flush=True)
    source = (
        str(Path(args.data_dir) / args.dataset)
        if args.data_dir else None
    )
    pool = load_pool(args.dataset, source=source)
    print(f"  {len(pool.passages):,} passages, {len(pool.queries):,} queries", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    max_tokens = 512

    def tokenize(texts: list[str]) -> list[list[str]]:
        return [tokenizer.tokenize(t)[:max_tokens] for t in texts]

    print("Tokenising passages...", flush=True)
    t0 = time.time()
    corpus_tokens = tokenize([p.text for p in pool.passages])
    print(f"  passages tokenised in {time.time() - t0:.1f}s", flush=True)

    print("Building BM25 index...", flush=True)
    t0 = time.time()
    bm25 = bm25s.BM25(k1=args.k1, b=args.b)
    bm25.index(corpus_tokens, show_progress=False)
    print(f"  index built in {time.time() - t0:.1f}s", flush=True)

    print("Tokenising queries...", flush=True)
    t0 = time.time()
    query_tokens = tokenize([q.text for q in pool.queries])
    print(f"  queries tokenised in {time.time() - t0:.1f}s", flush=True)

    def score_group(qi: int, rel_indices: list[int]) -> np.ndarray:
        all_scores = bm25.get_scores(query_tokens[qi])
        return np.asarray(all_scores, dtype=np.float32)[rel_indices]

    print("Aggregating LPR over groups...", flush=True)
    result = aggregate_lpr(pool, score_group)

    save_json(
        out_path,
        {
            "model_name": model_name,
            "backend": "bm25",
            "tokenizer": args.tokenizer,
            "k1": args.k1,
            "b": args.b,
            "dataset": args.dataset,
            **result,
        },
    )
    print(f"LPR = {result['aggregate_lpr']:.6f}", flush=True)


if __name__ == "__main__":
    main()
