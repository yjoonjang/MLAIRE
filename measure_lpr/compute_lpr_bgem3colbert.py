"""Recompute LPR for the BGE-M3 ColBERT (multi-vector) head.

Score = ColBERT MaxSim over FlagEmbedding token embeddings,
sum_i max_j (q_i . p_j), as in FlagEmbedding's ``colbert_score``.

Run from the colbert-sparse venv:

    CUDA_VISIBLE_DEVICES=0 .venv-colbert-sparse/bin/python \\
        measure_lpr/compute_lpr_bgem3colbert.py --dataset mlqa
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from FlagEmbedding import BGEM3FlagModel

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from measure_lpr._common import (  # noqa: E402
    aggregate_lpr,
    load_pool,
    output_path,
    save_json,
)


def _maxsim(q: np.ndarray, p: np.ndarray) -> float:
    return float((q @ p.T).max(axis=1).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["mlqa", "xquad", "belebele"])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "measure_lpr" / "results"))
    args = ap.parse_args()

    model_name = "bge-m3-colbert"
    out_path = output_path(Path(args.output_dir), args.dataset, model_name)
    if out_path.exists():
        print(f"Already done: {out_path}", flush=True)
        return

    print(f"=== bge-m3-colbert / {args.dataset} ===", flush=True)
    source = (
        str(Path(args.data_dir) / args.dataset)
        if args.data_dir else None
    )
    pool = load_pool(args.dataset, source=source)
    print(f"  {len(pool.passages):,} passages, {len(pool.queries):,} queries", flush=True)

    print("Loading BGE-M3 (colbert head)...", flush=True)
    model = BGEM3FlagModel(args.model, use_fp16=True)

    def encode(texts):
        out = model.encode(
            texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
        )
        return [np.asarray(v, dtype=np.float32) for v in out["colbert_vecs"]]

    print("Encoding passages (multi-vector)...", flush=True)
    t0 = time.time()
    p_reps = encode([p.text for p in pool.passages])
    print(f"  passages encoded in {time.time() - t0:.1f}s", flush=True)

    print("Encoding queries (multi-vector)...", flush=True)
    t0 = time.time()
    q_reps = encode([q.text for q in pool.queries])
    print(f"  queries encoded in {time.time() - t0:.1f}s", flush=True)

    def score_group(qi: int, rel_indices: list[int]) -> np.ndarray:
        q = q_reps[qi]
        return np.array([_maxsim(q, p_reps[idx]) for idx in rel_indices], dtype=np.float32)

    print("Aggregating LPR over groups...", flush=True)
    result = aggregate_lpr(pool, score_group)

    save_json(
        out_path,
        {
            "model_name": model_name,
            "backend": "colbert_bgem3",
            "encoder_model": args.model,
            "dataset": args.dataset,
            **result,
        },
    )
    print(f"LPR = {result['aggregate_lpr']:.6f}", flush=True)


if __name__ == "__main__":
    main()
