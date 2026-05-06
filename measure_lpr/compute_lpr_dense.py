"""Recompute LPR for a dense embedder using full group-wise score access.

For each query we score ONLY the relevant group's passages (typically
≤122 for Belebele) and take argmax — so the resulting LPR is top_k
independent and matches the formal definition.

Run via the right venv (paper uses ``.venv`` for Qwen3 / mE5 / PPLX):

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python measure_lpr/compute_lpr_dense.py \\
        --model intfloat/multilingual-e5-small --dataset mlqa
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from measure_lpr._common import (  # noqa: E402
    aggregate_lpr,
    load_pool,
    output_path,
    save_json,
)
from mlaire.evaluation.retriever import Retriever  # noqa: E402
from mlaire.models.dense import DenseLocalModel  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, choices=["mlqa", "xquad", "belebele"])
    ap.add_argument("--data-dir", default=None,
                    help="Local BEIR export root; default = HuggingFace Hub")
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "measure_lpr" / "results"))
    ap.add_argument("--cache-dir", default=str(REPO_ROOT / "measure_lpr" / ".embed_cache"))
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--no-flash-attn", action="store_true")
    args = ap.parse_args()

    out_path = output_path(Path(args.output_dir), args.dataset, args.model)
    if out_path.exists():
        print(f"Already done: {out_path}", flush=True)
        return

    print(f"=== {args.model} / {args.dataset} ===", flush=True)
    print("Loading pool...", flush=True)
    source = (
        str(Path(args.data_dir) / args.dataset)
        if args.data_dir else None
    )
    pool = load_pool(args.dataset, source=source)
    print(f"  {len(pool.passages):,} passages, {len(pool.queries):,} queries", flush=True)

    print("Loading model...", flush=True)
    model = DenseLocalModel(
        model_name=args.model,
        use_bf16=not args.no_bf16,
        use_flash_attn=not args.no_flash_attn,
    )

    # Reuse the existing Retriever cache so we don't re-encode if a prior
    # run already wrote embeddings for the same (model, passage_set) pair.
    retriever = Retriever(model=model, pool=pool, cache_dir=args.cache_dir)
    print("Encoding passages...", flush=True)
    t0 = time.time()
    p_embs = retriever._encode_passages()
    print(f"  passages encoded in {time.time() - t0:.1f}s, shape={p_embs.shape}", flush=True)

    print("Encoding queries...", flush=True)
    t0 = time.time()
    q_embs = retriever._encode_queries()
    print(f"  queries encoded in {time.time() - t0:.1f}s, shape={q_embs.shape}", flush=True)

    if p_embs.dtype != np.float32:
        p_embs = p_embs.astype(np.float32)
    if q_embs.dtype != np.float32:
        q_embs = q_embs.astype(np.float32)
    # Idempotent under cosine: re-normalising unit vectors is a no-op.
    p_embs = p_embs / (np.linalg.norm(p_embs, axis=1, keepdims=True) + 1e-9)
    q_embs = q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-9)

    def score_group(qi: int, rel_indices: list[int]) -> np.ndarray:
        return q_embs[qi] @ p_embs[rel_indices].T

    print("Aggregating LPR over groups...", flush=True)
    result = aggregate_lpr(pool, score_group)

    save_json(
        out_path,
        {
            "model_name": args.model,
            "backend": "dense",
            "dataset": args.dataset,
            **result,
        },
    )
    print(f"LPR = {result['aggregate_lpr']:.6f}", flush=True)


if __name__ == "__main__":
    main()
