"""Recompute LPR for OpenSearch neural-sparse (sentence-transformers SparseEncoder).

Run from the colbert-sparse venv:

    CUDA_VISIBLE_DEVICES=0 .venv-colbert-sparse/bin/python \\
        measure_lpr/compute_lpr_opensearch.py --dataset mlqa
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SparseEncoder

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
    ap.add_argument(
        "--model",
        default="opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "measure_lpr" / "results"))
    args = ap.parse_args()

    model_name = "opensearch_neural_sparse"
    out_path = output_path(Path(args.output_dir), args.dataset, model_name)
    if out_path.exists():
        print(f"Already done: {out_path}", flush=True)
        return

    print(f"=== opensearch-neural-sparse / {args.dataset} ===", flush=True)
    source = (
        str(Path(args.data_dir) / args.dataset)
        if args.data_dir else None
    )
    pool = load_pool(args.dataset, source=source)
    print(f"  {len(pool.passages):,} passages, {len(pool.queries):,} queries", flush=True)

    print("Loading SparseEncoder...", flush=True)
    encoder = SparseEncoder(args.model, device=args.device)

    print("Encoding passages (sparse)...", flush=True)
    t0 = time.time()
    p_sparse = encoder.encode_document(
        [p.text for p in pool.passages],
        batch_size=args.batch_size,
        convert_to_sparse_tensor=True,
        show_progress_bar=True,
    )
    print(f"  passages encoded in {time.time() - t0:.1f}s", flush=True)

    print("Encoding queries (sparse)...", flush=True)
    t0 = time.time()
    q_sparse = encoder.encode_query(
        [q.text for q in pool.queries],
        batch_size=args.batch_size,
        convert_to_sparse_tensor=True,
        show_progress_bar=True,
    )
    print(f"  queries encoded in {time.time() - t0:.1f}s", flush=True)

    def _to_indexable(t):
        # Strided tensors index fine. CSR / CSC need converting first
        # because Python int indexing of those sparse layouts is not
        # supported. Anything else: coerce to COO via a generic round-trip.
        if t.layout == torch.strided:
            return t
        csr = getattr(torch, "sparse_csr", None)
        csc = getattr(torch, "sparse_csc", None)
        if (csr is not None and t.layout == csr) or (csc is not None and t.layout == csc):
            return t.to_sparse_coo().coalesce()
        try:
            return t.coalesce()
        except (RuntimeError, TypeError):
            return t.to_sparse_coo().coalesce()

    p_sparse = _to_indexable(p_sparse)
    q_sparse = _to_indexable(q_sparse)

    def _row(t, i):
        return t[i].to_dense() if t.is_sparse else t[i]

    def score_group(qi: int, rel_indices: list[int]) -> np.ndarray:
        q_vec = _row(q_sparse, qi).float()
        p_block = torch.stack([_row(p_sparse, idx).float() for idx in rel_indices])
        return (p_block @ q_vec).cpu().numpy()

    print("Aggregating LPR over groups...", flush=True)
    result = aggregate_lpr(pool, score_group)

    save_json(
        out_path,
        {
            "model_name": model_name,
            "backend": "opensearch_sparse",
            "encoder_model": args.model,
            "dataset": args.dataset,
            **result,
        },
    )
    print(f"LPR = {result['aggregate_lpr']:.6f}", flush=True)


if __name__ == "__main__":
    main()
