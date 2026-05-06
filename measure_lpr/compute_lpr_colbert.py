"""Recompute LPR for ColBERT-style (multi-vector) retrievers via PyLate.

Supports:
  - jinaai/jina-colbert-v2     (--variant jina)
  - LiquidAI/LFM2-ColBERT-350M (--variant lfm2)

Late-interaction MaxSim is computed directly between query token
embeddings and the relevant group members' passage token embeddings,
without going through a PLAID approximate index — the per-query group is
small (≤122 for Belebele) so this is faster *and* more accurate than the
indexed retrieval path.

Run from the colbert-sparse venv:

    CUDA_VISIBLE_DEVICES=0 .venv-colbert-sparse/bin/python \\
        measure_lpr/compute_lpr_colbert.py --variant jina --dataset mlqa
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pylate import models

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from measure_lpr._common import (  # noqa: E402
    aggregate_lpr,
    load_pool,
    output_path,
    save_json,
)


VARIANTS = {
    "jina": {
        "display_name": "jina_colbert_v2",
        "model_name": "jinaai/jina-colbert-v2",
        "kwargs": dict(
            query_prefix="[QueryMarker]",
            document_prefix="[DocumentMarker]",
            attend_to_expansion_tokens=True,
            trust_remote_code=True,
        ),
    },
    "lfm2": {
        "display_name": "lfm2_colbert_350m",
        "model_name": "LiquidAI/LFM2-ColBERT-350M",
        "kwargs": dict(trust_remote_code=True),
    },
}


def _maxsim(q_tokens: torch.Tensor, p_tokens: torch.Tensor) -> float:
    """MaxSim score s(q, p) = sum_i max_j (q_i · p_j).

    Both tensors are 2-D ``(L, D)`` per-token embeddings as returned by
    ``pylate.models.ColBERT.encode``. PyLate's default output is a list
    of variable-length tensors, so summing over every query token is
    exactly what ColBERT MaxSim asks for. For ``jina-colbert-v2`` the
    ``[MASK]``-expansion query tokens are part of the returned tensor by
    design (``attend_to_expansion_tokens=True``) and are intentionally
    included in the sum.
    """
    if q_tokens.ndim != 2 or p_tokens.ndim != 2:
        raise ValueError(
            f"Expected 2-D token embeddings, got q={tuple(q_tokens.shape)}, "
            f"p={tuple(p_tokens.shape)}"
        )
    sim = q_tokens @ p_tokens.T  # (Lq, Lp)
    return float(sim.max(dim=1).values.sum().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--dataset", required=True, choices=["mlqa", "xquad", "belebele"])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "measure_lpr" / "results"))
    args = ap.parse_args()

    spec = VARIANTS[args.variant]
    out_path = output_path(Path(args.output_dir), args.dataset, spec["display_name"])
    if out_path.exists():
        print(f"Already done: {out_path}", flush=True)
        return

    print(f"=== {spec['display_name']} / {args.dataset} ===", flush=True)
    source = (
        str(Path(args.data_dir) / args.dataset)
        if args.data_dir else None
    )
    pool = load_pool(args.dataset, source=source)
    print(f"  {len(pool.passages):,} passages, {len(pool.queries):,} queries", flush=True)

    print("Loading ColBERT model...", flush=True)
    model = models.ColBERT(
        model_name_or_path=spec["model_name"],
        device=args.device,
        **spec["kwargs"],
    )

    print("Encoding passages (multi-vector)...", flush=True)
    t0 = time.time()
    doc_embs = model.encode(
        [p.text for p in pool.passages],
        batch_size=args.batch_size,
        is_query=False,
        show_progress_bar=True,
    )
    print(f"  passages encoded in {time.time() - t0:.1f}s", flush=True)

    print("Encoding queries (multi-vector)...", flush=True)
    t0 = time.time()
    q_embs = model.encode(
        [q.text for q in pool.queries],
        batch_size=args.batch_size,
        is_query=True,
        show_progress_bar=True,
    )
    print(f"  queries encoded in {time.time() - t0:.1f}s", flush=True)

    def _to_tensor(x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        return torch.as_tensor(x, dtype=torch.float32)

    def score_group(qi: int, rel_indices: list[int]) -> np.ndarray:
        q_t = _to_tensor(q_embs[qi])
        scores = np.empty(len(rel_indices), dtype=np.float32)
        for j, p_idx in enumerate(rel_indices):
            p_t = _to_tensor(doc_embs[p_idx])
            scores[j] = _maxsim(q_t, p_t)
        return scores

    print("Aggregating LPR over groups...", flush=True)
    result = aggregate_lpr(pool, score_group, progress_every=2000)

    save_json(
        out_path,
        {
            "model_name": spec["display_name"],
            "backend": "colbert",
            "encoder_model": spec["model_name"],
            "dataset": args.dataset,
            **result,
        },
    )
    print(f"LPR = {result['aggregate_lpr']:.6f}", flush=True)


if __name__ == "__main__":
    main()
