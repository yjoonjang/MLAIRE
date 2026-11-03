"""BGE-M3 ColBERT (multi-vector) head evaluation with exact full-pool MaxSim.

BGE-M3 is not a PyLate model, so its token embeddings are encoded with
FlagEmbedding (``return_colbert_vecs=True``) and retrieval is EXACT ColBERT
MaxSim over every passage, computed on the GPU, rather than a PLAID index:

    score(q, p) = sum_i max_j (q_i . p_j)

Token embeddings are L2-normalised (standard ColBERT convention). Passages
are sorted by token length and scored in GPU-resident groups of at most
``--doc-group`` passages; each group's per-query top-k is merged into a
running global top-k, so peak GPU memory is bounded by one group regardless
of pool size. Query embeddings stay on the CPU and are moved per batch.

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv-colbert-sparse/bin/python colbert/bge_m3_colbert_eval.py \\
        --config mlaire/configs/xquad.yaml \\
        --output results/bge_m3_colbert_xquad.json

For Belebele use ``--doc-group 8000`` (or smaller) to bound GPU memory.
"""

import argparse
import time

import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel

from _common import load_config, run_and_save
from mlaire.types import RetrievalPool


def make_bgem3_encoder(model_name: str, batch_size: int, max_length: int):
    model = BGEM3FlagModel(model_name, use_fp16=True)

    def encode(texts):
        out = model.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
        )
        embs = []
        for e in out["colbert_vecs"]:
            e = np.asarray(e, dtype=np.float32)
            n = np.linalg.norm(e, axis=1, keepdims=True)
            n[n == 0] = 1.0
            embs.append((e / n).astype(np.float16))
        return embs

    return encode


def _pad_stack(embs: list) -> tuple[torch.Tensor, torch.Tensor]:
    """List of (L_i, D) arrays -> CPU (N, Lmax, D) fp16 tensor + (N,) lengths."""
    lengths = torch.tensor([e.shape[0] for e in embs], dtype=torch.long)
    out = torch.zeros((len(embs), int(lengths.max()), embs[0].shape[1]), dtype=torch.float16)
    for i, e in enumerate(embs):
        out[i, : e.shape[0]] = torch.from_numpy(e)
    return out, lengths


def _build_blocks(doc_embs, gidx, d_block, device):
    """Pad each block of passages to its own max length; True in mask = padding."""
    blocks, max_lb = [], 0
    dim = doc_embs[0].shape[1]
    for bs in range(0, len(gidx), d_block):
        idxs = gidx[bs:bs + d_block]
        lb = max(doc_embs[i].shape[0] for i in idxs)
        max_lb = max(max_lb, lb)
        Db = torch.zeros((len(idxs), lb, dim), dtype=torch.float16)
        dpad = torch.ones((len(idxs), lb), dtype=torch.bool)
        for k, i in enumerate(idxs):
            li = doc_embs[i].shape[0]
            Db[k, :li] = torch.from_numpy(doc_embs[i])
            dpad[k, :li] = False
        blocks.append((Db.to(device), dpad.to(device)))
    return blocks, max_lb


def make_exact_retriever(encode, device="cuda", q_batch=64, d_block=2048,
                         max_sim_elems=2_000_000_000, doc_group=0):
    def retrieve_fn(pool: RetrievalPool, top_k: int) -> list:
        doc_meta = [(p.passage_id, p.group_id, p.language) for p in pool.passages]
        query_ids = [q.query_id for q in pool.queries]

        print(f"  Encoding {len(doc_meta)} passages...", flush=True)
        doc_embs = encode([p.text for p in pool.passages])
        print(f"  Encoding {len(query_ids)} queries...", flush=True)
        Q, qlen = _pad_stack(encode([q.text for q in pool.queries]))
        Nq, qlmax = Q.shape[0], Q.shape[1]
        Np = len(doc_embs)
        top_k = min(top_k, Np)
        G = doc_group if doc_group > 0 else Np
        qmask_full = (torch.arange(qlmax)[None, :] < qlen[:, None]).to(torch.float16)

        order = sorted(range(Np), key=lambda i: doc_embs[i].shape[0])
        run_vals = torch.full((Nq, top_k), float("-inf"), device=device)
        run_ids = torch.full((Nq, top_k), -1, dtype=torch.long, device=device)

        print(f"  Exact MaxSim: Np={Np} Nq={Nq} qlmax={qlmax} top_k={top_k} "
              f"doc_group={G} groups={(Np + G - 1) // G}", flush=True)
        t0 = time.time()
        for gstart in range(0, Np, G):
            gidx = order[gstart:gstart + G]
            gids_t = torch.tensor(gidx, dtype=torch.long, device=device)
            Cg = len(gidx)
            blocks, max_lb = _build_blocks(doc_embs, gidx, d_block, device)
            gk = min(top_k, Cg)
            b = max(1, min(q_batch, max_sim_elems // max(1, d_block * qlmax * max_lb)))
            for qs in range(0, Nq, b):
                qb = Q[qs:qs + b].to(device)
                qmask = qmask_full[qs:qs + b].to(device)
                scores = torch.empty((qb.shape[0], Cg), dtype=torch.float32, device=device)
                col = 0
                for Db, dpad in blocks:
                    c = Db.shape[0]
                    sim = torch.einsum("bqd,cpd->bcqp", qb, Db)
                    sim = sim.masked_fill(dpad[None, :, None, :], float("-inf"))
                    mx = (sim.max(dim=3).values * qmask[:, None, :]).sum(dim=2)
                    scores[:, col:col + c] = mx.to(torch.float32)
                    col += c
                gv, gi = torch.topk(scores, gk, dim=1)
                rows = slice(qs, qs + qb.shape[0])
                cat_v = torch.cat([run_vals[rows], gv], dim=1)
                cat_i = torch.cat([run_ids[rows], gids_t[gi]], dim=1)
                nv, ni = torch.topk(cat_v, top_k, dim=1)
                run_vals[rows] = nv
                run_ids[rows] = torch.gather(cat_i, 1, ni)
            del blocks
            torch.cuda.empty_cache()
            print(f"    group {gstart}-{gstart + Cg}/{Np} done  {time.time() - t0:.0f}s",
                  flush=True)

        rv, ri = run_vals.cpu().numpy(), run_ids.cpu().numpy()
        results = []
        for r in range(Nq):
            ranked = []
            for j, s in zip(ri[r], rv[r]):
                if j < 0:
                    continue
                pid, gid, lang = doc_meta[j]
                ranked.append((pid, gid, lang, float(s)))
            results.append((query_ids[r], ranked))
        return results

    return retrieve_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--q-batch", type=int, default=64)
    ap.add_argument("--d-block", type=int, default=2048)
    ap.add_argument("--max-sim-elems", type=int, default=2_000_000_000,
                    help="cap on the (B, C, Lq, Lp) similarity intermediate")
    ap.add_argument("--doc-group", type=int, default=0,
                    help="max passages held on GPU at once (0 = all)")
    args = ap.parse_args()

    encode = make_bgem3_encoder(args.model, args.batch_size, args.max_length)
    retrieve_fn = make_exact_retriever(
        encode, device=args.device, q_batch=args.q_batch, d_block=args.d_block,
        max_sim_elems=args.max_sim_elems, doc_group=args.doc_group,
    )
    config = load_config(args.config)
    run_and_save(config, "bge-m3-colbert", retrieve_fn, args.output)


if __name__ == "__main__":
    main()
