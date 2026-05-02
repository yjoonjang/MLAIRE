"""LFM2-ColBERT-350M evaluation using PyLate with PLAID index.

Model card: https://huggingface.co/LiquidAI/LFM2-ColBERT-350M
Covers 8 languages (en, ar, zh, fr, de, ja, ko, es).

Usage:
    .venv-colbert-sparse/bin/python COLBERT/lfm2_colbert_eval.py \\
        --config mlaire/configs/compact.yaml \\
        --output results_compact_colbert/lfm2_colbert_compact.json
"""

import argparse
import tempfile

from pylate import indexes, models, retrieve

from _common import load_config, run_and_save
from mlaire.types import RetrievalPool


def make_lfm2_retriever(model_name: str, device: str = "cuda", batch_size: int = 32):
    model = models.ColBERT(model_name_or_path=model_name, device=device)
    # LFM2 model card workaround: pad_token is not set by default
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    def retrieve_fn(pool: RetrievalPool, top_k: int) -> list:
        passage_lookup = {
            p.passage_id: (p.group_id, p.language) for p in pool.passages
        }
        doc_ids = [p.passage_id for p in pool.passages]
        doc_texts = [p.text for p in pool.passages]
        query_ids = [q.query_id for q in pool.queries]
        query_texts = [q.text for q in pool.queries]

        print(f"  Encoding {len(doc_texts)} passages...", flush=True)
        doc_embs = model.encode(
            doc_texts,
            batch_size=batch_size,
            is_query=False,
            show_progress_bar=True,
        )

        print(f"  Encoding {len(query_texts)} queries...", flush=True)
        query_embs = model.encode(
            query_texts,
            batch_size=batch_size,
            is_query=True,
            show_progress_bar=True,
        )

        # Build PLAID index in a temp folder (per-dataset — independent pools)
        with tempfile.TemporaryDirectory(prefix="pylate_lfm2_") as tmpdir:
            index = indexes.PLAID(
                index_folder=tmpdir,
                index_name="mlair_idx",
                override=True,
            )
            index.add_documents(
                documents_ids=doc_ids,
                documents_embeddings=doc_embs,
            )

            retriever = retrieve.ColBERT(index=index)
            print("  Retrieving...", flush=True)
            scores = retriever.retrieve(
                queries_embeddings=query_embs,
                k=top_k,
            )

        # scores: list[list[{"id": str, "score": float}]]
        results = []
        for qid, query_hits in zip(query_ids, scores):
            ranked = []
            for hit in query_hits:
                pid = hit["id"]
                sc = hit["score"]
                if pid in passage_lookup:
                    gid, lang = passage_lookup[pid]
                    ranked.append((pid, gid, lang, float(sc)))
            results.append((qid, ranked))
        return results

    return retrieve_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="LiquidAI/LFM2-ColBERT-350M")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    config = load_config(args.config)
    retrieve_fn = make_lfm2_retriever(args.model, args.device, args.batch_size)
    run_and_save(config, args.model, retrieve_fn, args.output)


if __name__ == "__main__":
    main()
