"""Jina ColBERT v2 evaluation using PyLate with PLAID index.

Model card: https://huggingface.co/jinaai/jina-colbert-v2
Covers 94 languages. Requires flash_attn and trust_remote_code.

The model card specifies these init args:
    query_prefix="[QueryMarker]"
    document_prefix="[DocumentMarker]"
    attend_to_expansion_tokens=True
    trust_remote_code=True

Usage:
    .venv-colbert-sparse/bin/python COLBERT/jina_colbert_eval.py \\
        --config mlaire/configs/compact.yaml \\
        --output results_compact_colbert/jina_colbert_v2_compact.json
"""

import argparse
import tempfile

from pylate import indexes, models, retrieve

from _common import load_config, run_and_save
from mlaire.types import RetrievalPool


def make_jina_retriever(
    model_name: str = "jinaai/jina-colbert-v2",
    device: str = "cuda",
    batch_size: int = 16,
):
    model = models.ColBERT(
        model_name_or_path=model_name,
        query_prefix="[QueryMarker]",
        document_prefix="[DocumentMarker]",
        attend_to_expansion_tokens=True,
        trust_remote_code=True,
        device=device,
    )

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

        with tempfile.TemporaryDirectory(prefix="pylate_jina_") as tmpdir:
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
    parser.add_argument("--model", default="jinaai/jina-colbert-v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    config = load_config(args.config)
    retrieve_fn = make_jina_retriever(args.model, args.device, args.batch_size)
    run_and_save(config, args.model, retrieve_fn, args.output)


if __name__ == "__main__":
    main()
