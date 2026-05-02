"""Export MLAIRE evaluation data into MTEB-style BEIR parquet datasets.

For each requested dataset (mlqa / xquad / belebele) this script

  1. instantiates the canonical loader (which pulls the raw data from the
     upstream Hugging Face datasets, deduplicates passages by group_id and
     language, and assigns canonical group / passage / query ids), and
  2. writes a Hugging Face dataset directory with three configs

         <out>/<dataset>/corpus/test-*.parquet      _id, text, title,
                                                   language, group_id, dataset
         <out>/<dataset>/queries/test-*.parquet     _id, text,
                                                   language, group_id, dataset
         <out>/<dataset>/qrels/test-*.parquet       query-id, corpus-id, score

  3. drops a ``README.md`` whose YAML front-matter declares the configs so
     ``datasets.load_dataset`` works directly:

         load_dataset("anonymous-submission-nips/mlaire-mlqa", "corpus")
         load_dataset("anonymous-submission-nips/mlaire-mlqa", "queries")
         load_dataset("anonymous-submission-nips/mlaire-mlqa", "qrels")

Run from the repo root:

    python scripts/export_to_beir.py --out ./data
    # or just one dataset:
    python scripts/export_to_beir.py --out ./data --datasets mlqa
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

# Make the local ``mlaire`` package importable when this script is run from
# the repo root without the package being pip-installed.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mlaire.data.loaders.belebele import BelebeleLoader  # noqa: E402
from mlaire.data.loaders.mlqa import MLQALoader  # noqa: E402
from mlaire.data.loaders.xquad import XQuADLoader  # noqa: E402
from mlaire.language_metadata import (  # noqa: E402
    BELEBELE_LANG_MAP,
    MLQA_LANGUAGES,
    XQUAD_LANGUAGES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


DATASET_SPECS = {
    "mlqa":     (MLQALoader,    list(MLQA_LANGUAGES)),
    "xquad":    (XQuADLoader,   list(XQUAD_LANGUAGES)),
    "belebele": (BelebeleLoader, sorted(set(BELEBELE_LANG_MAP.values()))),
}


# ------------------------------------------------------------------ schemas
CORPUS_SCHEMA = pa.schema([
    ("_id",       pa.string()),
    ("text",      pa.string()),
    ("title",     pa.string()),
    ("language",  pa.string()),
    ("group_id",  pa.string()),
    ("dataset",   pa.string()),
])

QUERIES_SCHEMA = pa.schema([
    ("_id",       pa.string()),
    ("text",      pa.string()),
    ("language",  pa.string()),
    ("group_id",  pa.string()),
    ("dataset",   pa.string()),
])

QRELS_SCHEMA = pa.schema([
    ("query-id",  pa.string()),
    ("corpus-id", pa.string()),
    ("score",     pa.int64()),
])

# Streaming qrels in batches keeps peak memory bounded for Belebele
# (~13.4M qrels rows otherwise allocates >1 GB at once).
QRELS_BATCH = 200_000


def _write_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def _build_corpus_table(passages) -> pa.Table:
    return pa.table(
        {
            "_id":      [p.passage_id for p in passages],
            "text":     [p.text for p in passages],
            "title":    ["" for _ in passages],
            "language": [p.language for p in passages],
            "group_id": [p.group_id for p in passages],
            "dataset":  [p.dataset for p in passages],
        },
        schema=CORPUS_SCHEMA,
    )


def _build_queries_table(queries) -> pa.Table:
    return pa.table(
        {
            "_id":      [q.query_id for q in queries],
            "text":     [q.text for q in queries],
            "language": [q.language for q in queries],
            "group_id": [q.group_id for q in queries],
            "dataset":  [q.dataset for q in queries],
        },
        schema=QUERIES_SCHEMA,
    )


def _stream_qrels(queries, gid_to_pids) -> Iterable[pa.RecordBatch]:
    """Yield ``QRELS_BATCH``-sized record batches of (query, corpus, 1) triples."""
    qid_buf: list[str] = []
    cid_buf: list[str] = []
    score_buf: list[int] = []

    def flush() -> pa.RecordBatch:
        return pa.RecordBatch.from_arrays(
            [pa.array(qid_buf), pa.array(cid_buf), pa.array(score_buf, type=pa.int64())],
            schema=QRELS_SCHEMA,
        )

    for q in queries:
        for pid in gid_to_pids.get(q.group_id, ()):
            qid_buf.append(q.query_id)
            cid_buf.append(pid)
            score_buf.append(1)
            if len(qid_buf) >= QRELS_BATCH:
                batch = flush()
                qid_buf.clear(); cid_buf.clear(); score_buf.clear()
                yield batch
    if qid_buf:
        yield flush()


# ------------------------------------------------------------- dataset card
LANG_LIST_FOR_CARD = {
    "mlqa":     list(MLQA_LANGUAGES),
    "xquad":    list(XQUAD_LANGUAGES),
    "belebele": sorted(set(BELEBELE_LANG_MAP.values())),
}

DEFAULT_TOPK = {"mlqa": 20, "xquad": 20, "belebele": 200}

DESCRIPTIONS = {
    "mlqa": (
        "MLQA reformatted for language-aware retrieval evaluation. Passages "
        "are deduplicated at the context level via union-find on the original "
        "MLQA ids. Relevance is encoded by group_id matching."
    ),
    "xquad": (
        "XQuAD reformatted for language-aware retrieval evaluation. Each "
        "passage appears once per language; relevance is encoded by group_id "
        "matching."
    ),
    "belebele": (
        "Belebele reformatted for language-aware retrieval evaluation. 488 "
        "underlying passages, each available in 122 languages (joined "
        "globally by the original `link` field). Relevance is encoded by "
        "group_id matching."
    ),
}


def _readme(name: str) -> str:
    langs = LANG_LIST_FOR_CARD[name]
    # HF requires the `language:` field to use ISO 639 (2-3 lowercase letter)
    # codes only. Variants such as ``bn_Latn`` / ``zh_Hant`` go into the
    # ``language_bcp47:`` field with hyphenated form (BCP-47).
    iso_codes = [l for l in langs if "_" not in l]
    bcp47_codes = [l.replace("_", "-") for l in langs if "_" in l]

    iso_yaml = "\n".join(f"  - {l}" for l in iso_codes)
    bcp47_yaml = (
        "\nlanguage_bcp47:\n" + "\n".join(f"  - {l}" for l in bcp47_codes)
        if bcp47_codes
        else ""
    )
    return f"""---
license: mit
task_categories:
  - text-retrieval
multilinguality: multilingual
language:
{iso_yaml}{bcp47_yaml}
pretty_name: MLAIRE-{name.upper()}
tags:
  - retrieval
  - multilingual
  - language-aware-ir
  - benchmark
  - beir
  - mlaire
configs:
  - config_name: corpus
    data_files:
      - split: test
        path: corpus/test-*.parquet
  - config_name: queries
    data_files:
      - split: test
        path: queries/test-*.parquet
  - config_name: qrels
    data_files:
      - split: test
        path: qrels/test-*.parquet
---

# MLAIRE-{name.upper()}

{DESCRIPTIONS[name]}

This repository is part of the **MLAIRE** benchmark, submitted anonymously
to the NeurIPS 2026 Evaluations & Datasets Track. Authors and affiliations
are withheld for double-blind review.

## Default top-k

Reported metrics in the paper use **top-{DEFAULT_TOPK[name]}**.

## Layout

```
corpus/test-*.parquet      _id, text, title, language, group_id, dataset
queries/test-*.parquet     _id, text, language, group_id, dataset
qrels/test-*.parquet       query-id, corpus-id, score=1   (binary, BEIR-standard)
```

`language` and `group_id` are stored explicitly on every row, so language-
aware metrics (LPR, Lang-nDCG, Lang-Recall) and the 4-way top-1 failure
decomposition (perfect / lang_fail / sem_fail / both_fail) can be
reconstructed without consulting any additional resource.

## Usage

```python
from datasets import load_dataset

corpus  = load_dataset("anonymous-submission-nips/mlaire-{name}", "corpus",  split="test")
queries = load_dataset("anonymous-submission-nips/mlaire-{name}", "queries", split="test")
qrels   = load_dataset("anonymous-submission-nips/mlaire-{name}", "qrels",   split="test")
```

Or simply install the companion package and run

```bash
mlaire --config mlaire/configs/{name}.yaml --model <hf-model-id>
```

which downloads these files automatically and reproduces every metric we
report in the paper.

## License

MIT (anonymous authors).
"""


# ------------------------------------------------------------------ export
def export_one(name: str, out_root: Path) -> None:
    loader_cls, langs = DATASET_SPECS[name]
    logger.info("[%s] loading %d languages ...", name, len(langs))
    loader = loader_cls(languages=langs)
    passages, queries = loader.load()
    logger.info("[%s] %d passages, %d queries", name, len(passages), len(queries))

    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)

    # ---- corpus ---------------------------------------------------------
    corpus_path = out / "corpus" / "test-00000-of-00001.parquet"
    _write_parquet(corpus_path, _build_corpus_table(passages))
    logger.info("[%s] wrote %s", name, corpus_path)

    # ---- queries --------------------------------------------------------
    queries_path = out / "queries" / "test-00000-of-00001.parquet"
    _write_parquet(queries_path, _build_queries_table(queries))
    logger.info("[%s] wrote %s", name, queries_path)

    # ---- qrels (streamed) ----------------------------------------------
    gid_to_pids: dict[str, list[str]] = defaultdict(list)
    for p in passages:
        gid_to_pids[p.group_id].append(p.passage_id)

    qrels_path = out / "qrels" / "test-00000-of-00001.parquet"
    qrels_path.parent.mkdir(parents=True, exist_ok=True)

    n_qrels = 0
    with pq.ParquetWriter(qrels_path, QRELS_SCHEMA, compression="zstd") as writer:
        for batch in _stream_qrels(queries, gid_to_pids):
            writer.write_batch(batch)
            n_qrels += batch.num_rows
    logger.info("[%s] wrote %d qrels rows to %s", name, n_qrels, qrels_path)

    # ---- README ---------------------------------------------------------
    (out / "README.md").write_text(_readme(name), encoding="utf-8")
    logger.info("[%s] wrote %s", name, out / "README.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./data", help="output root directory")
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_SPECS.keys()),
        choices=list(DATASET_SPECS.keys()),
    )
    args = ap.parse_args()

    out_root = Path(args.out)
    for name in args.datasets:
        export_one(name, out_root)
    logger.info("All requested datasets exported under %s", out_root)


if __name__ == "__main__":
    main()
