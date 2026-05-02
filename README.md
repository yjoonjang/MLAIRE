# MLAIRE

**M**ulti**L**ingual **A**nonymous **I**R **E**valuation suite — code
for the NeurIPS 2026 Evaluations & Datasets Track submission.

`MLAIRE` provides a unified, language-aware retrieval benchmark over three
multilingual question–answering corpora (MLQA, XQuAD, Belebele) and reports
both standard IR metrics (MRR, nDCG, Recall) and language-aware variants
(LPR, Lang-MRR, Lang-nDCG, Lang-Recall) along with the top-1 4-way failure
decomposition (perfect / lang_fail / sem_fail / both_fail).

---

## 1. Datasets

The three benchmarks are hosted as MTEB-style multi-config Hugging Face
datasets:

| Dataset  | Repo                                                    | Languages | Default top-k |
|----------|---------------------------------------------------------|----------:|--------------:|
| MLQA     | `anonymous-submission-nips/mlaire-mlqa`                 |         7 |            20 |
| XQuAD    | `anonymous-submission-nips/mlaire-xquad`                |        12 |            20 |
| Belebele | `anonymous-submission-nips/mlaire-belebele`             |       122 |           200 |

Each repo exposes three configs that load directly with `datasets.load_dataset`:

```python
from datasets import load_dataset

corpus  = load_dataset("anonymous-submission-nips/mlaire-mlqa", "corpus",  split="test")
queries = load_dataset("anonymous-submission-nips/mlaire-mlqa", "queries", split="test")
qrels   = load_dataset("anonymous-submission-nips/mlaire-mlqa", "qrels",   split="test")
```

**Schema:**

```
corpus    _id, text, title, language, group_id, dataset
queries   _id, text,        language, group_id, dataset
qrels     query-id, corpus-id, score=1   (binary, BEIR-standard)
```

Relevance is encoded by **group_id matching**: a query and a passage are
relevant iff their `group_id` agrees. Because every row carries an explicit
`language` field, the four language-aware metrics and the top-1 failure
decomposition are reconstructable at evaluation time from the binary qrels
alone.

---

## 2. Environment setup

Different model families require different transformers runtimes. Three
pinned environments are provided as shell scripts under `setup/`. They
expect `uv` (`pip install uv`) and Python 3.10:

```bash
# (a) dense embedders (E5, BGE, Qwen3, Jina-v3, …)
bash setup/venv_main.sh                # creates ./.venv

# (b) GTE / Qwen2 family (needs transformers 4.x)
bash setup/venv_gte.sh                 # creates ./.venv-gte

# (c) ColBERT + sparse retrievers (PyLate, bm25s, SparseEncoder)
bash setup/venv_colbert_sparse.sh      # creates ./.venv-colbert-sparse

# convenience: install all three sequentially
bash setup/venv_all.sh
```

Each script ends with a verification block that imports the key packages
and prints the installed versions, so a green run means you are ready to
evaluate.

After the main environment is built, install MLAIRE itself in editable
mode (it is a thin pure-Python package — no extra deps beyond what the
venv scripts install):

```bash
.venv/bin/pip install -e .
```

---

## 3. Running an evaluation

The entry point is the `mlaire` CLI. All configs assume the data is
fetched from the Hugging Face Hub repos listed above; no manual download
is needed.

### 3.1 Dense embedders

```bash
# MLQA — top-20
.venv/bin/mlaire \
    --config mlaire/configs/mlqa.yaml \
    --model intfloat/multilingual-e5-small \
    --output results/intfloat_multilingual-e5-small_mlqa.json

# XQuAD — top-20
.venv/bin/mlaire \
    --config mlaire/configs/xquad.yaml \
    --model intfloat/multilingual-e5-small \
    --output results/intfloat_multilingual-e5-small_xquad.json

# Belebele — top-200, all 122 languages
.venv/bin/mlaire \
    --config mlaire/configs/belebele.yaml \
    --model intfloat/multilingual-e5-small \
    --output results/intfloat_multilingual-e5-small_belebele.json
```

GPU selection follows the standard CUDA convention:

```bash
CUDA_VISIBLE_DEVICES=4 .venv/bin/mlaire \
    --config mlaire/configs/mlqa.yaml \
    --model Qwen/Qwen3-Embedding-8B
```

Models that need transformers 4.x (GTE family, Qwen2-instruct
embedders, jina-v3) should be run from the alternative environment:

```bash
.venv-gte/bin/mlaire \
    --config mlaire/configs/mlqa.yaml \
    --model Alibaba-NLP/gte-multilingual-base
```

### 3.2 Sparse retrievers (BM25, OpenSearch neural-sparse)

```bash
.venv-colbert-sparse/bin/python sparse/bm25_eval.py \
    --config mlaire/configs/mlqa.yaml \
    --output results/bm25_mlqa.json

.venv-colbert-sparse/bin/python sparse/opensearch_eval.py \
    --config mlaire/configs/xquad.yaml \
    --output results/opensearch_xquad.json
```

### 3.3 Late-interaction (ColBERT)

```bash
.venv-colbert-sparse/bin/python colbert/jina_colbert_eval.py \
    --config mlaire/configs/belebele.yaml \
    --output results/jina_colbert_v2_belebele.json

.venv-colbert-sparse/bin/python colbert/lfm2_colbert_eval.py \
    --config mlaire/configs/mlqa.yaml \
    --output results/lfm2_colbert_350m_mlqa.json
```

### 3.4 Output

Every evaluation writes a single JSON file with the structure

```jsonc
{
  "aggregate":    {"mrr": ..., "lpr": ..., "ndcg@20": ..., "recall@20": ..., "lang_ndcg@20": ..., ... },
  "per_language": {"en": {... per-lang metrics ...}, "de": {...}, ...},
  "per_dataset":  {"mlqa": {... per-dataset metrics ...}},
  "diagnostics":  {
      "failure_types":          {"perfect": ..., "lang_fail": ..., "sem_fail": ..., "both_fail": ...},
      "confusion_matrix":       {"<query-lang>": {"<retrieved-lang>": count}},
      "lpr_confusion_matrix":   {"<query-lang>": {"<first-relevant-lang>": count}},
      "score_distribution":     {"top_20": {"3": ..., "2": ..., "1": ..., "0": ...}}
  },
  "meta": {"model_name": ..., "datasets": [...], "languages": [...], "timestamp": ...}
}
```

---

## 4. Configuration reference

Each YAML in `mlaire/configs/` is a small declarative file:

```yaml
datasets:
  mlqa:
    languages: [en, ar, de, es, hi, vi, zh]   # subset (or "all" for Belebele)
k_values: [20]                                # k-set for nDCG / Recall reporting
retrieval_top_k: 200                          # rank depth used to compute MRR / LPR
output_dir: results
# data_dir: ./data                            # optional: load BEIR files from disk instead of HF
# use_canonical_loaders: true                 # optional: rederive from upstream HF datasets
```

`retrieval_top_k` is decoupled from `k_values` because MRR and LPR depend
on the *full* first-relevant rank, not on the truncated top-k — the
default of `2 * max(k_values)` is sometimes too shallow and the
ship-with-config values (200 / 200 / 2000) match the rank depth used to
produce the canonical paper numbers.

---

## 5. Reproducing the paper

A representative paper number can be reproduced end to end with no
additional arguments:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/mlaire \
    --config mlaire/configs/mlqa.yaml \
    --model intfloat/multilingual-e5-small \
    --output results/intfloat_multilingual-e5-small_mlqa.json
```

The resulting JSON's `aggregate.lpr`, `aggregate.ndcg@20`, etc. are the
exact numbers that underlie the corresponding row of Table 1 / Figure 6 /
Figure 7 in the paper. Bit-exact reproduction has been verified for
MLQA and XQuAD on `intfloat/multilingual-e5-small`; Belebele agrees to
better than 1 × 10⁻⁴ on every metric (residual cuBLAS non-determinism on
the larger 60k×110k similarity tensor).

---

## 6. Optional: regenerate data locally

If you would rather not depend on the Hugging Face Hub at all, the
canonical loaders can rederive the BEIR files from the upstream MLQA /
XQuAD / Belebele datasets:

```bash
# 1. Build local MTEB-style parquet files under ./data/<dataset>/
.venv/bin/python scripts/export_to_beir.py --out ./data

# 2. Sanity-check that they round-trip exactly through BeirLoader
.venv/bin/python scripts/verify_beir.py --source ./data

# 3. Point the CLI at the local copy
.venv/bin/mlaire --config mlaire/configs/mlqa.yaml \
    --model intfloat/multilingual-e5-small \
    --data-dir ./data
```
