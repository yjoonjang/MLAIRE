---
license: mit
task_categories:
  - text-retrieval
multilinguality: multilingual
language:
  - en
  - ar
  - de
  - es
  - hi
  - vi
  - zh
pretty_name: MLAIRE-MLQA
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

# MLAIRE-MLQA

MLQA reformatted for language-aware retrieval evaluation. Passages are deduplicated at the context level via union-find on the original MLQA ids. Relevance is encoded by group_id matching.

This repository is part of the **MLAIRE** benchmark, submitted anonymously
to the NeurIPS 2026 Evaluations & Datasets Track. Authors and affiliations
are withheld for double-blind review.

## Default top-k

Reported metrics in the paper use **top-20**.

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

corpus  = load_dataset("anonymous-submission-nips/mlaire-mlqa", "corpus",  split="test")
queries = load_dataset("anonymous-submission-nips/mlaire-mlqa", "queries", split="test")
qrels   = load_dataset("anonymous-submission-nips/mlaire-mlqa", "qrels",   split="test")
```

Or simply install the companion package and run

```bash
mlaire --config mlaire/configs/mlqa.yaml --model <hf-model-id>
```

which downloads these files automatically and reproduces every metric we
report in the paper.
