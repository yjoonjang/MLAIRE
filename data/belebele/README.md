---
license: mit
task_categories:
  - text-retrieval
multilinguality: multilingual
language:
  - acm
  - af
  - am
  - apc
  - ar
  - ars
  - ary
  - arz
  - as
  - az
  - bg
  - bm
  - bn
  - bo
  - ca
  - ceb
  - ckb
  - cs
  - da
  - de
  - el
  - en
  - es
  - et
  - eu
  - fa
  - fi
  - fr
  - fuv
  - gaz
  - gn
  - gu
  - ha
  - he
  - hi
  - hr
  - ht
  - hu
  - hy
  - id
  - ig
  - ilo
  - is
  - it
  - ja
  - jv
  - ka
  - kac
  - kea
  - kk
  - km
  - kn
  - ko
  - ky
  - lg
  - ln
  - lo
  - lt
  - luo
  - lv
  - mg
  - mi
  - mk
  - ml
  - mn
  - mr
  - ms
  - mt
  - my
  - nb
  - ne
  - nl
  - nso
  - ny
  - or
  - pa
  - pl
  - ps
  - pt
  - ro
  - ru
  - rw
  - sd
  - shn
  - si
  - sk
  - sl
  - sn
  - so
  - sq
  - sr
  - ss
  - st
  - su
  - sv
  - sw
  - ta
  - te
  - tg
  - th
  - ti
  - tl
  - tn
  - tr
  - ts
  - uk
  - ur
  - uz
  - vi
  - war
  - wo
  - xh
  - yo
  - zh
  - zu
language_bcp47:
  - ar-Latn
  - bn-Latn
  - hi-Latn
  - ne-Latn
  - si-Latn
  - ur-Latn
  - zh-Hant
pretty_name: MLAIRE-BELEBELE
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

# MLAIRE-BELEBELE

Belebele reformatted for language-aware retrieval evaluation. 488 underlying passages, each available in 122 languages (joined globally by the original `link` field). Relevance is encoded by group_id matching.

This repository is part of the **MLAIRE** benchmark, submitted anonymously
to the NeurIPS 2026 Evaluations & Datasets Track. Authors and affiliations
are withheld for double-blind review.

## Default top-k

Reported metrics in the paper use **top-200**.

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

corpus  = load_dataset("anonymous-submission-nips/mlaire-belebele", "corpus",  split="test")
queries = load_dataset("anonymous-submission-nips/mlaire-belebele", "queries", split="test")
qrels   = load_dataset("anonymous-submission-nips/mlaire-belebele", "qrels",   split="test")
```

Or simply install the companion package and run

```bash
mlaire --config mlaire/configs/belebele.yaml --model <hf-model-id>
```

which downloads these files automatically and reproduces every metric we
report in the paper.
