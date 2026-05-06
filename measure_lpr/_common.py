"""Shared utilities for the MLAIRE measure_lpr scripts.

This module is the source of truth for the canonical LPR definition

    LPR(q) = 1 if argmax_{p in group(q)} score(q, p) has same language as q

The per-backend scripts (compute_lpr_dense / _bm25 / _opensearch / _colbert)
provide a ``score_group(query_idx, rel_indices) -> np.ndarray`` callback
and ``aggregate_lpr`` in this module takes care of:

  * iterating every query in the pool,
  * indexing into its relevant group via ``Passage.group_id``,
  * argmax → language comparison,
  * accumulating per-language LPR + a confusion matrix
    {query_lang: {best_first_relevant_lang: count}}.

The headline ``aggregate_lpr`` is a query-weighted (micro) mean — this
matches what mlaire's main eval used to write before LPR was moved out
to this dedicated pipeline. ``aggregate_lpr_macro`` is also exposed for
transparency.
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mlaire.data.loaders.beir import BeirLoader  # noqa: E402
from mlaire.data.processors.pool_builder import PoolBuilder  # noqa: E402
from mlaire.language_metadata import (  # noqa: E402
    BELEBELE_LANG_MAP,
    MLQA_LANGUAGES,
    XQUAD_LANGUAGES,
    get_language_info,
)
from mlaire.types import RetrievalPool  # noqa: E402

# By default, fetch BEIR-format files from the public Hugging Face repos
# that ship alongside this code. Override with --data-dir on the per-
# backend script if you want to evaluate against a local export.
HF_REPO_PREFIX = "anonymous-submission-nips/mlaire"

LANGS_MAP = {
    "mlqa": list(MLQA_LANGUAGES),
    "xquad": list(XQUAD_LANGUAGES),
    "belebele": "all",  # all 122 in BELEBELE_LANG_MAP
}


def load_pool(dataset: str, source: str | None = None) -> RetrievalPool:
    """Load a RetrievalPool for the given dataset.

    Parameters
    ----------
    dataset
        ``mlqa`` / ``xquad`` / ``belebele``.
    source
        Either a Hugging Face dataset repo id (e.g.
        ``anonymous-submission-nips/mlaire-mlqa``) or a local directory
        with the MTEB-style parquet layout. ``None`` (default) resolves
        to ``HF_REPO_PREFIX-{dataset}``.
    """
    langs = LANGS_MAP[dataset]
    if langs == "all":
        langs = list(set(BELEBELE_LANG_MAP.values()))
    if source is None:
        source = f"{HF_REPO_PREFIX}-{dataset}"
    loader = BeirLoader(source=source, dataset=dataset, languages=langs)
    passages, queries = loader.load()
    builder = PoolBuilder()
    builder.add_passages(passages)
    builder.add_queries(queries)
    return builder.build()


def build_group_index(pool: RetrievalPool) -> dict[str, list[int]]:
    group_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(pool.passages):
        group_to_indices[p.group_id].append(i)
    return group_to_indices


# Backend signature: scoring against the relevant group only.
ScoreGroupFn = Callable[[int, list[int]], np.ndarray]


def aggregate_lpr(
    pool: RetrievalPool,
    score_group: ScoreGroupFn,
    *,
    progress_every: int = 5000,
) -> dict:
    group_to_indices = build_group_index(pool)

    per_lang_lpr: dict[str, list[float]] = defaultdict(list)
    lpr_confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    n_skipped = 0

    n_queries = len(pool.queries)
    t0 = time.time()
    for qi, query in enumerate(pool.queries):
        rel_indices = group_to_indices.get(query.group_id, [])
        if not rel_indices:
            # Legitimate skip: this query has no relevant passages in the pool.
            n_skipped += 1
            continue

        scores = score_group(qi, rel_indices)
        if scores is None or len(scores) == 0:
            # Backend returned no scores for a non-empty rel_indices — that's
            # almost certainly a bug (slicing / dtype mismatch), not a real
            # "no relevant" condition. Fail loudly so it doesn't silently
            # under-count LPR.
            raise RuntimeError(
                f"score_group returned an empty array for query {qi} "
                f"({query.query_id}, lang={query.language}, "
                f"group={query.group_id}, |rel|={len(rel_indices)}). "
                f"Check the backend's slicing logic."
            )

        best_local = int(np.argmax(scores))
        best_p_idx = rel_indices[best_local]
        best_p = pool.passages[best_p_idx]

        is_same_lang = best_p.language == query.language
        per_lang_lpr[query.language].append(1.0 if is_same_lang else 0.0)
        lpr_confusion[query.language][best_p.language] += 1

        if (qi + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(
                f"  [{qi + 1:>7}/{n_queries}] elapsed={elapsed:.1f}s",
                flush=True,
            )

    overall: list[float] = []
    for vals in per_lang_lpr.values():
        overall.extend(vals)
    micro = float(np.mean(overall)) if overall else 0.0
    macro = (
        float(np.mean([float(np.mean(v)) for v in per_lang_lpr.values()]))
        if per_lang_lpr else 0.0
    )

    def _safe_lang_info(lang: str) -> dict:
        try:
            return get_language_info(lang)
        except Exception:
            return {"family": "unknown", "script": "unknown", "resource": "unknown"}

    return {
        "aggregate_lpr": micro,            # query-weighted (micro) mean
        "aggregate_lpr_macro": macro,      # mean of per-language LPRs
        "n_queries_evaluated": len(overall),
        "n_queries_skipped": n_skipped,
        "per_language": {
            lang: {
                "lpr": float(np.mean(vals)),
                "n_queries": len(vals),
                **_safe_lang_info(lang),
            }
            for lang, vals in per_lang_lpr.items()
        },
        "lpr_confusion_matrix": {
            ql: dict(tgts) for ql, tgts in lpr_confusion.items()
        },
    }


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {path}", flush=True)


def output_path(out_root: Path, dataset: str, model_name: str) -> Path:
    safe = model_name.replace("/", "_")
    return out_root / dataset / f"{safe}.json"
