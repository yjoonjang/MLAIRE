"""Shared utilities for SPARSE evaluation scripts.

Loads BEIR-format data, runs a user-provided retrieval function, and feeds the
output into the existing mlaire Evaluator so metrics match dense results.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import yaml

# Make mlaire package importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mlaire.data.loaders.beir import BeirLoader  # noqa: E402
from mlaire.data.processors.pool_builder import PoolBuilder  # noqa: E402
from mlaire.evaluation.evaluator import Evaluator  # noqa: E402
from mlaire.language_metadata import get_language_info  # noqa: E402
from mlaire.types import RetrievalPool  # noqa: E402

# retrieve_fn(pool, top_k) -> list[(query_id, [(pid, gid, lang, score), ...])]
RetrieveFn = Callable[[RetrievalPool, int], list]


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_pools(config: dict) -> list[tuple[str, RetrievalPool]]:
    """Load every dataset declared in the config as a RetrievalPool.

    Source resolution mirrors ``mlaire.cli``:

      * ``config["data_dir"]`` set     -> local BEIR export at ``<dir>/<ds>/``
      * else                           -> Hugging Face Hub
                                          (``<hf_repo_prefix>-<ds>``)
    """
    data_dir = config.get("data_dir")
    hf_repo_prefix = config.get(
        "hf_repo_prefix", "anonymous-submission-nips/mlaire"
    )
    pools = []
    for ds_name, ds_config in config["datasets"].items():
        langs = ds_config.get("languages") if isinstance(ds_config, dict) else None
        lang_filter = None if langs in (None, "all") else list(langs)
        if data_dir:
            source: str = str(Path(data_dir) / ds_name)
        else:
            source = f"{hf_repo_prefix}-{ds_name}"
        loader = BeirLoader(source=source, dataset=ds_name, languages=lang_filter)
        passages, queries = loader.load()
        builder = PoolBuilder()
        builder.add_passages(passages)
        builder.add_queries(queries)
        pools.append((ds_name, builder.build()))
    return pools


def run_and_save(
    config: dict,
    model_name: str,
    retrieve_fn: RetrieveFn,
    output_path: str,
) -> dict:
    k_values = config.get("k_values", [10, 20, 50, 100])
    max_k = max(k_values)
    # See cli.run_evaluation: retrieval depth is decoupled from k-set so
    # that MRR / LPR can be computed against the full first-relevant rank.
    top_k = int(config.get("retrieval_top_k", max_k * 2))

    all_results = []
    all_queries = []
    group_counts: dict[str, int] = {}

    for ds_name, pool in load_pools(config):
        print(
            f"[{ds_name}] {len(pool.passages)} passages, {len(pool.queries)} queries",
            flush=True,
        )
        ds_results = retrieve_fn(pool, top_k)
        all_results.extend(ds_results)
        all_queries.extend(pool.queries)
        for p in pool.passages:
            group_counts[p.group_id] = group_counts.get(p.group_id, 0) + 1

    evaluator = Evaluator(k_values=k_values)
    eval_results = evaluator.evaluate(
        all_results, all_queries, group_counts=group_counts
    )

    all_langs = sorted({q.language for q in all_queries})
    eval_results["meta"] = {
        "benchmark_tier": config.get("tier", "compact"),
        "model_name": model_name,
        "datasets": list(config["datasets"].keys()),
        "languages": all_langs,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    for lang in eval_results.get("per_language", {}):
        info = get_language_info(lang)
        eval_results["per_language"][lang].update(info)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)

    agg = eval_results["aggregate"]
    print(f"\nSaved to {output_path}")
    print(
        f"MRR: {agg['mrr']:.4f}  "
        f"NDCG@10: {agg.get('ndcg@10', 0):.4f}  "
        f"LPR: {agg['lpr']:.4f}  "
        f"Lang-NDCG@10: {agg.get('lang_ndcg@10', 0):.4f}"
    )
    return eval_results
