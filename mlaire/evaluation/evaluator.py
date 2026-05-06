"""Standard retrieval-metric aggregation.

LPR (Language Preference Rate) and the ``lpr_confusion_matrix`` diagnostic
are deliberately NOT computed here. LPR is

    LPR(q) = 1 if argmax_{p in group(q)} score(q, p) has same language as q

i.e. it is computed *over the entire relevant group*, regardless of the
top-k retrieval cutoff this evaluator operates on. Mixing it in here would
make the value silently top-k dependent. Use the dedicated scripts under
``measure_lpr/`` (e.g. ``measure_lpr/queue_runner.py``) to compute LPR and
the language confusion matrix correctly.
"""

from collections import defaultdict

from mlaire.types import Query
from mlaire.metrics.mrr import compute_mrr, compute_lang_mrr
from mlaire.metrics.ndcg import compute_ndcg, compute_lang_ndcg
from mlaire.metrics.recall import compute_recall, compute_lang_recall
from mlaire.evaluation.diagnostics import (
    classify_failure,
    compute_confusion_matrix,
    compute_score_distribution,
)

RankedResult = tuple[str, str, str, float]
QueryResult = tuple[str, list[RankedResult]]


class Evaluator:
    def __init__(self, k_values: list[int] | None = None):
        self.k_values = k_values or [1, 5, 10, 20]

    def evaluate(
        self,
        retrieval_results: list[QueryResult],
        queries: list[Query],
        group_counts: dict[str, int] | None = None,
    ) -> dict:
        query_map = {q.query_id: q for q in queries}

        # Standard retrieval metrics (language-agnostic)
        all_mrr = []
        all_ndcg: dict[int, list[float]] = {k: [] for k in self.k_values}
        all_recall: dict[int, list[float]] = {k: [] for k in self.k_values}

        # Language-aware metrics (excluding LPR — see module docstring)
        all_lang_mrr = []
        all_lang_ndcg: dict[int, list[float]] = {k: [] for k in self.k_values}
        all_lang_recall: dict[int, list[float]] = {k: [] for k in self.k_values}

        failure_counts = {"perfect": 0, "lang_fail": 0, "sem_fail": 0, "both_fail": 0}
        lang_fail_cases: list[dict[str, str]] = []
        score_dist_per_k: dict[int, dict[int, int]] = {
            k: {3: 0, 2: 0, 1: 0, 0: 0} for k in self.k_values
        }

        def _make_metrics() -> dict[str, list[float]]:
            return {
                "mrr": [],
                "lang_mrr": [],
                **{f"ndcg@{k}": [] for k in self.k_values},
                **{f"recall@{k}": [] for k in self.k_values},
                **{f"lang_ndcg@{k}": [] for k in self.k_values},
                **{f"lang_recall@{k}": [] for k in self.k_values},
            }

        per_language: dict[str, dict[str, list[float]]] = defaultdict(_make_metrics)
        per_dataset: dict[str, dict[str, list[float]]] = defaultdict(_make_metrics)

        for query_id, ranked in retrieval_results:
            query = query_map[query_id]
            qlang = query.language
            gid = query.group_id
            dataset = query.dataset
            total_relevant = group_counts.get(gid) if group_counts else None

            mrr = compute_mrr(ranked, qlang, gid)
            lang_mrr = compute_lang_mrr(ranked, qlang, gid)
            all_mrr.append(mrr)
            all_lang_mrr.append(lang_mrr)

            per_language[qlang]["mrr"].append(mrr)
            per_language[qlang]["lang_mrr"].append(lang_mrr)
            per_dataset[dataset]["mrr"].append(mrr)
            per_dataset[dataset]["lang_mrr"].append(lang_mrr)

            for k in self.k_values:
                # Standard (language-agnostic)
                ndcg = compute_ndcg(ranked, qlang, gid, k, total_relevant=total_relevant)
                recall = compute_recall(ranked, gid, k, total_relevant=total_relevant)
                all_ndcg[k].append(ndcg)
                all_recall[k].append(recall)

                # Language-aware
                l_ndcg = compute_lang_ndcg(ranked, qlang, gid, k, total_relevant=total_relevant)
                l_recall = compute_lang_recall(ranked, qlang, gid, k)
                all_lang_ndcg[k].append(l_ndcg)
                all_lang_recall[k].append(l_recall)

                per_language[qlang][f"ndcg@{k}"].append(ndcg)
                per_language[qlang][f"recall@{k}"].append(recall)
                per_language[qlang][f"lang_ndcg@{k}"].append(l_ndcg)
                per_language[qlang][f"lang_recall@{k}"].append(l_recall)

                per_dataset[dataset][f"ndcg@{k}"].append(ndcg)
                per_dataset[dataset][f"recall@{k}"].append(recall)
                per_dataset[dataset][f"lang_ndcg@{k}"].append(l_ndcg)
                per_dataset[dataset][f"lang_recall@{k}"].append(l_recall)

            # Diagnostics — top-1 4-way classification + lang_fail confusion +
            # top-k score distribution. None of these are LPR-derived.
            if ranked:
                failure = classify_failure(ranked[0], qlang, gid)
                failure_counts[failure] += 1
                if failure == "lang_fail":
                    lang_fail_cases.append({
                        "query_lang": qlang,
                        "retrieved_lang": ranked[0][2],
                    })

            for k in self.k_values:
                dist = compute_score_distribution(ranked, qlang, gid, k)
                for grade, count in dist.items():
                    score_dist_per_k[k][grade] += count

        total = len(retrieval_results) or 1

        def _avg_metrics(metrics: dict[str, list[float]]) -> dict[str, float]:
            return {
                name: (sum(vals) / len(vals) if vals else 0.0)
                for name, vals in metrics.items()
            }

        per_lang_avg = {lang: _avg_metrics(m) for lang, m in per_language.items()}
        per_dataset_avg = {ds: _avg_metrics(m) for ds, m in per_dataset.items()}

        metric_names = ["mrr", "lang_mrr"]
        for k in self.k_values:
            metric_names += [f"ndcg@{k}", f"recall@{k}", f"lang_ndcg@{k}", f"lang_recall@{k}"]

        num_datasets = len(per_dataset_avg) or 1
        aggregate = {}
        for metric in metric_names:
            aggregate[metric] = sum(
                ds_metrics.get(metric, 0.0) for ds_metrics in per_dataset_avg.values()
            ) / num_datasets

        diagnostics = {
            "failure_types": {k: v / total for k, v in failure_counts.items()},
            "confusion_matrix": compute_confusion_matrix(lang_fail_cases),
            "score_distribution": {
                f"top_{k}": score_dist_per_k[k] for k in self.k_values
            },
        }

        return {
            "aggregate": aggregate,
            "per_language": per_lang_avg,
            "per_dataset": per_dataset_avg,
            "diagnostics": diagnostics,
        }
