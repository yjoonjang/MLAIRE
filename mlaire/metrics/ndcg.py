import math

RankedResult = tuple[str, str, str, float]  # (passage_id, group_id, language, score)


def _dcg(relevances: list[float], k: int) -> float:
    """Compute DCG@k."""
    total = 0.0
    for i, r in enumerate(relevances[:k]):
        total += (2**r - 1) / math.log2(i + 2)
    return total


def compute_ndcg(
    ranked_results: list[RankedResult],
    query_lang: str,
    relevant_group_id: str,
    k: int,
    total_relevant: int | None = None,
) -> float:
    """Compute standard NDCG@k using binary relevance (language-agnostic).

    A passage is relevant (1) if it belongs to the same group, regardless of language.
    IDCG is computed from the true total relevant count in the corpus (consistent with
    sentence-transformers InformationRetrievalEvaluator: true_relevances = [1] * len(relevant_docs)).
    """
    if total_relevant is None:
        total_relevant = sum(1 for _, gid, _, _ in ranked_results if gid == relevant_group_id)

    relevances = [
        1.0 if group_id == relevant_group_id else 0.0
        for _, group_id, lang, _ in ranked_results
    ]

    dcg = _dcg(relevances, k)
    # Ideal: top min(total_relevant, k) positions are all relevant
    ideal = [1.0] * total_relevant
    idcg = _dcg(ideal, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_lang_ndcg(
    ranked_results: list[RankedResult],
    query_lang: str,
    relevant_group_id: str,
    k: int,
    total_relevant: int | None = None,
) -> float:
    """Compute language-aware NDCG@k using graded relevance.

    Grading: 3=relevant+same_lang, 2=relevant+diff_lang, 0=irrelevant.
    IDCG places the same-lang relevant passage first (grade 3), then all
    other-lang relevant passages (grade 2), using the true total relevant count.
    """
    if total_relevant is None:
        total_relevant = sum(1 for _, gid, _, _ in ranked_results if gid == relevant_group_id)

    grades = []
    for _, group_id, lang, _ in ranked_results:
        if group_id == relevant_group_id:
            grades.append(3.0 if lang == query_lang else 2.0)
        else:
            grades.append(0.0)

    dcg = _dcg(grades, k)
    # Ideal: same-lang relevant first (grade 3), then other-lang relevant (grade 2)
    ideal = ([3.0] + [2.0] * (total_relevant - 1)) if total_relevant > 0 else []
    idcg = _dcg(ideal, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg
