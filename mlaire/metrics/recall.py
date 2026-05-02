RankedResult = tuple[str, str, str, float]  # (passage_id, group_id, language, score)


def compute_recall(
    ranked_results: list[RankedResult],
    relevant_group_id: str,
    k: int,
    total_relevant: int | None = None,
) -> float:
    """Recall@k: proportion of relevant passages (any language) in top-k.

    Denominator is the true total relevant count in the corpus (consistent with
    sentence-transformers InformationRetrievalEvaluator: num_correct / len(query_relevant_docs)).
    """
    if total_relevant is None:
        total_relevant = sum(
            1 for _, group_id, _, _ in ranked_results
            if group_id == relevant_group_id
        )
    if total_relevant == 0:
        return 0.0
    found = sum(
        1 for _, group_id, _, _ in ranked_results[:k]
        if group_id == relevant_group_id
    )
    return found / total_relevant


def compute_lang_recall(
    ranked_results: list[RankedResult],
    query_lang: str,
    relevant_group_id: str,
    k: int,
) -> float:
    """Lang-Recall@k: is the query-language relevant passage in top-k?"""
    for _, group_id, lang, _ in ranked_results[:k]:
        if group_id == relevant_group_id and lang == query_lang:
            return 1.0
    return 0.0
