RankedResult = tuple[str, str, str, float]  # (passage_id, group_id, language, score)


def compute_mrr(
    ranked_results: list[RankedResult],
    query_lang: str,
    relevant_group_id: str,
) -> float:
    """Compute MRR: reciprocal rank of first relevant passage (any language)."""
    for rank, (_, group_id, lang, _) in enumerate(ranked_results, start=1):
        if group_id == relevant_group_id:
            return 1.0 / rank
    return 0.0


def compute_lang_mrr(
    ranked_results: list[RankedResult],
    query_lang: str,
    relevant_group_id: str,
) -> float:
    """Compute Lang-MRR: reciprocal rank of first same-language relevant passage."""
    for rank, (_, group_id, lang, _) in enumerate(ranked_results, start=1):
        if group_id == relevant_group_id and lang == query_lang:
            return 1.0 / rank
    return 0.0
