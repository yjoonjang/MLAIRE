RankedResult = tuple[str, str, str, float]  # (passage_id, group_id, language, score)


def compute_lpr(
    ranked_results: list[RankedResult],
    query_lang: str,
    relevant_group_id: str,
) -> float:
    """Compute Language Preference Rate for a single query.

    Returns 1.0 if the query-language passage ranks first within
    the relevant semantic group, 0.0 otherwise.
    """
    # Find first passage in the relevant group
    for _, group_id, lang, _ in ranked_results:
        if group_id == relevant_group_id:
            return 1.0 if lang == query_lang else 0.0
    return 0.0
