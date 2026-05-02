from collections import defaultdict

RankedResult = tuple[str, str, str, float]


def classify_failure(
    top1: RankedResult,
    query_lang: str,
    relevant_group_id: str,
) -> str:
    """Classify top-1 result into: perfect, lang_fail, sem_fail, both_fail."""
    _, group_id, lang, _ = top1
    is_relevant = group_id == relevant_group_id
    is_same_lang = lang == query_lang

    if is_relevant and is_same_lang:
        return "perfect"
    elif is_relevant:
        return "lang_fail"
    elif is_same_lang:
        return "sem_fail"
    else:
        return "both_fail"


def compute_confusion_matrix(
    lang_fail_cases: list[dict[str, str]],
) -> dict[str, dict[str, int]]:
    """Build confusion matrix from lang_fail cases.

    Each case has: query_lang, retrieved_lang.
    Returns: {query_lang: {retrieved_lang: count}}.
    """
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for case in lang_fail_cases:
        matrix[case["query_lang"]][case["retrieved_lang"]] += 1
    return dict(matrix)


def compute_score_distribution(
    ranked_results: list[RankedResult],
    query_lang: str,
    relevant_group_id: str,
    k: int,
) -> dict[int, int]:
    """Count graded scores (3/2/1/0) in top-k results."""
    dist = {3: 0, 2: 0, 1: 0, 0: 0}
    for _, group_id, lang, _ in ranked_results[:k]:
        is_relevant = group_id == relevant_group_id
        is_same_lang = lang == query_lang
        if is_relevant and is_same_lang:
            dist[3] += 1
        elif is_relevant:
            dist[2] += 1
        elif is_same_lang:
            dist[1] += 1
        else:
            dist[0] += 1
    return dist
