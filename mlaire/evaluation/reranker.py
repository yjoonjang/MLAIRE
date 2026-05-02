RankedResult = tuple[str, str, str, float]


def language_aware_rerank(
    ranked_results: list[RankedResult],
    query_lang: str,
    alpha: float,
) -> list[RankedResult]:
    """Rerank by: alpha * relevance_score + (1 - alpha) * language_match."""
    reranked = []
    for pid, gid, lang, score in ranked_results:
        lang_score = 1.0 if lang == query_lang else 0.0
        new_score = alpha * score + (1 - alpha) * lang_score
        reranked.append((pid, gid, lang, new_score))

    reranked.sort(key=lambda x: -x[3])
    return reranked


def sweep_alpha(
    ranked_results: list[RankedResult],
    query_lang: str,
    alphas: list[float] | None = None,
) -> list[tuple[float, list[RankedResult]]]:
    """Run reranking with multiple alpha values."""
    if alphas is None:
        alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    return [
        (alpha, language_aware_rerank(ranked_results, query_lang, alpha))
        for alpha in alphas
    ]
