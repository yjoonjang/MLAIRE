from mlaire.types import Passage, Query, RetrievalPool


class PoolBuilder:
    def __init__(self):
        self._passages: list[Passage] = []
        self._queries: list[Query] = []

    def add_passages(self, passages: list[Passage]) -> None:
        self._passages.extend(passages)

    def add_queries(self, queries: list[Query]) -> None:
        self._queries.extend(queries)

    def build(
        self,
        languages: list[str] | None = None,
        datasets: list[str] | None = None,
    ) -> RetrievalPool:
        passages = self._passages
        queries = self._queries

        if languages is not None:
            lang_set = set(languages)
            passages = [p for p in passages if p.language in lang_set]
            queries = [q for q in queries if q.language in lang_set]

        if datasets is not None:
            ds_set = set(datasets)
            passages = [p for p in passages if p.dataset in ds_set]
            queries = [q for q in queries if q.dataset in ds_set]

        return RetrievalPool(passages=passages, queries=queries)
