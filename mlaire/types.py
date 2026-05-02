from dataclasses import dataclass, field


@dataclass(frozen=True)
class Passage:
    passage_id: str
    group_id: str
    language: str
    text: str
    dataset: str


@dataclass(frozen=True)
class Query:
    query_id: str
    group_id: str
    language: str
    text: str
    dataset: str


@dataclass
class RetrievalPool:
    passages: list[Passage]
    queries: list[Query]

    def __post_init__(self):
        self._group_index: dict[str, list[Passage]] = {}
        for p in self.passages:
            self._group_index.setdefault(p.group_id, []).append(p)

    def get_relevant_passages(self, query: Query) -> list[Passage]:
        return self._group_index.get(query.group_id, [])

    def get_same_lang_relevant(self, query: Query) -> list[Passage]:
        return [
            p for p in self.get_relevant_passages(query)
            if p.language == query.language
        ]
