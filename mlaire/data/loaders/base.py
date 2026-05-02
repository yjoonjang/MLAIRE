from abc import ABC, abstractmethod
from mlaire.types import Passage, Query


class BaseLoader(ABC):
    def __init__(self, languages: list[str]):
        self.languages = languages

    @abstractmethod
    def load(self) -> tuple[list[Passage], list[Query]]:
        """Load dataset and return (passages, queries)."""
        ...
