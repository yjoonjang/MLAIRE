from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def encode_queries(self, texts: list[str]) -> np.ndarray:
        ...

    @abstractmethod
    def encode_passages(self, texts: list[str]) -> np.ndarray:
        ...
