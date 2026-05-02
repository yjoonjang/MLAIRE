import logging
import os
import time

import numpy as np

try:
    import openai
except ImportError:
    openai = None  # type: ignore

try:
    import cohere
except ImportError:
    cohere = None  # type: ignore

from mlaire.models.base import BaseModel

logger = logging.getLogger(__name__)

_OPENAI_BATCH_SIZE = 2048
_COHERE_BATCH_SIZE = 96
_MAX_RETRIES = 3


class DenseAPIModel(BaseModel):
    def __init__(self, model_name: str, provider: str):
        super().__init__(model_name)
        self.provider = provider
        if provider == "openai":
            if openai is None:
                raise ImportError("openai package required: pip install openai")
            self._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        elif provider == "cohere":
            if cohere is None:
                raise ImportError("cohere package required: pip install cohere")
            self._client = cohere.ClientV2(api_key=os.environ.get("COHERE_API_KEY"))
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, input_type="query")

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, input_type="passage")

    def _encode(self, texts: list[str], input_type: str) -> np.ndarray:
        if self.provider == "openai":
            return self._encode_openai(texts)
        elif self.provider == "cohere":
            return self._encode_cohere(texts, input_type)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _encode_openai(self, texts: list[str]) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), _OPENAI_BATCH_SIZE):
            batch = texts[i : i + _OPENAI_BATCH_SIZE]
            logger.debug("OpenAI embed batch %d/%d (size=%d)", i // _OPENAI_BATCH_SIZE + 1, -(-len(texts) // _OPENAI_BATCH_SIZE), len(batch))
            response = self._retry(lambda b=batch: self._client.embeddings.create(model=self.model_name, input=b))
            all_embeddings.append(np.array([d.embedding for d in response.data]))
        return np.concatenate(all_embeddings, axis=0)

    def _encode_cohere(self, texts: list[str], input_type: str) -> np.ndarray:
        cohere_type = "search_query" if input_type == "query" else "search_document"
        all_embeddings = []
        for i in range(0, len(texts), _COHERE_BATCH_SIZE):
            batch = texts[i : i + _COHERE_BATCH_SIZE]
            logger.debug("Cohere embed batch %d/%d (size=%d)", i // _COHERE_BATCH_SIZE + 1, -(-len(texts) // _COHERE_BATCH_SIZE), len(batch))
            response = self._retry(
                lambda b=batch: self._client.embed(
                    model=self.model_name,
                    texts=b,
                    input_type=cohere_type,
                    embedding_types=["float"],
                )
            )
            all_embeddings.append(np.array(response.embeddings.float_))
        return np.concatenate(all_embeddings, axis=0)

    def _retry(self, fn, max_retries: int = _MAX_RETRIES):
        wait = 1.0
        for attempt in range(max_retries + 1):
            try:
                return fn()
            except Exception as exc:
                if attempt == max_retries:
                    raise
                logger.warning("API call failed (attempt %d/%d): %s — retrying in %.1fs", attempt + 1, max_retries + 1, exc, wait)
                time.sleep(wait)
                wait *= 2
