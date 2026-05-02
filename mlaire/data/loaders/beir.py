"""MTEB-style BEIR loader (parquet + multi-config layout).

Reads three Hugging Face dataset configs (``corpus`` / ``queries`` /
``qrels``) from either:

  * a Hugging Face Hub repo id (e.g.
    ``anonymous-submission-nips/mlaire-mlqa``) — files are auto-downloaded
    and cached by ``datasets.load_dataset``, or
  * a local directory containing the same MTEB-style layout
    (``corpus/test-*.parquet``, ``queries/test-*.parquet``,
    ``qrels/test-*.parquet``).

We deliberately do not use ``qrels`` at load time: every relevance judgement
encoded there is the deterministic consequence of
``Passage.group_id == Query.group_id``, which is already explicit on the
returned dataclasses. The ``qrels`` config is shipped only for downstream
BEIR/MTEB tooling that expects it.
"""

import logging
from pathlib import Path

from mlaire.data.loaders.base import BaseLoader
from mlaire.types import Passage, Query

logger = logging.getLogger(__name__)


def _looks_like_hf_repo(s: str) -> bool:
    """Heuristic: '<org>/<name>' shape and not an existing local path."""
    if Path(s).exists():
        return False
    return s.count("/") >= 1 and not s.startswith((".", "/"))


class BeirLoader(BaseLoader):
    """Load a single dataset (mlqa / xquad / belebele) in MTEB BEIR layout.

    Parameters
    ----------
    source : str or Path
        Local directory or HF Hub repo id. Backwards compatibility: the
        keyword ``data_dir`` is also accepted.
    dataset : str
        Dataset short name; recorded on every returned Passage / Query.
    languages : list[str], optional
        If provided, only entries whose ``language`` is in this list are
        returned. ``None`` (default) returns every language present in the
        files.
    """

    def __init__(
        self,
        source: str | Path | None = None,
        dataset: str = "",
        languages: list[str] | None = None,
        *,
        data_dir: str | Path | None = None,
    ):
        if source is None and data_dir is None:
            raise TypeError("BeirLoader requires `source` (or legacy `data_dir`).")
        if source is None:
            source = data_dir

        super().__init__(languages=list(languages) if languages else [])
        self.dataset = dataset
        self._source_str = str(source)
        self._is_hf = _looks_like_hf_repo(self._source_str)

    def _load_config(self, config_name: str):
        from datasets import load_dataset

        if self._is_hf:
            return load_dataset(self._source_str, config_name, split="test")
        # Local: point load_dataset at the parquet files for this config.
        local = Path(self._source_str) / config_name
        files = sorted(str(p) for p in local.glob("test-*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"No parquet files found at {local}/test-*.parquet"
            )
        return load_dataset("parquet", data_files=files, split="train")

    def load(self) -> tuple[list[Passage], list[Query]]:
        lang_filter = set(self.languages) if self.languages else None

        corpus = self._load_config("corpus")
        passages: list[Passage] = []
        for row in corpus:
            lang = row["language"]
            if lang_filter and lang not in lang_filter:
                continue
            passages.append(
                Passage(
                    passage_id=row["_id"],
                    group_id=row["group_id"],
                    language=lang,
                    text=row["text"],
                    dataset=row.get("dataset", self.dataset),
                )
            )

        queries_ds = self._load_config("queries")
        queries: list[Query] = []
        for row in queries_ds:
            lang = row["language"]
            if lang_filter and lang not in lang_filter:
                continue
            queries.append(
                Query(
                    query_id=row["_id"],
                    group_id=row["group_id"],
                    language=lang,
                    text=row["text"],
                    dataset=row.get("dataset", self.dataset),
                )
            )

        logger.info(
            "BeirLoader [%s]: %d passages, %d queries from %s",
            self.dataset, len(passages), len(queries), self._source_str,
        )
        return passages, queries
