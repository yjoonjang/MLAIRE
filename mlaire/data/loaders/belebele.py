import logging

from datasets import load_dataset

from mlaire.data.loaders.base import BaseLoader
from mlaire.language_metadata import BELEBELE_REVERSE_MAP
from mlaire.types import Passage, Query

logger = logging.getLogger(__name__)


class BelebeleLoader(BaseLoader):
    DATASET_NAME = "facebook/belebele"

    def load(self) -> tuple[list[Passage], list[Query]]:
        passages: list[Passage] = []
        queries: list[Query] = []

        # Cross-language group mapping: link -> group_id
        # Each unique link corresponds to one passage (488 unique passages).
        # Multiple questions can share the same passage.
        link_to_group: dict[str, str] = {}
        group_counter = 0
        # Track seen passages to avoid duplicates: (group_id, lang)
        seen_passages: set[tuple[str, str]] = set()

        for lang in self.languages:
            flores_code = BELEBELE_REVERSE_MAP.get(lang)
            if flores_code is None:
                continue

            try:
                ds = load_dataset(self.DATASET_NAME, flores_code)
            except (ValueError, FileNotFoundError) as e:
                logger.warning("Skipping %s (%s): %s", lang, flores_code, e)
                continue

            if hasattr(ds, "keys"):
                split = list(ds.keys())[0]
                ds = ds[split]

            for row in ds:
                link = row["link"]

                if link not in link_to_group:
                    link_to_group[link] = f"belebele_{group_counter}"
                    group_counter += 1

                group_id = link_to_group[link]

                # One passage per (group, language) — deduplicate
                passage_key = (group_id, lang)
                if passage_key not in seen_passages:
                    seen_passages.add(passage_key)
                    passages.append(Passage(
                        passage_id=f"{group_id}_{lang}",
                        group_id=group_id,
                        language=lang,
                        text=row["flores_passage"],
                        dataset="belebele",
                    ))

                queries.append(Query(
                    query_id=f"belebele_q{group_id}_{row['question_number']}_{lang}",
                    group_id=group_id,
                    language=lang,
                    text=row["question"],
                    dataset="belebele",
                ))

        return passages, queries
