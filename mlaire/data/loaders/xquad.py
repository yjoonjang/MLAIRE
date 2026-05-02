from datasets import load_dataset

from mlaire.data.loaders.base import BaseLoader
from mlaire.types import Passage, Query


class XQuADLoader(BaseLoader):
    DATASET_NAME = "google/xquad"

    def load(self) -> tuple[list[Passage], list[Query]]:
        passages: list[Passage] = []
        queries: list[Query] = []

        for lang in self.languages:
            config = f"xquad.{lang}"
            ds = load_dataset(self.DATASET_NAME, config, trust_remote_code=True)
            if hasattr(ds, "keys"):
                split = list(ds.keys())[0]
                ds = ds[split]

            seen_contexts: dict[str, int] = {}
            paragraph_idx = 0

            for row in ds:
                context = row["context"]

                if context not in seen_contexts:
                    seen_contexts[context] = paragraph_idx
                    idx = paragraph_idx
                    paragraph_idx += 1

                    group_id = f"xquad_{idx}"
                    passages.append(Passage(
                        passage_id=f"xquad_{idx}_{lang}",
                        group_id=group_id,
                        language=lang,
                        text=context,
                        dataset="xquad",
                    ))
                else:
                    idx = seen_contexts[context]
                    group_id = f"xquad_{idx}"

                queries.append(Query(
                    query_id=f"xquad_q{row['id']}_{lang}",
                    group_id=group_id,
                    language=lang,
                    text=row["question"],
                    dataset="xquad",
                ))

        return passages, queries
