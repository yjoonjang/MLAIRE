"""MLQA loader with context-level deduplication.

MLQA's ``id`` field identifies a (question, context) pair, not a context.
Multiple questions can share the same context, which means the naive
``id``-based passage construction produces duplicate passages (and therefore
duplicate group_ids) for a single underlying context.

This loader fixes the issue with a union-find pass: within each language,
``id``s that share identical context text are merged into one group, so
each (group, lang) pair corresponds to exactly one canonical passage.
"""

from collections import defaultdict

from datasets import load_dataset

from mlaire.data.loaders.base import BaseLoader
from mlaire.types import Passage, Query


class MLQALoader(BaseLoader):
    DATASET_NAME = "facebook/mlqa"

    def load(self) -> tuple[list[Passage], list[Query]]:
        # ---- Step 1: load all rows for all languages ------------------------
        rows_by_lang: dict[str, list] = {}
        for lang in self.languages:
            config = f"mlqa.{lang}.{lang}"
            ds = load_dataset(
                self.DATASET_NAME, config, split="test", trust_remote_code=True
            )
            rows_by_lang[lang] = list(ds)

        # ---- Step 2: Union-Find on MLQA ids ---------------------------------
        # Start: each id is its own group.
        # Merge: within a language, ids sharing identical context text → same group.
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for lang in self.languages:
            for row in rows_by_lang[lang]:
                mid = row["id"]
                if mid not in parent:
                    parent[mid] = mid

        for lang in self.languages:
            text_to_ids: dict[str, list[str]] = defaultdict(list)
            for row in rows_by_lang[lang]:
                text_to_ids[row["context"]].append(row["id"])
            for ids in text_to_ids.values():
                if len(ids) > 1:
                    anchor = ids[0]
                    for mid in ids[1:]:
                        union(anchor, mid)

        # ---- Step 3: assign canonical integer group ids ---------------------
        root_to_group: dict[str, str] = {}

        def get_group(mid: str) -> str:
            root = find(mid)
            if root not in root_to_group:
                root_to_group[root] = f"mlqa_{len(root_to_group)}"
            return root_to_group[root]

        # ---- Step 4: build deduplicated passages ----------------------------
        # One passage per (group, lang). Context text is consistent
        # within a (group, lang) pair (MLQA uses one translation per context).
        group_lang_text: dict[tuple[str, str], str] = {}
        for lang in self.languages:
            for row in rows_by_lang[lang]:
                g = get_group(row["id"])
                key = (g, lang)
                if key not in group_lang_text:
                    group_lang_text[key] = row["context"]

        passages: list[Passage] = []
        for (g, lang), text in group_lang_text.items():
            passages.append(
                Passage(
                    passage_id=f"{g}_{lang}",
                    group_id=g,
                    language=lang,
                    text=text,
                    dataset="mlqa",
                )
            )

        # ---- Step 5: build queries ------------------------------------------
        # One query per (row, lang). Query id uses the original MLQA id to
        # stay globally unique.
        queries: list[Query] = []
        seen_qids: set[str] = set()
        for lang in self.languages:
            for row in rows_by_lang[lang]:
                g = get_group(row["id"])
                qid = f"mlqa_q_{row['id']}_{lang}"
                if qid in seen_qids:
                    continue
                seen_qids.add(qid)
                queries.append(
                    Query(
                        query_id=qid,
                        group_id=g,
                        language=lang,
                        text=row["question"],
                        dataset="mlqa",
                    )
                )

        return passages, queries
