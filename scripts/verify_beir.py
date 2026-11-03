"""Verify the BEIR export against the canonical loaders.

For each dataset, loads (passages, queries) twice:

  A. via the canonical Loader (e.g. ``MLQALoader``), which pulls the raw
     data from the upstream Hugging Face datasets and applies our
     deduplication / id-assignment logic, and
  B. via ``BeirLoader``, which reads the BEIR-format files we just
     exported (or downloaded from the Hugging Face Hub).

Successful verification means every retrieval metric computed downstream
will be identical between the two paths, by construction: pool contents
are bit-identical at the (passage_id, group_id, language, text) level.

Usage:
    # Verify against the local export
    python scripts/verify_beir.py --source ./data

    # Verify against the uploaded HF dataset
    python scripts/verify_beir.py --source nlpai-lab/mlaire
    # (the script appends "-<dataset>" automatically)
"""

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mlaire.data.loaders.beir import BeirLoader  # noqa: E402
from mlaire.data.loaders.belebele import BelebeleLoader  # noqa: E402
from mlaire.data.loaders.mlqa import MLQALoader  # noqa: E402
from mlaire.data.loaders.xquad import XQuADLoader  # noqa: E402
from mlaire.language_metadata import (  # noqa: E402
    BELEBELE_LANG_MAP,
    MLQA_LANGUAGES,
    XQUAD_LANGUAGES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


CANONICAL = {
    "mlqa": (MLQALoader, list(MLQA_LANGUAGES)),
    "xquad": (XQuADLoader, list(XQUAD_LANGUAGES)),
    "belebele": (BelebeleLoader, sorted(set(BELEBELE_LANG_MAP.values()))),
}


def _key_passage(p):
    return (p.passage_id, p.group_id, p.language, p.text, p.dataset)


def _key_query(q):
    return (q.query_id, q.group_id, q.language, q.text, q.dataset)


def verify_one(name: str, source: str) -> bool:
    loader_cls, langs = CANONICAL[name]
    logger.info("[%s] loading canonical loader ...", name)
    canon_p, canon_q = loader_cls(languages=langs).load()

    if "/" in source and not Path(source).exists():
        beir_source = f"{source}-{name}"
    else:
        beir_source = str(Path(source) / name)

    logger.info("[%s] loading BeirLoader from %s ...", name, beir_source)
    beir_p, beir_q = BeirLoader(source=beir_source, dataset=name).load()

    a = sorted(_key_passage(p) for p in canon_p)
    b = sorted(_key_passage(p) for p in beir_p)
    if a != b:
        diff_count = sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
        logger.error("[%s] PASSAGES MISMATCH (canonical=%d, beir=%d, diff=%d)",
                     name, len(a), len(b), diff_count)
        # Show a few diffs
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                logger.error("  diff #%d:\n    canonical=%s\n    beir     =%s", i, x, y)
                if i >= 4:
                    break
        return False

    a = sorted(_key_query(q) for q in canon_q)
    b = sorted(_key_query(q) for q in beir_q)
    if a != b:
        logger.error("[%s] QUERIES MISMATCH (canonical=%d, beir=%d)", name, len(a), len(b))
        return False

    logger.info(
        "[%s] OK: %d passages, %d queries match exactly.",
        name, len(canon_p), len(canon_q),
    )
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        default="./data",
        help="local data root, OR HF Hub repo prefix (e.g. nlpai-lab/mlaire)",
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=list(CANONICAL.keys()),
        choices=list(CANONICAL.keys()),
    )
    args = ap.parse_args()

    all_ok = True
    for name in args.datasets:
        if not verify_one(name, args.source):
            all_ok = False

    if not all_ok:
        sys.exit("VERIFICATION FAILED")
    logger.info("All datasets verified.")


if __name__ == "__main__":
    main()
