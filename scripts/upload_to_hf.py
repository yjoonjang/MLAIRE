"""Upload the MTEB-style BEIR datasets to the Hugging Face Hub.

The script reads its access token from the ``HF_TOKEN`` environment
variable (never from disk) and pushes one repository per dataset:

    anonymous-submission-nips/mlaire-mlqa
    anonymous-submission-nips/mlaire-xquad
    anonymous-submission-nips/mlaire-belebele

Old layout artefacts (``corpus.jsonl``, ``queries.jsonl``,
``qrels/test.tsv``) from any previous upload are deleted before pushing
the new parquet files, so the resulting repos are clean MTEB-style
multi-config datasets.

Run after ``scripts/export_to_beir.py``:

    HF_TOKEN=... python scripts/upload_to_hf.py --data ./data
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


HF_USER = "anonymous-submission-nips"

# Files from the previous (raw JSONL/TSV) layout that should be removed
# before re-uploading the MTEB-style parquet files.
LEGACY_PATHS = ["corpus.jsonl", "queries.jsonl", "qrels/test.tsv"]


def _delete_legacy(api, repo_id: str) -> None:
    """Remove old-layout files that may exist in the repo from a prior push."""
    from huggingface_hub.utils import EntryNotFoundError

    for p in LEGACY_PATHS:
        try:
            api.delete_file(
                path_in_repo=p,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Remove legacy layout: {p}",
            )
            logger.info("[%s] removed legacy %s", repo_id.split("/")[-1], p)
        except EntryNotFoundError:
            pass
        except Exception as e:  # pragma: no cover  -- be tolerant of any 4xx
            logger.warning("[%s] could not delete %s: %s", repo_id, p, e)


def upload_one(name: str, data_root: Path, token: str, dry_run: bool) -> None:
    from huggingface_hub import HfApi, create_repo

    repo_id = f"{HF_USER}/mlaire-{name}"
    src = data_root / name
    if not src.exists():
        raise FileNotFoundError(
            f"{src} does not exist; run scripts/export_to_beir.py first."
        )

    files = sorted(p for p in src.rglob("*") if p.is_file())
    total_bytes = sum(p.stat().st_size for p in files)
    logger.info(
        "[%s] %d files, %.1f MB -> %s",
        name, len(files), total_bytes / 1e6, repo_id,
    )

    if dry_run:
        for p in files:
            logger.info(
                "    %s (%.1f kB)",
                p.relative_to(src),
                p.stat().st_size / 1024,
            )
        return

    api = HfApi(token=token)
    create_repo(
        repo_id,
        repo_type="dataset",
        exist_ok=True,
        token=token,
        private=False,
    )
    _delete_legacy(api, repo_id)

    api.upload_folder(
        folder_path=str(src),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Upload mlaire-{name} (MTEB-style parquet)",
    )
    logger.info(
        "[%s] uploaded -> https://huggingface.co/datasets/%s", name, repo_id
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data", help="root containing exported BEIR files")
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["mlqa", "xquad", "belebele"],
        choices=["mlqa", "xquad", "belebele"],
    )
    ap.add_argument("--dry-run", action="store_true", help="list files without uploading")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        sys.exit("HF_TOKEN env var is required (or pass --dry-run).")

    for name in args.datasets:
        upload_one(name, Path(args.data), token or "", args.dry_run)


if __name__ == "__main__":
    main()
