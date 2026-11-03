"""Multi-GPU queue runner for the LPR re-measurement jobs.

Reads ``measure_lpr/jobs.yaml``, expands each entry into per-dataset
subjobs, then runs them through one worker thread per GPU. Each worker
is pinned to a single GPU via ``CUDA_VISIBLE_DEVICES`` and dispatches
the job to the right venv + backend script.

Usage:
    python measure_lpr/queue_runner.py
    python measure_lpr/queue_runner.py --gpus 0,1,2,3
    python measure_lpr/queue_runner.py --datasets mlqa xquad     # subset
    python measure_lpr/queue_runner.py --skip-existing            # default on

Multi-server sharding (run on a shared output dir):
    # server A (4 GPUs) -> takes 4/10 of jobs
    python measure_lpr/queue_runner.py --gpus 0,1,2,3 \
        --shard 0-3/10 --out-root /data_x/yjoonjang/PAPERS/MLAIRBench/lpr_out
    # server B (3 GPUs) -> takes 3/10 of jobs
    python measure_lpr/queue_runner.py --gpus 0,1,2 \
        --shard 4-6/10 --out-root /data_x/yjoonjang/PAPERS/MLAIRBench/lpr_out
    # server C (3 GPUs) -> takes 3/10 of jobs
    python measure_lpr/queue_runner.py --gpus 0,1,2 \
        --shard 7-9/10 --out-root /data_x/yjoonjang/PAPERS/MLAIRBench/lpr_out

``--shard SPEC/N`` deterministically partitions the expanded job list
into N interleaved buckets (job index mod N) and this runner takes
only the buckets named in SPEC. SPEC accepts a single index, a comma
list, or hyphen ranges (e.g. ``3/10``, ``0,4,7/10``, ``0-3/10``,
``0,2-4,7/10``). Pick N = total GPUs across all servers and assign
each server a contiguous range proportional to its GPU count, so no
runner ever picks the same (model, dataset) pair as another.

``--out-root`` redirects ``results/`` and ``logs/`` to a shared FS so
``--skip-existing`` works across all runners.

Outputs land at ``<out-root>/results/{dataset}/{model_safe}.json`` so
runs can be resumed safely.
"""

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
JOBS_FILE = REPO_ROOT / "measure_lpr" / "jobs.yaml"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "measure_lpr" / "results"
DEFAULT_LOG_ROOT = REPO_ROOT / "measure_lpr" / "logs"


def _safe(name: str) -> str:
    return name.replace("/", "_")


def _build_cmd(job: dict, dataset: str) -> tuple[Path, list[str]]:
    """Return (python_executable, argv) for the given job + dataset."""
    venv_py = REPO_ROOT / job["venv"] / "bin" / "python"
    backend = job["backend"]
    if backend == "dense":
        script = REPO_ROOT / "measure_lpr" / "compute_lpr_dense.py"
        argv = [str(venv_py), str(script),
                "--model", job["model"], "--dataset", dataset]
    elif backend == "bm25":
        script = REPO_ROOT / "measure_lpr" / "compute_lpr_bm25.py"
        argv = [str(venv_py), str(script), "--dataset", dataset]
    elif backend == "opensearch":
        script = REPO_ROOT / "measure_lpr" / "compute_lpr_opensearch.py"
        argv = [str(venv_py), str(script), "--dataset", dataset]
    elif backend == "colbert":
        script = REPO_ROOT / "measure_lpr" / "compute_lpr_colbert.py"
        argv = [str(venv_py), str(script),
                "--variant", job["variant"], "--dataset", dataset]
    elif backend in ("bgem3sparse", "bgem3colbert"):
        script = REPO_ROOT / "measure_lpr" / f"compute_lpr_{backend}.py"
        argv = [str(venv_py), str(script), "--dataset", dataset]
    else:
        raise ValueError(f"unknown backend {backend}")
    return venv_py, argv


def _expand(jobs_cfg: dict, datasets_filter: list[str] | None) -> list[dict]:
    default_datasets = jobs_cfg["datasets"]
    out: list[dict] = []
    for job in jobs_cfg["jobs"]:
        per_job_ds = job.get("datasets", default_datasets)
        for ds in per_job_ds:
            if datasets_filter and ds not in datasets_filter:
                continue
            out.append({**job, "dataset": ds})
    return out


def _output_path(job: dict, results_root: Path) -> Path:
    return results_root / job["dataset"] / f"{_safe(job['model'])}.json"


def _worker(
    name: str,
    gpu_id: int,
    q: "queue.Queue",
    skip_existing: bool,
    results_root: Path,
    log_root: Path,
) -> None:
    while True:
        job = q.get()
        if job is None:
            q.task_done()
            return
        out_path = _output_path(job, results_root)
        log_path = log_root / job["dataset"] / f"{_safe(job['model'])}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if skip_existing and out_path.exists():
            print(f"[{name} GPU{gpu_id}] SKIP {job['model']} / {job['dataset']} (already done)",
                  flush=True)
            q.task_done()
            continue

        _, argv = _build_cmd(job, job["dataset"])
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONUNBUFFERED"] = "1"

        print(f"[{name} GPU{gpu_id}] START {job['backend']:<10} {job['model']} / {job['dataset']}",
              flush=True)
        t0 = time.time()
        try:
            with open(log_path, "w") as logf:
                proc = subprocess.run(
                    argv, env=env, stdout=logf, stderr=subprocess.STDOUT,
                    cwd=str(REPO_ROOT),
                )
            ok = proc.returncode == 0
        except Exception as e:  # pragma: no cover
            ok = False
            with open(log_path, "a") as logf:
                logf.write(f"\n[runner] exception: {e}\n")
        elapsed = time.time() - t0
        status = "OK " if ok else "FAIL"
        print(f"[{name} GPU{gpu_id}] {status}  {job['model']} / {job['dataset']} "
              f"({elapsed:.0f}s, log={log_path})", flush=True)
        q.task_done()


def _parse_shard(spec: str | None) -> tuple[set[int], int]:
    """Parse a shard spec.

    Returns (selected_indices, total_buckets).

    Accepted forms:
        None or ""          -> ({0}, 1)            (no sharding)
        "3/10"              -> ({3}, 10)
        "0,4,7/10"          -> ({0,4,7}, 10)
        "0-3/10"            -> ({0,1,2,3}, 10)
        "0,2-4,7/10"        -> ({0,2,3,4,7}, 10)
    """
    if not spec:
        return {0}, 1
    if "/" not in spec:
        raise SystemExit(f"--shard must look like 'SPEC/N', got {spec!r}")
    left, n_str = spec.rsplit("/", 1)
    try:
        n = int(n_str)
    except ValueError as e:
        raise SystemExit(f"--shard total must be an int, got {n_str!r} ({e})")
    if n < 1:
        raise SystemExit(f"--shard total must be >= 1, got {n}")

    selected: set[int] = set()
    for part in left.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a_str, b_str = part.split("-", 1)
            try:
                a, b = int(a_str), int(b_str)
            except ValueError as e:
                raise SystemExit(f"--shard range {part!r} not parseable ({e})")
            if a > b:
                a, b = b, a
            selected.update(range(a, b + 1))
        else:
            try:
                selected.add(int(part))
            except ValueError as e:
                raise SystemExit(f"--shard index {part!r} not parseable ({e})")

    if not selected:
        raise SystemExit(f"--shard {spec!r}: no indices selected")
    bad = [i for i in selected if not (0 <= i < n)]
    if bad:
        raise SystemExit(f"--shard {spec!r}: indices {bad} out of range [0,{n})")
    return selected, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3",
                    help="Comma-separated GPU IDs to use as workers (default: 0,1,2,3)")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="Restrict run to these datasets")
    ap.add_argument("--no-skip", action="store_true",
                    help="Force re-run even when an output file already exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print expanded job list and exit")
    ap.add_argument("--shard", default=None,
                    help="SPEC/N: take jobs whose (index mod N) is in SPEC. "
                         "SPEC supports single indices, comma lists, and "
                         "hyphen ranges (e.g. '3/10', '0,4,7/10', '0-3/10', "
                         "'0,2-4,7/10'). Use to split work across servers "
                         "proportionally to GPU count.")
    ap.add_argument("--out-root", default=None,
                    help="Override output base dir. Results land in "
                         "<out-root>/results/, logs in <out-root>/logs/. "
                         "Use a shared NFS path so --skip-existing dedups "
                         "across servers. Default: measure_lpr/ (per-repo).")
    args = ap.parse_args()

    if args.out_root:
        out_root = Path(args.out_root).resolve()
        results_root = out_root / "results"
        log_root = out_root / "logs"
    else:
        results_root = DEFAULT_RESULTS_ROOT
        log_root = DEFAULT_LOG_ROOT
    results_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(JOBS_FILE.read_text())
    jobs_all = _expand(cfg, args.datasets)

    shard_set, shard_n = _parse_shard(args.shard)
    if shard_n > 1:
        jobs = [j for k, j in enumerate(jobs_all) if (k % shard_n) in shard_set]
    else:
        jobs = jobs_all

    shard_label = (
        f"{','.join(str(i) for i in sorted(shard_set))}/{shard_n}"
        if shard_n > 1 else "all"
    )

    if args.dry_run:
        header = f"=== shard {shard_label}: {len(jobs)} of {len(jobs_all)} jobs ==="
        print(header)
        for j in jobs:
            done = " (done)" if _output_path(j, results_root).exists() else ""
            print(f"  [{j['backend']:<10}] {j['venv']:<22} {j['model']:<45} {j['dataset']}{done}")
        return

    gpu_ids = [int(g) for g in args.gpus.split(",") if g.strip()]
    print(
        f"Running shard {shard_label}: {len(jobs)} of {len(jobs_all)} jobs "
        f"on {len(gpu_ids)} GPUs ({gpu_ids}).",
        flush=True,
    )
    print(f"  results -> {results_root}", flush=True)
    print(f"  logs    -> {log_root}", flush=True)

    q: "queue.Queue" = queue.Queue()
    for j in jobs:
        q.put(j)
    for _ in gpu_ids:
        q.put(None)

    skip_existing = not args.no_skip
    threads = []
    for i, gpu in enumerate(gpu_ids):
        t = threading.Thread(
            target=_worker,
            args=(f"W{i}", gpu, q, skip_existing, results_root, log_root),
            daemon=True,
        )
        t.start()
        threads.append(t)
    q.join()
    for t in threads:
        t.join()
    print("All jobs done.", flush=True)


if __name__ == "__main__":
    main()
