"""MLAIRE CLI — single-model evaluation and multi-GPU pipeline."""

import json
import logging
import os
from datetime import datetime, timezone

import click
import yaml

from mlaire.models.model_configs import MODEL_CONFIGS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result skip / caching
# ---------------------------------------------------------------------------


def check_result_completed(output_path: str) -> bool:
    """Check if evaluation result already exists and is valid."""
    if os.path.exists(output_path):
        try:
            with open(output_path) as f:
                data = json.load(f)
            if data and "aggregate" in data and "meta" in data:
                return True
        except (json.JSONDecodeError, KeyError):
            pass
    return False


def _build_output_path(config: dict, model_name: str, output_dir: str | None = None) -> str:
    """Derive the default output JSON path for a model + config pair."""
    base_dir = output_dir or config.get("output_dir", "results")
    os.makedirs(base_dir, exist_ok=True)
    safe_name = model_name.replace("/", "_")
    return os.path.join(base_dir, f"{safe_name}_{config.get('tier', 'custom')}.json")


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def run_evaluation(
    config: dict,
    model_name: str,
    provider: str | None = None,
    *,
    model=None,
    cache_dir: str | None = None,
    use_bf16: bool = True,
    use_flash_attn: bool = True,
    device: str | None = None,
) -> dict:
    """Run the full evaluation pipeline.

    Parameters
    ----------
    config:
        Parsed YAML config dict.
    model_name:
        HuggingFace model ID or API model name.
    provider:
        API provider (``"openai"`` or ``"cohere"``). ``None`` for local models.
    model:
        Pre-loaded model instance. When provided, ``provider`` / device / bf16
        / flash_attn args are ignored (the caller already configured them).
    cache_dir:
        Directory for embedding cache files. ``None`` disables caching.
    use_bf16:
        Enable bfloat16 for local models (ignored when *model* is given).
    use_flash_attn:
        Enable flash-attention-2 for local models (ignored when *model* is given).
    device:
        PyTorch device string for local models (ignored when *model* is given).
    """
    from mlaire.data.loaders.belebele import BelebeleLoader
    from mlaire.data.loaders.xquad import XQuADLoader
    from mlaire.data.loaders.mlqa import MLQALoader
    from mlaire.data.loaders.beir import BeirLoader
    from mlaire.data.processors.pool_builder import PoolBuilder
    from mlaire.evaluation.retriever import Retriever
    from mlaire.evaluation.evaluator import Evaluator
    from mlaire.language_metadata import get_language_info, BELEBELE_LANG_MAP

    loader_map = {
        "belebele": BelebeleLoader,
        "xquad": XQuADLoader,
        "mlqa": MLQALoader,
    }

    # Load model if not provided
    if model is None:
        if provider:
            from mlaire.models.dense_api import DenseAPIModel

            model = DenseAPIModel(model_name=model_name, provider=provider)
        else:
            from mlaire.models.dense import DenseLocalModel

            model = DenseLocalModel(
                model_name=model_name,
                device=device,
                use_bf16=use_bf16,
                use_flash_attn=use_flash_attn,
            )

    # Evaluate each dataset in its own independent pool.
    # Mixing passages from different datasets into one pool would make
    # cross-dataset passages act as spurious distractors and distort metrics.
    max_k = max(config.get("k_values", [1, 5, 10, 20]))
    # Retrieval depth is decoupled from the k-set used for nDCG/Recall:
    # MRR and LPR look at the FULL ranked list to find the first relevant
    # passage, so retrieving only top-(2*max_k) underestimates them when
    # the relevant lives at a deeper rank. ``retrieval_top_k`` overrides
    # the default of 2*max_k and should match the depth used to produce
    # the canonical results we ship under ``results/``.
    retrieval_top_k = int(config.get("retrieval_top_k", max_k * 2))
    all_retrieval_results = []
    all_queries = []
    all_group_counts: dict[str, int] = {}

    # Default flow: pull BEIR-format files from the public Hugging Face repos
    # (anonymous-submission-nips/mlaire-<dataset>). The user can override
    # with config["data_dir"] (local export) or config["use_canonical_loaders"]
    # (rederive from the upstream HF datasets via MLQA/XQuAD/Belebele loaders).
    data_dir = config.get("data_dir")
    use_canonical = config.get("use_canonical_loaders", False)
    hf_repo_prefix = config.get(
        "hf_repo_prefix", "anonymous-submission-nips/mlaire"
    )

    for ds_name, ds_config in config["datasets"].items():
        langs = ds_config["languages"]
        if langs == "all":
            langs = list(set(BELEBELE_LANG_MAP.values()))
        lang_filter = None if langs == "all" else list(langs)

        if use_canonical:
            loader = loader_map[ds_name](languages=lang_filter or langs)
        elif data_dir:
            ds_path = os.path.join(data_dir, ds_name)
            loader = BeirLoader(
                source=ds_path, dataset=ds_name, languages=lang_filter,
            )
        else:
            repo_id = f"{hf_repo_prefix}-{ds_name}"
            loader = BeirLoader(
                source=repo_id, dataset=ds_name, languages=lang_filter,
            )
        passages, queries = loader.load()

        builder = PoolBuilder()
        builder.add_passages(passages)
        builder.add_queries(queries)
        pool = builder.build()

        logger.info("Dataset %s: %d passages, %d queries", ds_name, len(pool.passages), len(pool.queries))

        retriever = Retriever(model=model, pool=pool, cache_dir=cache_dir)
        results = retriever.retrieve_all(top_k=retrieval_top_k)

        all_retrieval_results.extend(results)
        all_queries.extend(pool.queries)
        for p in pool.passages:
            all_group_counts[p.group_id] = all_group_counts.get(p.group_id, 0) + 1

    # Evaluate once on the combined per-dataset results
    evaluator = Evaluator(k_values=config.get("k_values", [1, 5, 10, 20]))
    eval_results = evaluator.evaluate(all_retrieval_results, all_queries, group_counts=all_group_counts)

    # Add metadata
    all_langs = sorted({q.language for q in all_queries})
    eval_results["meta"] = {
        "benchmark_tier": config.get("tier", "custom"),
        "model_name": model_name,
        "datasets": list(config["datasets"].keys()),
        "languages": all_langs,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Add language metadata to per_language
    for lang in eval_results.get("per_language", {}):
        info = get_language_info(lang)
        eval_results["per_language"][lang].update(info)

    return eval_results


# ---------------------------------------------------------------------------
# Multi-GPU pipeline
# ---------------------------------------------------------------------------


def _detect_gpus() -> list[int]:
    """Return list of available CUDA GPU IDs."""
    try:
        import torch

        return list(range(torch.cuda.device_count()))
    except Exception:
        return []


def _gpu_worker(
    gpu_id: int,
    job_queue,
    result_queue,
    config: dict,
    use_bf16: bool,
    use_flash_attn: bool,
    skip_completed: bool,
    cache_dir: str | None,
):
    """Worker process that owns one GPU and processes models from *job_queue*."""
    import torch

    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    current_model = None
    current_model_name = None

    logger.info("GPU worker %d started on %s", gpu_id, device)

    while True:
        job = job_queue.get()
        if job is None:  # Poison pill
            logger.info("GPU worker %d shutting down", gpu_id)
            break

        model_name: str = job["model_name"]
        output_path: str = job["output_path"]

        if skip_completed and check_result_completed(output_path):
            logger.info("Skipping %s (result exists at %s)", model_name, output_path)
            result_queue.put({
                "model": model_name,
                "gpu": gpu_id,
                "status": "skipped",
                "output_path": output_path,
            })
            continue

        try:
            # Load model only when it changes
            if current_model_name != model_name:
                if current_model is not None:
                    del current_model
                    current_model = None
                    current_model_name = None
                    torch.cuda.empty_cache()
                    import gc

                    gc.collect()

                from mlaire.models.dense import DenseLocalModel

                current_model = DenseLocalModel(
                    model_name=model_name,
                    device=str(device),
                    use_bf16=use_bf16,
                    use_flash_attn=use_flash_attn,
                )
                current_model_name = model_name

            logger.info("GPU %d evaluating %s", gpu_id, model_name)
            results = run_evaluation(
                config,
                model_name,
                model=current_model,
                cache_dir=cache_dir,
            )

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            result_queue.put({
                "model": model_name,
                "gpu": gpu_id,
                "status": "success",
                "output_path": output_path,
            })

        except Exception as exc:
            logger.error("GPU %d failed on %s: %s", gpu_id, model_name, exc)
            result_queue.put({
                "model": model_name,
                "gpu": gpu_id,
                "status": "error",
                "error": str(exc),
            })

    # Cleanup
    if current_model is not None:
        del current_model
        import torch as _torch

        _torch.cuda.empty_cache()


def run_pipeline(
    config: dict,
    model_names: list[str],
    gpu_ids: list[int],
    *,
    use_bf16: bool = True,
    use_flash_attn: bool = True,
    skip_completed: bool = True,
    cache_dir: str | None = None,
    output_dir: str | None = None,
) -> list[dict]:
    """Run multiple models across GPUs using a queue worker pattern.

    Each GPU gets a dedicated worker process. Models are distributed round-robin
    across GPU job queues. Workers load a model once and evaluate, then move on.

    Parameters
    ----------
    config:
        Parsed YAML config dict.
    model_names:
        List of HuggingFace model IDs to evaluate.
    gpu_ids:
        CUDA device IDs to use. Auto-detected if empty.
    use_bf16 / use_flash_attn:
        Forwarded to ``DenseLocalModel``.
    skip_completed:
        Skip models whose result JSON already exists and is valid.
    cache_dir:
        Embedding cache directory.
    output_dir:
        Override output directory from config.
    """
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    if not gpu_ids:
        gpu_ids = _detect_gpus()
    if not gpu_ids:
        raise RuntimeError("No CUDA GPUs available for pipeline mode")

    num_gpus = len(gpu_ids)
    logger.info("Pipeline: %d model(s) across %d GPU(s) %s", len(model_names), num_gpus, gpu_ids)

    # Create per-GPU job queues and a shared result queue
    job_queues: list[mp.Queue] = [mp.Queue() for _ in gpu_ids]
    result_queue: mp.Queue = mp.Queue()

    # Distribute models round-robin across GPUs
    for i, model_name in enumerate(model_names):
        out_path = _build_output_path(config, model_name, output_dir)
        target_gpu_idx = i % num_gpus
        job_queues[target_gpu_idx].put({
            "model_name": model_name,
            "output_path": out_path,
        })

    # Send poison pills
    for q in job_queues:
        q.put(None)

    # Start workers
    workers: list[mp.Process] = []
    for idx, gid in enumerate(gpu_ids):
        p = mp.Process(
            target=_gpu_worker,
            args=(
                gid,
                job_queues[idx],
                result_queue,
                config,
                use_bf16,
                use_flash_attn,
                skip_completed,
                cache_dir,
            ),
            daemon=True,
        )
        p.start()
        workers.append(p)

    # Collect results
    collected: list[dict] = []
    for _ in range(len(model_names)):
        result = result_queue.get()
        collected.append(result)
        status = result["status"]
        model = result["model"]
        if status == "success":
            click.echo(f"  [GPU {result['gpu']}] {model} -> {result['output_path']}")
        elif status == "skipped":
            click.echo(f"  [GPU {result['gpu']}] {model} (skipped, result exists)")
        else:
            click.echo(f"  [GPU {result['gpu']}] {model} FAILED: {result.get('error', 'unknown')}")

    # Wait for all workers to finish
    for p in workers:
        p.join(timeout=30)

    return collected


# ---------------------------------------------------------------------------
# CLI display helpers
# ---------------------------------------------------------------------------


def _print_results(results: dict, k_values: list[int]) -> None:
    """Print aggregate metrics and failure diagnostics."""
    agg = results["aggregate"]
    click.echo("")
    click.echo("=== Standard Retrieval Metrics ===")
    click.echo(f"  MRR:  {agg['mrr']:.4f}")
    for k in k_values:
        click.echo(f"  NDCG@{k}:    {agg.get(f'ndcg@{k}', 0):.4f}    Recall@{k}:  {agg.get(f'recall@{k}', 0):.4f}")
    click.echo("")
    click.echo("=== Language-Aware Metrics ===")
    click.echo(f"  LPR:       {agg['lpr']:.4f}")
    click.echo(f"  Lang-MRR:  {agg.get('lang_mrr', 0):.4f}")
    for k in k_values:
        click.echo(f"  Lang-NDCG@{k}: {agg.get(f'lang_ndcg@{k}', 0):.4f}  Lang-Recall@{k}: {agg.get(f'lang_recall@{k}', 0):.4f}")
    click.echo("")
    click.echo("=== Failure Types ===")
    for ftype, ratio in results["diagnostics"]["failure_types"].items():
        click.echo(f"  {ftype}: {ratio:.4f}")


def _list_models() -> None:
    """Print all known model configs with capabilities."""
    click.echo("Supported models:")
    click.echo("")
    for name, cfg in MODEL_CONFIGS.items():
        flags = []
        if cfg.supports_bf16:
            flags.append("bf16")
        if cfg.supports_flash_attn:
            flags.append("flash_attn")
        flag_str = ", ".join(flags) if flags else "none"
        click.echo(f"  {name}")
        click.echo(f"    batch_size: {cfg.batch_size} | max_length: {cfg.max_length} | features: [{flag_str}]")
        if cfg.query_prefix:
            prefix_display = repr(cfg.query_prefix[:60])
            click.echo(f"    query_prefix: {prefix_display}")


# ---------------------------------------------------------------------------
# Click CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--config", "config_path", type=click.Path(exists=True), help="Path to config YAML")
@click.option("--model", "model_name", type=str, default=None, help="Single model name (HuggingFace ID or API model)")
@click.option("--models", "model_names", type=str, multiple=True, help="Multiple models for pipeline mode")
@click.option("--provider", type=click.Choice(["openai", "cohere"]), default=None, help="API provider (single-model only)")
@click.option("--output", "output_path", type=str, default=None, help="Output JSON path (single-model only)")
@click.option("--output-dir", "output_dir", type=str, default=None, help="Output directory (overrides config)")
@click.option("--gpus", type=str, default=None, help="Comma-separated GPU IDs for pipeline (e.g. '0,1,2')")
@click.option("--no-bf16", is_flag=True, default=False, help="Disable bfloat16 precision")
@click.option("--no-flash-attn", is_flag=True, default=False, help="Disable flash attention")
@click.option("--no-skip", is_flag=True, default=False, help="Force re-evaluation even if results exist")
@click.option("--cache-dir", type=str, default=None, help="Embedding cache directory")
@click.option("--data-dir", "data_dir", type=str, default=None, help="Local BEIR export root (overrides Hugging Face Hub fetch)")
@click.option("--list-models", is_flag=True, help="List supported models with configs")
def main(
    config_path,
    model_name,
    model_names,
    provider,
    output_path,
    output_dir,
    gpus,
    no_bf16,
    no_flash_attn,
    no_skip,
    cache_dir,
    data_dir,
    list_models,
):
    """MLAIRE: Multilingual Language-Aware Information Retrieval Benchmark

    \b
    Single model:
      mlaire --config compact.yaml --model intfloat/multilingual-e5-large

    \b
    Multi-model pipeline across GPUs:
      mlaire --config compact.yaml \\
          --models intfloat/multilingual-e5-large \\
          --models BAAI/bge-m3 \\
          --models jinaai/jina-embeddings-v3 \\
          --gpus 0,1,2
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if list_models:
        _list_models()
        return

    use_bf16 = not no_bf16
    use_flash_attn = not no_flash_attn
    skip_completed = not no_skip

    # Parse GPU IDs
    gpu_ids: list[int] = []
    if gpus is not None:
        gpu_ids = [int(g.strip()) for g in gpus.split(",") if g.strip()]

    # ----- Pipeline mode (--models) -----
    if model_names:
        if not config_path:
            click.echo("Error: --config is required. Use --help for usage.")
            raise SystemExit(1)

        with open(config_path) as f:
            config = yaml.safe_load(f)

        if data_dir is not None:
            config["data_dir"] = data_dir

        models_list = list(model_names)
        click.echo(
            f"Running MLAIRE pipeline ({config.get('tier', 'custom')}) "
            f"with {len(models_list)} model(s) across {len(gpu_ids) or 'auto'} GPU(s)..."
        )

        results = run_pipeline(
            config,
            models_list,
            gpu_ids,
            use_bf16=use_bf16,
            use_flash_attn=use_flash_attn,
            skip_completed=skip_completed,
            cache_dir=cache_dir,
            output_dir=output_dir,
        )

        # Summary
        success = sum(1 for r in results if r["status"] == "success")
        skipped = sum(1 for r in results if r["status"] == "skipped")
        failed = sum(1 for r in results if r["status"] == "error")
        click.echo("")
        click.echo(f"Pipeline complete: {success} succeeded, {skipped} skipped, {failed} failed")
        return

    # ----- Single model mode (--model) -----
    if not config_path or not model_name:
        click.echo("Error: --config and --model (or --models) are required. Use --help for usage.")
        raise SystemExit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if data_dir is not None:
        config["data_dir"] = data_dir

    # Determine output path
    if output_path is None:
        output_path = _build_output_path(config, model_name, output_dir)

    # Skip if result already completed
    if skip_completed and check_result_completed(output_path):
        click.echo(f"Skipping {model_name} (valid result exists at {output_path})")
        click.echo("Use --no-skip to force re-evaluation.")
        return

    click.echo(f"Running MLAIRE ({config.get('tier', 'custom')}) with {model_name}...")
    results = run_evaluation(
        config,
        model_name,
        provider,
        use_bf16=use_bf16,
        use_flash_attn=use_flash_attn,
        device=f"cuda:{gpu_ids[0]}" if gpu_ids else None,
        cache_dir=cache_dir,
    )

    # Save results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    click.echo(f"Results saved to {output_path}")
    _print_results(results, config.get("k_values", [1, 5, 10, 20]))


if __name__ == "__main__":
    main()
