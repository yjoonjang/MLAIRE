"""Dense embedding model with MTEB-first loading, bf16/flash-attn, and OOM retry."""

from __future__ import annotations

import gc
import logging
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from mlaire.models.base import BaseModel
from mlaire.models.model_configs import ModelConfig, get_model_config

# ---------------------------------------------------------------------------
# Transformers 5.x compatibility patch for custom GTE-Qwen2 modeling code
# ---------------------------------------------------------------------------
# GTE-Qwen2's custom modeling_qwen.py calls DynamicCache.from_legacy_cache()
# and cache.get_usable_length(), both removed/renamed in transformers >= 5.x.
# We add compat shims at import time so trust_remote_code=True works correctly
# (bidirectional attention is required for proper embedding quality).
try:
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "from_legacy_cache"):
        @classmethod  # type: ignore[misc]
        def _from_legacy_cache(
            cls, past_key_values=None
        ) -> "DynamicCache":
            """Compat shim: reconstruct DynamicCache from legacy tuple format."""
            cache = cls()
            if past_key_values is not None:
                for layer_idx in range(len(past_key_values)):
                    key_states, value_states = past_key_values[layer_idx]
                    cache.update(key_states, value_states, layer_idx)
            return cache

        DynamicCache.from_legacy_cache = _from_legacy_cache  # type: ignore[attr-defined]

    if not hasattr(DynamicCache, "get_usable_length"):
        def _get_usable_length(self, new_seq_len: int, layer_idx: int = 0) -> int:  # type: ignore[misc]
            """Compat shim: get_usable_length was renamed to get_seq_length in transformers 5.x."""
            return self.get_seq_length(layer_idx)

        DynamicCache.get_usable_length = _get_usable_length  # type: ignore[attr-defined]

except Exception:
    pass  # If DynamicCache is not available, skip patching

logger = logging.getLogger(__name__)

_MAX_OOM_RETRIES = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_oom_error(error: Exception) -> bool:
    """Check whether *error* is a CUDA out-of-memory error."""
    error_str = str(error).lower()
    return any(
        tok in error_str
        for tok in [
            "out of memory",
            "cuda out of memory",
            "oom",
            "cudamalloc failed",
            "failed to allocate",
        ]
    )


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_model_kwargs(
    config: ModelConfig,
    use_bf16: bool,
    use_flash_attn: bool,
) -> dict[str, Any]:
    """Build ``model_kwargs`` dict for SentenceTransformer / MTEB loading."""
    kwargs: dict[str, Any] = {}

    # Explicit dtype override (e.g., for models where FP32 produces NaN and BF16 collapses)
    if config.torch_dtype is not None:
        _dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        kwargs["torch_dtype"] = _dtype_map[config.torch_dtype]
        logger.info("Using explicit dtype: %s", config.torch_dtype)
    # bf16
    elif use_bf16 and config.supports_bf16 and torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16
        logger.info("Using bfloat16 precision")

    # flash-attention-2 (only if supported AND installed; otherwise let model use its default)
    if use_flash_attn and config.supports_flash_attn and torch.cuda.is_available():
        try:
            import flash_attn  # noqa: F401
            kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using flash_attention_2")
        except ImportError:
            # Don't force SDPA — some architectures (XLM-R, BERT) don't support it
            # in older transformers versions. Let the model use its own default.
            logger.info("flash_attn not installed, using model default attention")

    return kwargs


def _disable_kv_cache(model: SentenceTransformer) -> None:
    """Disable ``use_cache`` on the underlying HF model to avoid DynamicCache issues."""
    # Navigate: SentenceTransformer -> first module (Transformer) -> auto_model -> config
    hf_model = None
    first_mod = getattr(model, "_first_module", lambda: None)()
    if first_mod is not None:
        hf_model = getattr(first_mod, "auto_model", None)
    if hf_model is None:
        hf_model = getattr(model, "auto_model", None)
    if hf_model is None:
        # Last resort: walk all modules looking for a config with use_cache
        for module in model.modules():
            if hasattr(module, "config") and hasattr(module.config, "use_cache"):
                module.config.use_cache = False
                logger.debug("Disabled KV cache on %s (walk)", type(module).__name__)
                return
        return
    hf_config = getattr(hf_model, "config", None)
    if hf_config is not None and hasattr(hf_config, "use_cache"):
        hf_config.use_cache = False
        logger.debug("Disabled KV cache on %s", type(hf_model).__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _lookup_mteb_prompts(model_name: str) -> tuple[str, str] | None:
    """Look up query/passage prefixes from MTEB's model registry.

    Uses MTEB purely as a **read-only prompt registry** — no model loading.
    Returns ``(query_prefix, passage_prefix)`` or ``None`` if not found.
    """
    try:
        import mteb
        from functools import partial

        meta = mteb.get_model_meta(model_name)
        if meta is None or meta.loader is None:
            return None

        # MTEB stores prompts in functools.partial(loader, model_prompts={...})
        loader = meta.loader
        prompts: dict[str, str] = {}
        if isinstance(loader, partial) and "model_prompts" in (loader.keywords or {}):
            prompts = loader.keywords["model_prompts"]
        elif hasattr(loader, "keywords") and loader.keywords:
            prompts = loader.keywords.get("model_prompts", {})

        if not prompts:
            return None

        query_prefix = prompts.get("query", "")
        passage_prefix = prompts.get("passage", prompts.get("document", ""))
        logger.info(
            "Found MTEB prompts for %s: query=%r, passage=%r",
            model_name, query_prefix[:40], passage_prefix[:40],
        )
        return query_prefix, passage_prefix
    except Exception as exc:
        logger.debug("MTEB prompt lookup failed for %s: %s", model_name, exc)
        return None


def _load_sentence_transformer(
    model_name: str,
    device: str,
    model_kwargs: dict[str, Any],
    config: ModelConfig,
) -> SentenceTransformer:
    """Load model via SentenceTransformer with manual prefix handling."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    model = SentenceTransformer(
        model_name,
        device=device,
        model_kwargs=model_kwargs,
        trust_remote_code=config.trust_remote_code,
    )
    logger.info("Loaded %s via SentenceTransformer (manual prefixes)", model_name)
    return model


# ---------------------------------------------------------------------------
# DenseLocalModel
# ---------------------------------------------------------------------------

class DenseLocalModel(BaseModel):
    """Dense embedding model with MTEB-first loading, OOM retry, and bf16/flash-attn.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier (e.g. ``"intfloat/multilingual-e5-large"``).
    device:
        PyTorch device string. ``None`` auto-selects CUDA when available.
    use_bf16:
        Enable bfloat16 inference when the model supports it.
    use_flash_attn:
        Enable flash-attention-2 (falls back to SDPA if unavailable).
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        use_bf16: bool = True,
        use_flash_attn: bool = True,
    ) -> None:
        super().__init__(model_name)
        self.config = get_model_config(model_name)
        self._device = _resolve_device(device)
        self._batch_size = self.config.batch_size

        model_kwargs = _build_model_kwargs(self.config, use_bf16, use_flash_attn)

        # If config has no prefixes, try to look them up from MTEB registry
        if not self.config.query_prefix and not self.config.passage_prefix:
            mteb_prompts = _lookup_mteb_prompts(model_name)
            if mteb_prompts is not None:
                self.config.query_prefix = mteb_prompts[0]
                self.config.passage_prefix = mteb_prompts[1]

        # Always load via SentenceTransformer (reliable, no wrapper issues)
        self._model = _load_sentence_transformer(
            model_name, self._device, model_kwargs, self.config,
        )

        _disable_kv_cache(self._model)

    # ------------------------------------------------------------------
    # Encoding with OOM retry
    # ------------------------------------------------------------------

    def _encode_with_retry(
        self,
        texts: list[str],
        batch_size: int,
        **encode_kwargs: Any,
    ) -> np.ndarray:
        """Encode *texts*, halving batch_size on CUDA OOM (up to ``_MAX_OOM_RETRIES``)."""
        current_bs = batch_size
        for attempt in range(_MAX_OOM_RETRIES):
            try:
                return self._model.encode(
                    texts,
                    batch_size=current_bs,
                    normalize_embeddings=self.config.normalize,
                    show_progress_bar=True,
                    **encode_kwargs,
                )
            except Exception as exc:
                if not _is_oom_error(exc) or current_bs <= 1:
                    raise
                new_bs = max(1, current_bs // 2)
                logger.warning(
                    "OOM at batch_size=%d (attempt %d/%d), retrying with %d",
                    current_bs,
                    attempt + 1,
                    _MAX_OOM_RETRIES,
                    new_bs,
                )
                current_bs = new_bs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

        # Final attempt with batch_size=1
        return self._model.encode(
            texts,
            batch_size=1,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=True,
            **encode_kwargs,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        prefixed = [f"{self.config.query_prefix}{t}" for t in texts]
        return self._encode_with_retry(prefixed, batch_size=self._batch_size)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        prefixed = [f"{self.config.passage_prefix}{t}" for t in texts]
        return self._encode_with_retry(prefixed, batch_size=self._batch_size)
