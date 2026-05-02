from dataclasses import dataclass
import re


def _transformers_lt5() -> bool:
    """Return True if installed transformers version is < 5.x."""
    try:
        import transformers
        major = int(transformers.__version__.split(".")[0])
        return major < 5
    except Exception:
        return False


# True on transformers 4.x (bidirectional attention works),
# False on transformers 5.x (AttentionMaskConverter API changed → near-zero sims)
_GTE_QWEN2_TRUST_REMOTE = _transformers_lt5()


@dataclass
class ModelConfig:
    """Configuration for a dense embedding model.

    Fields:
        name: Model identifier (auto-set by get_model_config).
        batch_size: Encoding batch size.
        supports_bf16: Whether the model supports bfloat16 inference.
        supports_flash_attn: Whether the model supports flash-attention-2.
        normalize: Whether to L2-normalize output embeddings.
        max_length: Maximum input token length.
        query_prefix: Prefix prepended to queries (fallback when MTEB prompts unavailable).
        passage_prefix: Prefix prepended to passages (fallback when MTEB prompts unavailable).
        torch_dtype: Explicit dtype override ("float16", "float32", "bfloat16"). When set,
            overrides supports_bf16. Use for models that require a specific dtype
            (e.g., FP16-native models where FP32 produces NaN and BF16 collapses).
    """

    name: str = ""
    batch_size: int = 256
    supports_bf16: bool = True
    supports_flash_attn: bool = True
    normalize: bool = True
    max_length: int = 512
    query_prefix: str = ""
    passage_prefix: str = ""
    trust_remote_code: bool = True
    torch_dtype: str | None = None


# ---------------------------------------------------------------------------
# Instruction prefixes
# ---------------------------------------------------------------------------

# E5-instruct style (trailing space after "Query: ")
_E5_INSTRUCT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query: "
)

# Nemotron style
_NEMOTRON_INSTRUCT = (
    "Instruct: Given a question, retrieve passages that answer the question\n"
    "Query: "
)

# Qwen3 models (no trailing space after "Query:")
_QWEN3_INSTRUCT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query:"
)

# BGE-multilingual-gemma2 (<instruct>/<query> tags)
_BGE_GEMMA2_INSTRUCT = (
    "<instruct>Given a web search query, retrieve relevant passages that answer the query.\n"
    "<query>"
)

# ---------------------------------------------------------------------------
# Known model configurations
# ---------------------------------------------------------------------------

MODEL_CONFIGS: dict[str, ModelConfig] = {
    # === E5 family (XLM-R encoder, "query: "/"passage: " prefix) ===
    "intfloat/multilingual-e5-large": ModelConfig(
        query_prefix="query: ",
        passage_prefix="passage: ",
        max_length=512,
        supports_flash_attn=False,  # XLM-R encoder
    ),
    "intfloat/multilingual-e5-base": ModelConfig(
        query_prefix="query: ",
        passage_prefix="passage: ",
        max_length=512,
        supports_flash_attn=False,
    ),
    "intfloat/multilingual-e5-small": ModelConfig(
        query_prefix="query: ",
        passage_prefix="passage: ",
        max_length=512,
        supports_flash_attn=False,
    ),
    "intfloat/multilingual-e5-large-instruct": ModelConfig(
        query_prefix=_E5_INSTRUCT,
        passage_prefix="",
        max_length=512,
    ),
    # === BGE family ===
    "BAAI/bge-m3": ModelConfig(
        query_prefix="",
        passage_prefix="",
        max_length=8192,
        supports_flash_attn=False,  # XLM-R architecture
    ),
    "BAAI/bge-multilingual-gemma2": ModelConfig(
        query_prefix=_BGE_GEMMA2_INSTRUCT,
        passage_prefix="",
        max_length=4096,
        batch_size=4,
    ),
    # === Qwen3 Embedding family (decoder, "Instruct:.../Query:" no trailing space) ===
    "Qwen/Qwen3-Embedding-0.6B": ModelConfig(
        query_prefix=_QWEN3_INSTRUCT,
        passage_prefix="",
        max_length=8192,
    ),
    "Qwen/Qwen3-Embedding-4B": ModelConfig(
        query_prefix=_QWEN3_INSTRUCT,
        passage_prefix="",
        max_length=8192,
        batch_size=64,
    ),
    "Qwen/Qwen3-Embedding-8B": ModelConfig(
        query_prefix=_QWEN3_INSTRUCT,
        passage_prefix="",
        max_length=8192,
        batch_size=16,
    ),
    # === GTE family (Alibaba) ===
    "Alibaba-NLP/gte-multilingual-base": ModelConfig(
        query_prefix="",
        passage_prefix="",
        max_length=8192,
        supports_flash_attn=False,  # NewModel custom architecture does not support flash-attn-2
        supports_bf16=False,        # do not auto-apply BF16 (handled by torch_dtype)
        torch_dtype="float16",      # must load in FP16: FP32 → NaN (NTK-RoPE instability), BF16 → embedding collapse
    ),
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct": ModelConfig(
        query_prefix=_E5_INSTRUCT,
        passage_prefix="",
        max_length=8192,
        batch_size=32,
        supports_flash_attn=_GTE_QWEN2_TRUST_REMOTE,  # flash_attn works with custom code (transformers 4.x only)
        trust_remote_code=_GTE_QWEN2_TRUST_REMOTE,
        # trust_remote_code=True  on transformers 4.x: bidirectional attention works correctly.
        # trust_remote_code=False on transformers 5.x: AttentionMaskConverter API changed →
        #   near-zero cosine similarities with True; falls back to standard Qwen2 (causal).
    ),
    "Alibaba-NLP/gte-Qwen2-7B-instruct": ModelConfig(
        query_prefix=_E5_INSTRUCT,
        passage_prefix="",
        max_length=8192,
        batch_size=4,
        supports_flash_attn=_GTE_QWEN2_TRUST_REMOTE,  # flash_attn works with custom code (transformers 4.x only)
        trust_remote_code=_GTE_QWEN2_TRUST_REMOTE,
    ),
    # === Nomic v2 MoE (nomic-bert, "search_query: "/"search_document: " prefix) ===
    "nomic-ai/nomic-embed-text-v2-moe": ModelConfig(
        query_prefix="search_query: ",
        passage_prefix="search_document: ",
        max_length=8192,
        supports_flash_attn=False,  # nomic-bert custom architecture
    ),
    # === VoyageAI voyage-4-nano (task-prefixed retrieval) ===
    "voyageai/voyage-4-nano": ModelConfig(
        query_prefix="Represent the query for retrieving supporting documents: ",
        passage_prefix="Represent the document for retrieval: ",
        max_length=32768,
        batch_size=32,
    ),
    # === Microsoft Harrier (bidirectional, E5-instruct-style query prefix) ===
    "microsoft/harrier-oss-v1-0.6b": ModelConfig(
        query_prefix=(
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
            "Query: "
        ),
        passage_prefix="",
        max_length=8192,
        batch_size=64,
    ),
    "microsoft/harrier-oss-v1-270m": ModelConfig(
        query_prefix=(
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
            "Query: "
        ),
        passage_prefix="",
        max_length=8192,
        batch_size=128,
    ),
    # === IBM Granite (no prefix needed) ===
    "ibm-granite/granite-embedding-278m-multilingual": ModelConfig(
        query_prefix="",
        passage_prefix="",
        max_length=512,
        supports_flash_attn=False,
    ),
    # === Perplexity pplx-embed (no prefix needed) ===
    "perplexity-ai/pplx-embed-v1-0.6b": ModelConfig(
        query_prefix="",
        passage_prefix="",
        max_length=8192,
        batch_size=64,
    ),
    "perplexity-ai/pplx-embed-v1-4b": ModelConfig(
        query_prefix="",
        passage_prefix="",
        max_length=8192,
        batch_size=4,
    ),
    # === Snowflake Arctic (XLM-R encoder, "query: " prefix) ===
    "Snowflake/snowflake-arctic-embed-l-v2.0": ModelConfig(
        query_prefix="query: ",
        passage_prefix="",
        max_length=8192,
        supports_flash_attn=False,
    ),
    "Snowflake/snowflake-arctic-embed-m-v2.0": ModelConfig(
        query_prefix="query: ",
        passage_prefix="",
        max_length=8192,
        supports_flash_attn=False,
    ),
    # === LaBSE (BERT encoder, no prefix) ===
    "sentence-transformers/LaBSE": ModelConfig(
        query_prefix="",
        passage_prefix="",
        max_length=256,
        supports_flash_attn=False,
        supports_bf16=True,
    ),
    # === Google EmbeddingGemma (Gemma3 decoder, task/query format) ===
    "google/embeddinggemma-300m": ModelConfig(
        query_prefix="task: search result | query: ",
        passage_prefix="title: none | text: ",
        max_length=2048,
        batch_size=64,
    ),
    # === SONAR (Meta, no prefix) ===
    "facebook/SONAR": ModelConfig(
        query_prefix="",
        passage_prefix="",
        max_length=512,
        supports_flash_attn=False,
    ),
    # === Jina v3 (XLM-R + LoRA, task-specific prompts) ===
    "jinaai/jina-embeddings-v3": ModelConfig(
        query_prefix="Represent the query for retrieving evidence documents: ",
        passage_prefix="Represent the document for retrieval: ",
        max_length=8192,
        supports_flash_attn=False,  # XLM-R + LoRA does not support flash-attn-2
    ),
    # === Jina v5 (Qwen3 decoder, "Query: "/"Document: " prefix) ===
    "jinaai/jina-embeddings-v5-text-small-retrieval": ModelConfig(
        query_prefix="Query: ",
        passage_prefix="Document: ",
        max_length=8192,
    ),
    # === Jina v5 nano (EuroBERT encoder, does not support flash-attn-2) ===
    "jinaai/jina-embeddings-v5-text-nano-retrieval": ModelConfig(
        query_prefix="Query: ",
        passage_prefix="Document: ",
        max_length=8192,
        supports_flash_attn=False,  # EuroBertModel does not support flash-attn-2
    ),
    # === NVIDIA Nemotron (LLaMA-based bidirectional) ===
    "nvidia/llama-embed-nemotron-8b": ModelConfig(
        query_prefix=_NEMOTRON_INSTRUCT,
        passage_prefix="",
        max_length=32768,
        batch_size=4,
    ),
    "nvidia/llama-nemotron-embed-1b-v2": ModelConfig(
        query_prefix="query: ",
        passage_prefix="passage: ",
        max_length=32768,
        batch_size=32,
    ),
}

_DEFAULT_CONFIG = ModelConfig()

# Regex to detect parameter counts like "7b", "0.6B", "13B" in model names
_PARAM_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)[bB]")


def _infer_batch_size(model_name: str) -> int:
    """Infer a reasonable batch size from the model name (parameter count hints)."""
    match = _PARAM_SIZE_RE.search(model_name.split("/")[-1])
    if match:
        size_b = float(match.group(1))
        if size_b >= 13:
            return 2
        if size_b >= 7:
            return 4
        if size_b >= 3:
            return 16
        if size_b >= 1:
            return 32
    return 64


def _infer_flash_attn(model_name: str) -> bool:
    """Infer whether flash-attention is likely supported.

    Encoder-only architectures (BERT, XLM-R, LaBSE) typically do not support
    flash-attention-2. Decoder / newer transformer variants usually do.
    """
    name_lower = model_name.lower()
    encoder_only_hints = ["bert", "xlm-r", "labse", "sonar", "arctic", "e5-large", "e5-base", "e5-small"]
    return not any(hint in name_lower for hint in encoder_only_hints)


def get_model_config(model_name: str) -> ModelConfig:
    """Return the config for *model_name*, inferring sensible defaults for unknown models."""
    if model_name in MODEL_CONFIGS:
        config = MODEL_CONFIGS[model_name]
        # Ensure name is populated
        if not config.name:
            config.name = model_name
        return config

    # Unknown model: infer settings from name heuristics
    return ModelConfig(
        name=model_name,
        batch_size=_infer_batch_size(model_name),
        supports_flash_attn=_infer_flash_attn(model_name),
    )
