#!/bin/bash
# Create .venv-gte for GTE-Qwen2 family and jina-embeddings-v3.
#
# These models need transformers < 5 because their custom modeling code
# uses AttentionMaskConverter / DynamicCache APIs that changed in
# transformers 5.x (bidirectional attention breaks; GTE-Qwen2 collapses
# to causal attention; jina-v3's LoRA adapter dispatch is also affected).
#
# Environment:
#   Python 3.10, torch 2.8.0, transformers 4.57.6,
#   sentence-transformers 3.4.1, datasets 3.6.0, pyarrow 17.0.0
#
# Usage (run from project root):
#   bash setup/venv_gte.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

VENV=.venv-gte
PY=3.10
PYBIN="$PROJECT_ROOT/$VENV/bin/python"

echo "[1/3] Creating $VENV (Python $PY) ..."
uv venv "$VENV" --python "$PY" --clear

echo "[2/3] Installing GTE-compatible package versions ..."
uv pip install --python "$PYBIN" \
    "torch==2.8.0" \
    "transformers==4.57.6" \
    "sentence-transformers==3.4.1" \
    "datasets==3.6.0" \
    "pyarrow==17.0.0" \
    "huggingface-hub>=0.24.0,<1.0.0" \
    "accelerate>=0.31.0" \
    "peft>=0.13.0" \
    "numpy<2" \
    "click" \
    "pyyaml" \
    "fire" \
    "einops" \
    "setproctitle" \
    "mteb"

echo "[3/3] Installing flash-attn CUDA prebuilt wheel (from /tmp) ..."
FLASH_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
(cd /tmp && uv pip install --python "$PYBIN" --no-deps --force-reinstall "$FLASH_URL")

echo ""
echo "=== Verification ==="
"$PYBIN" - <<'PY'
import torch, transformers, sentence_transformers, datasets
print(f"torch:                  {torch.__version__}")
print(f"transformers:           {transformers.__version__}  (expect 4.57.x)")
print(f"sentence-transformers:  {sentence_transformers.__version__}  (expect 3.4.x)")
print(f"datasets:               {datasets.__version__}")
try:
    import flash_attn, flash_attn_2_cuda
    print(f"flash_attn:             {flash_attn.__version__} (CUDA kernel OK)")
except Exception as e:
    print(f"flash_attn: NOT WORKING — {e}")
print(f"CUDA available:         {torch.cuda.is_available()}")
PY

echo ""
echo "[DONE] .venv-gte is ready at $PROJECT_ROOT/$VENV"
