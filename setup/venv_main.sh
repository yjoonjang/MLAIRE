#!/bin/bash
# Create the main .venv for general dense embedding models.
#
# Environment:
#   Python 3.10, torch 2.8.0, transformers 5.5.0, sentence-transformers 5.3.0
#   flash_attn 2.8.3 (prebuilt wheel — installed from /tmp to bypass the
#   project's [tool.uv.extra-build-variables] setting that skips CUDA build).
#
# Usage (run from project root):
#   bash setup/venv_main.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

VENV=.venv
PY=3.10
PYBIN="$PROJECT_ROOT/$VENV/bin/python"

echo "[1/3] Creating $VENV (Python $PY) ..."
uv venv "$VENV" --python "$PY" --clear

echo "[2/3] Installing core dependencies (no project install — avoids setuptools autodiscovery) ..."
uv pip install --python "$PYBIN" \
    "datasets>=2.14.0,<4.0.0" \
    "einops>=0.8.2" \
    "fire>=0.7.1" \
    "huggingface-hub>=1.9.0" \
    "mteb>=2.12.9" \
    "numpy<2" \
    "nvitop>=1.6.2" \
    "peft>=0.18.1" \
    "pillow>=12.2.0" \
    "sentence-transformers[train]==5.3.0" \
    "setproctitle>=1.3.7" \
    "torch==2.8.0" \
    "torchvision>=0.23.0" \
    "transformers>=5.5.0" \
    "wandb>=0.24.0" \
    "click" \
    "pyyaml"

echo "[3/3] Installing flash-attn CUDA prebuilt wheel (from /tmp to bypass pyproject config) ..."
FLASH_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
(cd /tmp && uv pip install --python "$PYBIN" --no-deps --force-reinstall "$FLASH_URL")

echo ""
echo "=== Verification ==="
"$PYBIN" - <<'PY'
import torch, transformers, sentence_transformers, flash_attn, flash_attn_2_cuda
print(f"torch:                  {torch.__version__}")
print(f"transformers:           {transformers.__version__}")
print(f"sentence-transformers:  {sentence_transformers.__version__}")
print(f"flash_attn:             {flash_attn.__version__} (CUDA kernel OK)")
print(f"CUDA available:         {torch.cuda.is_available()}")
PY

echo ""
echo "[DONE] Main .venv is ready at $PROJECT_ROOT/$VENV"
