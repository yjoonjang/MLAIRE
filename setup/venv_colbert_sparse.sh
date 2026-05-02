#!/bin/bash
# Create .venv-colbert-sparse for multi-vector (pylate) and sparse
# (bm25s, sentence-transformers SparseEncoder) evaluations.
#
# Environment:
#   Python 3.10, torch 2.8.0, transformers 4.56.2 (pulled by pylate),
#   sentence-transformers 5.1.1 (SparseEncoder included), pylate 1.4.0,
#   bm25s 0.3.3, ranx, flash_attn 2.8.3 (prebuilt wheel from /tmp)
#
# Usage (run from project root):
#   bash setup/venv_colbert_sparse.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

VENV=.venv-colbert-sparse
PY=3.10
PYBIN="$PROJECT_ROOT/$VENV/bin/python"

echo "[1/4] Creating $VENV (Python $PY) ..."
uv venv "$VENV" --python "$PY" --clear

echo "[2/4] Installing pylate + evaluation deps ..."
uv pip install --python "$PYBIN" \
    "pylate>=1.4" \
    "bm25s" \
    "rank_bm25" \
    "ranx" \
    "click" \
    "pyyaml" \
    "einops" \
    "packaging" \
    "ninja" \
    "wheel"

echo "[3/4] Pinning torch 2.8.0 (for flash_attn wheel compatibility) ..."
uv pip install --python "$PYBIN" "torch==2.8.0"

echo "[4/4] Installing flash-attn CUDA prebuilt wheel (from /tmp) ..."
FLASH_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
(cd /tmp && uv pip install --python "$PYBIN" --no-deps --force-reinstall "$FLASH_URL")

echo ""
echo "=== Verification ==="
"$PYBIN" - <<'PY'
import torch, transformers
from sentence_transformers import SparseEncoder
from pylate.indexes import PLAID, Voyager
from rank_bm25 import BM25Okapi
import bm25s
print(f"torch:                  {torch.__version__}")
print(f"transformers:           {transformers.__version__}")
print(f"SparseEncoder:          OK")
print(f"pylate PLAID+Voyager:   OK")
print(f"bm25s / rank_bm25:      OK")
try:
    import flash_attn, flash_attn_2_cuda
    print(f"flash_attn:             {flash_attn.__version__} (CUDA kernel OK)")
except Exception as e:
    print(f"flash_attn: NOT WORKING — {e}")
print(f"CUDA available:         {torch.cuda.is_available()}")
PY

echo ""
echo "[DONE] .venv-colbert-sparse is ready at $PROJECT_ROOT/$VENV"
