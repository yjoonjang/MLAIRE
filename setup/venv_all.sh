#!/bin/bash
# Create all three venvs sequentially (main → gte → colbert-sparse).
#
# Usage (run from project root):
#   bash setup/venv_all.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

echo "########################################"
echo "# [1/3] Main .venv"
echo "########################################"
bash "$HERE/venv_main.sh"

echo ""
echo "########################################"
echo "# [2/3] .venv-gte"
echo "########################################"
bash "$HERE/venv_gte.sh"

echo ""
echo "########################################"
echo "# [3/3] .venv-colbert-sparse"
echo "########################################"
bash "$HERE/venv_colbert_sparse.sh"

echo ""
echo "All three venvs created."
