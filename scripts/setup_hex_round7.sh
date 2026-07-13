#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ROUND7_VENV:-$REPO_ROOT/.venv-round7}"

cd "$REPO_ROOT"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install -r requirements-round7.txt

python - <<'PY'
import sys
import torch
import transformers

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
PY

python -m unittest discover -s round7/tests -v

echo "Hex Round 7 environment created at $VENV"
