#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -x .venv-round7/bin/python ]]; then
  echo "Missing .venv-round7. Run: bash scripts/setup_hex_round7.sh" >&2
  exit 1
fi
if [[ ! -f .env ]]; then
  echo "Missing .env with HF_TOKEN and KAGGLE_API_TOKEN." >&2
  exit 1
fi

mkdir -p outputs/round7/logs
sbatch scripts/hex_round7.sbatch
