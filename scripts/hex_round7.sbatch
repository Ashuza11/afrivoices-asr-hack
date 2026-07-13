#!/usr/bin/env bash
#SBATCH --job-name=afrivoices-r7
#SBATCH --partition=a5000-48h
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=outputs/round7/logs/%x-%j.out
#SBATCH --error=outputs/round7/logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
VENV="${ROUND7_VENV:-$REPO_ROOT/.venv-round7}"

cd "$REPO_ROOT"
source "$VENV/bin/activate"
if [[ ! -f .env ]]; then
  echo "Missing $REPO_ROOT/.env" >&2
  exit 1
fi
set -a
source .env
set +a

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export ROUND7_CONFIG=round7/config.hex.yaml
export ROUND7_CLUSTER_CONFIG=round7/cluster.hex.yaml
export ROUND7_STAGE="${ROUND7_STAGE:-all}"

python - <<'PY'
import os
import platform
import torch

print(f"Host: {platform.node()}")
print(f"Slurm job: {os.environ.get('SLURM_JOB_ID')}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory GiB: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}")
PY

bash scripts/run_round7.sh
