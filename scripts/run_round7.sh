#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${ROUND7_CONFIG:-round7/config.yaml}"
CLUSTER_CONFIG="${ROUND7_CLUSTER_CONFIG:-}"
PYTHON="${ROUND7_PYTHON:-python3}"

args=(--config "$CONFIG" --stage all)
if [[ -n "$CLUSTER_CONFIG" ]]; then
  args+=(--cluster-config "$CLUSTER_CONFIG")
fi

"$PYTHON" -m round7.pipeline "${args[@]}"
