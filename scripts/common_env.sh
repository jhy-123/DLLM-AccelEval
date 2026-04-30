#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT_DIR/.env"
  set +a
fi

export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-0}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
PYTHON_BIN="${PYTHON_BIN:-python}"

run_python_eval() {
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON_BIN" eval.py "$@"
}

run_accelerate_eval() {
  if ! command -v "$ACCELERATE_BIN" >/dev/null 2>&1; then
    echo "accelerate binary not found: $ACCELERATE_BIN"
    exit 127
  fi

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$ACCELERATE_BIN" launch \
    --num_machines "${NUM_MACHINES:-1}" \
    --num_processes "${NUM_PROCESSES:-1}" \
    eval.py "$@"
}
