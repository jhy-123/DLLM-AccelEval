#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Llama 3.1 8B Instruct / Vanilla AR / GSM8K
accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    eval.py \
    dataset.name=gsm8k \
    batch_size=1 \
    seed=1234 \
    attn_implementation=flash_attention_2 \
    model=llama31-8b-inst \
    hydra.run.dir=./outputs/examples/ar/llama31-8b-inst-gsm8k

# Llama 3.1 8B Instruct / EAGLE-3 / GSM8K
accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    eval.py \
    attn_implementation=flash_attention_2 \
    model=eagle3 \
    model.base_model_path="${LLAMA31_8B_INST_PATH:-/path/to/llama3.1-8b-instruct}" \
    model.ea_model_path="${EAGLE3_LLAMA_EA_PATH:-/path/to/llama-eagle3}" \
    model.dtype=bfloat16 \
    model.max_length=24576 \
    model.total_token=6 \
    model.depth=4 \
    model.top_k=1 \
    model.draft_sliding_window=2048 \
    dataset.name=gsm8k \
    batch_size=1 \
    seed=1234 \
    hydra.run.dir=./outputs/examples/ar/eagle3-llama31-8b-inst-gsm8k
