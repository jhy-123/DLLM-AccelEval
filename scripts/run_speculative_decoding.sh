#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# LLaDA / LoPA / GSM8K
accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    eval.py \
    dataset.name=gsm8k \
    batch_size=1 \
    seed=1234 \
    attn_implementation=sdpa \
    flash_attention=true \
    generation=lopa \
    generation.k=5 \
    generation.threshold=0.9 \
    generation.block_length=32 \
    generation.gen_length=256 \
    generation.steps=256 \
    model=llada-inst \
    hydra.run.dir=./outputs/examples/speculative_decoding/llada-inst-lopa-gsm8k

# LLaDA / FreeDAVE / GSM8K
accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    eval.py \
    dataset.name=gsm8k \
    batch_size=1 \
    seed=1234 \
    attn_implementation=sdpa \
    flash_attention=true \
    generation=freedave \
    generation.draft_steps=4 \
    generation.block_length=32 \
    generation.gen_length=256 \
    generation.steps=256 \
    model=llada-inst \
    hydra.run.dir=./outputs/examples/speculative_decoding/llada-inst-freedave-gsm8k
