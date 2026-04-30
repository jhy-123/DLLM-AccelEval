#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# LLaDA / SparseD / GSM8K
accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    eval.py \
    dataset.name=gsm8k \
    batch_size=1 \
    seed=1234 \
    attn_implementation=sdpa \
    flash_attention=true \
    generation=vanilla \
    generation.sparsed=true \
    generation.sparsed_select=0.5 \
    generation.sparsed_skip=0.2 \
    generation.sparsed_block_size=32 \
    generation.block_length=32 \
    generation.gen_length=256 \
    generation.steps=256 \
    model=llada-inst \
    hydra.run.dir=./outputs/examples/sparse_computation/llada-inst-sparsed-gsm8k

# LLaDA / DPad / GSM8K
accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    eval.py \
    dataset.name=gsm8k \
    batch_size=1 \
    seed=1234 \
    attn_implementation=sdpa \
    flash_attention=true \
    generation=dpad \
    generation.dropout=gaussian \
    generation.remasking=low_confidence \
    generation.early_termination=true \
    generation.block_length=32 \
    generation.gen_length=256 \
    generation.steps=256 \
    model=llada-inst \
    hydra.run.dir=./outputs/examples/sparse_computation/llada-inst-dpad-gsm8k

# LLaDA / Sparse-dLLM / GSM8K
accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    eval.py \
    dataset.name=gsm8k \
    batch_size=1 \
    seed=1234 \
    attn_implementation=sdpa \
    flash_attention=true \
    generation=sparse \
    generation.keep_ratio=0.5 \
    generation.kernel_size=3 \
    generation.remasking=low_confidence \
    generation.early_termination=true \
    generation.block_length=32 \
    generation.gen_length=256 \
    generation.steps=256 \
    model=llada-inst \
    hydra.run.dir=./outputs/examples/sparse_computation/llada-inst-sparse-dllm-gsm8k
