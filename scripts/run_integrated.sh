#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# LLaDA / Prefix Cache + Confidence / GSM8K
accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    eval.py \
    dataset.name=gsm8k \
    batch_size=1 \
    seed=1234 \
    attn_implementation=sdpa \
    flash_attention=true \
    cache=prefix \
    generation=vanilla \
    generation.block_length=32 \
    generation.threshold=0.9 \
    generation.gen_length=128 \
    generation.steps=128 \
    model=llada-inst \
    hydra.run.dir=./outputs/examples/integrated/llada-inst-prefixcache-confidence-gsm8k

# LLaDA / dKVCache + Confidence / GSM8K
accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    eval.py \
    dataset.name=gsm8k \
    batch_size=1 \
    seed=1234 \
    attn_implementation=sdpa \
    flash_attention=true \
    cache=dkvcache \
    generation=vanilla \
    generation.block_length=32 \
    generation.threshold=0.9 \
    generation.cache_reloading_step=4 \
    generation.gen_length=128 \
    generation.steps=128 \
    model=llada-inst \
    hydra.run.dir=./outputs/examples/integrated/llada-inst-dkvcache-confidence-gsm8k

# LLaDA / SparseD + Confidence / GSM8K
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
    generation.threshold=0.9 \
    generation.gen_length=128 \
    generation.steps=128 \
    model=llada-inst \
    hydra.run.dir=./outputs/examples/integrated/llada-inst-sparsed-confidence-gsm8k

# LLaDA / dParallel + Confidence / GSM8K
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
    generation.block_length=32 \
    generation.threshold=0.9 \
    generation.gen_length=128 \
    generation.steps=128 \
    model=dparallel_llada-inst \
    hydra.run.dir=./outputs/examples/integrated/dparallel-llada-inst-confidence-gsm8k

# Fast-dLLM v2 / Prefix Cache / GSM8K
accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    eval.py \
    dataset.name=gsm8k \
    batch_size=1 \
    seed=1234 \
    attn_implementation=sdpa \
    flash_attention=true \
    cache=prefix \
    generation=fast_dllm_v2 \
    generation.block_length=32 \
    generation.threshold=0.9 \
    generation.gen_length=2048 \
    generation.steps=2048 \
    generation.small_block_size=8 \
    add_bos_token=false \
    model.generation.add_bos_token=false \
    model=fast_dllm_v2_7b-inst \
    hydra.run.dir=./outputs/examples/integrated/fastdllmv2-prefix-gsm8k
