#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# LLaDA / Top-k / GSM8K
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
    generation.gen_length=256 \
    generation.steps=128 \
    model=llada-inst \
    hydra.run.dir=./outputs/examples/decoding_methods/llada-inst-topk-gsm8k

# LLaDA / Confidence Threshold / GSM8K
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
    generation.threshold=0.9 \
    generation.block_length=32 \
    generation.gen_length=256 \
    generation.steps=256 \
    model=llada-inst \
    hydra.run.dir=./outputs/examples/decoding_methods/llada-inst-confidence-threshold-gsm8k

# LLaDA / UNCODE / GSM8K
accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    eval.py \
    dataset.name=gsm8k \
    batch_size=1 \
    seed=1234 \
    attn_implementation=sdpa \
    flash_attention=true \
    generation=pc_sampler \
    generation.debias=true \
    generation.clip_alpha=10 \
    generation.block_length=32 \
    generation.gen_length=256 \
    generation.steps=256 \
    model=llada-inst \
    hydra.run.dir=./outputs/examples/decoding_methods/llada-inst-uncode-gsm8k

# LLaDA / EB-Sampler / GSM8K
accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    eval.py \
    dataset.name=gsm8k \
    batch_size=1 \
    seed=1234 \
    attn_implementation=sdpa \
    flash_attention=true \
    generation=eb_sampler \
    generation.gamma=0.001 \
    generation.block_length=32 \
    generation.gen_length=256 \
    generation.steps=256 \
    model=llada-inst \
    hydra.run.dir=./outputs/examples/decoding_methods/llada-inst-eb-sampler-gsm8k

# LLaDA / KLASS / GSM8K
accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    eval.py \
    dataset.name=gsm8k \
    batch_size=1 \
    seed=1234 \
    attn_implementation=sdpa \
    flash_attention=true \
    generation=klass \
    generation.kl_threshold=0.01 \
    generation.kl_history_length=2 \
    generation.block_length=32 \
    generation.gen_length=256 \
    generation.steps=256 \
    model=llada-inst \
    hydra.run.dir=./outputs/examples/decoding_methods/llada-inst-klass-gsm8k

# LLaDA / WINO / GSM8K
accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    eval.py \
    dataset.name=gsm8k \
    batch_size=1 \
    seed=1234 \
    attn_implementation=sdpa \
    flash_attention=true \
    generation=wino \
    generation.wide_in_thres=0.7 \
    generation.narrow_out_thres=0.9 \
    generation.block_length=32 \
    generation.gen_length=256 \
    generation.steps=256 \
    model=llada-inst \
    hydra.run.dir=./outputs/examples/decoding_methods/llada-inst-wino-gsm8k

# LLaDA / dParallel / GSM8K
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
    generation.gen_length=256 \
    generation.steps=256 \
    model=dparallel_llada-inst \
    hydra.run.dir=./outputs/examples/decoding_methods/dparallel-llada-inst-gsm8k

# LLaDA / DAEDAL / GSM8K
accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    eval.py \
    dataset.name=gsm8k \
    batch_size=1 \
    seed=1234 \
    attn_implementation=sdpa \
    flash_attention=true \
    generation=daedal \
    generation.initial_gen_length=64 \
    generation.block_length=32 \
    generation.gen_length=256 \
    generation.steps=256 \
    generation.max_gen_length=2048 \
    model=llada-inst \
    hydra.run.dir=./outputs/examples/decoding_methods/llada-inst-daedal-gsm8k
