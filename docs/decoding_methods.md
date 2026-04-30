# Decoding Methods

Decoding methods accelerate DLLM inference by changing the denoising schedule, token selection rule, or early-exit policy.

## Supported Methods

| Method | Paper | Config |
| --- | --- | --- |
| Top-k | / | `generation=vanilla generation.steps=128` |
| Confidence Threshold | / | `generation=vanilla generation.threshold=0.9` |
| UNCODE | [Paper](https://openreview.net/pdf?id=oCdHXvyKLB) | `generation=pc_sampler` |
| EB-Sampler | [Paper](https://arxiv.org/pdf/2505.24857) | `generation=eb_sampler` |
| KLASS | [Paper](https://arxiv.org/pdf/2511.05664) | `generation=klass` |
| WINO | [Paper](https://arxiv.org/pdf/2507.18578) | `generation=wino` |
| dParallel | [Paper](https://arxiv.org/pdf/2509.26488) | `model=dparallel_llada-inst` |
| DAEDAL | [Paper](https://arxiv.org/pdf/2508.00819) | `generation=daedal` |

## Quick Start

Run all decoding-method examples on LLaDA and GSM8K:

```bash
bash scripts/run_decoding_methods.sh
```

## Method Examples

### Top-k

```bash
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
```

Note: this setting uses half as many denoising steps as the generation length.

### Confidence Threshold

```bash
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
```

Note: `threshold` fixes tokens once their confidence is high enough.

### UNCODE

```bash
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
```

Note: UNCODE is implemented as `pc_sampler`; `debias` enables probability correction and `clip_alpha` clips the correction strength.

### EB-Sampler

```bash
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
```

Note: `gamma` controls the evidence-bonus strength during token selection.

### KLASS

```bash
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
```

Note: `kl_threshold` is the convergence threshold and `kl_history_length` is the KL history window.

### WINO

```bash
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
```

Note: `wide_in_thres` expands the candidate update set and `narrow_out_thres` filters the final update set.

### dParallel

```bash
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
```

Note: dParallel uses the dParallel model wrapper while keeping the same vanilla generation config.

### DAEDAL

```bash
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
```

Note: `initial_gen_length` sets the first decoding budget and `max_gen_length` caps adaptive expansion.
