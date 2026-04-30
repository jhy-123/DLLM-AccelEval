# Speculative Decoding

Speculative decoding accelerates DLLM inference by drafting candidate tokens and verifying them with the target model.

## Supported Methods

| Method | Paper | Config |
| --- | --- | --- |
| LoPA | [Paper](https://arxiv.org/pdf/2512.16229) | `generation=lopa` |
| FreeDAVE | [Paper](https://arxiv.org/pdf/2510.00294) | `generation=freedave` |

## Quick Start

Run all speculative-decoding examples on LLaDA and GSM8K:

```bash
bash scripts/run_speculative_decoding.sh
```

## Method Examples

### LoPA

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
    generation=lopa \
    generation.k=5 \
    generation.threshold=0.9 \
    generation.block_length=32 \
    generation.gen_length=256 \
    generation.steps=256 \
    model=llada-inst \
    hydra.run.dir=./outputs/examples/speculative_decoding/llada-inst-lopa-gsm8k
```

Note: `k` sets the proposal width and `threshold` controls candidate acceptance.

### FreeDAVE

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
    generation=freedave \
    generation.draft_steps=4 \
    generation.block_length=32 \
    generation.gen_length=256 \
    generation.steps=256 \
    model=llada-inst \
    hydra.run.dir=./outputs/examples/speculative_decoding/llada-inst-freedave-gsm8k
```

Note: `draft_steps` controls how many draft denoising steps are proposed before verification.
