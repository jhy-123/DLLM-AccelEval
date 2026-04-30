# Integrated DLLM Setups

Integrated settings combine multiple DLLM acceleration choices and compare them against integrated model-level acceleration.

## Example Setups

This section shows representative combinations used by the examples; the same acceleration components can be applied to other supported DLLM model configs when compatible.

| Setup | Paper | Config |
| --- | --- | --- |
| Prefix Cache + Confidence | / | `cache=prefix generation=vanilla generation.threshold=0.9` |
| dKVCache + Confidence | / | `cache=dkvcache generation=vanilla generation.threshold=0.9` |
| SparseD + Confidence | / | `generation=vanilla generation.sparsed=true generation.threshold=0.9` |
| dParallel + Confidence | / | `model=dparallel_llada-inst generation=vanilla generation.threshold=0.9` |
| Fast-dLLM v2 + Prefix Cache | [Paper](https://arxiv.org/pdf/2509.26328) | `model=fast_dllm_v2_7b-inst generation=fast_dllm_v2 cache=prefix` |

## Quick Start

Run all integrated examples on GSM8K:

```bash
bash scripts/run_integrated.sh
```

## Method Examples

### LLaDA + Prefix Cache + Confidence

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
    cache=prefix \
    generation=vanilla \
    generation.block_length=32 \
    generation.threshold=0.9 \
    generation.gen_length=128 \
    generation.steps=128 \
    model=llada-inst \
    hydra.run.dir=./outputs/examples/integrated/llada-inst-prefixcache-confidence-gsm8k
```

Note: this combines prefix KV reuse with confidence-based early token fixing.

### LLaDA + dKVCache + Confidence

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
    cache=dkvcache \
    generation=vanilla \
    generation.block_length=32 \
    generation.threshold=0.9 \
    generation.cache_reloading_step=4 \
    generation.gen_length=128 \
    generation.steps=128 \
    model=llada-inst \
    hydra.run.dir=./outputs/examples/integrated/llada-inst-dkvcache-confidence-gsm8k
```

Note: `generation.cache_reloading_step=4` refreshes the KV cache every four decoding steps.

### LLaDA + SparseD + Confidence

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
```

Note: this combines sparse token updates with confidence-based token fixing.

### LLaDA + dParallel + Confidence

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
    generation.threshold=0.9 \
    generation.gen_length=128 \
    generation.steps=128 \
    model=dparallel_llada-inst \
    hydra.run.dir=./outputs/examples/integrated/dparallel-llada-inst-confidence-gsm8k
```

Note: dParallel changes the model wrapper and keeps the same confidence decoding rule.

### Fast-dLLM v2 + Prefix Cache

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
```

Note: Fast-dLLM v2 uses its own block-diffusion generation config; `small_block_size` controls the sub-block cache granularity.
