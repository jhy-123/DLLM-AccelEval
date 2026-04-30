# KV/Cache Management

KV/cache management methods accelerate DLLM inference by reusing intermediate states and selectively refreshing only the parts that need to be recomputed.

## Supported Methods

| Method | Paper | Config |
| --- | --- | --- |
| Prefix Cache / Dual Cache | [Paper](https://arxiv.org/pdf/2505.22618) | `cache=prefix` |
| dLLMCache | [Paper](https://arxiv.org/pdf/2506.06295) | `cache=dllm` |
| dKVCache | [Paper](https://arxiv.org/pdf/2505.15781) | `cache=dkvcache` |
| D2Cache | [Paper](https://arxiv.org/pdf/2509.23094) | `cache=d2cache` |
| SPA-Cache | [Paper](https://arxiv.org/pdf/2602.02544) | `cache=spacache` |

## Quick Start

Run the wrapper script to compare all currently configured KV/cache methods on the same model and dataset:

```bash
MODEL=llada DATASET_NAME=gsm8k bash scripts/run_kv_cache.sh
```

The script runs `prefix`, `dkvcache`, `dllm`, `d2cache`, and `spacache` for the selected model and dataset.

## Method Examples

### Prefix Cache

```bash
accelerate launch \
  --num_machines 1 \
  --num_processes 1 \
  eval.py \
  model=llada-inst \
  generation=vanilla \
  cache=prefix \
  dataset.name=gsm8k \
  batch_size=1 \
  seed=1234 \
  attn_implementation=sdpa \
  flash_attention=true \
  generation.block_length=32 \
  generation.gen_length=128 \
  generation.steps=128 \
  hydra.run.dir=./outputs/examples/kv_cache/llada-inst_prefix_gsm8k
```

Note: `cache.use_dual=true` enables the Dual Cache variant, which also caches tokens after the current decoding block. The default `cache=prefix` setting uses `cache.use_dual=false`.

### dKVCache

```bash
accelerate launch \
  --num_machines 1 \
  --num_processes 1 \
  eval.py \
  model=llada-inst \
  generation=vanilla \
  cache=dkvcache \
  dataset.name=gsm8k \
  batch_size=1 \
  seed=1234 \
  attn_implementation=sdpa \
  flash_attention=true \
  generation.block_length=32 \
  generation.cache_reloading_step=4 \
  generation.gen_length=128 \
  generation.steps=128 \
  hydra.run.dir=./outputs/examples/kv_cache/llada-inst_dkvcache_gsm8k
```

Note: `generation.cache_reloading_step` sets the KV refresh interval; this example uses `4`.

### dLLMCache

```bash
accelerate launch \
  --num_machines 1 \
  --num_processes 1 \
  eval.py \
  model=llada-inst \
  generation=vanilla \
  cache=dllm \
  dataset.name=gsm8k \
  batch_size=1 \
  seed=1234 \
  attn_implementation=sdpa \
  flash_attention=true \
  generation.block_length=32 \
  generation.gen_length=128 \
  generation.steps=128 \
  hydra.run.dir=./outputs/examples/kv_cache/llada-inst_dllm_gsm8k
```

Note: Defaults are `cache.kp=50`, `cache.kr=2`, and `cache.rou=0.25`; `kp` and `kr` are the update intervals for prompt-side and response-side KV caches, while `rou` is the fraction of response tokens adaptively updated.

### D2Cache

```bash
accelerate launch \
  --num_machines 1 \
  --num_processes 1 \
  eval.py \
  model=llada-inst \
  generation=vanilla \
  cache=d2cache \
  dataset.name=gsm8k \
  batch_size=1 \
  seed=1234 \
  attn_implementation=eager \
  flash_attention=true \
  generation.block_length=32 \
  generation.gen_length=128 \
  generation.steps=128 \
  hydra.run.dir=./outputs/examples/kv_cache/llada-inst_d2cache_gsm8k
```

Note: Defaults are `cache.rollout_p=0.1`, `cache.current_k=32`, `cache.sigma=10.0`, and `cache.inflate_w=0`; `rollout_p` is the top-p ratio for attention-rollout selection, `current_k` is the number of masked tokens to update, `sigma` is used for certainty-density calculation, and `inflate_w` is the mask-inflation window size. Use `attn_implementation=eager` to expose attention weights.

### SPA-Cache

```bash
accelerate launch \
  --num_machines 1 \
  --num_processes 1 \
  eval.py \
  model=llada-inst \
  generation=vanilla \
  cache=spacache \
  cache.svd_cache_dir=./svd_cache/llada \
  dataset.name=gsm8k \
  batch_size=1 \
  seed=1234 \
  attn_implementation=sdpa \
  flash_attention=true \
  generation.block_length=32 \
  generation.gen_length=128 \
  generation.steps=128 \
  hydra.run.dir=./outputs/examples/kv_cache/llada-inst_spacache_gsm8k
```

Note: Defaults are `cache.proxy_rank=128`, `cache.freq_dist=gaussian`, `cache.max_update_ratio=0.25`, and `cache.min_update_ratio=0.03125`; `cache.svd_cache_dir` points to the SVD proxy files.
