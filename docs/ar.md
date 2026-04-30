# AR Baselines

AR baselines provide the reference point for comparing optimized DLLMs and speculative AR decoding.

## Supported Baselines

| Baseline | Paper | Config |
| --- | --- | --- |
| Vanilla AR | / | `model=llama31-8b-inst` |
| EAGLE-3 | [Paper](https://arxiv.org/pdf/2503.01840) | `model=eagle3` |

## Quick Start

Run the vanilla AR and EAGLE-3 examples on GSM8K:

```bash
bash scripts/run_ar.sh
```

## Method Examples

### Vanilla AR

```bash
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
```

Note: this is the standard autoregressive baseline using the model config path.

### EAGLE-3

```bash
accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    eval.py \
    attn_implementation=flash_attention_2 \
    model=eagle3 \
    model.base_model_path="${LLAMA31_8B_INST_PATH}" \
    model.ea_model_path="${EAGLE3_LLAMA_EA_PATH}" \
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
```

Note: set `LLAMA31_8B_INST_PATH` and `EAGLE3_LLAMA_EA_PATH` before running the EAGLE-3 example.
