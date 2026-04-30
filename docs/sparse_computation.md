# Sparse Computation

Sparse computation reduces the number of token or block computations during DLLM denoising while keeping the same evaluation protocol.

## Supported Methods

| Method | Paper | Config |
| --- | --- | --- |
| SparseD | [Paper](https://arxiv.org/pdf/2509.24014) | `generation=vanilla generation.sparsed=true` |
| DPad | [Paper](https://arxiv.org/pdf/2508.14148) | `generation=dpad` |
| Sparse-dLLM | [Paper](https://arxiv.org/pdf/2508.02558) | `generation=sparse` |

## Quick Start

Run all sparse-computation examples on LLaDA and GSM8K:

```bash
bash scripts/run_sparse_computation.sh
```

## Method Examples

### SparseD

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
    generation.gen_length=256 \
    generation.steps=256 \
    model=llada-inst \
    hydra.run.dir=./outputs/examples/sparse_computation/llada-inst-sparsed-gsm8k
```

Note: `sparsed_select` controls the selected sparse-token ratio, `sparsed_skip` controls the skipped-token ratio, and `sparsed_block_size` sets the sparse block size.

### DPad

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
    generation=dpad \
    generation.dropout=gaussian \
    generation.remasking=low_confidence \
    generation.early_termination=true \
    generation.block_length=32 \
    generation.gen_length=256 \
    generation.steps=256 \
    model=llada-inst \
    hydra.run.dir=./outputs/examples/sparse_computation/llada-inst-dpad-gsm8k
```

Note: `dropout` chooses the sparse dropout pattern, `remasking` keeps the low-confidence remasking rule, and `early_termination` enables early stop when decoding has converged.

### Sparse-dLLM

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
```

Note: `keep_ratio` is the retained-token ratio for sparse updates, and `kernel_size` controls the local smoothing window.
