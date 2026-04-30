# DLLM-AccelEval

Evaluation code for **A Comparative Survey of Inference Acceleration for DLLMs against AR-LLMs: No Free Lunch**.

This repository organizes the experiments in the survey around two questions:

1. How do different inference-acceleration techniques behave on diffusion language models (DLLMs)?
2. How do optimized or integrated DLLM systems compare with autoregressive LLM (AR-LLM) baselines?

https://github.com/user-attachments/assets/1b30415c-5f40-40e1-aee7-56c1e74f3a6e

## Installation

```bash
git clone <repo-url>
cd DLLM-AccelEval

conda create -n dllm-acceleval python=3.10 -y
conda activate dllm-acceleval

pip install -r requirements.txt
pip install -e .
```

Create a local environment file and fill in model paths:

```bash
cp .env.example .env
```

Before running examples that rely on local model paths, export the required variables or load them from `.env`.

## Supported Methods

| Area | Documentation | Example script |
| --- | --- | --- |
| KV/cache management | [`docs/kv_cache.md`](docs/kv_cache.md) | `scripts/run_kv_cache.sh` |
| Sparse computation | [`docs/sparse_computation.md`](docs/sparse_computation.md) | `scripts/run_sparse_computation.sh` |
| Decoding methods | [`docs/decoding_methods.md`](docs/decoding_methods.md) | `scripts/run_decoding_methods.sh` |
| Speculative decoding | [`docs/speculative_decoding.md`](docs/speculative_decoding.md) | `scripts/run_speculative_decoding.sh` |
| Integrated DLLM systems | [`docs/integrated.md`](docs/integrated.md) | `scripts/run_integrated.sh` |
| AR baselines | [`docs/ar.md`](docs/ar.md) | `scripts/run_ar.sh` |

## Quick Start

Run the KV-cache examples:

```bash
bash scripts/run_kv_cache.sh
```

Run the sparse-computation examples:

```bash
bash scripts/run_sparse_computation.sh
```

Run the decoding-method examples:

```bash
bash scripts/run_decoding_methods.sh
```

Run the speculative-decoding examples:

```bash
bash scripts/run_speculative_decoding.sh
```

Run the integrated DLLM examples:

```bash
bash scripts/run_integrated.sh
```

Run the AR baseline examples:

```bash
bash scripts/run_ar.sh
```

Each script contains explicit example commands and writes Hydra outputs under `outputs/examples/`.

## Acknowledgements

This repository builds on and adapts ideas, interfaces, and evaluation utilities from several open research projects. We gratefully acknowledge the authors and maintainers of [d2Cache](https://github.com/Kamichanw/d2Cache), [Fast-dLLM v2](https://research.nvidia.com/labs/eai/publication/fast-dllm-v2/), [EAGLE-3](https://github.com/SafeAILab/EAGLE), and the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness). Their released methods and infrastructure provide important foundations for reproducible studies of efficient LLM inference.

