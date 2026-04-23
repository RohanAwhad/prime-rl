# Prime-RL Experiment Summary

## Setup

- **Machine**: 8x NVIDIA H100 80GB HBM3, CentOS Stream 9
- **CUDA**: Toolkit 12.4.1, Driver 550.90.07
- **PyTorch**: 2.10.0+cu128

## Problem: deep-gemm / CUDA Mismatch

The pre-built `deep-gemm` wheel pinned in `pyproject.toml` (`deep_gemm-2.3.0+477618c`) was compiled against **CUDA 13** (`libcudart.so.13`). Our system only has CUDA 12.4.

- **Building from source fails too** -- commit `477618c` uses CUDA 13-only APIs (`CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B`)
- **Minimum CUDA for any recent deep-gemm**: 12.8 (per [Issue #202](https://github.com/deepseek-ai/DeepGEMM/issues/202))
- **Upgrading CUDA toolkit to 13 is not viable** -- requires new NVIDIA driver (550 only supports up to 12.x), plus PyTorch and all CUDA extensions would need rebuilds

**Workaround**: `VLLM_USE_DEEP_GEMM=0` disables the deep-gemm import in vLLM entirely. This has no functional impact for non-FP8 models like Qwen3.

## Multi-GPU Configuration (6 Inference + 2 Training)

Key config additions to `rl.toml`:

```toml
[deployment]
num_train_gpus = 2
num_infer_gpus = 6

[inference.parallel]
dp = 6
```

- GPUs 0-5 are assigned to inference (6 data-parallel vLLM replicas)
- GPUs 6-7 are assigned to training (FSDP with world_size=2)
- The orchestrator runs on CPU, no GPU needed
- `inference.parallel.dp` must match `num_infer_gpus` when using TP=1
- Total must not exceed `gpus_per_node` (default 8)

## Scaling to Qwen3-4B

Changed model from `PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT` to `Qwen/Qwen3-4B-Instruct-2507`.

Added activation checkpointing to fit the larger model in memory:

```toml
[trainer.model.ac]
freq = 1
```

## Experiment Results

### Run Config

| Parameter | Value |
|---|---|
| Model | Qwen/Qwen3-4B-Instruct-2507 |
| Inference GPUs | 6 (dp=6) |
| Training GPUs | 2 (FSDP) |
| Steps | 100 |
| Batch size | 128 |
| Rollouts per example | 16 |
| Max completion tokens | 128 |
| Learning rate | 3e-6 |
| Activation checkpointing | freq=1 |

### Training Metrics

| Metric | Step 0 | Step ~10 | Step ~50 | Step 99 (final) |
|---|---|---|---|---|
| Loss | 0.0117 | -0.0006 | 0.0003 | -0.0001 |
| Entropy | 2.178 | 0.845 | 0.618 | 0.622 |
| Mismatch KL | 0.002 | 0.054 | 0.078 | 0.068 |
| Grad Norm | 0.899 | 1.261 | 1.791 | 1.371 |
| Throughput | 0 (warmup) | ~1243 tok/s | ~1040 tok/s | ~1107 tok/s |
| Peak Memory | 40.9 GiB | 41.7 GiB | 41.7 GiB | 41.7 GiB |
| Step Time | 56.7s (warmup) | ~1.9s | ~2.6s | ~2.0s |

- **Total wall time**: ~20 minutes for 100 steps (including warmup)
- **Final checkpoint written** after step 99

### Observations

- **Entropy dropped from 2.18 to ~0.55-0.62** over 100 steps -- model became more confident
- **Loss stabilized near zero** quickly, suggesting the task is learnable
- **Peak memory 41.7 GiB** out of 80 GiB per GPU -- headroom for larger batch sizes or longer sequences
- **Step 0 was slow (57s)** due to warmup/compilation; subsequent steps averaged ~2.2s
- **Throughput settled at ~1050-1110 tokens/s** with MFU ~1.4-1.5%
- **Run completed cleanly** with final checkpoint saved

## Final Config

```toml
max_steps = 100
seq_len = 2048

[deployment]
num_train_gpus = 2
num_infer_gpus = 6

[model]
name = "Qwen/Qwen3-4B-Instruct-2507"

[wandb]
project = "reverse-text"
name = "reverse-text-4b-8gpu"

[orchestrator]
batch_size = 128
rollouts_per_example = 16

[orchestrator.train.sampling]
max_completion_tokens = 128

[[orchestrator.train.env]]
id = "rohans_reverse_text@local"

[trainer.optim]
lr = 3e-6

[trainer.model.ac]
freq = 1

[ckpt]

[inference.parallel]
dp = 6
```

## How to Run

```bash
VLLM_USE_DEEP_GEMM=0 uv run rl @ examples/reverse_text/rl.toml --clean-output-dir
```
