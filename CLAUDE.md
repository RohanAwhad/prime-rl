# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PRIME-RL is an asynchronous reinforcement learning framework for LLMs at scale. The architecture uses an **orchestrator-based design** that coordinates three main components:

**Core architecture:**
- **Orchestrator** (`src/prime_rl/orchestrator/`): CPU process coordinating data flow; collects rollouts from inference, assembles batches, dispatches to trainer; relays updated weights back to inference
- **Trainer** (`src/prime_rl/trainer/`): FSDP2-based training (RL and SFT); produces updated policy from rollouts; supports AIPO, GRPO, GSPO, OPO, RLOO, CISPO objectives
- **Inference** (`src/prime_rl/inference/`): vLLM-based OpenAI-compatible server; generates rollouts; supports custom endpoints (`update_weights`, `reload_weights`)
- **Communication**: Orchestrator ↔ Inference (async via OpenAI client), Orchestrator ↔ Trainer (rollout batches), Trainer → Inference (checkpoint updates)

**Package name**: `prime_rl` (import name), repo name `prime-rl`

## Common Commands

### Environment Setup
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install dependencies (Python 3.12 required)
uv sync

# Optional: Install flash-attn for better performance
# (Pre-built wheels available; fast install)
uv sync --extra fa  # or just during initial sync

# Verify installation
uv run python -c "import flash_attn"

# Install pre-commit hooks
uv run pre-commit install
```

### Development Workflow

**SFT Training (supervised fine-tuning):**
```bash
# Single GPU
uv run sft @ configs/debug/sft/train.toml

# Multi-GPU (8 GPUs example)
uv run torchrun --nproc-per-node 8 src/prime_rl/trainer/sft/train.py @ configs/examples/reverse_text/sft/train.toml
```

**RL Training (reinforcement learning):**
```bash
# Start all components individually (multi-terminal):
# Terminal 1: Inference server
uv run inference @ configs/debug/infer.toml

# Terminal 2: RL Trainer
uv run trainer @ configs/debug/rl/train.toml

# Terminal 3: Orchestrator
uv run orchestrator @ configs/debug/orch.toml

# OR: Single-node all-in-one (spawns all three)
uv run rl \
  --trainer @ configs/debug/rl/train.toml \
  --orchestrator @ configs/debug/orch.toml \
  --inference @ configs/debug/infer.toml
```

**Evaluation:**
```bash
# Eval with API models (default: gpt-4.1-mini via OpenAI)
export OPENAI_API_KEY=...
uv run eval @ configs/debug/eval/single_env.toml

# Eval against local vLLM server
uv run eval @ configs/debug/eval/local_model.toml --client.base-url http://localhost:8000/v1

# Eval training checkpoints
uv run eval @ configs/debug/eval/single_env.toml --weights-dir outputs/weights --steps 100,200,300
```

**Synthetic Data Generation:**
```bash
export OPENAI_API_KEY=...
uv run synthesize @ configs/debug/synthesize/single_turn.toml
```

### Testing
```bash
# Full test suite
uv run pytest -v

# Unit tests only
uv run pytest tests/unit -v

# Integration tests only
uv run pytest tests/integration -v

# CPU-only tests (no GPU required)
uv run pytest -v -m "not gpu"

# Fast tests only (skip slow tests)
uv run pytest -v -m "not slow"

# Single test file or function
uv run pytest tests/unit/test_specific.py::test_function_name -v
```

### Linting & Formatting
```bash
# Run pre-commit manually (ruff format + ruff lint)
uv run pre-commit run --all-files

# Ruff directly (line length 120, import sorting enabled)
uv run ruff check src tests
uv run ruff format src tests
```

## Configuration System

All entrypoints use `pydantic-settings` with **layered precedence**:

1. **CLI args** (highest): `--key.subkey value` (e.g., `--model.name Qwen/Qwen3-8B`)
2. **Config files**: `@ path/to/config.toml` (space after `@` required)
3. **Environment vars**: `PRIME_KEY__SUBKEY=value` (double underscore for nesting)
4. **Defaults** (lowest): Defined in `src/prime_rl/{trainer,inference,orchestrator,eval}/config.py`

**Multiple config files merge left-to-right** (later overrides earlier):
```bash
uv run inference @ base.toml @ override.toml --model.name Qwen/Qwen3-32B
```

**View all config options:**
```bash
uv run rl --help
uv run sft --help
uv run trainer --help
uv run inference --help
uv run orchestrator --help
uv run eval --help
uv run synthesize --help
```

## Code Structure

### Entry Points (CLI Commands)
Defined in `pyproject.toml` `[project.scripts]`:
- `uv run rl`: All-in-one RL training launcher (`src/prime_rl/rl.py`)
- `uv run trainer`: RL trainer only (`src/prime_rl/trainer/rl/train.py`)
- `uv run orchestrator`: Orchestrator only (`src/prime_rl/orchestrator/orchestrator.py`)
- `uv run inference`: vLLM inference server (`src/prime_rl/inference/server.py`)
- `uv run sft`: SFT trainer (`src/prime_rl/trainer/sft/train.py`)
- `uv run eval`: Evaluation harness (`src/prime_rl/eval/eval.py`)
- `uv run synthesize`: Synthetic data generation (`src/prime_rl/synthesize/synthesize.py`)
- `uv run monitor`: Wandb training monitor (`src/prime_rl/monitor/wandb_monitor.py`)

### Core Modules

**`src/prime_rl/trainer/`:**
- `rl/`: RL-specific training (AIPO, GRPO, etc.)
  - `train.py`: Main RL training loop
  - `loss.py`: Loss implementations
  - `data.py`: Rollout data handling
- `sft/`: SFT-specific training
  - `train.py`: Main SFT training loop
  - `data.py`: Conversational dataset handling
- `config.py`: Shared trainer config (AdamConfig, ModelConfig, ParallelDims, etc.)
- `model.py`: Model initialization, FSDP setup
- `ckpt.py`: FSDP checkpoint save/load
- `lora.py`: LoRA training support
- `optim.py`: Optimizer setup
- `parallel_dims.py`: Parallelism configuration (DP, TP, PP, CP)
- `models/`: Custom model implementations (performance optimizations)

**`src/prime_rl/orchestrator/`:**
- `orchestrator.py`: Main orchestration logic
- `rollout_manager.py`: Rollout collection and batching
- `weight_relay.py`: Weight update coordination
- `config.py`: Orchestrator config schema

**`src/prime_rl/inference/`:**
- `server.py`: vLLM server with custom endpoints
- `config.py`: Inference config (SamplingConfig, ModelConfig, etc.)
- `vllm/`: vLLM integration and patches
- `patches.py`: Upstream vLLM bug fixes

**`src/prime_rl/eval/`:**
- `eval.py`: Evaluation harness
- `envs.py`: Environment integration (`verifiers`)
- `config.py`: Eval config schema

**`src/prime_rl/synthesize/`:**
- `synthesize.py`: Synthetic data generation
- `config.py`: Synthesis config schema

**`src/prime_rl/utils/`:**
- `pydantic_config.py`: Config parsing (TOML + CLI + env vars)
- `monitor.py`: Metrics/logging (wandb integration)
- `logger.py`: Loguru setup for structured logging
- `vf.py`: `verifiers` environment utilities
- `client.py`: OpenAI-compatible client helpers

### Data Flow

**RL Training Loop:**
1. **Inference**: Generates rollouts from policy π_{n-k} (async, k=`max_async_level`)
2. **Orchestrator**: Collects rollouts via OpenAI client, assembles batches, sends to trainer
3. **Trainer**: Computes AIPO/GRPO loss, optimizes policy → π_n, saves checkpoint
4. **Orchestrator**: Relays checkpoint to inference via `update_weights` endpoint
5. **Inference**: Reloads weights, continues generation

**SFT Training:**
- Loads conversational dataset (prompt-completion format)
- Tokenizes with loss masking (requires prefix-preserving chat template)
- Standard supervised training with FSDP

**Evaluation:**
- Runs `verifiers` environments against models/checkpoints
- Supports single/multi-environment evaluation
- Can evaluate API models, local models, or checkpoint progressions

### Parallelism Implementation

**Training (FSDP2):**
- Config: `ParallelDims` in `src/prime_rl/trainer/parallel_dims.py`
  - `dp`: Data parallelism (default: world_size / tp)
  - `tp`: Tensor parallelism (model sharding)
  - `pp`: Pipeline parallelism (layer-wise sharding, experimental)
  - `cp`: Context parallelism (sequence sharding for long contexts)
- Execution: `torchrun --nproc-per-node N` handles distributed setup
- BF16 mixed precision default
- Gradient accumulation via `micro_bs` (per-GPU) and `batch_size` (global)

**Inference (vLLM):**
- TP: vLLM native tensor parallelism (set via `--tensor-parallel-size` or `--parallel.tp`)
- DP: vLLM native data parallelism (multi-GPU with shared KV cache, set via `--parallel.dp`)
- Scheduling: vLLM continuous batching for throughput

## Verifiers Integration

Tasks/environments use the [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) library:
- Install from Environments Hub: `{env_org}/{env_id}` format auto-installs
- Examples: `gsm8k`, `hendrycks-math`, `alphabet-sort`, `wiki-search`
- Multi-turn and tool-calling support
- Orchestrator/eval use `verifiers` for rollout generation and scoring

## Async Training

PRIME-RL uses **asynchronous off-policy RL**:
- Inference generates from policy π_{n-k} while trainer optimizes π_n
- `max_async_level` (default: 2) controls max staleness
- Loss: AIPO (token-level importance sampling with clipping)
- See `docs/async.md` for detailed explanation

## Important Patterns

### Config Schema
When adding new config options:
1. Add field to appropriate config class (e.g., `TrainConfig` in `src/prime_rl/trainer/config.py`)
2. Use `Annotated[type, Field(...)]` for validation and defaults
3. Access via `config.section.field` in code
4. Update corresponding TOML examples in `configs/`

### Checkpoint Format
- Training checkpoints: FSDP state dicts (safetensors, HuggingFace-compatible)
- Save: `src/prime_rl/trainer/ckpt.py:save_checkpoint()`
- Load: `src/prime_rl/trainer/ckpt.py:load_checkpoint()`
- Inference reloads via `update_weights` endpoint (automatic FSDP → vLLM conversion)

### Environment Variables
Key env vars:
- `PRIME_*`: Config overrides (`PRIME_MODEL__NAME=Qwen/Qwen3-8B`)
- `OPENAI_API_KEY`: For eval/synthesize with OpenAI models
- `HF_TOKEN`: For gated HuggingFace models/datasets
- `WANDB_API_KEY`: For wandb logging
- `CUDA_VISIBLE_DEVICES`: GPU selection
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`: Required for multi-GPU vLLM (set automatically in most cases)

### Logging
- Uses `loguru` for structured logging
- Logs to stdout + optional file sinks
- Log level: `LOGGING_LEVEL` env var (default: INFO)
- Wandb integration: set `--wandb.enabled` and `--wandb.project`

### Wandb Monitoring Script
Fetch training metrics and samples from wandb runs.

**On GPU nodes (Linux):**
```bash
uv run monitor --list
uv run monitor <run_id> --project prime-rl
```

**On local machine (macOS):**
```bash
# Use uvx to bypass CUDA dependencies
uvx --with wandb --with pandas python src/prime_rl/monitor/wandb_monitor.py --list --project prime-rl-test
uvx --with wandb --with pandas python src/prime_rl/monitor/wandb_monitor.py <run_id> --project prime-rl-test
```

**Common options:**
```bash
# List recent runs
--list [--num-runs 20]

# Fetch specific run
<run_id> [--entity my-team] [--project prime-rl]

# Save plots
--save-dir ./plots

# Filter samples by reward threshold
--reward-threshold 0.3 --max-samples 20

# Show only metrics or samples
--metrics-only | --samples-only
```

**Output includes:**
- Metrics summary: loss/mean, entropy, mismatch_kl, reward/mean, val_reward, throughput, MFU
- Failed samples: messages with reward below threshold (for debugging model errors)
- Plots: loss.png, reward.png, optimizer.png (when `--save-dir` specified)

**Requires:** `WANDB_API_KEY` environment variable

## Development Notes

- **Python 3.12 required** (pinned in `pyproject.toml`)
- **Line length 120** (ruff config)
- **Type hints required**: Explicit types, use `jaxtyping` for tensors
- **Minimal diffs**: Avoid refactoring unrelated code
- **Tests use markers**: `@pytest.mark.gpu` (GPU-required), `@pytest.mark.slow` (long-running)
- **Pre-commit**: Runs ruff format + lint automatically
- **vLLM version**: Currently pinned to `0.10.2`
- **Flash attention**: Optional but recommended (`uv sync --extra fa`)

## Common Debugging Steps

1. **Training hangs/crashes**:
   - Check `ulimit -n` (should be ≥65536 for multi-GPU)
   - Verify FSDP config (dp × tp = world_size)
   - Check `CUDA_VISIBLE_DEVICES` matches `--nproc-per-node`

2. **Inference OOM**:
   - Reduce `--max-batch-size` or `--sampling.max-tokens`
   - Increase `--tensor-parallel-size` (TP)
   - Use smaller model or quantization

3. **Orchestrator connection failures**:
   - Verify inference server is running (`curl http://localhost:8000/health`)
   - Check `--client.base-url` matches inference server address
   - Ensure `update_weights` endpoint is accessible

4. **Eval/Synthesize environment errors**:
   - Verify environment installed (`uv run python -c "import {env_id}"`)
   - Check environment config matches expected format
   - Review `verifiers` docs for environment-specific requirements

5. **Config not applying**:
   - Check precedence order (CLI > TOML > env vars > defaults)
   - Verify TOML syntax (common: missing quotes, wrong nesting)
   - Use `--help` to see final merged config

## Training Examples

End-to-end examples in `examples/`:
- **Reverse Text** (`examples/reverse_text/`): Single-turn SFT+RL on Qwen3-0.6B (1 GPU, minutes)
- **Wordle** (`examples/wordle/`): Multi-turn RL for game playing (2-4 H100s, hours)
- **Alphabet Sort** (`examples/alphabet_sort/`): Multi-turn RL with LoRA (1 H100, ~1 hour)
- **Wiki Search** (`examples/wiki_search/`): Multi-turn with tool calling

Each example includes:
- SFT configs (warmup)
- RL configs (policy optimization)
- Eval configs (benchmarking)
- README with full instructions
