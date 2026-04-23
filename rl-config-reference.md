# RLConfig Reference

Root Pydantic model: `RLConfig` in `src/prime_rl/configs/rl.py:239`

## Top-level parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_steps` | `int \| None` | `None` | Maximum number of training steps. Propagated to trainer and orchestrator. |
| `seq_len` | `int \| None` | `None` | Shared sequence length. Propagates to `trainer.model.seq_len` and `orchestrator.seq_len`. |
| `max_model_len` | `int \| None` | `None` | Maximum model length. |
| `max_async_level` | `int \| None` | `None` | Async level for trainer and orchestrator. |
| `output_dir` | `Path` | `outputs` | Directory to store experiment outputs. |
| `clean_output_dir` | `bool` | `false` | Delete output directory before starting (required to overwrite previous checkpoints when not resuming). |
| `dry_run` | `bool` | `false` | Validate and dump resolved configs, then exit. |
| `bench` | `bool` | `false` | Benchmark mode. Sets trainer and orchestrator to bench mode, suffixes W&B project with `-bench`. |

## `[deployment]`

Two deployment types, discriminated by `type`:

### `SingleNodeDeploymentConfig` (default)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | `Literal["single_node"]` | `"single_node"` | Deployment type. |
| `num_train_gpus` | `int` | `1` | Number of training GPUs. |
| `num_infer_gpus` | `int` | `1` | Number of inference GPUs. |
| `num_teacher_gpus` | `int \| None` | `None` | Number of teacher inference GPUs. |
| `gpus_per_node` | `int` | `8` | GPUs per node. Total GPUs must not exceed this. |

### `MultiNodeDeploymentConfig`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | `Literal["multi_node"]` | `"multi_node"` | Deployment type. |
| `num_train_nodes` | `int` | required | Number of training nodes. |
| `num_infer_nodes` | `int` | required | Number of inference nodes per replica. Set to 0 to skip inference (requires fake data). |
| `num_infer_replicas` | `int` | `1` | Number of independent inference replicas. |
| `num_teacher_nodes` | `int \| None` | `None` | Number of teacher inference nodes (not yet supported). |
| `nodes_per_fsdp_group` | `int \| None` | `None` | Nodes per FSDP island. Auto-sets `trainer.dp_replicate`. |
| `gpus_per_node` | `int` | `8` | GPUs per node. |

## `[model]`

Shared model config propagated to trainer, orchestrator, and inference.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"Qwen/Qwen3-0.6B"` | HuggingFace model name. |
| `vlm` | `VLMConfig \| None` | `None` | Vision-language model configuration. |

## `[wandb]`

Shared W&B config propagated to trainer and orchestrator.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project` | `str \| None` | `"prime-rl"` | W&B project name. |
| `name` | `str \| None` | `None` | W&B run name. |
| `offline` | `bool \| None` | `false` | Run W&B in offline mode. |
| `shared` | `bool` | `true` | Log trainer and orchestrator metrics to a single W&B run. Incompatible with offline mode. |

## `[orchestrator]`

Defined in `src/prime_rl/configs/orchestrator.py` (`OrchestratorConfig`).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | `int \| None` | - | Batch size for orchestration. |
| `rollouts_per_example` | `int` | - | Number of rollouts per example. |
| `seq_len` | `int` | - | Sequence length for orchestrator. |
| `max_steps` | `int` | - | Maximum steps (set via shared `max_steps`). |
| `bench` | `bool` | `false` | Benchmark mode. |

### `[orchestrator.train.sampling]`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_completion_tokens` | `int` | - | Maximum tokens in generated completions. |

### `[[orchestrator.train.env]]`

List of environment configurations.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `id` | `str` | required | Environment identifier (e.g. `rohans_reverse_text@local`). |

## `[trainer]`

Defined in `src/prime_rl/configs/trainer.py` (`TrainerConfig`).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_steps` | `int` | - | Maximum training steps (set via shared `max_steps`). |
| `max_async_level` | `int` | - | Async level. |
| `enable_router_replay` | `bool` | - | Enable router replay for MoE models. |

### `[trainer.optim]`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | `float` | - | Learning rate. |

### `[trainer.model]`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | - | Model name (set via shared `model.name`). |
| `seq_len` | `int` | - | Sequence length for training. |
| `impl` | `str` | - | Model implementation (e.g. `"custom"`). |
| `cp` | `int` | - | Context parallelism size. |
| `dp_replicate` | `int` | - | Data parallel replication factor. |
| `trust_remote_code` | `bool` | - | Trust remote code when loading model. |

### `[trainer.model.ac]`

Activation checkpointing.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `freq` | `int` | - | Activation checkpointing frequency. |

### `[trainer.model.lora]`

LoRA (Low-Rank Adaptation) configuration (optional). Defined as `LoRAConfig` in `src/prime_rl/configs/trainer.py:111`. When set, the `auto_setup_lora` validator in `rl.py:691` propagates rank/alpha to `orchestrator.model.lora` and enables LoRA on the inference server.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rank` | `int` | `16` | Rank of the low-rank decomposition matrices. |
| `alpha` | `float` | `32.0` | LoRA scaling parameter. |
| `dropout` | `float` | `0.0` | LoRA dropout rate (0-1). |
| `target_modules` | `list[str]` | `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "experts"]` | Module names or regex patterns to apply LoRA to. |
| `modules_to_save` | `list[str]` | `[]` | Module names or regex patterns for modules to keep fully trainable (not frozen). |

### `[trainer.loss]`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | `str` | `"default"` | Loss type (`"default"` or `"sft"`). |
| `teacher_tau` | `float` | - | Teacher distillation temperature. |

## `[ckpt]`

Shared checkpoint config propagated to both trainer and orchestrator.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | `Path \| None` | `None` | Override directory for checkpoints and weight snapshots. |
| `interval` | `int \| None` | `None` | Checkpoint save interval (in steps). |
| `resume_step` | `int \| None` | `None` | Step to resume from. |
| `keep_last` | `int \| None` | `None` | Keep at most N recent checkpoints. |
| `keep_interval` | `int \| None` | `None` | Keep checkpoints at every N steps permanently. |

## `[inference]`

Defined in `src/prime_rl/configs/inference.py` (`InferenceConfig`). Optional; if `None`, the RL entrypoint will not start an inference server.

### `[inference.parallel]`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dp` | `int` | `1` | Data parallelism degree. |
| `tp` | `int` | `1` | Tensor parallelism degree. |

### `[inference.model]`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | - | Model name (set via shared `model.name`). |
| `chat_template` | `str \| None` | `None` | Custom chat template for vLLM. |

### `[inference.server]`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | `str \| None` | - | Server host. |
| `port` | `int` | `8000` | Server port. |

## `[weight_broadcast]`

Shared weight broadcast config for syncing weights between trainer and inference.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | `Literal["nccl", "filesystem"]` | `"filesystem"` | Broadcast backend. |
| `port` | `int` | `29501` | Port for NCCL broadcast. |
| `timeout` | `int` | `1200` | Timeout in seconds for NCCL broadcast. |
| `quantize_in_weight_transfer` | `bool` | `false` | Use FP8 quantized NCCL transfer. Requires `type = "nccl"` and `trainer.model.impl = "custom"`. |

## `[log]`

Shared logging config.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `level` | `str \| None` | `None` | Log level. Falls back to `PRIME_LOG_LEVEL` env var, then `"info"`. |
| `json_logging` | `bool` | `false` | Emit JSON logs for log aggregation. |

## `[tokenizer]`

Shared tokenizer config propagated to trainer, orchestrator, and inference.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str \| None` | `None` | Tokenizer name. Defaults to `model.name`. |
| `chat_template` | `str \| None` | `None` | Custom chat template. Also propagated to `inference.model.chat_template`. |
| `trust_remote_code` | `bool \| None` | `None` | Trust remote code. Defaults to `model.trust_remote_code`. |

## `[slurm]`

SLURM job submission config. If `None`, runs locally.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `template_path` | `Path \| None` | `None` | Custom SLURM template. Auto-detected based on deployment type if unset. |

## Config propagation

Top-level "shared" configs are propagated to subcomponents via model validators:

- `model.name` -> `trainer.model.name`, `orchestrator.model.name`, `inference.model.name`
- `wandb` -> `trainer.wandb`, `orchestrator.wandb`
- `ckpt` -> `trainer.ckpt`, `orchestrator.ckpt`
- `max_steps` -> `trainer.max_steps`, `orchestrator.max_steps`
- `seq_len` -> `trainer.model.seq_len`, `orchestrator.seq_len`
- `max_async_level` -> `trainer.max_async_level`, `orchestrator.max_async_level`
- `weight_broadcast` -> `trainer.weight_broadcast`, `orchestrator.weight_broadcast`, `inference.weight_broadcast`
- `tokenizer` -> `trainer.tokenizer`, `orchestrator.tokenizer`, `inference.model.chat_template`
- `output_dir` -> `trainer.output_dir`, `orchestrator.output_dir`
- `log` -> `trainer.log`, `orchestrator.log`
- `bench` -> `trainer.bench`, `orchestrator.bench`, `trainer.data.fake`

Per-component values explicitly set in the config always take precedence over shared values.
