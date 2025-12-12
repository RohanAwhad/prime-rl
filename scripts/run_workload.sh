#!/bin/bash
# Example workload script for running prime-rl inference and training on remote GPU node
# NOTE: This script runs INSIDE the tmux session created by deploy_to_remote.py
#
# User modification points:
# - Change config files (debug.toml vs simple_math.toml vs others)
# - Adjust GPU allocation (CUDA_VISIBLE_DEVICES)
# - Modify parallelism (--nproc_per_node for training, DP/TP flags for inference)
# - Comment out panes to run only inference or only training
# - Add additional panes for monitoring, logging, etc.

# Project directory
PROJECT_DIR="/workspace/home/lab/rawhad/2_Learnings/prime_rl_learnings/prime-rl"
cd "$PROJECT_DIR"

# WandB configuration - reads from local WANDB_API_KEY env var
# Set it locally: export WANDB_API_KEY="your-key" (get from https://wandb.ai/authorize)
export WANDB_API_KEY="${WANDB_API_KEY:-}"

echo "========================================"
echo "Prime-RL Workload Setup"
echo "========================================"
echo "Project directory: $PROJECT_DIR"
echo ""

# Setup environment first (before splitting panes)
echo "[*] Setting up uv environment..."
uv sync && uv sync --extra fa
echo "[✓] Environment setup complete"
echo ""

echo "[*] Creating training pane (right) and launching training..."
# Create right pane with training command directly
# This splits horizontally and runs training in the new pane
tmux split-window -h -c "$PROJECT_DIR" \
    "export CUDA_VISIBLE_DEVICES=6,7; ulimit -n 65536; echo 'Starting training worker...'; uv run torchrun --nproc_per_node=2 src/zeroband/train.py @ configs/training/simple_math.toml; exec bash"

# Select left pane (pane 0) for inference
tmux select-pane -t 0

echo ""
echo "========================================"
echo "Training launched in right pane!"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Inference: 6 GPUs (0-5) | configs/inference/simple_math.toml"
echo "  Training:  2 GPUs (6-7) | configs/training/simple_math.toml"
echo ""
echo "Now starting inference in this pane..."
echo ""

# Run inference in this pane (pane 0 / left)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export VLLM_WORKER_MULTIPROC_METHOD=spawn
echo "Starting inference worker..."
exec uv run python src/zeroband/infer.py @ configs/inference/simple_math.toml
