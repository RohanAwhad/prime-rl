#!/bin/bash
# Simple test script for running PRIME-RL end-to-end
# This script can run in two modes:
#   1. All-in-one mode (default): Single process spawns all components
#   2. Multi-pane mode: Separate tmux panes for inference, orchestrator, trainer

# Configuration
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

# WandB configuration (optional)
export WANDB_API_KEY="${WANDB_API_KEY:-}"

echo "========================================"
echo "PRIME-RL Test Workload"
echo "========================================"
echo "Project directory: $PROJECT_DIR"
echo ""

# Setup environment
echo "[*] Upgrading uv to latest version..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

echo "[*] Setting up uv environment..."
uv sync
echo "[✓] Environment ready"
echo ""

echo "========================================"
echo "Running all-in-one RL training"
echo "========================================"
echo "Config: configs/multi_reverse_text/rl.toml"
echo ""
echo "This will spawn all three components in a single process."
echo "Press Ctrl+C to stop all components."
echo ""

# Create logs directory
mkdir -p logs

# Run all-in-one with logging (spawns inference, orchestrator, trainer)
LOG_FILE="logs/run_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE"
echo ""

uv run rl @ configs/multi_reverse_text/rl.toml 2>&1 | tee "$LOG_FILE"

# Keep session alive after process exits
echo ""
echo "========================================"
echo "Process exited. Session kept alive."
echo "Log file: $LOG_FILE"
echo "========================================"
exec bash

