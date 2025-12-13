#!/bin/bash
# Simple test script for running PRIME-RL end-to-end
# This script can run in two modes:
#   1. All-in-one mode (default): Single process spawns all components
#   2. Multi-pane mode: Separate tmux panes for inference, orchestrator, trainer

# Configuration
MODE="${1:-all-in-one}"  # "all-in-one" or "multi-pane"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

# WandB configuration (optional)
export WANDB_API_KEY="${WANDB_API_KEY:-}"

echo "========================================"
echo "PRIME-RL Test Workload"
echo "========================================"
echo "Mode: $MODE"
echo "Project directory: $PROJECT_DIR"
echo ""

# Setup environment
echo "[*] Setting up uv environment..."
uv sync
echo "[✓] Environment ready"
echo ""

if [ "$MODE" = "all-in-one" ]; then
    echo "========================================"
    echo "Running all-in-one RL training"
    echo "========================================"
    echo "Config: configs/debug/rl/train.toml (trainer)"
    echo "        configs/debug/orch.toml (orchestrator)"
    echo "        configs/debug/infer.toml (inference)"
    echo ""
    echo "This will spawn all three components in a single process."
    echo "Press Ctrl+C to stop all components."
    echo ""

    # Run all-in-one (spawns inference, orchestrator, trainer)
    exec uv run rl \
        --trainer @ configs/debug/rl/train.toml \
        --orchestrator @ configs/debug/orch.toml \
        --inference @ configs/debug/infer.toml

elif [ "$MODE" = "multi-pane" ]; then
    echo "========================================"
    echo "Setting up multi-pane tmux session"
    echo "========================================"
    echo "Creating 3 panes: inference | orchestrator | trainer"
    echo ""

    # Check if running in tmux
    if [ -z "$TMUX" ]; then
        echo "[!] Not in tmux session. Starting new session..."
        tmux new-session -s prime-rl "$0 multi-pane-internal"
        exit 0
    fi

    # Create panes
    echo "[*] Creating inference pane (left)..."
    tmux split-window -h -c "$PROJECT_DIR"
    tmux select-pane -t 0

    echo "[*] Creating orchestrator pane (top-right)..."
    tmux select-pane -t 1
    tmux split-window -v -c "$PROJECT_DIR"

    # Launch components in each pane
    echo "[*] Launching inference server..."
    tmux send-keys -t 0 "echo 'Starting inference server...'; uv run inference @ configs/debug/infer.toml" C-m

    echo "[*] Launching orchestrator..."
    tmux send-keys -t 2 "sleep 5; echo 'Starting orchestrator...'; uv run orchestrator @ configs/debug/orch.toml" C-m

    echo "[*] Launching trainer..."
    tmux send-keys -t 1 "sleep 10; echo 'Starting trainer...'; uv run trainer @ configs/debug/rl/train.toml" C-m

    echo ""
    echo "========================================"
    echo "All components launched!"
    echo "========================================"
    echo "Layout:"
    echo "  Pane 0 (left):       Inference server"
    echo "  Pane 1 (bottom-right): Trainer"
    echo "  Pane 2 (top-right):   Orchestrator"
    echo ""
    echo "Use Ctrl+B then arrow keys to navigate panes"
    echo "Use Ctrl+C in each pane to stop components"
    echo ""

    # Select orchestrator pane to monitor
    tmux select-pane -t 2

else
    echo "[!] Invalid mode: $MODE"
    echo "Usage: $0 [all-in-one|multi-pane]"
    exit 1
fi
