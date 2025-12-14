#!/bin/bash
# Cascade LLM Training: M1 (frozen) -> M2 (trainable)
#
# Architecture:
#   Query -> M1 (port 8001, frozen) -> draft -> M2 (port 8000, trainable) -> refined answer -> reward
#
# This script:
#   1. Starts M1 inference server (frozen drafter) on port 8001
#   2. Waits for M1 to be healthy
#   3. Runs RL training (spawns M2 + trainer + orchestrator)
#   4. Handles cleanup on exit/interrupt

set -e

# Configuration
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

M1_PORT="${M1_PORT:-8003}"
M1_HEALTH_TIMEOUT="${M1_HEALTH_TIMEOUT:-120}"  # seconds

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "Cascade LLM Training"
echo "========================================"
echo "Architecture: Query -> M1 (frozen) -> M2 (trainable) -> Answer"
echo "Project directory: $PROJECT_DIR"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo -e "${YELLOW}[*] Stopping all processes...${NC}"
    if [ -n "$M1_PID" ]; then
        kill $M1_PID 2>/dev/null || true
        wait $M1_PID 2>/dev/null || true
        echo -e "${GREEN}[✓] M1 server stopped${NC}"
    fi
    echo -e "${GREEN}[✓] Cleanup complete${NC}"
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Create logs directory
mkdir -p logs

# Start M1 (frozen drafter) - runs in background on GPU 7 (separate from M2 on GPU 0, trainer on GPU 1)
M1_GPU="${M1_GPU:-7}"
echo -e "${YELLOW}[*] Starting M1 inference server (frozen) on port $M1_PORT, GPU $M1_GPU...${NC}"
M1_LOG="logs/m1_$(date +%Y%m%d_%H%M%S).log"
CUDA_VISIBLE_DEVICES=$M1_GPU uv run inference @ configs/cascade/m1_infer.toml > "$M1_LOG" 2>&1 &
M1_PID=$!
echo "    M1 PID: $M1_PID"
echo "    M1 Log: $M1_LOG"

# Wait for M1 to be healthy
echo -e "${YELLOW}[*] Waiting for M1 server to be healthy...${NC}"
HEALTH_URL="http://localhost:$M1_PORT/health"
for i in $(seq 1 $M1_HEALTH_TIMEOUT); do
    if curl -s "$HEALTH_URL" > /dev/null 2>&1; then
        echo -e "${GREEN}[✓] M1 server ready (took ${i}s)${NC}"
        break
    fi
    if ! kill -0 $M1_PID 2>/dev/null; then
        echo -e "${RED}[✗] M1 server process died. Check log: $M1_LOG${NC}"
        tail -20 "$M1_LOG"
        exit 1
    fi
    if [ $i -eq $M1_HEALTH_TIMEOUT ]; then
        echo -e "${RED}[✗] M1 server failed to start within ${M1_HEALTH_TIMEOUT}s${NC}"
        echo "Last 20 lines of log:"
        tail -20 "$M1_LOG"
        exit 1
    fi
    sleep 1
done

echo ""
echo "========================================"
echo "Running Cascade RL Training"
echo "========================================"
echo "M1 (frozen): http://localhost:$M1_PORT/v1 (GPU $M1_GPU)"
echo "M2 (trainable): Started by uv run rl"
echo ""
echo "Press Ctrl+C to stop all components."
echo ""

# Run RL training (spawns M2 + trainer + orchestrator)
RL_LOG="logs/cascade_rl_$(date +%Y%m%d_%H%M%S).log"
echo "RL Log: $RL_LOG"
echo ""

uv run rl @ configs/cascade/rl.toml 2>&1 | tee "$RL_LOG"

echo ""
echo "========================================"
echo "Training complete"
echo "========================================"
echo "M1 Log: $M1_LOG"
echo "RL Log: $RL_LOG"
