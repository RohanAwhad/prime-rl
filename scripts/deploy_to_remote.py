#!/usr/bin/env python3
"""
Remote deployment script for prime-rl on remote GPU nodes.

This script:
1. Creates a git archive of the current codebase
2. Transfers it to the remote GPU node
3. Extracts the archive
4. Launches the workload in a tmux session

Usage:
    python scripts/deploy_to_remote.py <node-id>
    python scripts/deploy_to_remote.py rh-h100-09
    python scripts/deploy_to_remote.py rh-h200-02
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# Configuration
REMOTE_BASE_DIR = "/workspace/home/lab/rawhad/2_Learnings/prime_rl_learnings"
REMOTE_PROJECT_DIR = f"{REMOTE_BASE_DIR}/prime-rl"
LOCAL_ARCHIVE_PATH = "/tmp/prime-rl-deploy.tar.gz"
TMUX_SESSION_NAME = "prime-rl"


def run_command(cmd: list[str], description: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and print status."""
    print(f"[*] {description}...")
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[✓] {description} completed")
        else:
            print(f"[✗] {description} failed")
            if result.stderr:
                print(f"    Error: {result.stderr}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"[✗] {description} failed with error:")
        print(f"    {e.stderr}")
        if check:
            sys.exit(1)
        return e


def main():
    """Main deployment workflow."""
    parser = argparse.ArgumentParser(description="Deploy prime-rl to a remote GPU node")
    parser.add_argument("node_id", help="Remote node ID (e.g., rh-h100-09, rh-h200-02)")
    args = parser.parse_args()

    remote_host = args.node_id

    print("=" * 60)
    print(f"Prime-RL Remote Deployment to {remote_host}")
    print("=" * 60)

    # Step 1: Create git archive
    run_command(
        ["git", "archive", "--format=tar.gz", "-o", LOCAL_ARCHIVE_PATH, "HEAD"],
        "Creating git archive"
    )

    # Step 2: Cleanup remote directory
    run_command(
        ["ssh", remote_host, f"rm -rf {REMOTE_PROJECT_DIR}"],
        f"Cleaning remote directory: {REMOTE_PROJECT_DIR}",
        check=False  # Don't fail if directory doesn't exist
    )

    # Step 3: Create parent directory
    run_command(
        ["ssh", remote_host, f"mkdir -p {REMOTE_BASE_DIR}"],
        f"Creating parent directory: {REMOTE_BASE_DIR}"
    )

    # Step 4: Transfer archive
    run_command(
        ["scp", LOCAL_ARCHIVE_PATH, f"{remote_host}:{REMOTE_BASE_DIR}/"],
        "Transferring archive to remote"
    )

    # Step 5: Extract archive on remote
    # Note: git archive creates files directly in the target directory (no subdirectory)
    # So we extract to a temp dir, then move
    extract_cmd = (
        f"cd {REMOTE_BASE_DIR} && "
        f"mkdir -p prime-rl && "
        f"tar -xzf prime-rl-deploy.tar.gz -C prime-rl && "
        f"rm prime-rl-deploy.tar.gz"
    )
    run_command(
        ["ssh", remote_host, extract_cmd],
        "Extracting archive on remote"
    )

    # Step 6: Transfer run script to remote
    local_run_script = Path(__file__).parent / "run_workload.sh"
    if not local_run_script.exists():
        print(f"[✗] Error: {local_run_script} not found!")
        sys.exit(1)

    run_command(
        ["scp", str(local_run_script), f"{remote_host}:{REMOTE_PROJECT_DIR}/"],
        "Transferring run_workload.sh to remote"
    )

    # Step 7: Make run script executable
    run_command(
        ["ssh", remote_host, f"chmod +x {REMOTE_PROJECT_DIR}/run_workload.sh"],
        "Making run_workload.sh executable"
    )

    # Step 8: Kill existing tmux session if it exists
    print(f"[*] Checking for existing tmux session '{TMUX_SESSION_NAME}'...")
    result = run_command(
        ["ssh", remote_host, f"tmux has-session -t {TMUX_SESSION_NAME}"],
        f"Checking for existing tmux session",
        check=False
    )

    if result.returncode == 0:
        print(f"[*] Killing existing tmux session '{TMUX_SESSION_NAME}'...")
        run_command(
            ["ssh", remote_host, f"tmux kill-session -t {TMUX_SESSION_NAME}"],
            f"Killing existing tmux session"
        )

    # Step 9: Launch tmux session with run script
    # Pass WANDB_API_KEY from local environment to remote
    wandb_api_key = os.environ.get("WANDB_API_KEY", "")
    if wandb_api_key:
        env_prefix = f"WANDB_API_KEY='{wandb_api_key}' "
        print(f"[✓] Passing WANDB_API_KEY to remote session")
    else:
        env_prefix = ""
        print(f"[!] WANDB_API_KEY not set locally - wandb may use remote's cached credentials")

    launch_cmd = (
        f"cd {REMOTE_PROJECT_DIR} && "
        f"tmux new-session -d -s {TMUX_SESSION_NAME} '{env_prefix}./run_workload.sh'"
    )
    run_command(
        ["ssh", remote_host, launch_cmd],
        f"Launching tmux session '{TMUX_SESSION_NAME}'"
    )

    # Cleanup local archive
    Path(LOCAL_ARCHIVE_PATH).unlink(missing_ok=True)
    print(f"[✓] Cleaned up local archive")

    print("=" * 60)
    print("Deployment complete!")
    print(f"")
    print(f"To attach to the tmux session:")
    print(f"  ssh {remote_host}")
    print(f"  tmux attach -t {TMUX_SESSION_NAME}")
    print(f"")
    print(f"To view session status:")
    print(f"  ssh {remote_host} 'tmux list-sessions'")
    print("=" * 60)


if __name__ == "__main__":
    main()
