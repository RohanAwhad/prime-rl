"""
Fetch training metrics and samples from wandb for analysis.

Usage (on GPU nodes / Linux):
    uv run monitor --list
    uv run monitor <run_id> --project prime-rl

Usage (on local macOS - bypasses CUDA deps):
    uvx --with wandb --with pandas python src/prime_rl/monitor/wandb_monitor.py --list --project prime-rl-test
    uvx --with wandb --with pandas python src/prime_rl/monitor/wandb_monitor.py <run_id> --project prime-rl-test

Options:
    --list              List recent runs
    --num-runs N        Number of runs to list (default: 10)
    --entity ENTITY     Wandb entity (default: from wandb config)
    --project PROJECT   Wandb project (default: prime-rl)
    --save-dir DIR      Save plots to directory
    --reward-threshold  Show samples below threshold (default: 0.5)
    --max-samples N     Max samples to show (default: 10)
    --metrics-only      Skip samples table
    --samples-only      Skip metrics
"""

import argparse
import os
import sys
from pathlib import Path

import wandb

# Lazy import pandas only when needed
pd = None


def get_pandas():
    global pd
    if pd is None:
        import pandas
        pd = pandas
    return pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch training metrics and samples from wandb")
    parser.add_argument("run_id", nargs="?", default=None, help="The wandb run ID to fetch")
    parser.add_argument("--list", "-l", action="store_true", help="List recent runs instead of fetching a specific run")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of runs to list (default: 10)")
    parser.add_argument("--entity", default=None, help="Wandb entity (team/user). If not set, uses default from wandb config.")
    parser.add_argument("--project", default="prime-rl", help="Wandb project name (default: prime-rl)")
    parser.add_argument("--save-dir", type=Path, default=None, help="Directory to save plots. If not set, displays summary only.")
    parser.add_argument("--reward-threshold", type=float, default=0.5, help="Show samples with reward below this threshold (default: 0.5)")
    parser.add_argument("--samples-only", action="store_true", help="Only show samples, skip metrics")
    parser.add_argument("--metrics-only", action="store_true", help="Only show metrics, skip samples")
    parser.add_argument("--max-samples", type=int, default=10, help="Maximum number of samples to display (default: 10)")
    parser.add_argument("--step-range", type=str, default=None, help="Step range to filter (e.g., '0:1000' or '500:')")
    return parser.parse_args()


def list_runs(entity: str | None, project: str, num_runs: int) -> None:
    """List recent runs from wandb."""
    api = wandb.Api()

    if entity is None:
        entity = api.default_entity

    print(f"Entity: {entity}")
    print(f"Project: {project}")
    print()

    try:
        runs = api.runs(f"{entity}/{project}", order="-created_at", per_page=num_runs)
    except wandb.errors.CommError as e:
        print(f"Error: Could not access project '{entity}/{project}'")
        print(f"  - Check entity and project are correct")
        print(f"  - Ensure WANDB_API_KEY has access")
        sys.exit(1)

    print(f"Recent runs (showing {num_runs}):")
    print("-" * 100)
    print(f"{'RUN ID':<12} | {'NAME':<30} | {'STATE':<10} | {'CREATED':<20}")
    print("-" * 100)

    for run in runs:
        name = run.name[:28] + ".." if len(run.name) > 30 else run.name
        created = str(run.created_at)[:19] if run.created_at else "N/A"
        print(f"{run.id:<12} | {name:<30} | {run.state:<10} | {created:<20}")

    print("-" * 100)
    print(f"\nTo fetch a run: uv run monitor <run_id>")


def get_run(entity: str | None, project: str, run_id: str) -> wandb.Api:
    """Get a wandb run by ID."""
    api = wandb.Api()
    if entity:
        run_path = f"{entity}/{project}/{run_id}"
    else:
        run_path = f"{project}/{run_id}"
    try:
        return api.run(run_path)
    except wandb.errors.CommError as e:
        # Try with default entity
        if entity is None:
            try:
                default_entity = api.default_entity
                run_path = f"{default_entity}/{project}/{run_id}"
                return api.run(run_path)
            except Exception:
                pass
        print(f"Error: Could not find run '{run_path}'")
        print(f"  - Check run ID is correct")
        print(f"  - Try specifying --entity explicitly")
        sys.exit(1)


def parse_step_range(step_range: str | None) -> tuple[int | None, int | None]:
    """Parse step range string like '0:1000' or '500:' into (start, end)."""
    if step_range is None:
        return None, None
    parts = step_range.split(":")
    start = int(parts[0]) if parts[0] else None
    end = int(parts[1]) if len(parts) > 1 and parts[1] else None
    return start, end


def fetch_metrics(run: wandb.apis.public.Run, step_start: int | None, step_end: int | None):
    """Fetch training metrics from a run."""
    pd = get_pandas()

    # Keys we want to extract (fetch all history, filter later)
    desired_keys = [
        "step", "_step",
        # Loss metrics
        "loss/mean", "loss/std", "entropy/mean", "mismatch_kl/mean", "max_vio/mean",
        # Reward metrics
        "reward/mean", "val_reward/mean",
        # Optimizer
        "optim/lr", "optim/grad_norm",
        # Performance
        "perf/throughput", "perf/mfu",
    ]

    # Fetch history without key filtering (keys param filters to rows with ALL keys)
    history = run.scan_history()
    rows = list(history)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Keep only columns we care about (plus any that exist)
    available_cols = [k for k in desired_keys if k in df.columns]
    if available_cols:
        df = df[available_cols].copy()

    # Use _step if step not available
    if "step" not in df.columns and "_step" in df.columns:
        df["step"] = df["_step"]

    # Filter by step range
    if "step" in df.columns:
        if step_start is not None:
            df = df[df["step"] >= step_start]
        if step_end is not None:
            df = df[df["step"] <= step_end]

    return df


def fetch_samples_table(run: wandb.apis.public.Run, table_name: str = "final-samples"):
    """Fetch samples table from a run."""
    pd = get_pandas()

    # Try to get the artifact
    try:
        # Check run summary for table reference
        if table_name in run.summary:
            table_ref = run.summary[table_name]
            if hasattr(table_ref, "get"):
                table = table_ref.get("table")
                if table is not None:
                    return pd.DataFrame(data=table.data, columns=table.columns)

        # Try fetching from logged tables
        for artifact in run.logged_artifacts():
            if table_name in artifact.name:
                table = artifact.get(table_name)
                if table is not None:
                    return pd.DataFrame(data=table.data, columns=table.columns)

        # Try direct table access from history
        history = run.scan_history(keys=[table_name])
        for row in history:
            if table_name in row and row[table_name] is not None:
                table_ref = row[table_name]
                if hasattr(table_ref, "get"):
                    return pd.DataFrame(data=table_ref.data, columns=table_ref.columns)

        return None
    except Exception as e:
        print(f"Warning: Could not fetch table '{table_name}': {e}")
        return None


def print_metrics_summary(df) -> None:
    """Print a summary of training metrics."""
    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)

    if df.empty:
        print("No metrics found.")
        return

    # Get step range
    if "step" in df.columns:
        print(f"Steps: {int(df['step'].min())} -> {int(df['step'].max())} ({len(df)} logged points)")

    print()

    # Loss metrics
    if "loss/mean" in df.columns:
        loss = df["loss/mean"].dropna()
        if not loss.empty:
            print("LOSS:")
            print(f"  loss/mean:     {loss.iloc[0]:.4f} -> {loss.iloc[-1]:.4f} (delta: {loss.iloc[-1] - loss.iloc[0]:+.4f})")

    if "entropy/mean" in df.columns:
        entropy = df["entropy/mean"].dropna()
        if not entropy.empty:
            print(f"  entropy/mean:  {entropy.iloc[0]:.4f} -> {entropy.iloc[-1]:.4f}")

    if "mismatch_kl/mean" in df.columns:
        kl = df["mismatch_kl/mean"].dropna()
        if not kl.empty:
            print(f"  mismatch_kl:   {kl.iloc[0]:.4f} -> {kl.iloc[-1]:.4f}")

    print()

    # Reward metrics
    if "reward/mean" in df.columns:
        reward = df["reward/mean"].dropna()
        if not reward.empty:
            print("REWARD:")
            print(f"  reward/mean:     {reward.iloc[0]:.4f} -> {reward.iloc[-1]:.4f} (delta: {reward.iloc[-1] - reward.iloc[0]:+.4f})")
            print(f"  reward/max:      {reward.max():.4f} (at step {int(df.loc[reward.idxmax(), 'step']) if 'step' in df.columns else 'N/A'})")

    if "val_reward/mean" in df.columns:
        val_reward = df["val_reward/mean"].dropna()
        if not val_reward.empty:
            print(f"  val_reward/mean: {val_reward.iloc[0]:.4f} -> {val_reward.iloc[-1]:.4f}")

    print()

    # Performance
    if "perf/throughput" in df.columns:
        throughput = df["perf/throughput"].dropna()
        if not throughput.empty:
            print("PERFORMANCE:")
            print(f"  throughput: {throughput.mean():.1f} tokens/s (avg), {throughput.max():.1f} (max)")

    if "perf/mfu" in df.columns:
        mfu = df["perf/mfu"].dropna()
        if not mfu.empty:
            print(f"  MFU:        {mfu.mean():.1f}% (avg)")

    print()


def print_samples(df, reward_threshold: float, max_samples: int) -> None:
    """Print samples filtered by reward threshold."""
    print("\n" + "=" * 60)
    print(f"SAMPLES (reward < {reward_threshold})")
    print("=" * 60)

    if df is None or df.empty:
        print("No samples table found.")
        print("  - Ensure wandb.log_extras.samples=true in config")
        print("  - Table is logged at end of training as 'final-samples'")
        return

    # Filter by reward
    if "reward" in df.columns:
        failed = df[df["reward"] < reward_threshold].copy()
    else:
        print("No 'reward' column in samples table.")
        return

    if failed.empty:
        print(f"No samples with reward < {reward_threshold}")
        return

    # Sort by reward (worst first)
    failed = failed.sort_values("reward", ascending=True)

    print(f"Found {len(failed)} samples with reward < {reward_threshold}")
    print(f"Showing first {min(max_samples, len(failed))}:\n")

    for i, (_, row) in enumerate(failed.head(max_samples).iterrows()):
        print("-" * 60)
        step = row.get("step", "N/A")
        example_id = row.get("example_id", "N/A")
        reward = row.get("reward", "N/A")

        print(f"[{i+1}] Step: {step} | Example: {example_id} | Reward: {reward}")
        print()

        if "messages" in row:
            messages = row["messages"]
            # Truncate long messages
            if len(str(messages)) > 2000:
                messages = str(messages)[:2000] + "... [truncated]"
            print(messages)

        print()


def save_plots(df, save_dir: Path) -> None:
    """Save metric plots to directory."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping plots")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    if df.empty or "step" not in df.columns:
        print("No data to plot.")
        return

    # Loss plot
    loss_cols = ["loss/mean", "entropy/mean", "mismatch_kl/mean"]
    available_loss = [c for c in loss_cols if c in df.columns and df[c].notna().any()]

    if available_loss:
        fig, ax = plt.subplots(figsize=(10, 6))
        for col in available_loss:
            data = df[["step", col]].dropna()
            ax.plot(data["step"], data[col], label=col)
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.set_title("Training Loss Metrics")
        ax.legend()
        ax.grid(True, alpha=0.3)

        loss_path = save_dir / "loss.png"
        fig.savefig(loss_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {loss_path}")

    # Reward plot
    reward_cols = ["reward/mean", "val_reward/mean"]
    available_reward = [c for c in reward_cols if c in df.columns and df[c].notna().any()]

    if available_reward:
        fig, ax = plt.subplots(figsize=(10, 6))
        for col in available_reward:
            data = df[["step", col]].dropna()
            ax.plot(data["step"], data[col], label=col)
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        ax.set_title("Training Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)

        reward_path = save_dir / "reward.png"
        fig.savefig(reward_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {reward_path}")

    # Grad norm + LR plot
    if "optim/grad_norm" in df.columns and df["optim/grad_norm"].notna().any():
        fig, ax1 = plt.subplots(figsize=(10, 6))

        data = df[["step", "optim/grad_norm"]].dropna()
        ax1.plot(data["step"], data["optim/grad_norm"], "b-", label="grad_norm")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Grad Norm", color="b")
        ax1.tick_params(axis="y", labelcolor="b")

        if "optim/lr" in df.columns and df["optim/lr"].notna().any():
            ax2 = ax1.twinx()
            lr_data = df[["step", "optim/lr"]].dropna()
            ax2.plot(lr_data["step"], lr_data["optim/lr"], "r-", label="lr")
            ax2.set_ylabel("Learning Rate", color="r")
            ax2.tick_params(axis="y", labelcolor="r")

        ax1.set_title("Optimizer Metrics")
        ax1.grid(True, alpha=0.3)

        optim_path = save_dir / "optimizer.png"
        fig.savefig(optim_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {optim_path}")


def main():
    # Check for API key
    if not os.environ.get("WANDB_API_KEY"):
        print("Warning: WANDB_API_KEY not set in environment")

    args = parse_args()

    # Handle --list mode
    if args.list:
        list_runs(args.entity, args.project, args.num_runs)
        return

    # Require run_id if not listing
    if args.run_id is None:
        print("Error: run_id is required (or use --list to see recent runs)")
        sys.exit(1)

    print(f"Fetching run: {args.run_id}")
    print(f"Project: {args.project}")
    if args.entity:
        print(f"Entity: {args.entity}")

    # Get run
    run = get_run(args.entity, args.project, args.run_id)
    print(f"Run name: {run.name}")
    print(f"Run state: {run.state}")

    step_start, step_end = parse_step_range(args.step_range)

    # Fetch and display metrics
    if not args.samples_only:
        print("\nFetching metrics...")
        metrics_df = fetch_metrics(run, step_start, step_end)
        print_metrics_summary(metrics_df)

        if args.save_dir:
            print(f"\nSaving plots to {args.save_dir}...")
            save_plots(metrics_df, args.save_dir)

    # Fetch and display samples
    if not args.metrics_only:
        print("\nFetching samples table...")
        samples_df = fetch_samples_table(run, "final-samples")
        if samples_df is None:
            # Try incremental samples table
            samples_df = fetch_samples_table(run, "samples")
        print_samples(samples_df, args.reward_threshold, args.max_samples)


if __name__ == "__main__":
    main()
