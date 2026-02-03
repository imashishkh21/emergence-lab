#!/usr/bin/env python3
"""CLI for computing all emergence metrics from a checkpoint.

Usage:
    python scripts/compute_emergence_metrics.py \\
        --checkpoint checkpoints/params.pkl \\
        --output results/emergence_metrics.json

    # Dry run (show what would be computed)
    python scripts/compute_emergence_metrics.py --dry-run \\
        --checkpoint checkpoints/params.pkl --output /tmp/test.json

    # Skip surrogate tests (faster)
    python scripts/compute_emergence_metrics.py \\
        --checkpoint checkpoints/params.pkl --output results.json --skip-surrogates

    # Custom window size for windowed analysis
    python scripts/compute_emergence_metrics.py \\
        --checkpoint checkpoints/params.pkl --output results.json \\
        --window-size 500000 --window-overlap 0.25

Reference: US-015 in Phase 5 PRD.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load checkpoint from pickle file.

    Args:
        checkpoint_path: Path to checkpoint pickle file.

    Returns:
        Dict with 'params', optionally 'agent_params' and 'config'.
    """
    with open(checkpoint_path, "rb") as f:
        return pickle.load(f)


def generate_synthetic_trajectory(
    n_timesteps: int = 1000,
    n_agents: int = 8,
    grid_size: int = 20,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate synthetic trajectory for testing.

    Creates realistic-looking trajectory data for testing the emergence
    metrics pipeline when no real checkpoint is available.

    Args:
        n_timesteps: Number of timesteps.
        n_agents: Number of agents.
        grid_size: Environment grid size.
        seed: Random seed.

    Returns:
        Trajectory dict with all required keys.
    """
    rng = np.random.default_rng(seed)

    # Actions: 6 possible actions (stay, up, down, left, right, reproduce)
    actions = rng.integers(0, 6, size=(n_timesteps, n_agents))

    # Positions: agents move on grid
    positions = np.zeros((n_timesteps, n_agents, 2), dtype=np.int32)
    positions[0] = rng.integers(0, grid_size, size=(n_agents, 2))
    for t in range(1, n_timesteps):
        # Random walk with some correlation
        delta = rng.integers(-1, 2, size=(n_agents, 2))
        positions[t] = np.clip(positions[t - 1] + delta, 0, grid_size - 1)

    # Rewards: sparse food collection
    rewards = np.zeros((n_timesteps, n_agents), dtype=np.float32)
    for t in range(n_timesteps):
        # Some agents collect food each step
        n_collectors = rng.integers(0, n_agents // 2)
        collectors = rng.choice(n_agents, size=n_collectors, replace=False)
        rewards[t, collectors] = 1.0

    # Alive mask: all agents alive
    alive_mask = np.ones((n_timesteps, n_agents), dtype=bool)

    # Energy: starts high, drains slowly
    energy = np.zeros((n_timesteps, n_agents), dtype=np.float32)
    energy[0] = 100.0
    for t in range(1, n_timesteps):
        energy[t] = np.clip(energy[t - 1] - 0.5 + rewards[t] * 10, 0, 200)

    # Field: 4-channel field that evolves
    field = rng.random((n_timesteps, grid_size, grid_size, 4)).astype(np.float32)
    # Add some temporal correlation
    for t in range(1, n_timesteps):
        field[t] = 0.9 * field[t - 1] + 0.1 * field[t]

    return {
        "actions": actions,
        "positions": positions,
        "rewards": rewards,
        "alive_mask": alive_mask,
        "energy": energy,
        "field": field,
    }


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Compute all emergence metrics from a checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint pickle file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1000000,
        help="Window size for windowed analysis (default: 1000000 = 1M steps)",
    )
    parser.add_argument(
        "--window-overlap",
        type=float,
        default=0.5,
        help="Overlap fraction between windows (default: 0.5 = 50%%)",
    )
    parser.add_argument(
        "--n-surrogates",
        type=int,
        default=100,
        help="Number of surrogate samples (default: 100)",
    )
    parser.add_argument(
        "--skip-surrogates",
        action="store_true",
        help="Skip surrogate significance tests (faster)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be computed without running",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output",
    )
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Use synthetic trajectory data (for testing)",
    )
    parser.add_argument(
        "--synthetic-timesteps",
        type=int,
        default=1000,
        help="Number of timesteps for synthetic trajectory (default: 1000)",
    )
    parser.add_argument(
        "--synthetic-agents",
        type=int,
        default=8,
        help="Number of agents for synthetic trajectory (default: 8)",
    )

    args = parser.parse_args()

    # Dry run mode
    if args.dry_run:
        print("DRY RUN - Would compute emergence metrics:")
        print(f"  Checkpoint: {args.checkpoint}")
        print(f"  Output: {args.output}")
        print(f"  Window size: {args.window_size:,} steps")
        print(f"  Window overlap: {args.window_overlap * 100:.0f}%")
        print(f"  Surrogates: {args.n_surrogates if not args.skip_surrogates else 'SKIPPED'}")
        print()
        print("Metrics to compute:")
        print("  1. O-Information (synergy vs redundancy)")
        print("  2. PID Median Synergy (agent-field-outcome)")
        print("  3. Causal Emergence EI Gap (macro vs micro)")
        print("  4. Rosas Psi (emergent causation)")
        print("  5. Mean Transfer Entropy (agent coordination)")
        print("  6. Specialization Score (behavioral divergence)")
        print("  7. Division of Labor (task role differentiation)")
        return 0

    # Import here to avoid slow import on --help or --dry-run
    try:
        from src.analysis.emergence_report import (
            compute_all_emergence_metrics,
            compute_windowed_metrics,
            print_emergence_report,
            report_to_json,
        )
    except ImportError as e:
        print(f"ERROR: Failed to import emergence_report module: {e}")
        print("Make sure you have installed all Phase 5 dependencies:")
        print("  pip install -e '.[phase5]'")
        return 1

    # Load or generate trajectory
    if args.use_synthetic:
        print(f"Generating synthetic trajectory ({args.synthetic_timesteps} steps, {args.synthetic_agents} agents)...")
        trajectory = generate_synthetic_trajectory(
            n_timesteps=args.synthetic_timesteps,
            n_agents=args.synthetic_agents,
        )
        agent_params = None
        checkpoint_path = "synthetic"
    else:
        # Check checkpoint exists
        checkpoint_path = args.checkpoint
        if not Path(checkpoint_path).exists():
            print(f"ERROR: Checkpoint not found: {checkpoint_path}")
            return 1

        print(f"Loading checkpoint: {checkpoint_path}")
        try:
            checkpoint = load_checkpoint(checkpoint_path)
        except Exception as e:
            print(f"ERROR: Failed to load checkpoint: {e}")
            return 1

        # Extract agent params if available
        agent_params = checkpoint.get("agent_params")

        # Need trajectory data - check if available in checkpoint
        if "trajectory" in checkpoint:
            trajectory = checkpoint["trajectory"]
        else:
            # Generate synthetic trajectory for demonstration
            print("WARNING: Checkpoint does not contain trajectory data.")
            print("Generating synthetic trajectory for demonstration...")
            trajectory = generate_synthetic_trajectory(
                n_timesteps=args.synthetic_timesteps,
                n_agents=args.synthetic_agents,
            )

    # Compute metrics
    print()
    print("Computing emergence metrics...")
    if args.verbose:
        print(f"  Surrogate tests: {'ENABLED' if not args.skip_surrogates else 'DISABLED'}")
        print(f"  N surrogates: {args.n_surrogates}")

    try:
        report = compute_all_emergence_metrics(
            trajectory=trajectory,
            agent_params=agent_params,
            run_surrogates=not args.skip_surrogates,
            n_surrogates=args.n_surrogates,
            checkpoint_path=checkpoint_path,
        )
    except Exception as e:
        print(f"ERROR: Failed to compute metrics: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    # Windowed analysis if trajectory is long enough
    n_timesteps = report.n_timesteps
    if n_timesteps >= args.window_size:
        print(f"Computing windowed metrics (window={args.window_size:,}, overlap={args.window_overlap})...")
        try:
            windowed = compute_windowed_metrics(
                trajectory=trajectory,
                agent_params=agent_params,
                window_size=args.window_size,
                overlap=args.window_overlap,
                run_surrogates=False,  # Skip surrogates in windows for speed
            )
            report.windowed_results = windowed
            print(f"  Computed {len(windowed)} windows")
        except Exception as e:
            print(f"WARNING: Windowed analysis failed: {e}")
    else:
        print(f"Skipping windowed analysis (trajectory too short: {n_timesteps} < {args.window_size})")

    # Print report
    print()
    print_emergence_report(report)

    # Save to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        json_str = report_to_json(report)
        output_path.write_text(json_str)
        print(f"Saved report to: {output_path}")
    except Exception as e:
        print(f"ERROR: Failed to save report: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
