#!/usr/bin/env python3
"""Run full baselines comparison experiment.

Compares all methods (Ours, IPPO, ACO-Fixed, ACO-Hybrid, MAPPO) across
environment configs and seeds with the same compute budget.

Methods:
    - ours: Full emergence system (field + evolution)
    - ippo: Independent PPO with no field
    - aco_fixed: Ant Colony Optimization with hardcoded rules (no NN)
    - aco_hybrid: ACO rules for field writes + NN for movement
    - mappo: Multi-Agent PPO with centralized critic

Reference: Agarwal et al. (2021), "Deep RL at the Edge of the Statistical Precipice"

Usage:
    # Dry run to verify setup
    python scripts/run_baselines_comparison.py --dry-run

    # Run subset of methods
    python scripts/run_baselines_comparison.py --dry-run --methods ours,ippo

    # Run with checkpoint for "ours" method
    python scripts/run_baselines_comparison.py --checkpoint checkpoints/params.pkl

    # Quick test with fewer seeds
    python scripts/run_baselines_comparison.py --checkpoint checkpoints/params.pkl \
        --n-seeds 3 --n-episodes 5

    # Run only on specific environment configs
    python scripts/run_baselines_comparison.py --checkpoint checkpoints/params.pkl \
        --env-configs standard,food_scarcity
"""

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Setup JAX before imports
import jax

from src.experiments.baselines import (
    ALL_METHODS,
    BaselinesComparisonResult,
    MethodName,
    compare_baselines_results,
    print_baselines_comparison,
    run_baselines_comparison,
    save_baselines_result,
)
from src.experiments.configs import list_env_configs


def load_checkpoint(checkpoint_path: str) -> tuple[dict[str, Any], Any]:
    """Load checkpoint and return params and optional config.

    Args:
        checkpoint_path: Path to the checkpoint pickle file.

    Returns:
        Tuple of (params dict, config or None).
    """
    with open(checkpoint_path, "rb") as f:
        checkpoint_data = pickle.load(f)

    if isinstance(checkpoint_data, dict):
        params = checkpoint_data.get("params", checkpoint_data)
        config = checkpoint_data.get("config", None)
    else:
        params = checkpoint_data
        config = None

    return params, config


def parse_methods(methods_str: str) -> list[MethodName]:
    """Parse comma-separated methods string.

    Args:
        methods_str: Comma-separated method names.

    Returns:
        List of validated method names.

    Raises:
        ValueError: If any method name is invalid.
    """
    methods: list[MethodName] = []
    for m in methods_str.split(","):
        m = m.strip()
        if m not in ALL_METHODS:
            raise ValueError(
                f"Unknown method '{m}'. Valid methods: {', '.join(ALL_METHODS)}"
            )
        methods.append(m)  # type: ignore
    return methods


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run full baselines comparison experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for 'ours' method (optional)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without running experiments",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated list of methods (default: all)",
    )
    parser.add_argument(
        "--env-configs",
        type=str,
        default=None,
        help="Comma-separated list of env configs (default: all)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=20,
        help="Number of seeds per method (default: 20)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Episodes per seed (default: 10)",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Base seed offset (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/baselines",
        help="Directory to save results (default: results/baselines)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed progress",
    )

    args = parser.parse_args()

    # Parse methods
    if args.methods:
        try:
            methods = parse_methods(args.methods)
        except ValueError as e:
            print(f"ERROR: {e}")
            return 1
    else:
        methods = list(ALL_METHODS)

    # Parse env configs
    if args.env_configs:
        env_configs = [c.strip() for c in args.env_configs.split(",")]
        valid_configs = list_env_configs()
        for c in env_configs:
            if c not in valid_configs:
                print(f"ERROR: Unknown env config '{c}'")
                print(f"Valid configs: {', '.join(valid_configs)}")
                return 1
    else:
        env_configs = list_env_configs()

    # Setup output directory
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")

    # Print configuration
    print("=" * 70)
    print("Baselines Comparison Experiment")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint or 'None (random init for ours)'}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Env configs: {', '.join(env_configs)}")
    print(f"Seeds: {args.n_seeds}, Episodes per seed: {args.n_episodes}")
    print(f"Seed offset: {args.seed_offset}")
    print(f"Output directory: {output_dir}")
    print(f"Dry run: {args.dry_run}")

    # Calculate total experiments
    total_experiments = len(methods) * len(env_configs) * args.n_seeds
    total_episodes = total_experiments * args.n_episodes
    print(f"\nTotal experiment runs: {total_experiments}")
    print(f"Total episodes: {total_episodes}")

    if args.dry_run:
        print("\n[DRY RUN] Would run the experiments described above")
        print("\nExperiment matrix:")
        for env_config in env_configs:
            print(f"\n  {env_config}:")
            for method in methods:
                print(f"    - {method}: {args.n_seeds} seeds × {args.n_episodes} episodes")
        return 0

    # Load checkpoint if provided (for "ours" method)
    ours_params = None
    if args.checkpoint:
        if not Path(args.checkpoint).exists():
            print(f"ERROR: Checkpoint not found: {args.checkpoint}")
            return 1

        print("\nLoading checkpoint...")
        ours_params, _ = load_checkpoint(args.checkpoint)
        print("  Checkpoint loaded successfully")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiments for each environment config
    all_results: dict[str, BaselinesComparisonResult] = {}

    for env_config_name in env_configs:
        print(f"\n{'='*60}")
        print(f"Environment: {env_config_name}")
        print(f"{'='*60}")

        result = run_baselines_comparison(
            env_config_name=env_config_name,
            methods=methods,
            n_seeds=args.n_seeds,
            n_episodes=args.n_episodes,
            paired_seeds=True,
            seed_offset=args.seed_offset,
            ours_params=ours_params,
            verbose=args.verbose,
        )

        all_results[env_config_name] = result

        # Print summary
        print_baselines_comparison(result)

        # Save per-environment results
        env_output_file = output_dir / f"baselines_{env_config_name}.pkl"
        save_baselines_result(result, env_output_file)
        print(f"Saved results to {env_output_file}")

    # Cross-environment comparison
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("Cross-Environment Summary")
        print("=" * 70)

        comparison = compare_baselines_results(all_results)

        print(f"\nEnvironments tested: {', '.join(comparison['env_configs'])}")
        print(f"\nWins per method (best IQM per environment):")
        for method, wins in sorted(
            comparison["method_wins"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            print(f"  {method}: {wins} wins")

        print(f"\nAverage IQM across environments:")
        for method in comparison["overall_ranking"]:
            avg_iqm = comparison["method_avg_iqm"][method]
            print(f"  {method}: {avg_iqm:.2f}")

        print(f"\nOverall best method: {comparison['overall_best']}")

    # Save combined results
    combined_file = output_dir / "baselines_all_configs.pkl"
    with open(combined_file, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nSaved combined results to {combined_file}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
