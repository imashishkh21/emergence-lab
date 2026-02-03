#!/usr/bin/env python3
"""Run stigmergy ablation experiments at scale.

This script runs 6 field conditions × 20 seeds × 3 environment configs,
producing statistical results for the Phase 5 publication.

6 Field Conditions:
    1. normal: Field operates normally (baseline)
    2. zeroed: Field zeroed every step (no information)
    3. random: Field randomized every step (noise)
    4. frozen: Field initialized but never updated (static structure)
    5. no_field: Field removed from observations (zero-padded)
    6. write_only: Agents write but read zeros (test write-side value)

3 Environment Configs:
    - standard: Default balanced environment
    - hidden_resources: Requires coordination (hidden food enabled)
    - food_scarcity: Scarce resources (num_food=5)

Usage:
    # Dry run to verify setup
    python scripts/run_stigmergy_ablation.py --dry-run --checkpoint checkpoints/params.pkl

    # Run all conditions (N=20 seeds)
    python scripts/run_stigmergy_ablation.py --checkpoint checkpoints/params.pkl

    # Run specific conditions and env configs
    python scripts/run_stigmergy_ablation.py --checkpoint checkpoints/params.pkl \
        --conditions normal,zeroed,frozen --env-configs standard,food_scarcity

    # Run with specific training steps checkpoint
    python scripts/run_stigmergy_ablation.py --checkpoint checkpoints/params_5M.pkl --steps 5M
"""

import argparse
import pickle
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from src.agents.network import ActorCritic
from src.analysis.ablation import (
    ALL_FIELD_CONDITIONS,
    ExtendedAblationResult,
    ExtendedFieldCondition,
    extended_ablation_test,
)
from src.configs import Config
from src.environment.obs import obs_dim
from src.experiments.configs import get_env_config, list_env_configs


def load_checkpoint(checkpoint_path: str) -> tuple[dict, Config | None]:
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


def create_network(config: Config) -> ActorCritic:
    """Create the network architecture from config.

    Args:
        config: Configuration object.

    Returns:
        ActorCritic network module.
    """
    return ActorCritic(
        hidden_dims=tuple(config.agent.hidden_dims),
        num_actions=6,
    )


def run_ablation_for_env_config(
    network: ActorCritic,
    params: dict,
    base_config: Config,
    env_config_name: str,
    conditions: list[ExtendedFieldCondition],
    n_seeds: int,
    output_dir: Path,
    dry_run: bool = False,
) -> dict[str, ExtendedAblationResult] | None:
    """Run ablation for a single environment configuration.

    Args:
        network: ActorCritic network module.
        params: Network parameters.
        base_config: Base configuration.
        env_config_name: Name of environment config ("standard", "hidden_resources", "food_scarcity").
        conditions: List of field conditions to test.
        n_seeds: Number of seeds per condition.
        output_dir: Directory to save results.
        dry_run: If True, only print what would be done.

    Returns:
        Results dict if not dry_run, None otherwise.
    """
    print(f"\n{'='*60}")
    print(f"Environment: {env_config_name}")
    print(f"Conditions: {conditions}")
    print(f"Seeds per condition: {n_seeds}")
    print(f"{'='*60}")

    if dry_run:
        print("[DRY RUN] Would run extended ablation test")
        return None

    # Get environment-specific config
    config = get_env_config(env_config_name, base_config)

    # Verify network with dummy forward pass
    observation_dim = obs_dim(config)
    dummy_obs = jnp.zeros((observation_dim,))
    try:
        network.apply(params, dummy_obs)
    except Exception as e:
        print(f"ERROR: Network verification failed: {e}")
        return None

    # Run ablation for each seed and aggregate
    all_results: dict[str, list[ExtendedAblationResult]] = {c: [] for c in conditions}

    for seed in range(n_seeds):
        print(f"  Running seed {seed + 1}/{n_seeds}...", end=" ", flush=True)

        seed_results = extended_ablation_test(
            network=network,
            params=params,
            config=config,
            num_episodes=1,  # 1 episode per seed for efficiency
            seed=seed,
            conditions=conditions,
        )

        for condition in conditions:
            all_results[condition].append(seed_results[condition])

        print("done")

    # Aggregate results across seeds
    aggregated: dict[str, ExtendedAblationResult] = {}
    for condition in conditions:
        condition_results = all_results[condition]
        all_rewards = [r.mean_reward for r in condition_results]
        rewards_arr = np.array(all_rewards)

        all_populations = np.array([r.final_population for r in condition_results])
        all_births = np.array([r.total_births for r in condition_results])
        all_deaths = np.array([r.total_deaths for r in condition_results])
        all_survivals = np.array([r.survival_rate for r in condition_results])

        aggregated[condition] = ExtendedAblationResult(
            condition=condition,
            mean_reward=float(np.mean(rewards_arr)),
            std_reward=float(np.std(rewards_arr)),
            episode_rewards=all_rewards,
            final_population=float(np.mean(all_populations)),
            total_births=float(np.mean(all_births)),
            total_deaths=float(np.mean(all_deaths)),
            survival_rate=float(np.mean(all_survivals)),
        )

    # Save results
    output_file = output_dir / f"ablation_{env_config_name}.pkl"
    with open(output_file, "wb") as f:
        pickle.dump({k: asdict(v) for k, v in aggregated.items()}, f)
    print(f"  Saved results to {output_file}")

    return aggregated


def print_summary(
    all_env_results: dict[str, dict[str, ExtendedAblationResult]],
) -> None:
    """Print summary of all results across environments.

    Args:
        all_env_results: Dict mapping env_config_name to condition results.
    """
    print("\n" + "=" * 80)
    print("SUMMARY: Stigmergy Ablation Results Across Environments")
    print("=" * 80)

    # Header
    envs = list(all_env_results.keys())
    header = f"{'Condition':<12}"
    for env in envs:
        header += f" {env[:15]:>15}"
    print(header)
    print("-" * 80)

    # Rows for each condition
    condition_order = ["normal", "zeroed", "random", "frozen", "no_field", "write_only"]
    for condition in condition_order:
        row = f"{condition:<12}"
        for env in envs:
            if condition in all_env_results[env]:
                r = all_env_results[env][condition]
                row += f" {r.mean_reward:>15.2f}"
            else:
                row += f" {'N/A':>15}"
        print(row)

    print("=" * 80)

    # Key findings
    print("\nKey Findings:")
    for env, results in all_env_results.items():
        print(f"\n  {env}:")
        if "normal" in results and "zeroed" in results:
            diff = results["normal"].mean_reward - results["zeroed"].mean_reward
            pct = (diff / abs(results["zeroed"].mean_reward) * 100) if results["zeroed"].mean_reward != 0 else 0
            print(f"    Field benefit (normal - zeroed): {diff:+.2f} ({pct:+.1f}%)")
        if "normal" in results and "no_field" in results:
            diff = results["normal"].mean_reward - results["no_field"].mean_reward
            print(f"    Reading benefit (normal - no_field): {diff:+.2f}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run stigmergy ablation experiments at scale",
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
        "--dry-run",
        action="store_true",
        help="Print what would be done without running experiments",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default=None,
        help="Comma-separated list of conditions (default: all 6)",
    )
    parser.add_argument(
        "--env-configs",
        type=str,
        default=None,
        help="Comma-separated list of env configs (default: all 3)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=20,
        help="Number of seeds per condition (default: 20)",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Checkpoint milestone label (e.g., '1M', '5M', '10M') for output naming",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ablation",
        help="Directory to save results (default: results/ablation)",
    )

    args = parser.parse_args()

    # Parse conditions
    if args.conditions:
        conditions: list[ExtendedFieldCondition] = []
        for c in args.conditions.split(","):
            c = c.strip()
            if c not in ALL_FIELD_CONDITIONS:
                print(f"ERROR: Unknown condition '{c}'")
                print(f"Valid conditions: {ALL_FIELD_CONDITIONS}")
                return 1
            conditions.append(c)  # type: ignore
    else:
        conditions = list(ALL_FIELD_CONDITIONS)

    # Parse env configs
    if args.env_configs:
        env_configs = [c.strip() for c in args.env_configs.split(",")]
        for ec in env_configs:
            if ec not in list_env_configs():
                print(f"ERROR: Unknown env config '{ec}'")
                print(f"Valid configs: {list_env_configs()}")
                return 1
    else:
        env_configs = list_env_configs()

    # Setup output directory
    output_dir = Path(args.output_dir)
    if args.steps:
        output_dir = output_dir / f"checkpoint_{args.steps}"
    output_dir = output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("Stigmergy Ablation Experiment")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Conditions: {conditions}")
    print(f"Env configs: {env_configs}")
    print(f"Seeds per condition: {args.n_seeds}")
    print(f"Output directory: {output_dir}")
    print(f"Dry run: {args.dry_run}")

    # Verify checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return 1

    if args.dry_run:
        print("\n[DRY RUN] Would load checkpoint and run experiments")
        print(f"[DRY RUN] Total experiments: {len(conditions)} conditions × {len(env_configs)} envs × {args.n_seeds} seeds")
        print(f"[DRY RUN] = {len(conditions) * len(env_configs) * args.n_seeds} total runs")
        return 0

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print("\nLoading checkpoint...")
    params, saved_config = load_checkpoint(args.checkpoint)
    base_config = saved_config if saved_config else Config()
    print("  Checkpoint loaded successfully")

    # Create network
    network = create_network(base_config)

    # Run ablation for each environment config
    all_env_results: dict[str, dict[str, ExtendedAblationResult]] = {}

    for env_config_name in env_configs:
        results = run_ablation_for_env_config(
            network=network,
            params=params,
            base_config=base_config,
            env_config_name=env_config_name,
            conditions=conditions,
            n_seeds=args.n_seeds,
            output_dir=output_dir,
            dry_run=args.dry_run,
        )
        if results:
            all_env_results[env_config_name] = results

    # Print summary
    if all_env_results:
        print_summary(all_env_results)

        # Save combined results
        combined_file = output_dir / "ablation_all_envs.pkl"
        combined_data = {
            env: {k: asdict(v) for k, v in results.items()}
            for env, results in all_env_results.items()
        }
        with open(combined_file, "wb") as f:
            pickle.dump(combined_data, f)
        print(f"\nSaved combined results to {combined_file}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
