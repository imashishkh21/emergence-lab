#!/usr/bin/env python3
"""Run superlinear scaling experiments.

This script tests N = 1, 2, 4, 8, 16, 32 agents with 3 field conditions
and 20 seeds to analyze whether field-mediated coordination provides
superlinear scaling benefits.

The killer chart:
    X = number of agents
    Y = per-agent food rate
    3 curves: normal > zeroed > no_field (proves field helps at scale)

Power law fit: log(F_total) = alpha * log(N) + c
    - alpha > 1.0 => superlinear (field helps more with more agents)
    - alpha = 1.0 => linear (no coordination benefit)
    - alpha < 1.0 => sublinear (crowding hurts)

Reference: Hamann (2018), "Swarm Robotics: A Formal Approach"

Usage:
    # Dry run to verify setup
    python scripts/run_scaling_experiment.py --dry-run

    # Run all conditions with checkpoint (N=20 seeds)
    python scripts/run_scaling_experiment.py --checkpoint checkpoints/params.pkl

    # Run specific conditions and agent counts
    python scripts/run_scaling_experiment.py --checkpoint checkpoints/params.pkl \
        --conditions normal,zeroed --n-agents 1,4,8,16

    # Quick test with fewer seeds
    python scripts/run_scaling_experiment.py --checkpoint checkpoints/params.pkl \
        --n-seeds 3 --n-episodes 5
"""

import argparse
import pickle
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from src.agents.network import ActorCritic
from src.agents.policy import get_deterministic_actions
from src.analysis.ablation import _replace_field, _zero_field_obs
from src.analysis.scaling import (
    ScalingAnalysis,
    ScalingResult,
    aggregate_scaling_results,
    print_scaling_analysis,
    print_scaling_comparison,
)
from src.configs import Config
from src.environment.env import reset, step
from src.environment.obs import get_observations, obs_dim
from src.field.field import create_field

# Scaling experiment field conditions (subset of extended conditions)
ScalingFieldCondition = Literal["normal", "zeroed", "no_field"]
DEFAULT_N_AGENTS = [1, 2, 4, 8, 16, 32]
DEFAULT_CONDITIONS: list[ScalingFieldCondition] = ["normal", "zeroed", "no_field"]


def create_config_with_n_agents(base_config: Config, n_agents: int) -> Config:
    """Create a config with a specific number of agents.

    Args:
        base_config: Base configuration to modify.
        n_agents: Number of agents to use.

    Returns:
        New Config with modified agent count and max_agents.
    """
    # Ensure max_agents is at least n_agents
    max_agents = max(base_config.evolution.max_agents, n_agents)

    return Config(
        env=base_config.env.__class__(
            grid_size=base_config.env.grid_size,
            num_agents=n_agents,
            num_food=base_config.env.num_food,
            max_steps=base_config.env.max_steps,
            observation_radius=base_config.env.observation_radius,
            food_respawn_prob=base_config.env.food_respawn_prob,
            hidden_food=base_config.env.hidden_food,
        ),
        field=base_config.field,
        agent=base_config.agent,
        train=base_config.train,
        log=base_config.log,
        analysis=base_config.analysis,
        evolution=base_config.evolution.__class__(
            enabled=False,  # Disable evolution for scaling experiments
            max_agents=max_agents,
            starting_energy=base_config.evolution.starting_energy,
            energy_per_step=0,  # No energy drain
            food_energy=base_config.evolution.food_energy,
            reproduce_threshold=1000000,  # Never reproduce
            reproduce_cost=base_config.evolution.reproduce_cost,
            mutation_std=base_config.evolution.mutation_std,
        ),
        specialization=base_config.specialization,
    )


def run_scaling_episode(
    network: ActorCritic,
    params: dict,
    config: Config,
    key: jax.Array,
    condition: ScalingFieldCondition,
) -> dict:
    """Run a single episode and return results for scaling analysis.

    Args:
        network: ActorCritic network module.
        params: Network parameters.
        config: Configuration with specific n_agents.
        key: PRNG key.
        condition: Field condition ("normal", "zeroed", "no_field").

    Returns:
        Dict with "total_food", "total_reward", "n_agents".
    """
    key, reset_key = jax.random.split(key)
    state = reset(reset_key, config)

    total_reward = 0.0
    total_food = 0

    for t in range(config.env.max_steps):
        # Apply field condition before observation
        if condition == "zeroed":
            zero_field = create_field(
                config.env.grid_size,
                config.env.grid_size,
                config.field.num_channels,
            )
            # Replace field in state
            state = _replace_field(state, zero_field)

        # Get observations
        obs = get_observations(state, config)

        # For no_field condition: zero out field observations
        if condition == "no_field":
            obs = _zero_field_obs(obs, config)

        # Add batch dim: (1, max_agents, obs_dim)
        obs_batched = obs[None, :, :]

        # Get deterministic actions
        actions = get_deterministic_actions(network, params, obs_batched)
        actions = actions[0]

        # Count food before step
        pre_food = int(jnp.sum(state.food_collected.astype(jnp.int32)))

        # Step environment
        state, rewards, done, info = step(state, actions, config)

        # Count food after step
        post_food = int(jnp.sum(state.food_collected.astype(jnp.int32)))
        food_this_step = max(0, post_food - pre_food)

        total_food += food_this_step
        total_reward += float(jnp.sum(rewards))

        if bool(done):
            break

    return {
        "total_food": float(total_food),
        "total_reward": total_reward,
        "n_agents": config.env.num_agents,
    }


def evaluate_scaling(
    network: ActorCritic,
    params: dict,
    base_config: Config,
    n_agents: int,
    condition: ScalingFieldCondition,
    n_episodes: int,
    seed: int,
) -> ScalingResult:
    """Evaluate scaling for a specific agent count and condition.

    Args:
        network: ActorCritic network module.
        params: Network parameters.
        base_config: Base configuration.
        n_agents: Number of agents.
        condition: Field condition.
        n_episodes: Number of episodes to run.
        seed: Random seed.

    Returns:
        ScalingResult with aggregated statistics.
    """
    config = create_config_with_n_agents(base_config, n_agents)
    base_key = jax.random.PRNGKey(seed)

    episode_foods: list[float] = []
    episode_rewards: list[float] = []

    for ep in range(n_episodes):
        ep_key = jax.random.fold_in(base_key, ep)
        result = run_scaling_episode(network, params, config, ep_key, condition)
        episode_foods.append(result["total_food"])
        episode_rewards.append(result["total_reward"])

    total_food = float(np.mean(episode_foods))
    total_reward = float(np.mean(episode_rewards))
    per_agent_food = total_food / n_agents if n_agents > 0 else 0.0
    per_agent_reward = total_reward / n_agents if n_agents > 0 else 0.0

    # Efficiency will be computed later when we have solo baseline
    efficiency = 1.0  # Placeholder

    return ScalingResult(
        n_agents=n_agents,
        field_condition=condition,
        total_food=total_food,
        per_agent_food=per_agent_food,
        efficiency=efficiency,
        total_reward=total_reward,
        per_agent_reward=per_agent_reward,
        seed=seed,
        episode_foods=episode_foods,
    )


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
        num_actions=config.agent.num_actions,
    )


def run_scaling_for_condition(
    network: ActorCritic,
    params: dict,
    base_config: Config,
    condition: ScalingFieldCondition,
    n_agents_list: list[int],
    n_seeds: int,
    n_episodes: int,
    output_dir: Path,
    dry_run: bool = False,
) -> ScalingAnalysis | None:
    """Run scaling experiment for a single field condition.

    Args:
        network: ActorCritic network module.
        params: Network parameters.
        base_config: Base configuration.
        condition: Field condition.
        n_agents_list: List of agent counts to test.
        n_seeds: Number of seeds.
        n_episodes: Episodes per seed/N combination.
        output_dir: Directory for results.
        dry_run: If True, only print what would be done.

    Returns:
        ScalingAnalysis if not dry_run, None otherwise.
    """
    print(f"\n{'='*60}")
    print(f"Condition: {condition}")
    print(f"Agent counts: {n_agents_list}")
    print(f"Seeds: {n_seeds}, Episodes per seed: {n_episodes}")
    print(f"{'='*60}")

    if dry_run:
        total_runs = len(n_agents_list) * n_seeds
        print(f"[DRY RUN] Would run {total_runs} experiments")
        return None

    all_results: list[ScalingResult] = []

    for n_agents in n_agents_list:
        print(f"\n  N={n_agents:>2} agents:", end=" ", flush=True)

        for seed in range(n_seeds):
            result = evaluate_scaling(
                network=network,
                params=params,
                base_config=base_config,
                n_agents=n_agents,
                condition=condition,
                n_episodes=n_episodes,
                seed=seed,
            )
            all_results.append(result)
            print(".", end="", flush=True)

        print(" done")

    # Compute solo baseline and update efficiencies
    solo_results = [r for r in all_results if r.n_agents == 1]
    if solo_results:
        solo_food = float(np.mean([r.total_food for r in solo_results]))
    else:
        # Estimate from smallest N
        min_n = min(r.n_agents for r in all_results)
        min_n_results = [r for r in all_results if r.n_agents == min_n]
        solo_food = float(np.mean([r.total_food for r in min_n_results])) / min_n

    # Update efficiencies
    for r in all_results:
        if solo_food > 0:
            r.efficiency = r.total_food / (r.n_agents * solo_food)

    # Aggregate results
    analysis = aggregate_scaling_results(all_results, n_agents_list, solo_food)

    # Save results
    output_file = output_dir / f"scaling_{condition}.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(asdict(analysis), f)
    print(f"  Saved results to {output_file}")

    return analysis


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run superlinear scaling experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint pickle file (optional for dry-run)",
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
        help="Comma-separated list of conditions (default: normal,zeroed,no_field)",
    )
    parser.add_argument(
        "--n-agents",
        type=str,
        default=None,
        help="Comma-separated list of agent counts (default: 1,2,4,8,16,32)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=20,
        help="Number of seeds per condition/N (default: 20)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Episodes per seed (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/scaling",
        help="Directory to save results (default: results/scaling)",
    )

    args = parser.parse_args()

    # Parse conditions
    if args.conditions:
        conditions: list[ScalingFieldCondition] = []
        for c in args.conditions.split(","):
            c = c.strip()
            if c not in ["normal", "zeroed", "no_field"]:
                print(f"ERROR: Unknown condition '{c}'")
                print("Valid conditions: normal, zeroed, no_field")
                return 1
            conditions.append(c)  # type: ignore
    else:
        conditions = list(DEFAULT_CONDITIONS)

    # Parse agent counts
    if args.n_agents:
        n_agents_list = [int(n.strip()) for n in args.n_agents.split(",")]
    else:
        n_agents_list = list(DEFAULT_N_AGENTS)

    # Setup output directory
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("Superlinear Scaling Experiment")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Conditions: {conditions}")
    print(f"Agent counts: {n_agents_list}")
    print(f"Seeds: {args.n_seeds}, Episodes per seed: {args.n_episodes}")
    print(f"Output directory: {output_dir}")
    print(f"Dry run: {args.dry_run}")

    # Calculate total experiments
    total_experiments = len(conditions) * len(n_agents_list) * args.n_seeds
    print(f"\nTotal experiments: {total_experiments}")

    if args.dry_run:
        print("\n[DRY RUN] Would run the experiments described above")
        return 0

    # Verify checkpoint exists
    if args.checkpoint is None:
        print("ERROR: --checkpoint required for actual run")
        return 1

    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return 1

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print("\nLoading checkpoint...")
    params, saved_config = load_checkpoint(args.checkpoint)
    base_config = saved_config if saved_config else Config()
    print("  Checkpoint loaded successfully")

    # Create network
    network = create_network(base_config)

    # Verify network works with different agent counts
    print("  Verifying network...")
    test_config = create_config_with_n_agents(base_config, max(n_agents_list))
    observation_dim = obs_dim(test_config)
    dummy_obs = jnp.zeros((observation_dim,))
    try:
        network.apply(params, dummy_obs)
    except Exception as e:
        print(f"ERROR: Network verification failed: {e}")
        return 1
    print("  Network verified")

    # Run scaling experiments for each condition
    all_analyses: dict[str, ScalingAnalysis] = {}

    for condition in conditions:
        analysis = run_scaling_for_condition(
            network=network,
            params=params,
            base_config=base_config,
            condition=condition,
            n_agents_list=n_agents_list,
            n_seeds=args.n_seeds,
            n_episodes=args.n_episodes,
            output_dir=output_dir,
            dry_run=args.dry_run,
        )
        if analysis:
            all_analyses[condition] = analysis

    # Print individual analyses
    for _cond, analysis in all_analyses.items():
        print_scaling_analysis(analysis)

    # Print comparison
    if len(all_analyses) > 1:
        print_scaling_comparison(all_analyses)

    # Save combined results
    if all_analyses:
        combined_file = output_dir / "scaling_all_conditions.pkl"
        combined_data = {c: asdict(a) for c, a in all_analyses.items()}
        with open(combined_file, "wb") as f:
            pickle.dump(combined_data, f)
        print(f"\nSaved combined results to {combined_file}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
