#!/usr/bin/env python
"""Train a model briefly and run specialization ablation analysis.

Compares populations with divergent (specialized) weights vs uniform (cloned)
weights vs random weights to test whether weight diversity helps collective
performance.

Usage:
    python scripts/run_specialization_ablation.py
    python scripts/run_specialization_ablation.py --iterations 100 --num-episodes 10
    python scripts/run_specialization_ablation.py --checkpoint checkpoints/params.pkl --skip-training
"""

import argparse
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

from src.agents.network import ActorCritic
from src.analysis.ablation import (
    specialization_ablation_test,
    print_specialization_ablation_results,
)
from src.analysis.specialization import (
    compute_weight_divergence,
    specialization_score,
    extract_behavior_features,
)
from src.analysis.trajectory import record_episode
from src.configs import Config
from src.environment.obs import obs_dim
from src.training.train import create_train_state, train_step


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train briefly and run specialization ablation test"
    )
    parser.add_argument(
        "--iterations", type=int, default=50,
        help="Number of training iterations (default: 50)",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=10,
        help="Number of ablation episodes per condition (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint (saves after training, loads with --skip-training)",
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip training and load from --checkpoint instead",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (optional)",
    )
    args = parser.parse_args()

    # Load or create config
    if args.config is not None:
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    config.log.wandb = False
    config.train.seed = args.seed

    print("=" * 70)
    print("Emergence Lab -- Specialization Ablation")
    print("=" * 70)

    if args.skip_training:
        # Load from checkpoint
        if args.checkpoint is None:
            print("ERROR: --skip-training requires --checkpoint")
            return
        print(f"Loading checkpoint from {args.checkpoint}...")
        with open(args.checkpoint, "rb") as f:
            checkpoint_data = pickle.load(f)

        if isinstance(checkpoint_data, dict) and "agent_params" in checkpoint_data:
            # Structured checkpoint (from this script's own save)
            params = checkpoint_data["params"]
            trained_agent_params = checkpoint_data.get("agent_params")
            trained_alive_mask = checkpoint_data.get("alive_mask")
            if trained_alive_mask is not None:
                trained_alive_mask = jnp.array(trained_alive_mask)
        else:
            # Raw Flax variables dict (from train.py)
            params = checkpoint_data
            trained_agent_params = None
            trained_alive_mask = None

        observation_dim = obs_dim(config)
        network = ActorCritic(
            hidden_dims=tuple(config.agent.hidden_dims),
            num_actions=config.agent.num_actions,
        )
        # Verify params
        dummy_obs = jnp.zeros((observation_dim,))
        network.apply(params, dummy_obs)
        print("Checkpoint loaded and verified.")
    else:
        # --- Phase 1: Training ---
        steps_per_iter = (
            config.train.num_envs * config.train.num_steps * config.evolution.max_agents
        )
        config.train.total_steps = args.iterations * steps_per_iter

        print(f"Training iterations: {args.iterations}")
        print(f"Steps per iteration: {steps_per_iter}")
        print(f"Total env steps: {config.train.total_steps}")
        print(f"Ablation episodes per condition: {args.num_episodes}")
        print(f"Seed: {args.seed}")
        print()

        print("Phase 1: Training")
        print("-" * 40)

        key = jax.random.PRNGKey(config.train.seed)
        print("Initializing training state...")
        runner_state = create_train_state(config, key)
        print("Training state initialized.")

        # JIT compile train_step
        print("JIT compiling train_step...")

        @jax.jit
        def jit_train_step(rs):
            return train_step(rs, config)

        t0 = time.time()
        runner_state, metrics = jit_train_step(runner_state)
        jax.block_until_ready(metrics)
        jit_time = time.time() - t0
        print(f"JIT compilation done ({jit_time:.1f}s)")

        # Training loop
        t_start = time.time()
        for i in range(1, args.iterations):
            runner_state, metrics = jit_train_step(runner_state)

            if i % 10 == 0 or i == args.iterations - 1:
                reward = float(metrics["mean_reward"])
                loss = float(metrics["total_loss"])
                entropy = float(metrics["entropy"])
                print(
                    f"  iter {i:>4d}/{args.iterations} | "
                    f"reward={reward:.4f} | loss={loss:.4f} | entropy={entropy:.4f}"
                )

        jax.block_until_ready(metrics)
        train_time = time.time() - t_start
        print(f"\nTraining complete ({train_time:.1f}s)")

        params = runner_state.params

        # Extract per-agent params and alive mask from first env
        trained_agent_params = None
        trained_alive_mask = None
        if config.evolution.enabled and runner_state.env_state.agent_params is not None:
            # agent_params shape: (num_envs, max_agents, ...)
            # Take first env
            trained_agent_params = jax.tree_util.tree_map(
                lambda x: x[0], runner_state.env_state.agent_params
            )
            trained_alive_mask = runner_state.env_state.agent_alive[0]

        # Save checkpoint if requested
        if args.checkpoint is not None:
            checkpoint_dir = os.path.dirname(args.checkpoint)
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_data = {
                "params": params,
                "agent_params": trained_agent_params,
                "alive_mask": (
                    np.asarray(trained_alive_mask)
                    if trained_alive_mask is not None
                    else None
                ),
            }
            with open(args.checkpoint, "wb") as f:
                pickle.dump(checkpoint_data, f)
            print(f"Checkpoint saved to {args.checkpoint}")

        network = ActorCritic(
            hidden_dims=tuple(config.agent.hidden_dims),
            num_actions=config.agent.num_actions,
        )

    # --- Phase 2: Pre-ablation Specialization Analysis ---
    print()
    print("Phase 2: Specialization Analysis")
    print("-" * 40)

    if trained_agent_params is not None and trained_alive_mask is not None:
        # Compute weight divergence
        div_result = compute_weight_divergence(
            trained_agent_params,
            np.asarray(trained_alive_mask),
        )
        print(f"Weight divergence (mean): {div_result['mean_divergence']:.6f}")
        print(f"Weight divergence (max):  {div_result['max_divergence']:.6f}")
        print(f"Alive agents: {len(div_result['agent_indices'])}")

        # Record a trajectory and compute specialization score
        print("Recording evaluation trajectory...")
        traj_key = jax.random.PRNGKey(args.seed + 1000)
        trajectory = record_episode(network, params, config, traj_key)
        features = extract_behavior_features(trajectory)
        spec_result = specialization_score(
            features,
            agent_params=trained_agent_params,
            alive_mask=np.asarray(trained_alive_mask),
        )
        print(f"Specialization score: {spec_result['score']:.4f}")
        print(f"  Silhouette component:  {spec_result['silhouette_component']:.4f}")
        print(f"  Divergence component:  {spec_result['divergence_component']:.4f}")
        print(f"  Variance component:    {spec_result['variance_component']:.4f}")
        print(f"  Optimal clusters (k):  {spec_result['optimal_k']}")
    else:
        print("No per-agent params available — evolution may be disabled.")
        print("Running ablation with shared params only.")

    # --- Phase 3: Specialization Ablation ---
    print()
    print("Phase 3: Specialization Ablation")
    print("-" * 40)

    if trained_agent_params is None or trained_alive_mask is None:
        print("Skipping specialization ablation — no per-agent params available.")
        print("Enable evolution (evolution.enabled=True) and train to get per-agent params.")
        return

    print(f"Running {args.num_episodes} episodes per condition...")
    print("Conditions: divergent (trained), uniform (cloned), random_weights")
    print()

    results = specialization_ablation_test(
        network=network,
        params=params,
        config=config,
        trained_agent_params=trained_agent_params,
        trained_alive_mask=trained_alive_mask,
        num_episodes=args.num_episodes,
        seed=args.seed,
    )

    print_specialization_ablation_results(results)

    # --- Summary ---
    print()
    print("Summary")
    print("-" * 40)
    div_r = results["divergent"]
    uni_r = results["uniform"]
    rnd_r = results["random_weights"]

    print(f"Divergent (specialized):  reward={div_r.mean_reward:.2f}, food={div_r.food_collected:.1f}")
    print(f"Uniform (cloned):         reward={uni_r.mean_reward:.2f}, food={uni_r.food_collected:.1f}")
    print(f"Random weights:           reward={rnd_r.mean_reward:.2f}, food={rnd_r.food_collected:.1f}")

    if div_r.food_collected > uni_r.food_collected and div_r.food_collected > rnd_r.food_collected:
        print("\nConclusion: Specialization (weight diversity) IMPROVES collective performance!")
    elif uni_r.food_collected > div_r.food_collected:
        print("\nConclusion: Uniform weights outperform — specialization may not help (yet).")
        print("Try training for longer or with higher mutation rates.")
    else:
        print("\nConclusion: Results are mixed — more training may reveal clearer patterns.")


if __name__ == "__main__":
    main()
