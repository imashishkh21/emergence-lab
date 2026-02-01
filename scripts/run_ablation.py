#!/usr/bin/env python
"""Train a model for a short run and then run ablation analysis.

Usage:
    python scripts/run_ablation.py
    python scripts/run_ablation.py --iterations 100 --num-episodes 10
"""

import argparse
import os
import pickle
import sys
import time

import jax
import jax.numpy as jnp

from src.configs import Config
from src.training.train import create_train_state, train_step
from src.agents.network import ActorCritic
from src.environment.obs import obs_dim
from src.analysis.ablation import (
    ablation_test,
    evolution_ablation_test,
    print_ablation_results,
    print_evolution_ablation_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train briefly and run ablation test")
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
        "--checkpoint", type=str, default="checkpoints/params.pkl",
        help="Path to save/load checkpoint (default: checkpoints/params.pkl)",
    )
    parser.add_argument(
        "--evolution", action="store_true",
        help="Run evolution ablation (2x2: field x evolution) instead of field-only ablation",
    )
    args = parser.parse_args()

    # Configure for a quick training run with W&B disabled
    config = Config()
    config.log.wandb = False
    config.train.seed = args.seed

    # Compute how many total_steps we need for the desired iterations
    steps_per_iter = (
        config.train.num_envs * config.train.num_steps * config.evolution.max_agents
    )
    config.train.total_steps = args.iterations * steps_per_iter

    print("=" * 60)
    print("Emergence Lab — Ablation Runner")
    print("=" * 60)
    print(f"Training iterations: {args.iterations}")
    print(f"Steps per iteration: {steps_per_iter}")
    print(f"Total env steps: {config.train.total_steps}")
    print(f"Ablation episodes per condition: {args.num_episodes}")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    print()

    # --- Phase 1: Training ---
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

    # Print final metrics
    print("\nFinal training metrics:")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {float(v):.6f}")

    # --- Phase 2: Save checkpoint ---
    print()
    print("Phase 2: Saving checkpoint")
    print("-" * 40)

    checkpoint_dir = os.path.dirname(args.checkpoint)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    params = runner_state.params
    with open(args.checkpoint, "wb") as f:
        pickle.dump(params, f)
    print(f"Params saved to {args.checkpoint}")

    # --- Phase 3: Ablation test ---
    print()
    print("Phase 3: Ablation Test")
    print("-" * 40)

    network = ActorCritic(
        hidden_dims=tuple(config.agent.hidden_dims),
        num_actions=6,
    )

    # Verify params with dummy forward pass
    observation_dim = obs_dim(config)
    dummy_obs = jnp.zeros((observation_dim,))
    network.apply(params, dummy_obs)

    if args.evolution:
        print(f"Running EVOLUTION ablation with {args.num_episodes} episodes per condition...")
        print("Conditions: field+evolution, field_only, evolution_only, neither")
        print()

        results = evolution_ablation_test(
            network=network,
            params=params,
            config=config,
            num_episodes=args.num_episodes,
            seed=args.seed,
        )

        print_evolution_ablation_results(results)

        # Interpretation
        print()
        fe = results["field+evolution"].mean_reward
        fo = results["field_only"].mean_reward
        eo = results["evolution_only"].mean_reward
        ne = results["neither"].mean_reward

        print("Interpretation:")
        field_effect = (fe - eo + fo - ne) / 2
        evo_effect = (fe - fo + eo - ne) / 2
        if field_effect > 0 and evo_effect > 0:
            print("  Both field and evolution contribute positively.")
        elif field_effect > 0:
            print("  Field helps, but evolution may not (yet).")
        elif evo_effect > 0:
            print("  Evolution helps, but field may not (yet).")
        else:
            print("  Neither clearly helps — more training may be needed.")

        if fe > max(fo, eo, ne):
            print("  Field + Evolution together is the BEST condition (synergy).")
    else:
        print(f"Running ablation with {args.num_episodes} episodes per condition...")
        print("Conditions: normal, zeroed, random")
        print()

        results = ablation_test(
            network=network,
            params=params,
            config=config,
            num_episodes=args.num_episodes,
            seed=args.seed,
        )

        print_ablation_results(results)

        # Interpretation
        print()
        normal = results["normal"].mean_reward
        zeroed = results["zeroed"].mean_reward
        random = results["random"].mean_reward

        print("Interpretation:")
        if normal > zeroed and normal > random:
            print("  The field carries useful information — agents perform BETTER with it.")
        elif normal < zeroed:
            print("  The field may be HURTING agents — they do better without it.")
        elif abs(normal - zeroed) < results["normal"].std_reward:
            print("  No significant difference — the field may not yet carry useful info.")
            print("  (Try training for more iterations.)")
        else:
            print("  Mixed results — more training or episodes may clarify.")


if __name__ == "__main__":
    main()
