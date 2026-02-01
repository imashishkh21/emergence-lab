"""Ablation test to evaluate whether the shared field matters.

Tests three conditions:
    1. Normal: field operates normally (diffusion + decay + agent writes)
    2. Zeroed: field values are zeroed every step (agents see no field info)
    3. Random: field values are randomized every step (agents see noise)

Compares mean rewards across conditions to determine if field contributes.
"""

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from src.agents.network import ActorCritic
from src.agents.policy import get_deterministic_actions
from src.configs import Config
from src.environment.env import reset, step
from src.environment.obs import get_observations, obs_dim
from src.environment.state import EnvState
from src.field.field import FieldState, create_field


FieldCondition = Literal["normal", "zeroed", "random"]


@dataclass
class AblationResult:
    """Results from one ablation condition.

    Attributes:
        condition: Name of the ablation condition.
        mean_reward: Mean total episode reward across episodes.
        std_reward: Standard deviation of total episode rewards.
        episode_rewards: Per-episode total rewards.
    """
    condition: str
    mean_reward: float
    std_reward: float
    episode_rewards: list[float]


def _run_episode(
    network: ActorCritic,
    params: dict,
    config: Config,
    key: jax.Array,
    condition: FieldCondition,
) -> float:
    """Run a single episode under the given field condition.

    Args:
        network: ActorCritic network module.
        params: Network parameters.
        config: Master configuration.
        key: PRNG key for environment reset.
        condition: One of "normal", "zeroed", "random".

    Returns:
        Total episode reward (summed across all agents and steps).
    """
    key, reset_key = jax.random.split(key)
    state = reset(reset_key, config)
    total_reward = 0.0

    for t in range(config.env.max_steps):
        # Apply field ablation before observation
        if condition == "zeroed":
            zero_field = create_field(
                config.env.grid_size,
                config.env.grid_size,
                config.field.num_channels,
            )
            state = _replace_field(state, zero_field)
        elif condition == "random":
            key, noise_key = jax.random.split(key)
            noise_values = jax.random.uniform(
                noise_key,
                shape=(config.env.grid_size, config.env.grid_size, config.field.num_channels),
            )
            random_field = FieldState(values=noise_values)
            state = _replace_field(state, random_field)

        # Get observations
        obs = get_observations(state, config)

        # Add batch dim: (1, num_agents, obs_dim)
        obs_batched = obs[None, :, :]

        # Get deterministic actions: (1, num_agents) -> (num_agents,)
        actions = get_deterministic_actions(network, params, obs_batched)
        actions = actions[0]

        # Step environment
        state, rewards, done, _info = step(state, actions, config)

        # Sum rewards across agents for this step
        total_reward += float(jnp.sum(rewards))

        if bool(done):
            break

    return total_reward


def _replace_field(state: EnvState, field_state: FieldState) -> EnvState:
    """Return a copy of state with a replaced field_state."""
    return EnvState(
        agent_positions=state.agent_positions,
        food_positions=state.food_positions,
        food_collected=state.food_collected,
        field_state=field_state,
        step=state.step,
        key=state.key,
        agent_energy=state.agent_energy,
        agent_alive=state.agent_alive,
        agent_ids=state.agent_ids,
        agent_parent_ids=state.agent_parent_ids,
        next_agent_id=state.next_agent_id,
        agent_params=state.agent_params,
    )


def ablation_test(
    network: ActorCritic,
    params: dict,
    config: Config,
    num_episodes: int = 20,
    seed: int = 0,
) -> dict[str, AblationResult]:
    """Run ablation test comparing normal, zeroed, and random field conditions.

    Args:
        network: ActorCritic network module.
        params: Network parameters.
        config: Master configuration.
        num_episodes: Number of episodes per condition.
        seed: Base random seed.

    Returns:
        Dictionary mapping condition name to AblationResult.
    """
    conditions: list[FieldCondition] = ["normal", "zeroed", "random"]
    results: dict[str, AblationResult] = {}

    for condition in conditions:
        episode_rewards: list[float] = []
        base_key = jax.random.PRNGKey(seed)

        for ep in range(num_episodes):
            ep_key = jax.random.fold_in(base_key, ep)
            reward = _run_episode(network, params, config, ep_key, condition)
            episode_rewards.append(reward)

        rewards_arr = np.array(episode_rewards)
        results[condition] = AblationResult(
            condition=condition,
            mean_reward=float(np.mean(rewards_arr)),
            std_reward=float(np.std(rewards_arr)),
            episode_rewards=episode_rewards,
        )

    return results


def print_ablation_results(results: dict[str, AblationResult]) -> None:
    """Pretty-print ablation test results."""
    print("=" * 60)
    print("Ablation Test Results")
    print("=" * 60)
    print(f"{'Condition':<12} {'Mean Reward':>12} {'Std':>10} {'Episodes':>10}")
    print("-" * 44)
    for name in ["normal", "zeroed", "random"]:
        if name in results:
            r = results[name]
            print(f"{r.condition:<12} {r.mean_reward:>12.2f} {r.std_reward:>10.2f} {len(r.episode_rewards):>10}")
    print("=" * 60)

    # Statistical comparison
    if "normal" in results and "zeroed" in results:
        diff = results["normal"].mean_reward - results["zeroed"].mean_reward
        print(f"\nNormal - Zeroed field gap: {diff:+.2f}")
    if "normal" in results and "random" in results:
        diff = results["normal"].mean_reward - results["random"].mean_reward
        print(f"Normal - Random field gap: {diff:+.2f}")


def main() -> None:
    """CLI entry point for ablation testing.

    Usage:
        python -m src.analysis.ablation --checkpoint=path/to/params.pkl
        python -m src.analysis.ablation --checkpoint=path/to/params.pkl --num-episodes=50
    """
    import argparse
    import pickle

    parser = argparse.ArgumentParser(description="Run ablation test on trained model")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to saved parameters (pickle file)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (optional, uses defaults if not provided)",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=20,
        help="Number of episodes per condition",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed",
    )
    args = parser.parse_args()

    # Load config
    if args.config is not None:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    # Load checkpoint params
    with open(args.checkpoint, "rb") as f:
        params = pickle.load(f)

    # Create network
    observation_dim = obs_dim(config)
    network = ActorCritic(
        hidden_dims=tuple(config.agent.hidden_dims),
        num_actions=6,
    )

    # Verify params by running a dummy forward pass
    dummy_obs = jnp.zeros((observation_dim,))
    network.apply(params, dummy_obs)

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Running ablation test with {args.num_episodes} episodes per condition...")
    print()

    results = ablation_test(
        network=network,
        params=params,
        config=config,
        num_episodes=args.num_episodes,
        seed=args.seed,
    )

    print_ablation_results(results)


if __name__ == "__main__":
    main()
