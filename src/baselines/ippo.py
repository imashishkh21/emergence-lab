"""IPPO Baseline: Independent PPO with no field and no evolution.

This is the simplest baseline: agents use PPO with individual rewards but have
no shared communication medium (field zeroed) and no evolution (population
remains constant). This represents "no communication at all" lower bound.

Key characteristics:
    - Field disabled: decay_rate=1.0 (zeroes out field instantly)
    - Evolution disabled: population stays at initial num_agents
    - Shared parameters: all agents use the same policy weights
    - Individual rewards: each agent optimizes for its own food collection
"""

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from src.agents.network import ActorCritic
from src.agents.policy import sample_actions
from src.configs import Config
from src.environment.env import reset, step
from src.environment.obs import get_observations, obs_dim


def ippo_config(base_config: Config | None = None) -> Config:
    """Create a config for IPPO baseline.

    IPPO disables the field (decay_rate=1.0) and evolution.
    Agents still observe the field (zeros) to maintain observation shape
    compatibility, but the field provides no information.

    Args:
        base_config: Optional base config to modify. If None, uses default Config.

    Returns:
        Config with field zeroed and evolution disabled.
    """
    if base_config is None:
        base_config = Config()

    # Disable field: any residual decays instantly
    field_config = replace(
        base_config.field,
        decay_rate=1.0,  # Full decay each step = field is always zeros
    )

    # Disable evolution: no births, deaths, or reproduction
    # Keep max_agents same as num_agents (no empty slots needed)
    evolution_config = replace(
        base_config.evolution,
        enabled=False,
        # Infinite energy so agents never die
        starting_energy=1000000,
        energy_per_step=0,
        max_energy=1000000,
        # Reproduction impossible
        reproduce_threshold=10000000,
        # Keep max_agents = num_agents for consistency
        max_agents=base_config.env.num_agents,
        min_agents=base_config.env.num_agents,
    )

    return replace(
        base_config,
        field=field_config,
        evolution=evolution_config,
    )


def run_ippo_episode(
    network: ActorCritic,
    params: Any,
    config: Config,
    key: jax.Array,
    deterministic: bool = False,
) -> dict[str, Any]:
    """Run a single episode using IPPO (no field, no evolution).

    Args:
        network: ActorCritic network module.
        params: Network parameters (shared across all agents).
        config: Configuration (should be from ippo_config()).
        key: JAX PRNG key.
        deterministic: If True, use greedy actions instead of sampling.

    Returns:
        Standardized result dict with:
            - total_reward: Sum of all rewards across all agents and steps
            - food_collected: Total food items collected in the episode
            - final_population: Number of alive agents at episode end
            - per_agent_rewards: List of total reward per agent over the episode
    """
    # Initialize environment
    key, reset_key = jax.random.split(key)
    state = reset(reset_key, config)

    # Track metrics
    total_reward = 0.0
    total_food = 0.0
    num_agents = config.env.num_agents
    per_agent_rewards = np.zeros(num_agents)

    # Run episode
    for _ in range(config.env.max_steps):
        # Get observations (add batch dimension for single env)
        obs = get_observations(state, config)
        obs = obs[None, :, :]  # (1, max_agents, obs_dim)

        # Sample or get deterministic actions
        key, action_key = jax.random.split(key)
        if deterministic:
            from src.agents.policy import get_deterministic_actions

            actions = get_deterministic_actions(network, params, obs)
        else:
            actions, _, _, _ = sample_actions(network, params, obs, action_key)

        # Remove batch dimension
        actions = actions[0]  # (max_agents,)

        # Step environment
        state, rewards, done, info = step(state, actions, config)

        # Accumulate metrics
        rewards_np = np.array(rewards)
        total_reward += float(np.sum(rewards_np))
        total_food += float(info["food_collected_this_step"])
        per_agent_rewards[:num_agents] += rewards_np[:num_agents]

        if done:
            break

    # Final population (should be same as initial for IPPO)
    final_population = int(np.sum(np.array(state.agent_alive)))

    return {
        "total_reward": total_reward,
        "food_collected": total_food,
        "final_population": final_population,
        "per_agent_rewards": per_agent_rewards.tolist(),
    }


def evaluate_ippo(
    network: ActorCritic,
    params: Any,
    config: Config,
    n_episodes: int,
    seed: int,
    deterministic: bool = False,
) -> dict[str, Any]:
    """Evaluate IPPO over multiple episodes.

    Args:
        network: ActorCritic network module.
        params: Network parameters (shared across all agents).
        config: Configuration (should be from ippo_config()).
        n_episodes: Number of episodes to run.
        seed: Random seed for reproducibility.
        deterministic: If True, use greedy actions instead of sampling.

    Returns:
        Aggregated result dict with:
            - total_reward: Mean total reward across episodes
            - total_reward_std: Standard deviation of total reward
            - food_collected: Mean food collected across episodes
            - food_collected_std: Standard deviation of food collected
            - final_population: Mean final population across episodes
            - per_agent_rewards: Mean reward per agent across episodes
            - episode_rewards: List of total rewards per episode
            - episode_food: List of food collected per episode
            - n_episodes: Number of episodes run
    """
    key = jax.random.PRNGKey(seed)

    episode_rewards = []
    episode_food = []
    episode_populations = []
    all_per_agent_rewards = []

    for _ in range(n_episodes):
        key, episode_key = jax.random.split(key)
        result = run_ippo_episode(
            network, params, config, episode_key, deterministic=deterministic
        )
        episode_rewards.append(result["total_reward"])
        episode_food.append(result["food_collected"])
        episode_populations.append(result["final_population"])
        all_per_agent_rewards.append(result["per_agent_rewards"])

    # Compute statistics
    rewards_array = np.array(episode_rewards)
    food_array = np.array(episode_food)
    pop_array = np.array(episode_populations)
    per_agent_array = np.array(all_per_agent_rewards)

    return {
        "total_reward": float(np.mean(rewards_array)),
        "total_reward_std": float(np.std(rewards_array)),
        "food_collected": float(np.mean(food_array)),
        "food_collected_std": float(np.std(food_array)),
        "final_population": float(np.mean(pop_array)),
        "per_agent_rewards": np.mean(per_agent_array, axis=0).tolist(),
        "episode_rewards": episode_rewards,
        "episode_food": episode_food,
        "n_episodes": n_episodes,
    }


def create_ippo_network(config: Config) -> ActorCritic:
    """Create an ActorCritic network for IPPO.

    Uses the standard ActorCritic with shared weights across all agents.

    Args:
        config: Configuration object.

    Returns:
        ActorCritic network module.
    """
    return ActorCritic(
        hidden_dims=config.agent.hidden_dims,
        num_actions=config.agent.num_actions,
        agent_embed_dim=config.agent.agent_embed_dim,
        n_agents=config.env.num_agents,
    )


def init_ippo_params(
    network: ActorCritic,
    config: Config,
    key: jax.Array,
) -> Any:
    """Initialize IPPO network parameters.

    Args:
        network: ActorCritic network module.
        config: Configuration object.
        key: JAX PRNG key for initialization.

    Returns:
        Initialized network parameters.
    """
    # Get observation dimension for this config
    observation_dim = obs_dim(config)

    # Create dummy observation
    dummy_obs = jnp.zeros((observation_dim,))

    # Initialize parameters
    params = network.init(key, dummy_obs)

    return params
