"""ACO Baselines: Ant Colony Optimization with hardcoded pheromone rules.

This module provides two ACO-based baselines:

1. ACO-Fixed: Uses hardcoded pheromone rules (no neural network)
   - Agents deposit pheromone on food collection
   - Movement follows field gradient with softmax
   - Classic ant colony optimization approach

2. ACO-Hybrid: Neural network for movement, hardcoded field writes
   - NN decides movement based on observations
   - Field writes follow hardcoded ACO rules (deposit on food collection)
   - Isolates the value of LEARNING to write (our system) vs hardcoded rules

ACO Parameters (Dorigo & Stutzle 2004):
    alpha = 1.0  (pheromone importance)
    beta = 2.0   (heuristic importance - distance to food)
    rho = 0.5    (evaporation rate - mapped to field decay_rate)
    Q = 1.0      (deposit quantity)

Formula for movement probability:
    p_ij = (tau_ij^alpha * eta_ij^beta) / sum_k (tau_ik^alpha * eta_ik^beta)

    where:
    - tau_ij = pheromone intensity at direction j
    - eta_ij = heuristic (1/distance to nearest food in direction j)
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
from src.field.ops import write_local

# ACO parameters from Dorigo & Stutzle (2004)
ACO_ALPHA = 1.0  # Pheromone importance
ACO_BETA = 2.0   # Heuristic importance (distance to food)
ACO_RHO = 0.5    # Evaporation rate
ACO_Q = 1.0      # Deposit quantity


def aco_config(base_config: Config | None = None) -> Config:
    """Create a config for ACO baselines.

    ACO uses the field as a pheromone grid. The field is enabled but with
    ACO-specific parameters. Evolution is disabled.

    Args:
        base_config: Optional base config to modify. If None, uses default Config.

    Returns:
        Config with ACO-tuned field parameters and evolution disabled.
    """
    if base_config is None:
        base_config = Config()

    # Enable field with ACO parameters
    # rho (evaporation) maps to decay_rate
    field_config = replace(
        base_config.field,
        decay_rate=ACO_RHO,  # 50% evaporation per step
        diffusion_rate=0.1,  # Some spreading like real pheromone
    )

    # Disable evolution: no births, deaths, or reproduction
    evolution_config = replace(
        base_config.evolution,
        enabled=False,
        starting_energy=1000000,
        energy_per_step=0,
        max_energy=1000000,
        reproduce_threshold=10000000,
        max_agents=base_config.env.num_agents,
        min_agents=base_config.env.num_agents,
    )

    return replace(
        base_config,
        field=field_config,
        evolution=evolution_config,
    )


def _compute_food_heuristic(
    positions: jnp.ndarray,
    food_positions: jnp.ndarray,
    food_collected: jnp.ndarray,
    grid_size: int,
) -> jnp.ndarray:
    """Compute heuristic values for each direction based on food proximity.

    Returns eta values for each agent and each movement direction.
    Higher values indicate food is closer in that direction.

    Args:
        positions: Agent positions (max_agents, 2).
        food_positions: Food positions (num_food, 2).
        food_collected: Boolean mask of collected food (num_food,).
        grid_size: Grid dimension.

    Returns:
        Heuristic values (max_agents, 5) for directions [stay, up, down, left, right].
    """
    # Movement deltas: stay, up, down, left, right
    deltas = jnp.array([
        [0, 0],   # stay
        [-1, 0],  # up
        [1, 0],   # down
        [0, -1],  # left
        [0, 1],   # right
    ], dtype=jnp.int32)

    # For each agent and direction, compute distance to nearest uncollected food
    heuristics = []

    for d in range(5):
        # Hypothetical position if agent moves in direction d
        hyp_pos = positions + deltas[d]
        hyp_pos = jnp.clip(hyp_pos, 0, grid_size - 1)

        # Distance from hypothetical position to all food
        # Manhattan distance (simpler than Euclidean for grid)
        food_dist = jnp.abs(hyp_pos[:, 0:1] - food_positions[:, 0]) + \
                    jnp.abs(hyp_pos[:, 1:2] - food_positions[:, 1])  # (num_agents, num_food)

        # Mask out collected food with large distance
        food_dist = jnp.where(food_collected, 1e6, food_dist)

        # Minimum distance to any uncollected food
        min_dist = jnp.min(food_dist, axis=1)  # (num_agents,)

        # Heuristic: 1 / (1 + distance) to avoid division by zero
        eta = 1.0 / (1.0 + min_dist)
        heuristics.append(eta)

    return jnp.stack(heuristics, axis=1)  # (num_agents, 5)


def _compute_field_pheromone(
    positions: jnp.ndarray,
    field_values: jnp.ndarray,
    grid_size: int,
) -> jnp.ndarray:
    """Compute pheromone values for each direction from field.

    Reads field values at neighboring cells to determine pheromone intensity
    in each direction.

    Args:
        positions: Agent positions (max_agents, 2).
        field_values: Field values (H, W, C).
        grid_size: Grid dimension.

    Returns:
        Pheromone values (max_agents, 5) for directions [stay, up, down, left, right].
    """
    # Movement deltas: stay, up, down, left, right
    deltas = jnp.array([
        [0, 0],   # stay
        [-1, 0],  # up
        [1, 0],   # down
        [0, -1],  # left
        [0, 1],   # right
    ], dtype=jnp.int32)

    pheromones = []

    for d in range(5):
        # Position to read field from
        read_pos = positions + deltas[d]
        read_pos = jnp.clip(read_pos, 0, grid_size - 1)

        # Read field values at those positions (sum across channels)
        field_at_pos = field_values[read_pos[:, 0], read_pos[:, 1]]  # (num_agents, C)
        tau = jnp.sum(field_at_pos, axis=1)  # (num_agents,)

        # Add small constant to avoid zero pheromone
        tau = tau + 0.01
        pheromones.append(tau)

    return jnp.stack(pheromones, axis=1)  # (num_agents, 5)


def _aco_movement_probabilities(
    tau: jnp.ndarray,
    eta: jnp.ndarray,
    alpha: float = ACO_ALPHA,
    beta: float = ACO_BETA,
) -> jnp.ndarray:
    """Compute ACO movement probabilities.

    Uses the classic ACO formula:
        p_ij = (tau_ij^alpha * eta_ij^beta) / sum_k (tau_ik^alpha * eta_ik^beta)

    Args:
        tau: Pheromone values (num_agents, 5).
        eta: Heuristic values (num_agents, 5).
        alpha: Pheromone importance weight.
        beta: Heuristic importance weight.

    Returns:
        Movement probabilities (num_agents, 5).
    """
    # Compute numerator: tau^alpha * eta^beta
    numerator = jnp.power(tau, alpha) * jnp.power(eta, beta)

    # Normalize to get probabilities
    denominator = jnp.sum(numerator, axis=1, keepdims=True)
    probs = numerator / (denominator + 1e-10)

    return probs


def _sample_aco_actions(
    probs: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """Sample actions from ACO movement probabilities.

    Args:
        probs: Movement probabilities (num_agents, 5).
        key: JAX PRNG key.

    Returns:
        Sampled actions (num_agents,) in range [0, 4] (5 movement actions, no reproduce).
    """
    # Sample from categorical distribution
    num_agents = probs.shape[0]
    keys = jax.random.split(key, num_agents)

    def sample_one(key_i, prob_i):
        return jax.random.choice(key_i, 5, p=prob_i)

    actions = jax.vmap(sample_one)(keys, probs)
    return jnp.asarray(actions, dtype=jnp.int32)


def run_aco_fixed_episode(
    config: Config,
    key: jax.Array,
) -> dict[str, Any]:
    """Run a single episode using ACO-Fixed (no neural network).

    Agents use hardcoded ACO rules:
    - Movement: follow pheromone gradient + food heuristic
    - Field writes: deposit pheromone when collecting food

    Args:
        config: Configuration (should be from aco_config()).
        key: JAX PRNG key.

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
    max_agents = config.evolution.max_agents
    per_agent_rewards = np.zeros(num_agents)
    grid_size = config.env.grid_size

    # Track which agents collected food last step (for pheromone deposit)
    # We'll compute this after each step
    prev_food_collected = np.array(state.food_collected)

    # Run episode
    for _ in range(config.env.max_steps):
        # Get field values for pheromone reading
        field_values = state.field_state.values

        # Compute pheromone values for each direction
        tau = _compute_field_pheromone(
            state.agent_positions,
            field_values,
            grid_size,
        )

        # Compute food heuristic for each direction
        eta = _compute_food_heuristic(
            state.agent_positions,
            state.food_positions,
            state.food_collected,
            grid_size,
        )

        # Compute movement probabilities
        probs = _aco_movement_probabilities(tau, eta)

        # Sample actions (only movement, no reproduce action=5)
        key, action_key = jax.random.split(key)
        actions = _sample_aco_actions(probs, action_key)

        # Mask dead agents to action 0 (stay)
        actions = jnp.where(state.agent_alive, actions, 0)

        # Record food collected before step
        prev_food_collected = np.array(state.food_collected)

        # Step environment (this handles field dynamics internally)
        state, rewards, done, info = step(state, actions, config)

        # Determine which agents collected food this step
        # (food that was not collected before but is collected now)
        curr_food_collected = np.array(state.food_collected)
        newly_collected_food = curr_food_collected & ~prev_food_collected

        # For ACO: deposit extra pheromone where food was collected
        # This is in addition to the standard field writes in step()
        # Agents that collected food deposit extra pheromone
        if np.any(newly_collected_food):
            # Find which agents are near newly collected food
            food_collectors = np.zeros(max_agents, dtype=bool)
            agent_pos_np = np.array(state.agent_positions)
            food_pos_np = np.array(state.food_positions)

            for f_idx in range(len(newly_collected_food)):
                if newly_collected_food[f_idx]:
                    # Find agents within collection distance of this food
                    f_pos = food_pos_np[f_idx]
                    for a_idx in range(max_agents):
                        if state.agent_alive[a_idx]:
                            a_pos = agent_pos_np[a_idx]
                            if max(abs(a_pos[0] - f_pos[0]), abs(a_pos[1] - f_pos[1])) <= 1:
                                food_collectors[a_idx] = True

            # Deposit extra pheromone for collectors
            # This reinforces the path that led to food
            if np.any(food_collectors):
                extra_deposit = jnp.zeros(
                    (max_agents, config.field.num_channels),
                    dtype=jnp.float32,
                )
                extra_deposit = extra_deposit.at[food_collectors].set(ACO_Q * 2.0)
                new_field = write_local(
                    state.field_state,
                    state.agent_positions,
                    extra_deposit,
                )
                # Update state with extra pheromone
                state = state.replace(field_state=new_field)  # type: ignore[attr-defined]

        # Accumulate metrics
        rewards_np = np.array(rewards)
        total_reward += float(np.sum(rewards_np))
        total_food += float(info["food_collected_this_step"])
        per_agent_rewards[:num_agents] += rewards_np[:num_agents]

        if done:
            break

    # Final population
    final_population = int(np.sum(np.array(state.agent_alive)))

    return {
        "total_reward": total_reward,
        "food_collected": total_food,
        "final_population": final_population,
        "per_agent_rewards": per_agent_rewards.tolist(),
    }


def run_aco_hybrid_episode(
    network: ActorCritic,
    params: Any,
    config: Config,
    key: jax.Array,
    deterministic: bool = False,
) -> dict[str, Any]:
    """Run a single episode using ACO-Hybrid (NN movement, hardcoded writes).

    Agents use:
    - Movement: Neural network decides actions
    - Field writes: Hardcoded ACO rules (deposit on food collection)

    This isolates the value of LEARNING the write behavior.

    Args:
        network: ActorCritic network module.
        params: Network parameters (shared across all agents).
        config: Configuration (should be from aco_config()).
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
    max_agents = config.evolution.max_agents
    per_agent_rewards = np.zeros(num_agents)

    # Track which agents collected food last step
    prev_food_collected = np.array(state.food_collected)

    # Run episode
    for _ in range(config.env.max_steps):
        # Get observations (add batch dimension for single env)
        obs = get_observations(state, config)
        obs = obs[None, :, :]  # (1, max_agents, obs_dim)

        # Use neural network for movement decisions
        key, action_key = jax.random.split(key)
        if deterministic:
            from src.agents.policy import get_deterministic_actions
            actions = get_deterministic_actions(network, params, obs)
        else:
            actions, _, _, _ = sample_actions(network, params, obs, action_key)

        # Remove batch dimension
        actions = actions[0]  # (max_agents,)

        # Record food state before step
        prev_food_collected = np.array(state.food_collected)

        # Step environment
        state, rewards, done, info = step(state, actions, config)

        # Determine which agents collected food this step
        curr_food_collected = np.array(state.food_collected)
        newly_collected_food = curr_food_collected & ~prev_food_collected

        # ACO-style pheromone deposit: extra deposit when food is collected
        if np.any(newly_collected_food):
            food_collectors = np.zeros(max_agents, dtype=bool)
            agent_pos_np = np.array(state.agent_positions)
            food_pos_np = np.array(state.food_positions)

            for f_idx in range(len(newly_collected_food)):
                if newly_collected_food[f_idx]:
                    f_pos = food_pos_np[f_idx]
                    for a_idx in range(max_agents):
                        if state.agent_alive[a_idx]:
                            a_pos = agent_pos_np[a_idx]
                            if max(abs(a_pos[0] - f_pos[0]), abs(a_pos[1] - f_pos[1])) <= 1:
                                food_collectors[a_idx] = True

            if np.any(food_collectors):
                extra_deposit = jnp.zeros(
                    (max_agents, config.field.num_channels),
                    dtype=jnp.float32,
                )
                extra_deposit = extra_deposit.at[food_collectors].set(ACO_Q * 2.0)
                new_field = write_local(
                    state.field_state,
                    state.agent_positions,
                    extra_deposit,
                )
                state = state.replace(field_state=new_field)  # type: ignore[attr-defined]

        # Accumulate metrics
        rewards_np = np.array(rewards)
        total_reward += float(np.sum(rewards_np))
        total_food += float(info["food_collected_this_step"])
        per_agent_rewards[:num_agents] += rewards_np[:num_agents]

        if done:
            break

    final_population = int(np.sum(np.array(state.agent_alive)))

    return {
        "total_reward": total_reward,
        "food_collected": total_food,
        "final_population": final_population,
        "per_agent_rewards": per_agent_rewards.tolist(),
    }


def evaluate_aco_fixed(
    config: Config,
    n_episodes: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate ACO-Fixed over multiple episodes.

    Args:
        config: Configuration (should be from aco_config()).
        n_episodes: Number of episodes to run.
        seed: Random seed for reproducibility.

    Returns:
        Aggregated result dict with statistics.
    """
    key = jax.random.PRNGKey(seed)

    episode_rewards = []
    episode_food = []
    episode_populations = []
    all_per_agent_rewards = []

    for _ in range(n_episodes):
        key, episode_key = jax.random.split(key)
        result = run_aco_fixed_episode(config, episode_key)
        episode_rewards.append(result["total_reward"])
        episode_food.append(result["food_collected"])
        episode_populations.append(result["final_population"])
        all_per_agent_rewards.append(result["per_agent_rewards"])

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


def evaluate_aco_hybrid(
    network: ActorCritic,
    params: Any,
    config: Config,
    n_episodes: int,
    seed: int,
    deterministic: bool = False,
) -> dict[str, Any]:
    """Evaluate ACO-Hybrid over multiple episodes.

    Args:
        network: ActorCritic network module.
        params: Network parameters (shared across all agents).
        config: Configuration (should be from aco_config()).
        n_episodes: Number of episodes to run.
        seed: Random seed for reproducibility.
        deterministic: If True, use greedy actions instead of sampling.

    Returns:
        Aggregated result dict with statistics.
    """
    key = jax.random.PRNGKey(seed)

    episode_rewards = []
    episode_food = []
    episode_populations = []
    all_per_agent_rewards = []

    for _ in range(n_episodes):
        key, episode_key = jax.random.split(key)
        result = run_aco_hybrid_episode(
            network, params, config, episode_key, deterministic=deterministic
        )
        episode_rewards.append(result["total_reward"])
        episode_food.append(result["food_collected"])
        episode_populations.append(result["final_population"])
        all_per_agent_rewards.append(result["per_agent_rewards"])

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


def create_aco_hybrid_network(config: Config) -> ActorCritic:
    """Create an ActorCritic network for ACO-Hybrid.

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


def init_aco_hybrid_params(
    network: ActorCritic,
    config: Config,
    key: jax.Array,
) -> Any:
    """Initialize ACO-Hybrid network parameters.

    Args:
        network: ActorCritic network module.
        config: Configuration object.
        key: JAX PRNG key for initialization.

    Returns:
        Initialized network parameters.
    """
    observation_dim = obs_dim(config)
    dummy_obs = jnp.zeros((observation_dim,))
    params = network.init(key, dummy_obs)
    return params
