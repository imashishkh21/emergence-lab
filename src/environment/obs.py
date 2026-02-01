"""Observation function for agents."""

import jax
import jax.numpy as jnp

from src.configs import Config
from src.environment.state import EnvState
from src.field.ops import read_local


# Number of nearest food items each agent observes.
_K_NEAREST_FOOD = 5


def obs_dim(config: Config) -> int:
    """Compute the observation dimension for a single agent.

    Components:
        - own position (2): normalized row, col
        - own energy (1): normalized to [0, 1] by max_energy
        - local field values: (2*obs_radius+1)^2 * num_channels
        - relative food positions (K_NEAREST_FOOD * 3): dx, dy, available flag per food

    Returns:
        Total observation size per agent.
    """
    radius = config.env.observation_radius
    patch_size = (2 * radius + 1) ** 2
    field_dim = patch_size * config.field.num_channels
    food_dim = _K_NEAREST_FOOD * 3  # (dx, dy, available) per food slot
    return 3 + field_dim + food_dim


def get_observations(state: EnvState, config: Config) -> jnp.ndarray:
    """Build observations for all agents (including dead slots).

    Each alive agent observes:
        1. Own position normalized to [-1, 1].
        2. Own energy normalized to [0, 1] by max_energy.
        3. Local field values within observation_radius (flattened).
        4. Relative positions of K nearest uncollected food items,
           normalized by grid_size to [-1, 1]. If fewer than K food items
           are visible/available, remaining slots are zeroed out with a 0
           availability flag.

    Dead agents receive all-zero observations.

    Args:
        state: Current environment state.
        config: Master configuration.

    Returns:
        Observations array of shape (max_agents, obs_dim).
    """
    grid_size = config.env.grid_size
    max_agents = config.evolution.max_agents
    radius = config.env.observation_radius
    max_energy = jnp.float32(config.evolution.max_energy)

    all_positions = state.agent_positions  # (max_agents, 2)

    # --- 1. Normalized own position ---
    # Map [0, grid_size-1] -> [-1, 1]
    norm_pos = (all_positions.astype(jnp.float32) / (grid_size - 1)) * 2.0 - 1.0  # (max_agents, 2)

    # --- 2. Normalized own energy ---
    # Map [0, max_energy] -> [0, 1]
    norm_energy = jnp.clip(state.agent_energy / max_energy, 0.0, 1.0)  # (max_agents,)
    norm_energy = norm_energy[:, None]  # (max_agents, 1)

    # --- 3. Local field values ---
    field_obs = read_local(state.field_state, all_positions, radius)  # (max_agents, patch*C)
    # Clamp field observations to [-1, 1]
    field_obs = jnp.clip(field_obs, -1.0, 1.0)

    # --- 4. Relative food positions (K nearest uncollected) ---
    food_obs = _compute_food_obs(state, config)  # (max_agents, K*3)

    # Concatenate all components
    obs = jnp.concatenate([norm_pos, norm_energy, field_obs, food_obs], axis=-1)  # (max_agents, obs_dim)

    # Zero out observations for dead agents
    alive_mask = state.agent_alive[:, None]  # (max_agents, 1)
    obs = jnp.where(alive_mask, obs, 0.0)

    return obs


def _compute_food_obs(state: EnvState, config: Config) -> jnp.ndarray:
    """Compute relative food observations for all agents.

    For each agent, find the K nearest uncollected food items and return
    their relative positions (normalized to [-1, 1]) and an availability
    flag (1.0 if food exists, 0.0 for padding).

    Dead agents will have their food observations zeroed out by the caller
    (get_observations applies the alive mask after concatenation).

    Returns:
        Array of shape (max_agents, K_NEAREST_FOOD * 3).
    """
    grid_size = config.env.grid_size
    max_agents = config.evolution.max_agents
    k = _K_NEAREST_FOOD

    agent_pos_f = state.agent_positions.astype(jnp.float32)  # (max_agents, 2)
    food_pos_f = state.food_positions.astype(jnp.float32)    # (F, 2)

    # (max_agents, 1, 2) - (1, F, 2) -> (max_agents, F, 2)
    rel_pos = food_pos_f[None, :, :] - agent_pos_f[:, None, :]

    # Normalize relative positions to [-1, 1] by grid_size
    rel_pos_norm = rel_pos / grid_size  # values in roughly [-1, 1]
    rel_pos_norm = jnp.clip(rel_pos_norm, -1.0, 1.0)

    # Distances for sorting (Manhattan distance)
    distances = jnp.sum(jnp.abs(rel_pos), axis=-1)  # (max_agents, F)

    # Mask out collected food by setting their distance to a large value
    collected_mask = state.food_collected  # (F,)
    large_dist = jnp.float32(grid_size * 2 + 1)
    distances = jnp.where(collected_mask[None, :], large_dist, distances)

    # Check if food is within observation radius (using Chebyshev distance)
    obs_radius = config.env.observation_radius
    chebyshev = jnp.max(jnp.abs(rel_pos), axis=-1)  # (max_agents, F)
    out_of_range = chebyshev > obs_radius
    distances = jnp.where(out_of_range, large_dist, distances)

    # Get indices of K nearest food items per agent
    # argsort along food axis, take first K
    sorted_indices = jnp.argsort(distances, axis=-1)[:, :k]  # (max_agents, K)

    # Gather the relative positions and distances for sorted food
    # Use advanced indexing: for each agent i, gather food sorted_indices[i]
    agent_idx = jnp.arange(max_agents)[:, None]  # (max_agents, 1)
    nearest_rel = rel_pos_norm[agent_idx, sorted_indices]  # (max_agents, K, 2)
    nearest_dist = distances[agent_idx, sorted_indices]     # (max_agents, K)

    # Availability flag: 1.0 if distance < large_dist, else 0.0
    available = (nearest_dist < large_dist).astype(jnp.float32)  # (max_agents, K)

    # Zero out positions for unavailable food
    nearest_rel = nearest_rel * available[:, :, None]

    # Concatenate: (dx, dy, available) per food slot -> (max_agents, K, 3)
    food_features = jnp.concatenate(
        [nearest_rel, available[:, :, None]], axis=-1
    )  # (max_agents, K, 3)

    # Flatten to (max_agents, K * 3)
    return food_features.reshape(max_agents, k * 3)
