"""Observation function for agents — biological pheromone system."""

import jax
import jax.numpy as jnp

from src.configs import Config
from src.environment.state import EnvState

# Number of nearest food items each agent observes.
_K_NEAREST_FOOD = 5


def obs_dim(config: Config) -> int:
    """Compute the observation dimension for a single agent.

    Components:
        - own position (2): normalized row, col
        - own energy (1): normalized to [0, 1]
        - has_food flag (1): 0 or 1
        - nest compass (2): noisy direction to nest center
        - field spatial (spatial_mult * num_channels): 3x3 patch or 5-point cross
        - field temporal (num_channels): dC/dt per channel
        - relative food positions (K * 3): dx, dy, available per food

    Returns:
        Total observation size per agent.
    """
    num_ch = config.field.num_channels
    spatial_mult = 9 if config.field.field_spatial_patch else 5
    food_dim = _K_NEAREST_FOOD * 3
    return 2 + 1 + 1 + 2 + (spatial_mult * num_ch) + num_ch + food_dim


def get_observations(state: EnvState, config: Config) -> jnp.ndarray:
    """Build observations for all agents (including dead slots).

    Each alive agent observes:
        1. Own position normalized to [-1, 1].
        2. Own energy normalized to [0, 1].
        3. Has food flag (0 or 1).
        4. Noisy compass pointing toward nest center.
        5. Field spatial gradients (3x3 patch or N, S, E, W, center per channel).
        6. Field temporal derivative (change since last step per channel).
        7. K nearest uncollected food items (dx, dy, available).

    Dead agents receive all-zero observations.

    Args:
        state: Current environment state.
        config: Master configuration.

    Returns:
        Observations array of shape (max_agents, obs_dim).
    """
    grid_size = config.env.grid_size
    max_energy = jnp.float32(config.evolution.max_energy)

    all_positions = state.agent_positions  # (max_agents, 2)

    # 1. Normalized position: [0, grid_size-1] -> [-1, 1]
    norm_pos = (all_positions.astype(jnp.float32) / (grid_size - 1)) * 2.0 - 1.0

    # 2. Normalized energy: [0, max_energy] -> [0, 1]
    norm_energy = jnp.clip(state.agent_energy / max_energy, 0.0, 1.0)[:, None]

    # 3. Has food flag
    has_food_obs = state.has_food.astype(jnp.float32)[:, None]

    # 4. Nest compass with distance-dependent noise
    compass = _compute_compass(state, config)  # (max_agents, 2)

    # 5. Field spatial: 3x3 patch or 5-point cross per channel
    field_spatial = _compute_field_spatial(state, config)

    # 6. Field temporal derivative
    center_values = state.field_state.values[
        all_positions[:, 0], all_positions[:, 1]
    ]  # (max_agents, C)
    field_temporal = center_values - state.prev_field_at_pos  # (max_agents, C)

    # 7. Food observations: exact positions, passive odor, nest-only compass, or nothing
    if config.env.nest_only_compass:
        nest_food_dir = _compute_nest_food_compass(state, config)  # (max_agents, 2)
        # Pad to 15 dims to maintain food_dim = K*3
        food_obs = jnp.concatenate(
            [nest_food_dir, jnp.zeros((config.evolution.max_agents, _K_NEAREST_FOOD * 3 - 2))],
            axis=-1,
        )  # (max_agents, K*3)
    elif config.env.food_obs_enabled:
        food_obs = _compute_food_obs(state, config)  # (max_agents, K*3)
    elif config.env.food_odor_enabled:
        odor = _compute_food_odor(state, config)  # (max_agents, 5)
        # Pad to 15 dims to maintain obs_dim
        food_obs = jnp.concatenate(
            [odor, jnp.zeros((config.evolution.max_agents, _K_NEAREST_FOOD * 3 - 5))],
            axis=-1,
        )  # (max_agents, K*3)
    else:
        food_obs = jnp.zeros(
            (config.evolution.max_agents, _K_NEAREST_FOOD * 3)
        )  # (max_agents, K*3)

    obs = jnp.concatenate(
        [norm_pos, norm_energy, has_food_obs, compass,
         field_spatial, field_temporal, food_obs],
        axis=-1,
    )

    # Zero out observations for dead agents
    alive_mask = state.agent_alive[:, None]
    obs = jnp.where(alive_mask, obs, 0.0)
    return obs


def _compute_compass(state: EnvState, config: Config) -> jnp.ndarray:
    """Compute noisy compass pointing toward nest center.

    Noise increases with distance from nest (path integration error).
    When compass_cutoff_radius > 0, agents beyond that Chebyshev distance
    from nest receive a random unit vector instead of the true direction.
    Uses jax.random.fold_in(state.key, state.step) for deterministic
    per-step noise without consuming or mutating the PRNG key.

    Returns:
        Array of shape (max_agents, 2) with compass values in [-1, 1].
    """
    grid_size = config.env.grid_size
    nest_center = jnp.array([grid_size // 2, grid_size // 2], dtype=jnp.float32)
    nest_center_int = jnp.array([grid_size // 2, grid_size // 2], dtype=jnp.int32)
    agent_pos = state.agent_positions.astype(jnp.float32)

    # True direction to nest, normalized by grid_size
    true_delta = (nest_center[None, :] - agent_pos) / grid_size  # roughly [-1, 1]

    # Distance-dependent noise
    dist = jnp.sqrt(
        jnp.sum((nest_center[None, :] - agent_pos) ** 2, axis=-1, keepdims=True)
    )
    noise_std = config.nest.compass_noise_rate * dist / grid_size

    # Deterministic per-step noise key
    noise_key = jax.random.fold_in(state.key, state.step)
    noise = jax.random.normal(noise_key, shape=true_delta.shape) * noise_std
    compass = jnp.clip(true_delta + noise, -1.0, 1.0)

    if config.nest.compass_cutoff_radius > 0:
        # Chebyshev distance to nest center
        cheby_dist = jnp.max(
            jnp.abs(state.agent_positions - nest_center_int[None, :]), axis=-1
        )  # (max_agents,)
        beyond_cutoff = cheby_dist > config.nest.compass_cutoff_radius  # (max_agents,)

        # Random unit vector for agents beyond cutoff
        rand_key = jax.random.fold_in(state.key, state.step + 1000000)
        rand_angles = jax.random.uniform(
            rand_key, shape=(agent_pos.shape[0],), minval=0.0, maxval=2.0 * jnp.pi
        )
        rand_compass = jnp.stack(
            [jnp.cos(rand_angles), jnp.sin(rand_angles)], axis=-1
        )  # (max_agents, 2)
        # Normalize to similar magnitude as true compass
        rand_compass = rand_compass / grid_size

        compass = jnp.where(beyond_cutoff[:, None], rand_compass, compass)

    return compass


def _compute_field_spatial(state: EnvState, config: Config) -> jnp.ndarray:
    """Compute field spatial features per channel.

    When config.field.field_spatial_patch is True, returns a 3x3 patch
    (9 values per channel). Otherwise returns the 5-point cross
    (N, S, E, W, center per channel).

    Returns:
        Array of shape (max_agents, spatial_mult * num_channels)
        where spatial_mult is 9 (patch) or 5 (cross).
    """
    h, w, c = state.field_state.values.shape
    pos = state.agent_positions
    rows, cols = pos[:, 0], pos[:, 1]

    if config.field.field_spatial_patch:
        # 3x3 patch: 9 positions per agent
        # Row offsets: -1, 0, +1; Col offsets: -1, 0, +1
        # Layout: row-major (top-left to bottom-right), channels contiguous per position
        patches = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r = jnp.clip(rows + dr, 0, h - 1)
                c_idx = jnp.clip(cols + dc, 0, w - 1)
                patches.append(state.field_state.values[r, c_idx])  # (A, C)

        # Stack: (A, 9, C) -> reshape to (A, 9*C)
        spatial = jnp.stack(patches, axis=1)
        return spatial.reshape(pos.shape[0], 9 * c)
    else:
        # 5-point cross: center + 4 cardinal neighbors
        center = state.field_state.values[rows, cols]  # (A, C)

        north = state.field_state.values[jnp.clip(rows - 1, 0, h - 1), cols]
        south = state.field_state.values[jnp.clip(rows + 1, 0, h - 1), cols]
        west = state.field_state.values[rows, jnp.clip(cols - 1, 0, w - 1)]
        east = state.field_state.values[rows, jnp.clip(cols + 1, 0, w - 1)]

        # Stack: (A, 5, C) -> flatten to (A, 5*C)
        spatial = jnp.stack([north, south, east, west, center], axis=1)
        return spatial.reshape(pos.shape[0], 5 * c)


def _compute_nest_food_compass(state: EnvState, config: Config) -> jnp.ndarray:
    """Compute direction to nearest uncollected food, only when inside nest.

    Agents inside the nest area (Chebyshev distance to center <= radius)
    receive a unit vector pointing toward the nearest uncollected food.
    Agents outside the nest receive (0, 0), forcing reliance on the
    pheromone field for navigation.

    Returns:
        Array of shape (max_agents, 2) with direction values in [-1, 1].
    """
    grid_size = config.env.grid_size
    max_agents = config.evolution.max_agents
    nest_center = jnp.array([grid_size // 2, grid_size // 2], dtype=jnp.int32)

    agent_pos = state.agent_positions  # (max_agents, 2)

    # Chebyshev distance to nest center
    nest_dist = jnp.max(jnp.abs(agent_pos - nest_center[None, :]), axis=-1)  # (A,)
    in_nest = nest_dist <= config.nest.radius  # (A,) bool

    # Compute direction to nearest uncollected food
    agent_pos_f = agent_pos.astype(jnp.float32)  # (A, 2)
    food_pos_f = state.food_positions.astype(jnp.float32)  # (F, 2)

    # Relative positions: (A, F, 2)
    rel_pos = food_pos_f[None, :, :] - agent_pos_f[:, None, :]

    # Euclidean distances: (A, F)
    distances = jnp.sqrt(jnp.sum(rel_pos ** 2, axis=-1) + 1e-8)

    # Mask out collected food with large distance
    large_dist = jnp.float32(grid_size * 3)
    distances = jnp.where(state.food_collected[None, :], large_dist, distances)

    # Index of nearest food per agent
    nearest_idx = jnp.argmin(distances, axis=-1)  # (A,)
    nearest_dist = distances[jnp.arange(max_agents), nearest_idx]  # (A,)

    # Direction vector to nearest food
    nearest_rel = rel_pos[jnp.arange(max_agents), nearest_idx]  # (A, 2)

    # Normalize to unit vector (avoid div by zero)
    norm = jnp.sqrt(jnp.sum(nearest_rel ** 2, axis=-1, keepdims=True) + 1e-8)
    direction = nearest_rel / norm  # (A, 2), unit vector

    # Zero out if no uncollected food exists
    has_food_available = (nearest_dist < large_dist)  # (A,)
    direction = jnp.where(has_food_available[:, None], direction, 0.0)

    # Zero out if agent is outside nest
    direction = jnp.where(in_nest[:, None], direction, 0.0)

    return direction


def _compute_food_odor(state: EnvState, config: Config) -> jnp.ndarray:
    """Compute passive food odor at each agent's position.

    Each uncollected food emits exp(-dist/lambda). Agents sense total odor
    at center + N/S/E/W positions (5 scalar values). This gives a vague
    sense of "food is nearby" without revealing exact positions.

    Returns:
        Array of shape (max_agents, 5).
    """
    grid_size = config.env.grid_size
    lam = config.env.food_odor_lambda

    pos = state.agent_positions
    rows, cols = pos[:, 0], pos[:, 1]

    # 5 sample points per agent: center, N, S, E, W
    sample_rows = jnp.stack([
        rows,
        jnp.clip(rows - 1, 0, grid_size - 1),
        jnp.clip(rows + 1, 0, grid_size - 1),
        rows,
        rows,
    ], axis=-1)  # (A, 5)
    sample_cols = jnp.stack([
        cols,
        cols,
        cols,
        jnp.clip(cols + 1, 0, grid_size - 1),
        jnp.clip(cols - 1, 0, grid_size - 1),
    ], axis=-1)  # (A, 5)

    food_pos_f = state.food_positions.astype(jnp.float32)  # (F, 2)
    collected = state.food_collected  # (F,)

    # Distances: (A, 5, 1) vs (1, 1, F) -> (A, 5, F)
    dr = sample_rows[:, :, None].astype(jnp.float32) - food_pos_f[None, None, :, 0]
    dc = sample_cols[:, :, None].astype(jnp.float32) - food_pos_f[None, None, :, 1]
    dist = jnp.sqrt(dr ** 2 + dc ** 2)  # (A, 5, F)

    odor_per_food = jnp.exp(-dist / lam)  # (A, 5, F)

    # Zero out collected food
    odor_per_food = jnp.where(collected[None, None, :], 0.0, odor_per_food)

    # Sum over food sources
    odor = jnp.sum(odor_per_food, axis=-1)  # (A, 5)

    # Normalize to roughly [0, 1]
    odor = odor / max(1.0, float(config.env.num_food))

    return odor


def _compute_food_obs(state: EnvState, config: Config) -> jnp.ndarray:
    """Compute relative food observations for all agents.

    For each agent, find the K nearest uncollected food items and return
    their relative positions (normalized to [-1, 1]) and an availability
    flag (1.0 if food exists, 0.0 for padding).

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
    rel_pos_norm = rel_pos / grid_size
    rel_pos_norm = jnp.clip(rel_pos_norm, -1.0, 1.0)

    # Distances for sorting (Manhattan distance)
    distances = jnp.sum(jnp.abs(rel_pos), axis=-1)  # (max_agents, F)

    # Mask out collected food by setting their distance to a large value
    collected_mask = state.food_collected  # (F,)
    large_dist = jnp.float32(grid_size * 2 + 1)
    distances = jnp.where(collected_mask[None, :], large_dist, distances)

    # Get indices of K nearest food items per agent
    # Handle case where num_food < K by padding distances
    num_food = state.food_positions.shape[0]
    if num_food < k:
        pad_size = k - num_food
        distances = jnp.concatenate(
            [distances, jnp.full((max_agents, pad_size), large_dist)], axis=-1
        )
        rel_pos_norm = jnp.concatenate(
            [rel_pos_norm, jnp.zeros((max_agents, pad_size, 2))], axis=1
        )

    sorted_indices = jnp.argsort(distances, axis=-1)[:, :k]  # (max_agents, K)

    # Gather the relative positions and distances for sorted food
    agent_idx = jnp.arange(max_agents)[:, None]
    nearest_rel = rel_pos_norm[agent_idx, sorted_indices]  # (max_agents, K, 2)
    nearest_dist = distances[agent_idx, sorted_indices]     # (max_agents, K)

    # Availability flag: 1.0 if distance < large_dist, else 0.0
    available = (nearest_dist < large_dist).astype(jnp.float32)

    # Zero out positions for unavailable food
    nearest_rel = nearest_rel * available[:, :, None]

    # Concatenate: (dx, dy, available) per food slot -> (max_agents, K, 3)
    food_features = jnp.concatenate(
        [nearest_rel, available[:, :, None]], axis=-1
    )

    return food_features.reshape(max_agents, k * 3)
