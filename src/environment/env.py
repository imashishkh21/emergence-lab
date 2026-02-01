"""Environment reset and step functions."""

from typing import Any

import jax
import jax.numpy as jnp

from src.configs import Config
from src.environment.state import EnvState
from src.field.field import create_field
from src.field.dynamics import step_field
from src.field.ops import write_local


def reset(key: jax.Array, config: Config) -> EnvState:
    """Create a fresh environment state for a new episode.

    Agent positions are random and non-overlapping. Food positions are random.
    The field is initialized to zeros. Evolution fields are initialized with
    the first num_agents slots active.

    Args:
        key: JAX PRNG key.
        config: Master configuration object.

    Returns:
        A freshly initialized EnvState.
    """
    k1, k2, k3 = jax.random.split(key, 3)

    grid_size = config.env.grid_size
    num_agents = config.env.num_agents
    num_food = config.env.num_food
    max_agents = config.evolution.max_agents

    # Generate non-overlapping agent positions by sampling from all grid cells.
    total_cells = grid_size * grid_size
    # Random permutation of cell indices, take first num_agents
    perm = jax.random.permutation(k1, total_cells)
    agent_indices = perm[:num_agents]
    agent_rows = agent_indices // grid_size
    agent_cols = agent_indices % grid_size
    active_positions = jnp.stack([agent_rows, agent_cols], axis=-1)

    # Pad agent_positions to (max_agents, 2) — dead slots get (0, 0)
    agent_positions = jnp.zeros((max_agents, 2), dtype=jnp.int32)
    agent_positions = agent_positions.at[:num_agents].set(active_positions)

    # Random food positions (may overlap with each other or agents — that's fine)
    food_positions = jax.random.randint(
        k2, shape=(num_food, 2), minval=0, maxval=grid_size
    )

    # No food collected yet
    food_collected = jnp.zeros((num_food,), dtype=jnp.bool_)

    # Fresh field initialized to zeros
    field_state = create_field(
        height=grid_size,
        width=grid_size,
        channels=config.field.num_channels,
    )

    # Evolution fields
    agent_energy = jnp.zeros((max_agents,), dtype=jnp.float32)
    agent_energy = agent_energy.at[:num_agents].set(
        float(config.evolution.starting_energy)
    )

    agent_alive = jnp.zeros((max_agents,), dtype=jnp.bool_)
    agent_alive = agent_alive.at[:num_agents].set(True)

    agent_ids = jnp.full((max_agents,), -1, dtype=jnp.int32)
    agent_ids = agent_ids.at[:num_agents].set(jnp.arange(num_agents, dtype=jnp.int32))

    agent_parent_ids = jnp.full((max_agents,), -1, dtype=jnp.int32)

    next_agent_id = jnp.int32(num_agents)

    return EnvState(
        agent_positions=agent_positions,
        food_positions=food_positions,
        food_collected=food_collected,
        field_state=field_state,
        step=jnp.int32(0),
        key=k3,
        agent_energy=agent_energy,
        agent_alive=agent_alive,
        agent_ids=agent_ids,
        agent_parent_ids=agent_parent_ids,
        next_agent_id=next_agent_id,
    )


def step(
    state: EnvState, actions: jnp.ndarray, config: Config
) -> tuple[EnvState, jnp.ndarray, jnp.ndarray, dict[str, Any]]:
    """Advance the environment by one timestep.

    Args:
        state: Current environment state.
        actions: Integer actions for each agent, shape (num_agents,) or (max_agents,).
            0=stay, 1=up, 2=down, 3=left, 4=right.
        config: Master configuration object.

    Returns:
        Tuple of (new_state, rewards, dones, info).
    """
    grid_size = config.env.grid_size
    max_agents = config.evolution.max_agents

    # Pad actions to (max_agents,) if needed — dead/inactive slots get action 0 (stay)
    padded_actions = jnp.zeros((max_agents,), dtype=jnp.int32)
    padded_actions = padded_actions.at[: actions.shape[0]].set(actions)

    # --- 1. Move agents ---
    # Action deltas: 0=stay, 1=up(-row), 2=down(+row), 3=left(-col), 4=right(+col)
    action_deltas = jnp.array([
        [0, 0],   # stay
        [-1, 0],  # up
        [1, 0],   # down
        [0, -1],  # left
        [0, 1],   # right
    ], dtype=jnp.int32)

    deltas = action_deltas[padded_actions]  # (max_agents, 2)
    new_positions = state.agent_positions + deltas

    # Clip to grid boundaries
    new_positions = jnp.clip(new_positions, 0, grid_size - 1)

    # Dead agents stay at their position (masked out)
    new_positions = jnp.where(
        state.agent_alive[:, None], new_positions, state.agent_positions
    )

    # --- 2. Food collection ---
    # Only alive agents can collect food
    agent_rows = new_positions[:, 0:1]  # (max_agents, 1)
    agent_cols = new_positions[:, 1:2]  # (max_agents, 1)
    food_rows = state.food_positions[:, 0]  # (F,)
    food_cols = state.food_positions[:, 1]  # (F,)

    row_dist = jnp.abs(agent_rows - food_rows[None, :])  # (max_agents, F)
    col_dist = jnp.abs(agent_cols - food_cols[None, :])  # (max_agents, F)
    chebyshev_dist = jnp.maximum(row_dist, col_dist)  # (max_agents, F)

    # Food is collectible if within 1 cell AND not already collected AND agent alive
    within_range = chebyshev_dist <= 1  # (max_agents, F)
    not_collected = ~state.food_collected  # (F,)
    alive_mask = state.agent_alive[:, None]  # (max_agents, 1)
    collectible = within_range & not_collected[None, :] & alive_mask  # (max_agents, F)

    # Any alive agent adjacent to uncollected food collects it
    newly_collected = jnp.any(collectible, axis=0)  # (F,)
    food_collected = state.food_collected | newly_collected

    # --- 3. Compute reward ---
    # +1 per food collected this step, shared across team
    num_collected = jnp.sum(newly_collected.astype(jnp.float32))
    rewards = jnp.full((config.env.num_agents,), num_collected)

    # --- 4. Update field ---
    # Step field dynamics (diffuse + decay)
    field_state = step_field(
        state.field_state,
        diffusion_rate=config.field.diffusion_rate,
        decay_rate=config.field.decay_rate,
    )

    # Only alive agents write their presence to the field
    write_values = jnp.ones(
        (max_agents, config.field.num_channels),
        dtype=jnp.float32,
    ) * config.field.write_strength
    # Zero out write values for dead agents
    write_values = write_values * state.agent_alive[:, None]
    field_state = write_local(field_state, new_positions, write_values)

    # --- 5. Energy drain ---
    # Subtract energy_per_step from alive agents, clamp to 0
    energy_drain = jnp.where(
        state.agent_alive,
        jnp.maximum(state.agent_energy - config.evolution.energy_per_step, 0.0),
        state.agent_energy,
    )
    new_energy = energy_drain

    # --- 6. Advance step counter and check done ---
    new_step = state.step + 1
    done = new_step >= config.env.max_steps

    # --- 7. Split PRNG key ---
    new_key, _ = jax.random.split(state.key)

    new_state = EnvState(
        agent_positions=new_positions,
        food_positions=state.food_positions,
        food_collected=food_collected,
        field_state=field_state,
        step=new_step,
        key=new_key,
        agent_energy=new_energy,
        agent_alive=state.agent_alive,
        agent_ids=state.agent_ids,
        agent_parent_ids=state.agent_parent_ids,
        next_agent_id=state.next_agent_id,
    )

    info: dict[str, Any] = {
        "food_collected_this_step": num_collected,
        "total_food_collected": jnp.sum(food_collected.astype(jnp.float32)),
    }

    return new_state, rewards, done, info
