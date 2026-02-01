"""Environment reset and step functions."""

import jax
import jax.numpy as jnp

from src.configs import Config
from src.environment.state import EnvState
from src.field.field import create_field


def reset(key: jax.Array, config: Config) -> EnvState:
    """Create a fresh environment state for a new episode.

    Agent positions are random and non-overlapping. Food positions are random.
    The field is initialized to zeros.

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

    # Generate non-overlapping agent positions by sampling from all grid cells.
    total_cells = grid_size * grid_size
    # Random permutation of cell indices, take first num_agents
    perm = jax.random.permutation(k1, total_cells)
    agent_indices = perm[:num_agents]
    agent_rows = agent_indices // grid_size
    agent_cols = agent_indices % grid_size
    agent_positions = jnp.stack([agent_rows, agent_cols], axis=-1)

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

    return EnvState(
        agent_positions=agent_positions,
        food_positions=food_positions,
        food_collected=food_collected,
        field_state=field_state,
        step=jnp.int32(0),
        key=k3,
    )
