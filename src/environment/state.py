"""EnvState dataclass for tracking simulation state."""

import flax.struct
import jax
import jax.numpy as jnp

from src.field.field import FieldState, create_field


@flax.struct.dataclass
class EnvState:
    """Full environment state for the emergence lab simulation.

    Attributes:
        agent_positions: Integer positions of agents, shape (num_agents, 2) as (row, col).
        food_positions: Integer positions of food items, shape (num_food, 2) as (row, col).
        food_collected: Boolean mask of collected food, shape (num_food,).
        field_state: The shared field that agents read/write.
        step: Current timestep in the episode.
        key: JAX PRNG key for stochastic operations.
    """
    agent_positions: jnp.ndarray   # (num_agents, 2)
    food_positions: jnp.ndarray    # (num_food, 2)
    food_collected: jnp.ndarray    # (num_food,) bool
    field_state: FieldState
    step: jnp.ndarray              # scalar int
    key: jax.Array                 # PRNG key


def create_env_state(key: jax.Array, config: "src.configs.Config") -> EnvState:  # type: ignore[name-defined]
    """Create an initial EnvState from config.

    Agent and food positions are randomly placed on the grid.
    The field is initialized to zeros.

    Args:
        key: JAX PRNG key.
        config: Master configuration object.

    Returns:
        A freshly initialized EnvState.
    """
    from src.configs import Config  # local import to avoid circular

    k1, k2, k3 = jax.random.split(key, 3)

    grid_size = config.env.grid_size
    num_agents = config.env.num_agents
    num_food = config.env.num_food

    # Random agent positions (row, col) within grid
    agent_positions = jax.random.randint(
        k1, shape=(num_agents, 2), minval=0, maxval=grid_size
    )

    # Random food positions (row, col) within grid
    food_positions = jax.random.randint(
        k2, shape=(num_food, 2), minval=0, maxval=grid_size
    )

    # No food collected yet
    food_collected = jnp.zeros((num_food,), dtype=jnp.bool_)

    # Fresh field
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
