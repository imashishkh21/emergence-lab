"""EnvState dataclass for tracking simulation state."""

from typing import Any

import flax.struct
import jax
import jax.numpy as jnp

from src.field.field import FieldState, create_field


@flax.struct.dataclass
class EnvState:
    """Full environment state for the emergence lab simulation.

    Attributes:
        agent_positions: Integer positions of agents, shape (max_agents, 2) as (row, col).
            Dead agents are padded with (0, 0).
        food_positions: Integer positions of food items, shape (num_food, 2) as (row, col).
        food_collected: Boolean mask of collected food, shape (num_food,).
        field_state: The shared field that agents read/write.
        step: Current timestep in the episode.
        key: JAX PRNG key for stochastic operations.
        agent_energy: Energy per agent, shape (max_agents,). 0 for dead/inactive slots.
        agent_alive: Boolean alive mask, shape (max_agents,). True for active agents.
        agent_ids: Unique ID per agent, shape (max_agents,). -1 for empty slots.
        agent_parent_ids: Parent agent ID, shape (max_agents,). -1 if original or empty.
        next_agent_id: Scalar counter for assigning unique IDs to new agents.
        agent_birth_step: Step when each agent was born, shape (max_agents,).
            0 for original agents, -1 for empty slots.
        agent_params: Per-agent network parameters. Pytree where each leaf has
            shape (max_agents, ...). None when evolution is disabled.
    """
    agent_positions: jnp.ndarray   # (max_agents, 2)
    food_positions: jnp.ndarray    # (num_food, 2)
    food_collected: jnp.ndarray    # (num_food,) bool
    field_state: FieldState
    step: jnp.ndarray              # scalar int
    key: jax.Array                 # PRNG key
    agent_energy: jnp.ndarray      # (max_agents,)
    agent_alive: jnp.ndarray       # (max_agents,) bool
    agent_ids: jnp.ndarray         # (max_agents,)
    agent_parent_ids: jnp.ndarray  # (max_agents,)
    next_agent_id: jnp.ndarray     # scalar int
    agent_birth_step: jnp.ndarray  # (max_agents,) int
    agent_params: Any = None       # per-agent params pytree, (max_agents, ...)


def create_env_state(key: jax.Array, config: "src.configs.Config") -> EnvState:  # type: ignore[name-defined]
    """Create an initial EnvState from config.

    Agent and food positions are randomly placed on the grid.
    The field is initialized to zeros. Evolution fields are initialized
    with the first num_agents slots active.

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
    max_agents = config.evolution.max_agents

    # Random agent positions (row, col) within grid — padded to max_agents
    agent_positions = jnp.zeros((max_agents, 2), dtype=jnp.int32)
    active_positions = jax.random.randint(
        k1, shape=(num_agents, 2), minval=0, maxval=grid_size
    )
    agent_positions = agent_positions.at[:num_agents].set(active_positions)

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

    # Birth step: 0 for original agents, -1 for empty slots
    agent_birth_step = jnp.full((max_agents,), -1, dtype=jnp.int32)
    agent_birth_step = agent_birth_step.at[:num_agents].set(0)

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
        agent_birth_step=agent_birth_step,
    )
