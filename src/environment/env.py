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
            0=stay, 1=up, 2=down, 3=left, 4=right, 5=reproduce.
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
    # Action deltas: 0=stay, 1=up(-row), 2=down(+row), 3=left(-col), 4=right(+col), 5=reproduce(stay)
    action_deltas = jnp.array([
        [0, 0],   # stay
        [-1, 0],  # up
        [1, 0],   # down
        [0, -1],  # left
        [0, 1],   # right
        [0, 0],   # reproduce (stay in place)
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
    # Only alive agents can collect food; closest alive agent gets energy
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

    # Determine closest alive agent per food item for energy assignment
    # Use large distance for non-collectible pairs so they lose argmin
    masked_dist = jnp.where(collectible, chebyshev_dist, jnp.float32(1e6))
    # For each food, find the closest agent (argmin over agents axis)
    closest_agent = jnp.argmin(masked_dist, axis=0)  # (F,)

    # Count food collected per agent: for each newly collected food,
    # the closest agent gets food_energy
    food_energy_val = jnp.float32(config.evolution.food_energy)
    max_energy_val = jnp.float32(config.evolution.max_energy)

    # Build per-agent energy gain: sum food_energy for each food assigned to agent
    # Use one-hot encoding: (F, max_agents) where each row has a 1 at closest_agent
    agent_food_mask = (
        jax.nn.one_hot(closest_agent, max_agents)  # (F, max_agents)
        * newly_collected[:, None]  # only count newly collected food
    )
    food_per_agent = jnp.sum(agent_food_mask, axis=0)  # (max_agents,)
    energy_gained = food_per_agent * food_energy_val  # (max_agents,)

    # Add energy to agents, cap at max_energy (only for alive agents)
    energy_after_food = jnp.where(
        state.agent_alive,
        jnp.minimum(state.agent_energy + energy_gained, max_energy_val),
        state.agent_energy,
    )

    # --- 3. Compute reward ---
    # Individual reward: each agent gets reward equal to energy gained
    # Slice to (num_agents,) for compatibility with training
    rewards = energy_gained[:config.env.num_agents]

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
    # Subtract energy_per_step from alive agents (after food energy), clamp to 0
    energy_drain = jnp.where(
        state.agent_alive,
        jnp.maximum(energy_after_food - config.evolution.energy_per_step, 0.0),
        energy_after_food,
    )
    new_energy = energy_drain

    # --- 6. Death from starvation ---
    # Agents with energy <= 0 die (only check previously alive agents)
    starved = state.agent_alive & (new_energy <= 0)
    new_alive = state.agent_alive & ~starved
    death_count = jnp.sum(starved.astype(jnp.int32))

    # --- 7. Reproduction ---
    # Action 5 = attempt reproduction
    # Conditions: agent alive, chose action 5, energy >= threshold, free slot exists
    reproduce_threshold = jnp.float32(config.evolution.reproduce_threshold)
    reproduce_cost = jnp.float32(config.evolution.reproduce_cost)

    wants_reproduce = (padded_actions == 5)  # (max_agents,)
    can_reproduce = (
        new_alive
        & wants_reproduce
        & (new_energy >= reproduce_threshold)
    )  # (max_agents,)

    # Check if there are free slots (any slot where agent_alive == False)
    num_free_slots = jnp.sum((~new_alive).astype(jnp.int32))
    any_free_slot = num_free_slots > 0

    # Only allow reproduction if there is at least one free slot
    # Process one reproduction at a time using scan to handle slot allocation
    # For simplicity, process agents in order; each successful reproduction
    # fills one slot and reduces available slots.

    def _process_reproductions(carry, agent_idx):
        """Process one agent's reproduction attempt sequentially."""
        alive, energy, ids, parent_ids, positions, next_id, key = carry

        # Check if this agent wants and can reproduce
        eligible = (
            can_reproduce[agent_idx]
            & (jnp.sum((~alive).astype(jnp.int32)) > 0)  # free slot exists
        )

        # Find first free slot
        free_mask = ~alive  # (max_agents,)
        # Use argmax on free_mask to get first True index; if none free, returns 0
        # but eligible check ensures at least one free slot
        free_slot = jnp.argmax(free_mask.astype(jnp.int32))

        # Deduct reproduce_cost from parent
        new_energy_val = jnp.where(eligible, energy[agent_idx] - reproduce_cost, energy[agent_idx])
        energy = energy.at[agent_idx].set(new_energy_val)

        # Spawn offspring in free slot
        key, spawn_key = jax.random.split(key)
        # Offspring position: random adjacent cell of parent (clipped to grid)
        offset = jax.random.randint(spawn_key, (2,), -1, 2)
        child_pos = jnp.clip(positions[agent_idx] + offset, 0, grid_size - 1)

        # Update offspring slot (only if eligible)
        alive = jnp.where(eligible, alive.at[free_slot].set(True), alive)
        energy = jnp.where(eligible, energy.at[free_slot].set(reproduce_cost), energy)
        positions = jnp.where(eligible, positions.at[free_slot].set(child_pos), positions)
        ids = jnp.where(eligible, ids.at[free_slot].set(next_id), ids)
        parent_ids = jnp.where(eligible, parent_ids.at[free_slot].set(ids[agent_idx]), parent_ids)
        next_id = jnp.where(eligible, next_id + 1, next_id)

        return (alive, energy, ids, parent_ids, positions, next_id, key), eligible

    # Split key for reproduction
    repro_key, post_repro_key = jax.random.split(state.key)

    init_carry = (new_alive, new_energy, state.agent_ids, state.agent_parent_ids,
                  new_positions, state.next_agent_id, repro_key)
    (new_alive, new_energy, new_ids, new_parent_ids, new_positions, new_next_id, _), births_per_agent = (
        jax.lax.scan(_process_reproductions, init_carry, jnp.arange(max_agents))
    )
    birth_count = jnp.sum(births_per_agent.astype(jnp.int32))
    new_key = post_repro_key

    # --- 8. Advance step counter and check done ---
    new_step = state.step + 1
    done = new_step >= config.env.max_steps

    # --- 9. Split PRNG key ---
    # (key already split during reproduction above)

    new_state = EnvState(
        agent_positions=new_positions,
        food_positions=state.food_positions,
        food_collected=food_collected,
        field_state=field_state,
        step=new_step,
        key=new_key,
        agent_energy=new_energy,
        agent_alive=new_alive,
        agent_ids=new_ids,
        agent_parent_ids=new_parent_ids,
        next_agent_id=new_next_id,
    )

    num_collected = jnp.sum(newly_collected.astype(jnp.float32))
    info: dict[str, Any] = {
        "food_collected_this_step": num_collected,
        "total_food_collected": jnp.sum(food_collected.astype(jnp.float32)),
        "deaths_this_step": death_count,
        "births_this_step": birth_count,
    }

    return new_state, rewards, done, info
