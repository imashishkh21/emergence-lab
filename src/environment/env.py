"""Environment reset and step functions."""

from typing import Any

import jax
import jax.numpy as jnp

from src.agents.reproduction import (
    compute_per_leaf_mutation_rates,
    mutate_agent_params,
    mutate_agent_params_layered,
)
from src.configs import Config
from src.environment.state import EnvState
from src.field.dynamics import step_field
from src.field.field import create_field
from src.field.ops import write_local


def reset(key: jax.Array, config: Config) -> EnvState:
    """Create a fresh environment state for a new episode.

    Agents spawn within the nest area (center of grid). Food positions are
    random. The field is initialized with Ch1 (territory) seeded around the
    nest. Evolution fields are initialized with the first num_agents slots
    active.

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

    # Nest area: center of grid
    nest_center_r = grid_size // 2
    nest_center_c = grid_size // 2
    nest_radius = config.nest.radius  # default 2, so 5x5 area

    # Spawn agents within nest area (non-overlapping)
    nest_min = max(0, nest_center_r - nest_radius)
    nest_max_val = min(grid_size - 1, nest_center_r + nest_radius)
    nest_width = nest_max_val - nest_min + 1
    nest_cells = nest_width * nest_width
    perm = jax.random.permutation(k1, nest_cells)
    agent_indices = perm[:num_agents]
    agent_rows = nest_min + agent_indices // nest_width
    agent_cols = nest_min + agent_indices % nest_width
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

    # Fresh field initialized to zeros, then seed Ch1 (territory) around nest
    field_state = create_field(
        height=grid_size,
        width=grid_size,
        channels=config.field.num_channels,
    )
    # Initialize territory channel (Ch1) in (2*nest_radius+3) x (2*nest_radius+3) area
    border = nest_radius + 1
    r_min = max(0, nest_center_r - border)
    r_max = min(grid_size, nest_center_r + border + 1)
    c_min = max(0, nest_center_c - border)
    c_max = min(grid_size, nest_center_c + border + 1)
    territory_init = field_state.values.at[r_min:r_max, c_min:c_max, 1].set(1.0)
    from src.field.field import FieldState
    field_state = FieldState(values=territory_init)

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

    # Initialize hidden food fields if enabled
    hidden_food_positions = None
    hidden_food_revealed = None
    hidden_food_reveal_timer = None
    hidden_food_collected = None

    if config.env.hidden_food.enabled:
        num_hidden = config.env.hidden_food.num_hidden
        k3, hidden_key = jax.random.split(k3)
        hidden_food_positions = jax.random.randint(
            hidden_key, shape=(num_hidden, 2), minval=0, maxval=grid_size
        )
        hidden_food_revealed = jnp.zeros((num_hidden,), dtype=jnp.bool_)
        hidden_food_reveal_timer = jnp.zeros((num_hidden,), dtype=jnp.int32)
        hidden_food_collected = jnp.zeros((num_hidden,), dtype=jnp.bool_)

    # Pheromone system fields
    has_food = jnp.zeros((max_agents,), dtype=jnp.bool_)
    prev_field_at_pos = jnp.zeros((max_agents, config.field.num_channels), dtype=jnp.float32)
    laden_cooldown = jnp.zeros((max_agents,), dtype=jnp.bool_)

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
        has_food=has_food,
        prev_field_at_pos=prev_field_at_pos,
        laden_cooldown=laden_cooldown,
        hidden_food_positions=hidden_food_positions,
        hidden_food_revealed=hidden_food_revealed,
        hidden_food_reveal_timer=hidden_food_reveal_timer,
        hidden_food_collected=hidden_food_collected,
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

    # Laden agents in cooldown are frozen in place (move every other step)
    laden_frozen = state.has_food & state.laden_cooldown & state.agent_alive
    padded_actions = jnp.where(laden_frozen, 0, padded_actions)  # force stay

    deltas = action_deltas[padded_actions]  # (max_agents, 2)
    new_positions = state.agent_positions + deltas

    # Clip to grid boundaries
    new_positions = jnp.clip(new_positions, 0, grid_size - 1)

    # Dead agents stay at their position (masked out)
    new_positions = jnp.where(
        state.agent_alive[:, None], new_positions, state.agent_positions
    )

    # --- 2. Food collection ---
    # Only alive agents NOT already carrying food can collect
    agent_rows = new_positions[:, 0:1]  # (max_agents, 1)
    agent_cols = new_positions[:, 1:2]  # (max_agents, 1)
    food_rows = state.food_positions[:, 0]  # (F,)
    food_cols = state.food_positions[:, 1]  # (F,)

    row_dist = jnp.abs(agent_rows - food_rows[None, :])  # (max_agents, F)
    col_dist = jnp.abs(agent_cols - food_cols[None, :])  # (max_agents, F)
    chebyshev_dist = jnp.maximum(row_dist, col_dist)  # (max_agents, F)

    # Food is collectible if within 1 cell AND not already collected AND agent alive AND not carrying food
    within_range = chebyshev_dist <= 1  # (max_agents, F)
    not_collected = ~state.food_collected  # (F,)
    alive_mask = state.agent_alive[:, None]  # (max_agents, 1)
    not_carrying = (~state.has_food)[:, None]  # (max_agents, 1)
    collectible = within_range & not_collected[None, :] & alive_mask & not_carrying  # (max_agents, F)

    # Any eligible agent adjacent to uncollected food collects it
    newly_collected = jnp.any(collectible, axis=0)  # (F,)
    food_collected = state.food_collected | newly_collected

    # Determine closest eligible agent per food item for energy assignment
    masked_dist = jnp.where(collectible, chebyshev_dist, jnp.float32(1e6))
    closest_agent = jnp.argmin(masked_dist, axis=0)  # (F,)

    # Count food collected per agent
    food_energy_val = jnp.float32(config.evolution.food_energy)
    max_energy_val = jnp.float32(config.evolution.max_energy)

    agent_food_mask = (
        jax.nn.one_hot(closest_agent, max_agents)  # (F, max_agents)
        * newly_collected[:, None]  # only count newly collected food
    )
    food_per_agent = jnp.sum(agent_food_mask, axis=0)  # (max_agents,)

    # Set has_food for agents that picked up food
    pickup_mask = (food_per_agent > 0) & state.agent_alive & ~state.has_food
    has_food = jnp.where(pickup_mask, True, state.has_food)

    # Crop refuel: fill energy to max on food pickup (biological: ant fills
    # personal stomach at food source to fuel the return trip)
    energy_after_food = jnp.where(
        pickup_mask,
        max_energy_val,
        state.agent_energy,
    )

    # Initialize laden_cooldown to False on new pickup
    laden_cooldown = jnp.where(pickup_mask, False, state.laden_cooldown)

    # Pickup reward (PPO signal — partial)
    pickup_reward = food_per_agent * food_energy_val * config.nest.pickup_reward_fraction

    # Toggle laden_cooldown for agents carrying food (move/write alternation)
    laden_cooldown = jnp.where(has_food, ~laden_cooldown, laden_cooldown)

    # --- 3. Food respawn ---
    # Collected food has food_respawn_prob chance to respawn at a random location
    respawn_key, remaining_key = jax.random.split(state.key)
    num_food = config.env.num_food
    roll_key, pos_key = jax.random.split(respawn_key)
    respawn_rolls = jax.random.uniform(roll_key, shape=(num_food,))
    # Food respawns if: it was collected (either previously or this step) AND roll < prob
    respawns = food_collected & (respawn_rolls < config.env.food_respawn_prob)
    # Generate new random positions for respawning food
    new_food_positions = jax.random.randint(
        pos_key, shape=(num_food, 2), minval=0, maxval=grid_size
    )
    # Replace positions of respawning food; keep others unchanged
    food_positions = jnp.where(
        respawns[:, None], new_food_positions, state.food_positions
    )
    # Mark respawned food as not collected
    food_collected = food_collected & ~respawns

    # --- 4. Hidden food reveal and collection (if enabled) ---
    # Hidden food fields (carry through unchanged if disabled)
    hidden_food_positions = state.hidden_food_positions
    hidden_food_revealed = state.hidden_food_revealed
    hidden_food_reveal_timer = state.hidden_food_reveal_timer
    hidden_food_collected = state.hidden_food_collected
    hidden_food_energy_gained = jnp.zeros((max_agents,), dtype=jnp.float32)

    if config.env.hidden_food.enabled:
        hf_config = config.env.hidden_food
        num_hidden = hf_config.num_hidden
        required_agents = hf_config.required_agents
        reveal_distance = hf_config.reveal_distance
        reveal_duration = hf_config.reveal_duration
        value_multiplier = hf_config.hidden_food_value_multiplier

        # Type assertions for mypy (we know these are not None when enabled)
        assert hidden_food_positions is not None
        assert hidden_food_revealed is not None
        assert hidden_food_reveal_timer is not None
        assert hidden_food_collected is not None

        # Compute distances from all agents to all hidden food
        # new_positions: (max_agents, 2), hidden_food_positions: (num_hidden, 2)
        hf_rows = hidden_food_positions[:, 0]  # (num_hidden,)
        hf_cols = hidden_food_positions[:, 1]  # (num_hidden,)
        agent_rows_hf = new_positions[:, 0:1]  # (max_agents, 1)
        agent_cols_hf = new_positions[:, 1:2]  # (max_agents, 1)

        hf_row_dist = jnp.abs(agent_rows_hf - hf_rows[None, :])  # (max_agents, num_hidden)
        hf_col_dist = jnp.abs(agent_cols_hf - hf_cols[None, :])  # (max_agents, num_hidden)
        hf_chebyshev = jnp.maximum(hf_row_dist, hf_col_dist)  # (max_agents, num_hidden)

        # Count alive agents within reveal_distance of each hidden food
        within_reveal = (hf_chebyshev <= reveal_distance) & state.agent_alive[:, None]  # (max_agents, num_hidden)
        agents_near_hf = jnp.sum(within_reveal.astype(jnp.int32), axis=0)  # (num_hidden,)

        # Check reveal condition: >= required_agents near hidden food
        should_reveal = (agents_near_hf >= required_agents) & ~hidden_food_collected

        # Update reveal timer: if newly revealed, set to reveal_duration
        # If already revealed, decrement timer (but not below 0)
        new_reveal_timer = jnp.where(
            should_reveal & ~hidden_food_revealed,
            reveal_duration,
            jnp.maximum(hidden_food_reveal_timer - 1, 0),
        )
        # Keep timer at reveal_duration if continuously revealed
        new_reveal_timer = jnp.where(
            should_reveal & hidden_food_revealed,
            reveal_duration,
            new_reveal_timer,
        )

        # Food is revealed if timer > 0 or just triggered
        new_revealed = (new_reveal_timer > 0) | should_reveal

        # Hidden food collection: within distance 1 AND revealed AND not collected
        within_collect = (hf_chebyshev <= 1)  # (max_agents, num_hidden)
        hf_collectible = within_collect & new_revealed[None, :] & ~hidden_food_collected[None, :] & state.agent_alive[:, None]

        # Any agent can collect revealed hidden food
        hf_newly_collected = jnp.any(hf_collectible, axis=0)  # (num_hidden,)

        # Determine which agent collects each hidden food (closest)
        hf_masked_dist = jnp.where(hf_collectible, hf_chebyshev, jnp.float32(1e6))
        hf_closest_agent = jnp.argmin(hf_masked_dist, axis=0)  # (num_hidden,)

        # Calculate hidden food energy: food_energy * value_multiplier
        hf_energy_val = jnp.float32(config.evolution.food_energy * value_multiplier)

        # Build per-agent hidden food energy gain
        hf_agent_mask = (
            jax.nn.one_hot(hf_closest_agent, max_agents)  # (num_hidden, max_agents)
            * hf_newly_collected[:, None]
        )
        hf_per_agent = jnp.sum(hf_agent_mask, axis=0)  # (max_agents,)
        hidden_food_energy_gained = hf_per_agent * hf_energy_val

        # Add hidden food energy to agents
        energy_after_food = jnp.where(
            state.agent_alive,
            jnp.minimum(energy_after_food + hidden_food_energy_gained, max_energy_val),
            energy_after_food,
        )

        # Update hidden food collected status
        new_hf_collected = hidden_food_collected | hf_newly_collected

        # When hidden food timer expires (reaches 0) and food was revealed but not collected,
        # re-hide and respawn at new position
        timer_expired = (new_reveal_timer == 0) & hidden_food_revealed & ~new_hf_collected

        # Generate new positions for expired hidden food
        remaining_key, hf_respawn_key = jax.random.split(remaining_key)
        new_hf_positions = jax.random.randint(
            hf_respawn_key, shape=(num_hidden, 2), minval=0, maxval=grid_size
        )
        hidden_food_positions = jnp.where(
            timer_expired[:, None], new_hf_positions, hidden_food_positions
        )

        # When hidden food is collected, respawn at new position and reset state
        remaining_key, hf_collected_respawn_key = jax.random.split(remaining_key)
        collected_new_positions = jax.random.randint(
            hf_collected_respawn_key, shape=(num_hidden, 2), minval=0, maxval=grid_size
        )
        hidden_food_positions = jnp.where(
            hf_newly_collected[:, None], collected_new_positions, hidden_food_positions
        )

        # Reset collected status for respawned food (both timer-expired and collected)
        hidden_food_collected = jnp.where(timer_expired | hf_newly_collected, False, new_hf_collected)
        hidden_food_revealed = jnp.where(timer_expired | hf_newly_collected, False, new_revealed)
        hidden_food_reveal_timer = jnp.where(timer_expired | hf_newly_collected, 0, new_reveal_timer)

    # --- 5. Nest delivery ---
    # Agents carrying food who return to the nest area deliver it
    nest_center_r = config.env.grid_size // 2
    nest_center_c = config.env.grid_size // 2
    nest_r = config.nest.radius

    # Check which agents are in nest area (Chebyshev distance <= radius)
    agent_nest_dr = jnp.abs(new_positions[:, 0] - nest_center_r)
    agent_nest_dc = jnp.abs(new_positions[:, 1] - nest_center_c)
    in_nest = (agent_nest_dr <= nest_r) & (agent_nest_dc <= nest_r)

    # Delivery: agent has_food AND in nest area AND alive
    delivering = has_food & in_nest & state.agent_alive

    # Delivery energy: 95% of food_energy
    delivery_energy = jnp.where(
        delivering, food_energy_val * config.nest.food_delivery_fraction, 0.0
    )
    energy_after_food = jnp.where(
        state.agent_alive,
        jnp.minimum(energy_after_food + delivery_energy, max_energy_val),
        energy_after_food,
    )

    # Reset has_food and laden_cooldown on delivery
    has_food = jnp.where(delivering, False, has_food)
    laden_cooldown = jnp.where(delivering, False, laden_cooldown)

    # Delivery reward (PPO signal)
    delivery_reward = jnp.where(
        delivering, food_energy_val * config.nest.delivery_reward_fraction, 0.0
    )

    # --- 6. Compute reward ---
    # Pickup reward + delivery reward + hidden food energy
    # Full (max_agents,) shape — dead agents get 0 reward
    rewards = jnp.where(
        state.agent_alive, pickup_reward + delivery_reward + hidden_food_energy_gained, 0.0
    )

    # --- 7. Update field ---
    # Step field dynamics (diffuse + decay) with per-channel rates if available
    if config.field.channel_diffusion_rates is not None:
        diffusion_rate = jnp.array(config.field.channel_diffusion_rates)
    else:
        diffusion_rate = config.field.diffusion_rate
    if config.field.channel_decay_rates is not None:
        decay_rate = jnp.array(config.field.channel_decay_rates)
    else:
        decay_rate = config.field.decay_rate
    field_state = step_field(
        state.field_state,
        diffusion_rate=diffusion_rate,
        decay_rate=decay_rate,
    )

    # --- Channel-specific field writes ---
    num_ch = config.field.num_channels

    # Ch1 (territory): ALL alive agents write passively
    territory_strength = config.field.territory_write_strength  # default 0.01
    territory_values = jnp.zeros((max_agents, num_ch), dtype=jnp.float32)
    territory_values = territory_values.at[:, 1].set(territory_strength)
    territory_values = territory_values * state.agent_alive[:, None]
    field_state = write_local(field_state, new_positions, territory_values, cap=config.field.field_value_cap)

    # Ch0 (recruitment): ONLY laden agents during write phase
    # laden_cooldown was toggled above: agents whose cooldown is now True are in "write" phase
    write_phase = has_food & laden_cooldown & state.agent_alive
    recruit_values = jnp.zeros((max_agents, num_ch), dtype=jnp.float32)
    recruit_values = recruit_values.at[:, 0].set(1.0)
    recruit_values = recruit_values * write_phase[:, None]
    field_state = write_local(field_state, new_positions, recruit_values, cap=config.field.field_value_cap)

    # --- 8. Energy drain ---
    # Subtract energy_per_step from alive agents, but write steps are free
    # (biological: pheromone deposit is chemically cheap, no locomotion cost)
    is_write_step = write_phase  # laden agents in write phase don't move
    energy_cost = jnp.where(is_write_step, 0.0, config.evolution.energy_per_step)
    energy_drain = jnp.where(
        state.agent_alive,
        jnp.maximum(energy_after_food - energy_cost, 0.0),
        energy_after_food,
    )
    new_energy = energy_drain

    # --- 9. Death from starvation ---
    # Agents with energy <= 0 die (only check previously alive agents)
    starved = state.agent_alive & (new_energy <= 0)
    new_alive = state.agent_alive & ~starved
    death_count = jnp.sum(starved.astype(jnp.int32))

    # Clear has_food and laden_cooldown on death
    has_food = jnp.where(starved, False, has_food)
    laden_cooldown = jnp.where(starved, False, laden_cooldown)

    # --- 10. Reproduction ---
    # Auto-reproduction: alive + energy >= threshold + free slot
    reproduce_threshold = jnp.float32(config.evolution.reproduce_threshold)
    reproduce_cost = jnp.float32(config.evolution.reproduce_cost)

    can_reproduce = (
        new_alive
        & in_nest  # Must be in nest area
        & (new_energy > reproduce_threshold)
    )  # (max_agents,)

    # Only allow reproduction if there is at least one free slot
    # Process one reproduction at a time using scan to handle slot allocation
    # For simplicity, process agents in order; each successful reproduction
    # fills one slot and reduces available slots.

    has_agent_params = state.agent_params is not None
    mutation_std = config.evolution.mutation_std

    # Pre-compute per-leaf mutation rates if layer-specific rates are configured
    layer_rates = getattr(config.specialization, 'layer_mutation_rates', None)
    use_layered = has_agent_params and layer_rates is not None
    if use_layered:
        per_leaf_rates = compute_per_leaf_mutation_rates(
            state.agent_params, mutation_std, layer_rates
        )

    current_step = state.step

    # Pre-compute nest bounds for child spawn (Python-level for JIT closure)
    nest_spawn_min = max(0, grid_size // 2 - config.nest.radius)
    nest_spawn_max = min(grid_size - 1, grid_size // 2 + config.nest.radius)

    if has_agent_params:
        def _process_reproductions(carry, agent_idx):
            """Process one agent's reproduction attempt sequentially (with params)."""
            alive, energy, ids, parent_ids, positions, next_id, key, ag_params, birth_steps = carry

            eligible = (
                can_reproduce[agent_idx]
                & (jnp.sum((~alive).astype(jnp.int32)) > 0)
            )

            free_mask = ~alive
            free_slot = jnp.argmax(free_mask.astype(jnp.int32))

            new_energy_val = jnp.where(eligible, energy[agent_idx] - reproduce_cost, energy[agent_idx])
            energy = energy.at[agent_idx].set(new_energy_val)

            key, spawn_key, mutate_key = jax.random.split(key, 3)
            child_pos = jax.random.randint(spawn_key, (2,), nest_spawn_min, nest_spawn_max + 1)

            alive = jnp.where(eligible, alive.at[free_slot].set(True), alive)
            energy = jnp.where(eligible, energy.at[free_slot].set(reproduce_cost), energy)
            positions = jnp.where(eligible, positions.at[free_slot].set(child_pos), positions)
            ids = jnp.where(eligible, ids.at[free_slot].set(next_id), ids)
            parent_ids = jnp.where(eligible, parent_ids.at[free_slot].set(ids[agent_idx]), parent_ids)
            birth_steps = jnp.where(eligible, birth_steps.at[free_slot].set(current_step), birth_steps)
            next_id = jnp.where(eligible, next_id + 1, next_id)

            # Mutate parent params -> child params
            if use_layered:
                mutated = mutate_agent_params_layered(
                    ag_params, agent_idx, free_slot, mutate_key, per_leaf_rates
                )
            else:
                mutated = mutate_agent_params(
                    ag_params, agent_idx, free_slot, mutate_key, mutation_std
                )
            ag_params = jax.tree.map(
                lambda orig, mut: jnp.where(eligible, mut, orig),
                ag_params,
                mutated,
            )

            return (alive, energy, ids, parent_ids, positions, next_id, key, ag_params, birth_steps), eligible

        repro_key, post_repro_key = jax.random.split(remaining_key)

        init_carry = (new_alive, new_energy, state.agent_ids, state.agent_parent_ids,
                      new_positions, state.next_agent_id, repro_key, state.agent_params, state.agent_birth_step)
        (new_alive, new_energy, new_ids, new_parent_ids, new_positions, new_next_id, _, new_agent_params, new_birth_steps), births_per_agent = (
            jax.lax.scan(_process_reproductions, init_carry, jnp.arange(max_agents))
        )
    else:
        def _process_reproductions_no_params(carry, agent_idx):
            """Process one agent's reproduction attempt sequentially (no params)."""
            alive, energy, ids, parent_ids, positions, next_id, key, birth_steps = carry

            eligible = (
                can_reproduce[agent_idx]
                & (jnp.sum((~alive).astype(jnp.int32)) > 0)
            )

            free_mask = ~alive
            free_slot = jnp.argmax(free_mask.astype(jnp.int32))

            new_energy_val = jnp.where(eligible, energy[agent_idx] - reproduce_cost, energy[agent_idx])
            energy = energy.at[agent_idx].set(new_energy_val)

            key, spawn_key = jax.random.split(key)
            child_pos = jax.random.randint(spawn_key, (2,), nest_spawn_min, nest_spawn_max + 1)

            alive = jnp.where(eligible, alive.at[free_slot].set(True), alive)
            energy = jnp.where(eligible, energy.at[free_slot].set(reproduce_cost), energy)
            positions = jnp.where(eligible, positions.at[free_slot].set(child_pos), positions)
            ids = jnp.where(eligible, ids.at[free_slot].set(next_id), ids)
            parent_ids = jnp.where(eligible, parent_ids.at[free_slot].set(ids[agent_idx]), parent_ids)
            birth_steps = jnp.where(eligible, birth_steps.at[free_slot].set(current_step), birth_steps)
            next_id = jnp.where(eligible, next_id + 1, next_id)

            return (alive, energy, ids, parent_ids, positions, next_id, key, birth_steps), eligible

        repro_key, post_repro_key = jax.random.split(remaining_key)

        init_carry_np = (new_alive, new_energy, state.agent_ids, state.agent_parent_ids,
                         new_positions, state.next_agent_id, repro_key, state.agent_birth_step)
        (new_alive, new_energy, new_ids, new_parent_ids, new_positions, new_next_id, _, new_birth_steps), births_per_agent = (
            jax.lax.scan(_process_reproductions_no_params, init_carry_np, jnp.arange(max_agents))
        )
        new_agent_params = None

    birth_count = jnp.sum(births_per_agent.astype(jnp.int32))
    new_key = post_repro_key

    # --- 11. Advance step counter and check done ---
    new_step = state.step + 1
    done = new_step >= config.env.max_steps

    # --- 12. Split PRNG key ---
    # (key already split during reproduction above)

    new_state = EnvState(
        agent_positions=new_positions,
        food_positions=food_positions,
        food_collected=food_collected,
        field_state=field_state,
        step=new_step,
        key=new_key,
        agent_energy=new_energy,
        agent_alive=new_alive,
        agent_ids=new_ids,
        agent_parent_ids=new_parent_ids,
        next_agent_id=new_next_id,
        agent_birth_step=new_birth_steps,
        agent_params=new_agent_params,
        has_food=has_food,
        prev_field_at_pos=field_state.values[new_positions[:, 0], new_positions[:, 1]],
        laden_cooldown=laden_cooldown,
        hidden_food_positions=hidden_food_positions,
        hidden_food_revealed=hidden_food_revealed,
        hidden_food_reveal_timer=hidden_food_reveal_timer,
        hidden_food_collected=hidden_food_collected,
    )

    num_collected = jnp.sum(newly_collected.astype(jnp.float32))
    # Track hidden food collected this step if enabled
    hf_collected_this_step = jnp.float32(0.0)
    if config.env.hidden_food.enabled:
        # Count hidden food collected this step (need to recompute since we respawned)
        # We can infer from hidden_food_energy_gained: if > 0, food was collected
        hf_collected_this_step = jnp.sum((hidden_food_energy_gained > 0).astype(jnp.float32))

    info: dict[str, Any] = {
        "food_collected_this_step": num_collected,
        "total_food_collected": jnp.sum(food_collected.astype(jnp.float32)),
        "deaths_this_step": death_count,
        "births_this_step": birth_count,
        "hidden_food_collected_this_step": hf_collected_this_step,
    }

    return new_state, rewards, done, info
