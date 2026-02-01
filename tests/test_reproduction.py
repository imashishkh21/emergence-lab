"""Tests for reproduction mechanics (US-007, US-008, US-009)."""

import jax
import jax.numpy as jnp

from src.agents.network import ActorCritic
from src.agents.reproduction import copy_agent_params, mutate_agent_params, mutate_params
from src.configs import Config
from src.environment.env import reset, step
from src.environment.state import EnvState
from src.field.field import create_field


def _make_state_with_positions(config, agent_pos, food_pos, energy=None):
    """Helper to create a controlled EnvState with specific positions."""
    max_agents = config.evolution.max_agents
    num_food = len(food_pos)
    config.env.num_food = num_food
    num_agents = len(agent_pos)
    config.env.num_agents = num_agents
    grid_size = config.env.grid_size
    key = jax.random.PRNGKey(0)

    agent_positions = jnp.zeros((max_agents, 2), dtype=jnp.int32)
    agent_positions = agent_positions.at[:num_agents].set(
        jnp.array(agent_pos, dtype=jnp.int32)
    )

    food_positions = jnp.array(food_pos, dtype=jnp.int32)
    food_collected = jnp.zeros((num_food,), dtype=jnp.bool_)

    field_state = create_field(
        height=grid_size, width=grid_size, channels=config.field.num_channels
    )

    if energy is None:
        energy_vals = jnp.float32(config.evolution.starting_energy)
    else:
        energy_vals = jnp.array(energy, dtype=jnp.float32)

    agent_energy = jnp.zeros((max_agents,), dtype=jnp.float32)
    if isinstance(energy_vals, jnp.ndarray) and energy_vals.ndim > 0:
        agent_energy = agent_energy.at[:num_agents].set(energy_vals[:num_agents])
    else:
        agent_energy = agent_energy.at[:num_agents].set(energy_vals)

    agent_alive = jnp.zeros((max_agents,), dtype=jnp.bool_)
    agent_alive = agent_alive.at[:num_agents].set(True)

    agent_ids = jnp.full((max_agents,), -1, dtype=jnp.int32)
    agent_ids = agent_ids.at[:num_agents].set(
        jnp.arange(num_agents, dtype=jnp.int32)
    )

    agent_parent_ids = jnp.full((max_agents,), -1, dtype=jnp.int32)
    next_agent_id = jnp.int32(num_agents)

    agent_birth_step = jnp.full((max_agents,), -1, dtype=jnp.int32)
    agent_birth_step = agent_birth_step.at[:num_agents].set(0)

    return EnvState(
        agent_positions=agent_positions,
        food_positions=food_positions,
        food_collected=food_collected,
        field_state=field_state,
        step=jnp.int32(0),
        key=key,
        agent_energy=agent_energy,
        agent_alive=agent_alive,
        agent_ids=agent_ids,
        agent_parent_ids=agent_parent_ids,
        next_agent_id=next_agent_id,
        agent_birth_step=agent_birth_step,
    )


class TestReproductionAction:
    """Tests for US-007: Reproduction action."""

    def test_reproduce_action_exists(self):
        """Test that action 5 (reproduce) is valid and agent stays in place."""
        config = Config()
        config.env.grid_size = 20
        config.evolution.starting_energy = 100
        config.evolution.food_energy = 0
        config.evolution.energy_per_step = 0  # No drain for simplicity
        config.evolution.reproduce_threshold = 200  # Too high to trigger

        state = _make_state_with_positions(
            config,
            agent_pos=[[5, 5]],
            food_pos=[[19, 19]],  # Far away
        )

        # Action 5 = reproduce (but threshold too high, so no spawn)
        actions = jnp.array([5], dtype=jnp.int32)
        new_state, _, _, _ = step(state, actions, config)

        # Agent should stay in place (reproduce = stay)
        assert jnp.all(new_state.agent_positions[0] == jnp.array([5, 5]))

    def test_reproduce_deducts_energy(self):
        """Test that successful reproduction deducts reproduce_cost from parent."""
        config = Config()
        config.env.grid_size = 20
        config.evolution.starting_energy = 160
        config.evolution.food_energy = 0
        config.evolution.energy_per_step = 0
        config.evolution.reproduce_threshold = 150
        config.evolution.reproduce_cost = 80
        config.evolution.max_agents = 8

        state = _make_state_with_positions(
            config,
            agent_pos=[[5, 5]],
            food_pos=[[19, 19]],
            energy=[160.0],
        )

        actions = jnp.array([5], dtype=jnp.int32)
        new_state, _, _, info = step(state, actions, config)

        # Parent energy should be 160 - 80 = 80
        assert jnp.isclose(new_state.agent_energy[0], 80.0)
        # A birth should have occurred
        assert info["births_this_step"] == 1

    def test_reproduce_fails_below_threshold(self):
        """Test that reproduction fails when energy < threshold."""
        config = Config()
        config.env.grid_size = 20
        config.evolution.starting_energy = 100
        config.evolution.food_energy = 0
        config.evolution.energy_per_step = 0
        config.evolution.reproduce_threshold = 150
        config.evolution.reproduce_cost = 80
        config.evolution.max_agents = 8

        state = _make_state_with_positions(
            config,
            agent_pos=[[5, 5]],
            food_pos=[[19, 19]],
            energy=[100.0],  # Below threshold of 150
        )

        actions = jnp.array([5], dtype=jnp.int32)
        new_state, _, _, info = step(state, actions, config)

        # Energy should be unchanged (no drain, no reproduction)
        assert jnp.isclose(new_state.agent_energy[0], 100.0)
        # No births
        assert info["births_this_step"] == 0

    def test_reproduce_fails_no_free_slots(self):
        """Test that reproduction fails when all slots are occupied."""
        config = Config()
        config.env.grid_size = 20
        config.evolution.starting_energy = 200
        config.evolution.food_energy = 0
        config.evolution.energy_per_step = 0
        config.evolution.reproduce_threshold = 150
        config.evolution.reproduce_cost = 80
        # max_agents = num_agents, so no free slots
        config.evolution.max_agents = 2

        state = _make_state_with_positions(
            config,
            agent_pos=[[5, 5], [10, 10]],
            food_pos=[[19, 19]],
            energy=[200.0, 200.0],
        )

        actions = jnp.array([5, 0], dtype=jnp.int32)
        new_state, _, _, info = step(state, actions, config)

        # Energy should be unchanged — reproduction failed
        assert jnp.isclose(new_state.agent_energy[0], 200.0)
        # No births
        assert info["births_this_step"] == 0

    def test_reproduce_spawns_offspring(self):
        """Test that successful reproduction creates an offspring in a free slot."""
        config = Config()
        config.env.grid_size = 20
        config.evolution.starting_energy = 160
        config.evolution.food_energy = 0
        config.evolution.energy_per_step = 0
        config.evolution.reproduce_threshold = 150
        config.evolution.reproduce_cost = 80
        config.evolution.max_agents = 4

        # 1 agent, 3 free slots
        state = _make_state_with_positions(
            config,
            agent_pos=[[10, 10]],
            food_pos=[[19, 19]],
            energy=[160.0],
        )

        actions = jnp.array([5], dtype=jnp.int32)
        new_state, _, _, info = step(state, actions, config)

        # A birth should have occurred
        assert info["births_this_step"] == 1

        # Offspring should be in slot 1 (first free slot)
        assert new_state.agent_alive[1]
        # Offspring energy = reproduce_cost
        assert jnp.isclose(new_state.agent_energy[1], 80.0)
        # Offspring ID = next_agent_id (which was 1 since 1 agent existed)
        assert new_state.agent_ids[1] == 1
        # Offspring parent ID = parent's ID (0)
        assert new_state.agent_parent_ids[1] == 0
        # next_agent_id should have incremented
        assert new_state.next_agent_id == 2

    def test_reproduce_offspring_near_parent(self):
        """Test that offspring spawns near parent position."""
        config = Config()
        config.env.grid_size = 20
        config.evolution.starting_energy = 160
        config.evolution.food_energy = 0
        config.evolution.energy_per_step = 0
        config.evolution.reproduce_threshold = 150
        config.evolution.reproduce_cost = 80
        config.evolution.max_agents = 4

        state = _make_state_with_positions(
            config,
            agent_pos=[[10, 10]],
            food_pos=[[19, 19]],
            energy=[160.0],
        )

        actions = jnp.array([5], dtype=jnp.int32)
        new_state, _, _, _ = step(state, actions, config)

        # Offspring should be within 1 cell of parent (offset is -1 to 1)
        parent_pos = new_state.agent_positions[0]
        child_pos = new_state.agent_positions[1]
        dist = jnp.abs(parent_pos - child_pos)
        assert jnp.all(dist <= 1), f"Child at {child_pos} too far from parent at {parent_pos}"

    def test_reproduce_dead_agent_cannot_reproduce(self):
        """Test that dead agents cannot reproduce even with action 5."""
        config = Config()
        config.env.grid_size = 20
        config.evolution.starting_energy = 200
        config.evolution.food_energy = 0
        config.evolution.energy_per_step = 0
        config.evolution.reproduce_threshold = 150
        config.evolution.reproduce_cost = 80
        config.evolution.max_agents = 4

        state = _make_state_with_positions(
            config,
            agent_pos=[[10, 10]],
            food_pos=[[19, 19]],
            energy=[200.0],
        )

        # Kill the agent
        new_alive = state.agent_alive.at[0].set(False)
        state = state.replace(agent_alive=new_alive)

        actions = jnp.array([5], dtype=jnp.int32)
        new_state, _, _, info = step(state, actions, config)

        # No births — agent is dead
        assert info["births_this_step"] == 0

    def test_reproduce_no_action_5_no_birth(self):
        """Test that agents choosing non-reproduce actions don't trigger reproduction."""
        config = Config()
        config.env.grid_size = 20
        config.evolution.starting_energy = 200
        config.evolution.food_energy = 0
        config.evolution.energy_per_step = 0
        config.evolution.reproduce_threshold = 150
        config.evolution.reproduce_cost = 80
        config.evolution.max_agents = 4

        state = _make_state_with_positions(
            config,
            agent_pos=[[10, 10]],
            food_pos=[[19, 19]],
            energy=[200.0],
        )

        # Action 0 = stay, not reproduce
        actions = jnp.array([0], dtype=jnp.int32)
        new_state, _, _, info = step(state, actions, config)

        # No births
        assert info["births_this_step"] == 0
        # Energy unchanged
        assert jnp.isclose(new_state.agent_energy[0], 200.0)

    def test_reproduce_multiple_agents(self):
        """Test that multiple agents can reproduce in the same step."""
        config = Config()
        config.env.grid_size = 20
        config.evolution.starting_energy = 160
        config.evolution.food_energy = 0
        config.evolution.energy_per_step = 0
        config.evolution.reproduce_threshold = 150
        config.evolution.reproduce_cost = 80
        config.evolution.max_agents = 8

        state = _make_state_with_positions(
            config,
            agent_pos=[[5, 5], [15, 15]],
            food_pos=[[19, 19]],
            energy=[160.0, 160.0],
        )

        # Both agents reproduce
        actions = jnp.array([5, 5], dtype=jnp.int32)
        new_state, _, _, info = step(state, actions, config)

        # Two births should have occurred
        assert info["births_this_step"] == 2
        # Both parents should have lost energy
        assert jnp.isclose(new_state.agent_energy[0], 80.0)
        assert jnp.isclose(new_state.agent_energy[1], 80.0)
        # Two new agents should be alive
        alive_count = jnp.sum(new_state.agent_alive.astype(jnp.int32))
        assert alive_count == 4  # 2 original + 2 offspring

    def test_reproduce_jit_compatible(self):
        """Test that reproduction works under JIT compilation."""
        config = Config()
        config.env.grid_size = 20
        config.evolution.starting_energy = 160
        config.evolution.food_energy = 0
        config.evolution.energy_per_step = 0
        config.evolution.reproduce_threshold = 150
        config.evolution.reproduce_cost = 80
        config.evolution.max_agents = 4

        state = _make_state_with_positions(
            config,
            agent_pos=[[10, 10]],
            food_pos=[[19, 19]],
            energy=[160.0],
        )

        actions = jnp.array([5], dtype=jnp.int32)

        @jax.jit
        def jit_step(s, a):
            return step(s, a, config)

        new_state, _, _, info = jit_step(state, actions)

        assert info["births_this_step"] == 1
        assert jnp.isclose(new_state.agent_energy[0], 80.0)
        assert new_state.agent_alive[1]

    def test_reproduce_with_energy_drain(self):
        """Test reproduction interacts correctly with energy drain."""
        config = Config()
        config.env.grid_size = 20
        config.evolution.starting_energy = 160
        config.evolution.food_energy = 0
        config.evolution.energy_per_step = 5
        config.evolution.reproduce_threshold = 150
        config.evolution.reproduce_cost = 80
        config.evolution.max_agents = 4

        state = _make_state_with_positions(
            config,
            agent_pos=[[10, 10]],
            food_pos=[[19, 19]],
            energy=[160.0],
        )

        actions = jnp.array([5], dtype=jnp.int32)
        new_state, _, _, info = step(state, actions, config)

        # Energy drain happens before reproduction check:
        # 160 - 5 = 155, then 155 >= 150 so reproduce, 155 - 80 = 75
        assert info["births_this_step"] == 1
        assert jnp.isclose(new_state.agent_energy[0], 75.0)


class TestSpawn:
    """Tests for US-008: Offspring spawning."""

    def test_spawn_fills_first_empty_slot(self):
        """Offspring is placed in the first slot where agent_alive == False."""
        config = Config()
        config.env.grid_size = 20
        config.evolution.starting_energy = 160
        config.evolution.food_energy = 0
        config.evolution.energy_per_step = 0
        config.evolution.reproduce_threshold = 150
        config.evolution.reproduce_cost = 80
        config.evolution.max_agents = 8

        # 2 agents alive at slots 0 and 1; slots 2-7 are free
        state = _make_state_with_positions(
            config,
            agent_pos=[[5, 5], [15, 15]],
            food_pos=[[19, 19]],
            energy=[160.0, 100.0],
        )

        # Only agent 0 reproduces (agent 1 below threshold)
        actions = jnp.array([5, 0], dtype=jnp.int32)
        new_state, _, _, info = step(state, actions, config)

        assert info["births_this_step"] == 1
        # Offspring should be in slot 2 (first free slot after 0, 1)
        assert new_state.agent_alive[2]
        # Slots 3-7 should still be empty
        assert not jnp.any(new_state.agent_alive[3:])

    def test_spawn_position_adjacent_to_parent(self):
        """Offspring position is within 1 cell (random adjacent) of parent."""
        config = Config()
        config.env.grid_size = 20
        config.evolution.starting_energy = 160
        config.evolution.food_energy = 0
        config.evolution.energy_per_step = 0
        config.evolution.reproduce_threshold = 150
        config.evolution.reproduce_cost = 80
        config.evolution.max_agents = 4

        state = _make_state_with_positions(
            config,
            agent_pos=[[10, 10]],
            food_pos=[[19, 19]],
            energy=[160.0],
        )

        actions = jnp.array([5], dtype=jnp.int32)
        new_state, _, _, _ = step(state, actions, config)

        parent_pos = new_state.agent_positions[0]
        child_pos = new_state.agent_positions[1]
        dist = jnp.abs(parent_pos - child_pos)
        assert jnp.all(dist <= 1), f"Child {child_pos} not adjacent to parent {parent_pos}"

    def test_spawn_position_clipped_to_grid(self):
        """Offspring position is clipped to grid boundaries when parent is at edge."""
        config = Config()
        config.env.grid_size = 20
        config.evolution.starting_energy = 160
        config.evolution.food_energy = 0
        config.evolution.energy_per_step = 0
        config.evolution.reproduce_threshold = 150
        config.evolution.reproduce_cost = 80
        config.evolution.max_agents = 4

        # Parent at corner (0, 0) — offspring can't go negative
        state = _make_state_with_positions(
            config,
            agent_pos=[[0, 0]],
            food_pos=[[19, 19]],
            energy=[160.0],
        )

        actions = jnp.array([5], dtype=jnp.int32)
        new_state, _, _, _ = step(state, actions, config)

        child_pos = new_state.agent_positions[1]
        assert jnp.all(child_pos >= 0), f"Child position {child_pos} out of bounds (negative)"
        assert jnp.all(child_pos < 20), f"Child position {child_pos} out of bounds (>= grid_size)"

    def test_spawn_energy_equals_reproduce_cost(self):
        """Offspring receives energy equal to reproduce_cost."""
        config = Config()
        config.env.grid_size = 20
        config.evolution.starting_energy = 160
        config.evolution.food_energy = 0
        config.evolution.energy_per_step = 0
        config.evolution.reproduce_threshold = 150
        config.evolution.reproduce_cost = 80
        config.evolution.max_agents = 4

        state = _make_state_with_positions(
            config,
            agent_pos=[[10, 10]],
            food_pos=[[19, 19]],
            energy=[160.0],
        )

        actions = jnp.array([5], dtype=jnp.int32)
        new_state, _, _, _ = step(state, actions, config)

        assert jnp.isclose(new_state.agent_energy[1], 80.0)

    def test_spawn_alive_flag_set(self):
        """Offspring has agent_alive set to True."""
        config = Config()
        config.env.grid_size = 20
        config.evolution.starting_energy = 160
        config.evolution.food_energy = 0
        config.evolution.energy_per_step = 0
        config.evolution.reproduce_threshold = 150
        config.evolution.reproduce_cost = 80
        config.evolution.max_agents = 4

        state = _make_state_with_positions(
            config,
            agent_pos=[[10, 10]],
            food_pos=[[19, 19]],
            energy=[160.0],
        )

        actions = jnp.array([5], dtype=jnp.int32)
        new_state, _, _, _ = step(state, actions, config)

        assert new_state.agent_alive[1]

    def test_spawn_gets_unique_id(self):
        """Offspring gets agent_ids[slot] = next_agent_id, counter incremented."""
        config = Config()
        config.env.grid_size = 20
        config.evolution.starting_energy = 160
        config.evolution.food_energy = 0
        config.evolution.energy_per_step = 0
        config.evolution.reproduce_threshold = 150
        config.evolution.reproduce_cost = 80
        config.evolution.max_agents = 4

        state = _make_state_with_positions(
            config,
            agent_pos=[[10, 10]],
            food_pos=[[19, 19]],
            energy=[160.0],
        )
        # next_agent_id starts at 1 (num_agents=1)
        assert state.next_agent_id == 1

        actions = jnp.array([5], dtype=jnp.int32)
        new_state, _, _, _ = step(state, actions, config)

        # Offspring ID should be 1 (the value of next_agent_id before birth)
        assert new_state.agent_ids[1] == 1
        # Counter should now be 2
        assert new_state.next_agent_id == 2

    def test_spawn_parent_id_set(self):
        """Offspring's agent_parent_ids is set to the parent's agent ID."""
        config = Config()
        config.env.grid_size = 20
        config.evolution.starting_energy = 160
        config.evolution.food_energy = 0
        config.evolution.energy_per_step = 0
        config.evolution.reproduce_threshold = 150
        config.evolution.reproduce_cost = 80
        config.evolution.max_agents = 4

        state = _make_state_with_positions(
            config,
            agent_pos=[[10, 10]],
            food_pos=[[19, 19]],
            energy=[160.0],
        )

        actions = jnp.array([5], dtype=jnp.int32)
        new_state, _, _, _ = step(state, actions, config)

        # Parent is agent 0 with ID 0
        assert new_state.agent_parent_ids[1] == 0

    def test_spawn_reuses_dead_agent_slot(self):
        """Offspring can spawn into a slot previously occupied by a dead agent."""
        config = Config()
        config.env.grid_size = 20
        config.evolution.starting_energy = 160
        config.evolution.food_energy = 0
        config.evolution.energy_per_step = 0
        config.evolution.reproduce_threshold = 150
        config.evolution.reproduce_cost = 80
        config.evolution.max_agents = 4

        # 3 agents: agent 1 is dead (slot reusable)
        state = _make_state_with_positions(
            config,
            agent_pos=[[5, 5], [10, 10], [15, 15]],
            food_pos=[[19, 19]],
            energy=[160.0, 50.0, 100.0],
        )
        # Kill agent 1
        new_alive = state.agent_alive.at[1].set(False)
        new_energy = state.agent_energy.at[1].set(0.0)
        state = state.replace(agent_alive=new_alive, agent_energy=new_energy)

        # Agent 0 reproduces
        actions = jnp.array([5, 0, 0], dtype=jnp.int32)
        new_state, _, _, info = step(state, actions, config)

        assert info["births_this_step"] == 1
        # Offspring should be in slot 1 (the dead agent's slot — first free)
        assert new_state.agent_alive[1]
        assert jnp.isclose(new_state.agent_energy[1], 80.0)
        # New agent ID should be next_agent_id (3, since 3 agents were initialized)
        assert new_state.agent_ids[1] == 3
        assert new_state.agent_parent_ids[1] == 0

    def test_spawn_multiple_generations(self):
        """Test that offspring can themselves reproduce (multi-generation)."""
        config = Config()
        config.env.grid_size = 20
        config.evolution.starting_energy = 160
        config.evolution.food_energy = 0
        config.evolution.energy_per_step = 0
        config.evolution.reproduce_threshold = 150
        config.evolution.reproduce_cost = 80
        config.evolution.max_agents = 8

        state = _make_state_with_positions(
            config,
            agent_pos=[[10, 10]],
            food_pos=[[19, 19]],
            energy=[160.0],
        )

        # Gen 1: agent 0 reproduces -> offspring at slot 1
        actions = jnp.array([5], dtype=jnp.int32)
        state, _, _, info = step(state, actions, config)
        assert info["births_this_step"] == 1
        assert state.agent_alive[1]

        # Give offspring enough energy to reproduce
        new_energy = state.agent_energy.at[1].set(160.0)
        state = state.replace(agent_energy=new_energy)

        # Gen 2: offspring (slot 1) reproduces -> grandchild at slot 2
        actions_gen2 = jnp.array([0, 5], dtype=jnp.int32)
        state, _, _, info = step(state, actions_gen2, config)
        assert info["births_this_step"] == 1
        assert state.agent_alive[2]
        # Grandchild's parent should be agent at slot 1 (ID=1)
        assert state.agent_parent_ids[2] == 1

    def test_spawn_jit_compatible(self):
        """Test that spawning works correctly under JIT."""
        config = Config()
        config.env.grid_size = 20
        config.evolution.starting_energy = 160
        config.evolution.food_energy = 0
        config.evolution.energy_per_step = 0
        config.evolution.reproduce_threshold = 150
        config.evolution.reproduce_cost = 80
        config.evolution.max_agents = 4

        state = _make_state_with_positions(
            config,
            agent_pos=[[10, 10]],
            food_pos=[[19, 19]],
            energy=[160.0],
        )

        actions = jnp.array([5], dtype=jnp.int32)

        @jax.jit
        def jit_step(s, a):
            return step(s, a, config)

        new_state, _, _, info = jit_step(state, actions)

        assert info["births_this_step"] == 1
        assert new_state.agent_alive[1]
        assert jnp.isclose(new_state.agent_energy[1], 80.0)
        assert new_state.agent_ids[1] == 1
        assert new_state.agent_parent_ids[1] == 0
        assert new_state.next_agent_id == 2


def _make_per_agent_params(config, network, key):
    """Create per-agent params: each leaf has shape (max_agents, ...)."""
    from src.environment.obs import obs_dim

    max_agents = config.evolution.max_agents
    observation_dim = obs_dim(config)
    dummy_obs = jnp.zeros((observation_dim,))
    single_params = network.init(key, dummy_obs)
    return jax.tree.map(
        lambda leaf: jnp.broadcast_to(leaf[None], (max_agents,) + leaf.shape).copy(),
        single_params,
    )


def _make_state_with_params(config, agent_pos, food_pos, energy, network, key):
    """Helper to create a controlled EnvState with per-agent params."""
    max_agents = config.evolution.max_agents
    num_food = len(food_pos)
    config.env.num_food = num_food
    num_agents = len(agent_pos)
    config.env.num_agents = num_agents
    grid_size = config.env.grid_size

    k1, k2 = jax.random.split(key)

    agent_positions = jnp.zeros((max_agents, 2), dtype=jnp.int32)
    agent_positions = agent_positions.at[:num_agents].set(
        jnp.array(agent_pos, dtype=jnp.int32)
    )

    food_positions = jnp.array(food_pos, dtype=jnp.int32)
    food_collected = jnp.zeros((num_food,), dtype=jnp.bool_)

    field_state = create_field(
        height=grid_size, width=grid_size, channels=config.field.num_channels
    )

    energy_vals = jnp.array(energy, dtype=jnp.float32)
    agent_energy = jnp.zeros((max_agents,), dtype=jnp.float32)
    agent_energy = agent_energy.at[:num_agents].set(energy_vals[:num_agents])

    agent_alive = jnp.zeros((max_agents,), dtype=jnp.bool_)
    agent_alive = agent_alive.at[:num_agents].set(True)

    agent_ids = jnp.full((max_agents,), -1, dtype=jnp.int32)
    agent_ids = agent_ids.at[:num_agents].set(
        jnp.arange(num_agents, dtype=jnp.int32)
    )

    agent_parent_ids = jnp.full((max_agents,), -1, dtype=jnp.int32)
    next_agent_id = jnp.int32(num_agents)

    agent_birth_step = jnp.full((max_agents,), -1, dtype=jnp.int32)
    agent_birth_step = agent_birth_step.at[:num_agents].set(0)

    agent_params = _make_per_agent_params(config, network, k1)

    return EnvState(
        agent_positions=agent_positions,
        food_positions=food_positions,
        food_collected=food_collected,
        field_state=field_state,
        step=jnp.int32(0),
        key=k2,
        agent_energy=agent_energy,
        agent_alive=agent_alive,
        agent_ids=agent_ids,
        agent_parent_ids=agent_parent_ids,
        next_agent_id=next_agent_id,
        agent_birth_step=agent_birth_step,
        agent_params=agent_params,
    )


class TestInheritance:
    """Tests for US-009: Weight inheritance with mutation."""

    def test_mutate_params_adds_noise(self):
        """mutate_params adds Gaussian noise to all leaves."""
        key = jax.random.PRNGKey(0)
        network = ActorCritic(hidden_dims=(16,), num_actions=6)
        dummy_obs = jnp.zeros((10,))
        params = network.init(key, dummy_obs)

        mutated = mutate_params(params, jax.random.PRNGKey(1), 0.01)

        # Params should differ after mutation
        leaves_orig = jax.tree_util.tree_leaves(params)
        leaves_mut = jax.tree_util.tree_leaves(mutated)
        for orig, mut in zip(leaves_orig, leaves_mut):
            assert not jnp.allclose(orig, mut), "Mutated params should differ from original"
            # Difference should be small (std=0.01)
            assert jnp.max(jnp.abs(orig - mut)) < 0.1

    def test_mutate_params_zero_std_no_change(self):
        """mutate_params with std=0 returns identical params."""
        key = jax.random.PRNGKey(0)
        network = ActorCritic(hidden_dims=(16,), num_actions=6)
        dummy_obs = jnp.zeros((10,))
        params = network.init(key, dummy_obs)

        mutated = mutate_params(params, jax.random.PRNGKey(1), 0.0)

        leaves_orig = jax.tree_util.tree_leaves(params)
        leaves_mut = jax.tree_util.tree_leaves(mutated)
        for orig, mut in zip(leaves_orig, leaves_mut):
            assert jnp.allclose(orig, mut)

    def test_copy_agent_params(self):
        """copy_agent_params copies one agent slot to another."""
        config = Config()
        config.env.grid_size = 10
        config.evolution.max_agents = 4
        network = ActorCritic(hidden_dims=(16,), num_actions=6)
        key = jax.random.PRNGKey(0)

        per_agent = _make_per_agent_params(config, network, key)

        # Modify slot 0 to have distinct values
        per_agent = jax.tree.map(
            lambda leaf: leaf.at[0].set(leaf[0] * 2.0 + 1.0),
            per_agent,
        )

        # Copy slot 0 -> slot 2
        copied = copy_agent_params(per_agent, parent_idx=0, child_idx=2)

        leaves = jax.tree_util.tree_leaves(copied)
        for leaf in leaves:
            assert jnp.allclose(leaf[0], leaf[2]), "Child slot should match parent"

    def test_mutate_agent_params_copies_with_noise(self):
        """mutate_agent_params copies parent to child with added noise."""
        config = Config()
        config.env.grid_size = 10
        config.evolution.max_agents = 4
        network = ActorCritic(hidden_dims=(16,), num_actions=6)
        key = jax.random.PRNGKey(0)

        per_agent = _make_per_agent_params(config, network, key)

        # Modify slot 0 to be distinct
        per_agent = jax.tree.map(
            lambda leaf: leaf.at[0].set(leaf[0] * 3.0 + 0.5),
            per_agent,
        )

        mutated = mutate_agent_params(
            per_agent, parent_idx=0, child_idx=2,
            key=jax.random.PRNGKey(42), mutation_std=0.01,
        )

        # Child should be close to parent but not identical
        leaves_orig = jax.tree_util.tree_leaves(per_agent)
        leaves_mut = jax.tree_util.tree_leaves(mutated)
        for orig, mut in zip(leaves_orig, leaves_mut):
            parent_vals = orig[0]
            child_vals = mut[2]
            assert not jnp.allclose(parent_vals, child_vals), "Child should differ from parent"
            assert jnp.allclose(parent_vals, child_vals, atol=0.1), "Child should be close to parent"
            # Slot 1 should be unchanged
            assert jnp.allclose(orig[1], mut[1]), "Other slots should be unchanged"

    def test_reproduction_inherits_params(self):
        """Reproduction in step() copies and mutates parent params to child."""
        config = Config()
        config.env.grid_size = 20
        config.evolution.starting_energy = 160
        config.evolution.food_energy = 0
        config.evolution.energy_per_step = 0
        config.evolution.reproduce_threshold = 150
        config.evolution.reproduce_cost = 80
        config.evolution.max_agents = 4
        config.evolution.mutation_std = 0.01

        network = ActorCritic(hidden_dims=(16,), num_actions=6)
        key = jax.random.PRNGKey(0)

        state = _make_state_with_params(
            config,
            agent_pos=[[10, 10]],
            food_pos=[[19, 19]],
            energy=[160.0],
            network=network,
            key=key,
        )

        actions = jnp.array([5], dtype=jnp.int32)
        new_state, _, _, info = step(state, actions, config)

        assert info["births_this_step"] == 1
        assert new_state.agent_params is not None

        # Child params (slot 1) should be close to parent params (slot 0) but not identical
        parent_leaves = jax.tree_util.tree_leaves(
            jax.tree.map(lambda x: x[0], new_state.agent_params)
        )
        child_leaves = jax.tree_util.tree_leaves(
            jax.tree.map(lambda x: x[1], new_state.agent_params)
        )
        for p, c in zip(parent_leaves, child_leaves):
            assert not jnp.allclose(p, c), "Child params should differ from parent (mutation)"
            assert jnp.allclose(p, c, atol=0.1), "Child params should be close to parent"

    def test_reproduction_without_params_still_works(self):
        """Reproduction without agent_params (None) works as before."""
        config = Config()
        config.env.grid_size = 20
        config.evolution.starting_energy = 160
        config.evolution.food_energy = 0
        config.evolution.energy_per_step = 0
        config.evolution.reproduce_threshold = 150
        config.evolution.reproduce_cost = 80
        config.evolution.max_agents = 4

        state = _make_state_with_positions(
            config,
            agent_pos=[[10, 10]],
            food_pos=[[19, 19]],
            energy=[160.0],
        )

        assert state.agent_params is None
        actions = jnp.array([5], dtype=jnp.int32)
        new_state, _, _, info = step(state, actions, config)

        assert info["births_this_step"] == 1
        assert new_state.agent_params is None

    def test_inheritance_jit_compatible(self):
        """Weight inheritance works under JIT compilation."""
        config = Config()
        config.env.grid_size = 20
        config.evolution.starting_energy = 160
        config.evolution.food_energy = 0
        config.evolution.energy_per_step = 0
        config.evolution.reproduce_threshold = 150
        config.evolution.reproduce_cost = 80
        config.evolution.max_agents = 4
        config.evolution.mutation_std = 0.01

        network = ActorCritic(hidden_dims=(16,), num_actions=6)
        key = jax.random.PRNGKey(0)

        state = _make_state_with_params(
            config,
            agent_pos=[[10, 10]],
            food_pos=[[19, 19]],
            energy=[160.0],
            network=network,
            key=key,
        )

        actions = jnp.array([5], dtype=jnp.int32)

        @jax.jit
        def jit_step(s, a):
            return step(s, a, config)

        new_state, _, _, info = jit_step(state, actions)

        assert info["births_this_step"] == 1
        assert new_state.agent_params is not None

        # Verify child params differ from parent
        parent_leaves = jax.tree_util.tree_leaves(
            jax.tree.map(lambda x: x[0], new_state.agent_params)
        )
        child_leaves = jax.tree_util.tree_leaves(
            jax.tree.map(lambda x: x[1], new_state.agent_params)
        )
        for p, c in zip(parent_leaves, child_leaves):
            assert not jnp.allclose(p, c)

    def test_multi_generation_inheritance(self):
        """Test that weight inheritance works across multiple generations."""
        config = Config()
        config.env.grid_size = 20
        config.evolution.starting_energy = 160
        config.evolution.food_energy = 0
        config.evolution.energy_per_step = 0
        config.evolution.reproduce_threshold = 150
        config.evolution.reproduce_cost = 80
        config.evolution.max_agents = 8
        config.evolution.mutation_std = 0.01

        network = ActorCritic(hidden_dims=(16,), num_actions=6)
        key = jax.random.PRNGKey(0)

        state = _make_state_with_params(
            config,
            agent_pos=[[10, 10]],
            food_pos=[[19, 19]],
            energy=[160.0],
            network=network,
            key=key,
        )

        # Gen 1: agent 0 reproduces -> offspring at slot 1
        actions = jnp.array([5], dtype=jnp.int32)
        state, _, _, info = step(state, actions, config)
        assert info["births_this_step"] == 1

        # Give offspring enough energy to reproduce
        new_energy = state.agent_energy.at[1].set(160.0)
        state = state.replace(agent_energy=new_energy)

        # Gen 2: offspring (slot 1) reproduces -> grandchild at slot 2
        actions_gen2 = jnp.array([0, 5], dtype=jnp.int32)
        state, _, _, info = step(state, actions_gen2, config)
        assert info["births_this_step"] == 1

        # Grandchild (slot 2) should be close to parent (slot 1),
        # and parent (slot 1) close to grandparent (slot 0),
        # but grandchild should have accumulated more divergence from grandparent
        gp_leaves = jax.tree_util.tree_leaves(
            jax.tree.map(lambda x: x[0], state.agent_params)
        )
        p_leaves = jax.tree_util.tree_leaves(
            jax.tree.map(lambda x: x[1], state.agent_params)
        )
        gc_leaves = jax.tree_util.tree_leaves(
            jax.tree.map(lambda x: x[2], state.agent_params)
        )
        for gp, p, gc in zip(gp_leaves, p_leaves, gc_leaves):
            # Parent close to grandparent
            assert jnp.allclose(gp, p, atol=0.1)
            # Grandchild close to parent
            assert jnp.allclose(p, gc, atol=0.1)

    def test_mutate_params_jit_compatible(self):
        """mutate_params works under JIT."""
        key = jax.random.PRNGKey(0)
        network = ActorCritic(hidden_dims=(16,), num_actions=6)
        dummy_obs = jnp.zeros((10,))
        params = network.init(key, dummy_obs)

        @jax.jit
        def jit_mutate(p, k):
            return mutate_params(p, k, 0.01)

        mutated = jit_mutate(params, jax.random.PRNGKey(1))
        leaves_orig = jax.tree_util.tree_leaves(params)
        leaves_mut = jax.tree_util.tree_leaves(mutated)
        for orig, mut in zip(leaves_orig, leaves_mut):
            assert not jnp.allclose(orig, mut)
