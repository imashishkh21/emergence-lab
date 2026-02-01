"""Tests for energy mechanics (US-004, US-005, US-006)."""

import jax
import jax.numpy as jnp

from src.configs import Config
from src.environment.env import reset, step


class TestEnergyDrain:
    """Tests for US-004: Energy drain per step."""

    def test_energy_decreases_each_step(self):
        """Test that alive agents lose energy_per_step each step."""
        config = Config()
        key = jax.random.PRNGKey(42)

        state = reset(key, config)
        initial_energy = state.agent_energy.copy()

        actions = jnp.zeros((config.env.num_agents,), dtype=jnp.int32)
        new_state, _, _, _ = step(state, actions, config)

        # Alive agents should have lost energy_per_step
        alive_mask = state.agent_alive
        expected_energy = jnp.where(
            alive_mask,
            initial_energy - config.evolution.energy_per_step,
            initial_energy,
        )
        assert jnp.allclose(new_state.agent_energy, expected_energy)

    def test_energy_drain_only_alive_agents(self):
        """Test that dead agents don't lose energy."""
        config = Config()
        key = jax.random.PRNGKey(42)

        state = reset(key, config)

        # Dead slots should have 0 energy and stay at 0
        dead_energy_before = state.agent_energy[config.env.num_agents:]
        assert jnp.all(dead_energy_before == 0.0)

        actions = jnp.zeros((config.env.num_agents,), dtype=jnp.int32)
        new_state, _, _, _ = step(state, actions, config)

        dead_energy_after = new_state.agent_energy[config.env.num_agents:]
        assert jnp.all(dead_energy_after == 0.0)

    def test_energy_cannot_go_below_zero(self):
        """Test that energy is clamped at 0."""
        config = Config()
        config.evolution.energy_per_step = 200  # More than starting_energy
        key = jax.random.PRNGKey(42)

        state = reset(key, config)

        actions = jnp.zeros((config.env.num_agents,), dtype=jnp.int32)
        new_state, _, _, _ = step(state, actions, config)

        # Energy should be 0, not negative
        assert jnp.all(new_state.agent_energy >= 0.0)
        # Alive agents should have exactly 0
        alive_energy = new_state.agent_energy[:config.env.num_agents]
        assert jnp.all(alive_energy == 0.0)

    def test_energy_drain_multiple_steps(self):
        """Test energy drains correctly over multiple steps."""
        config = Config()
        config.evolution.starting_energy = 10
        config.evolution.energy_per_step = 3
        key = jax.random.PRNGKey(42)

        state = reset(key, config)
        actions = jnp.zeros((config.env.num_agents,), dtype=jnp.int32)

        # Step 1: 10 - 3 = 7
        state, _, _, _ = step(state, actions, config)
        alive_energy = state.agent_energy[:config.env.num_agents]
        assert jnp.allclose(alive_energy, 7.0)

        # Step 2: 7 - 3 = 4
        state, _, _, _ = step(state, actions, config)
        alive_energy = state.agent_energy[:config.env.num_agents]
        assert jnp.allclose(alive_energy, 4.0)

        # Step 3: 4 - 3 = 1
        state, _, _, _ = step(state, actions, config)
        alive_energy = state.agent_energy[:config.env.num_agents]
        assert jnp.allclose(alive_energy, 1.0)

        # Step 4: 1 - 3 = 0 (clamped)
        state, _, _, _ = step(state, actions, config)
        alive_energy = state.agent_energy[:config.env.num_agents]
        assert jnp.allclose(alive_energy, 0.0)

    def test_energy_drain_jit_compatible(self):
        """Test that energy drain works under JIT."""
        config = Config()
        key = jax.random.PRNGKey(42)

        state = reset(key, config)
        actions = jnp.zeros((config.env.num_agents,), dtype=jnp.int32)

        @jax.jit
        def jit_step(s, a):
            return step(s, a, config)

        new_state, _, _, _ = jit_step(state, actions)

        alive_energy = new_state.agent_energy[:config.env.num_agents]
        expected = config.evolution.starting_energy - config.evolution.energy_per_step
        assert jnp.allclose(alive_energy, float(expected))


class TestDeath:
    """Tests for US-005: Death from starvation."""

    def test_death_when_energy_zero(self):
        """Test that agents die when energy reaches 0."""
        config = Config()
        # Set energy so agents die after one step
        config.evolution.starting_energy = 1
        config.evolution.energy_per_step = 1
        key = jax.random.PRNGKey(42)

        state = reset(key, config)
        # Verify agents start alive
        assert jnp.all(state.agent_alive[:config.env.num_agents])

        actions = jnp.zeros((config.env.num_agents,), dtype=jnp.int32)
        new_state, _, _, info = step(state, actions, config)

        # All original agents should now be dead (energy 1 - 1 = 0)
        assert not jnp.any(new_state.agent_alive[:config.env.num_agents])
        # Energy should be 0
        assert jnp.all(new_state.agent_energy[:config.env.num_agents] == 0.0)

    def test_death_count_in_info(self):
        """Test that death count is reported in info dict."""
        config = Config()
        config.evolution.starting_energy = 1
        config.evolution.energy_per_step = 1
        key = jax.random.PRNGKey(42)

        state = reset(key, config)
        actions = jnp.zeros((config.env.num_agents,), dtype=jnp.int32)
        _, _, _, info = step(state, actions, config)

        assert "deaths_this_step" in info
        # All agents should die
        assert info["deaths_this_step"] == config.env.num_agents

    def test_death_no_deaths_with_enough_energy(self):
        """Test no deaths when agents have plenty of energy."""
        config = Config()
        config.evolution.starting_energy = 100
        config.evolution.energy_per_step = 1
        key = jax.random.PRNGKey(42)

        state = reset(key, config)
        actions = jnp.zeros((config.env.num_agents,), dtype=jnp.int32)
        new_state, _, _, info = step(state, actions, config)

        # All agents should still be alive
        assert jnp.all(new_state.agent_alive[:config.env.num_agents])
        assert info["deaths_this_step"] == 0

    def test_dead_agents_stay_in_arrays(self):
        """Test that dead agents remain in arrays but are masked out."""
        config = Config()
        config.evolution.starting_energy = 1
        config.evolution.energy_per_step = 1
        key = jax.random.PRNGKey(42)

        state = reset(key, config)
        original_positions = state.agent_positions.copy()
        original_ids = state.agent_ids.copy()

        actions = jnp.zeros((config.env.num_agents,), dtype=jnp.int32)
        new_state, _, _, _ = step(state, actions, config)

        # Agents are dead but their IDs remain
        assert jnp.all(new_state.agent_ids[:config.env.num_agents] == original_ids[:config.env.num_agents])
        # Alive mask is False
        assert not jnp.any(new_state.agent_alive[:config.env.num_agents])

    def test_dead_agents_dont_move(self):
        """Test that dead agents cannot move on subsequent steps."""
        config = Config()
        config.evolution.starting_energy = 1
        config.evolution.energy_per_step = 1
        key = jax.random.PRNGKey(42)

        state = reset(key, config)
        actions = jnp.zeros((config.env.num_agents,), dtype=jnp.int32)

        # Step 1: agents die
        state, _, _, _ = step(state, actions, config)
        positions_after_death = state.agent_positions.copy()

        # Step 2: try to move dead agents (action 1 = up)
        move_actions = jnp.ones((config.env.num_agents,), dtype=jnp.int32)
        state, _, _, _ = step(state, move_actions, config)

        # Dead agents should not have moved
        assert jnp.all(state.agent_positions[:config.env.num_agents] == positions_after_death[:config.env.num_agents])

    def test_partial_death(self):
        """Test that only some agents die when they have different energy levels."""
        config = Config()
        config.evolution.energy_per_step = 5
        key = jax.random.PRNGKey(42)

        state = reset(key, config)
        num_agents = config.env.num_agents
        max_agents = config.evolution.max_agents

        # Manually set different energy levels: first half low, second half high
        half = num_agents // 2
        new_energy = state.agent_energy.at[:half].set(3.0)  # Will die (3 - 5 = 0)
        new_energy = new_energy.at[half:num_agents].set(100.0)  # Will survive
        state = state.replace(agent_energy=new_energy)

        actions = jnp.zeros((num_agents,), dtype=jnp.int32)
        new_state, _, _, info = step(state, actions, config)

        # First half should be dead
        assert not jnp.any(new_state.agent_alive[:half])
        # Second half should be alive
        assert jnp.all(new_state.agent_alive[half:num_agents])
        # Death count should equal half
        assert info["deaths_this_step"] == half

    def test_death_jit_compatible(self):
        """Test that death logic works under JIT."""
        config = Config()
        config.evolution.starting_energy = 1
        config.evolution.energy_per_step = 1
        key = jax.random.PRNGKey(42)

        state = reset(key, config)
        actions = jnp.zeros((config.env.num_agents,), dtype=jnp.int32)

        @jax.jit
        def jit_step(s, a):
            return step(s, a, config)

        new_state, _, _, info = jit_step(state, actions)

        # All agents should be dead
        assert not jnp.any(new_state.agent_alive[:config.env.num_agents])
        assert info["deaths_this_step"] == config.env.num_agents
