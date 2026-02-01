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
