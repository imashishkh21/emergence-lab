"""Tests for US-010: Update Observations for Variable Population."""

import jax
import jax.numpy as jnp
import pytest

from src.configs import Config
from src.environment.env import reset, step
from src.environment.obs import get_observations, obs_dim


def _make_config(**kwargs):
    """Create a config with small grid for testing."""
    config = Config()
    config.env.grid_size = 10
    config.env.num_agents = 4
    config.env.num_food = 5
    config.env.observation_radius = 3
    config.field.num_channels = 2
    config.evolution.max_agents = 8
    config.evolution.starting_energy = 100
    config.evolution.max_energy = 200
    config.evolution.food_energy = 50
    config.evolution.energy_per_step = 1
    for k, v in kwargs.items():
        parts = k.split(".")
        obj = config
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], v)
    return config


class TestObsDim:
    """Tests for obs_dim with energy component."""

    def test_obs_dim_includes_energy(self):
        """obs_dim should include 1 extra dimension for energy."""
        config = _make_config()
        dim = obs_dim(config)
        radius = config.env.observation_radius
        patch_size = (2 * radius + 1) ** 2
        field_dim = patch_size * config.field.num_channels
        food_dim = 5 * 3  # K_NEAREST_FOOD * 3
        expected = 3 + field_dim + food_dim  # 2 pos + 1 energy + field + food
        assert dim == expected

    def test_obs_dim_changed_from_phase1(self):
        """obs_dim should be 1 more than it would be without energy."""
        config = _make_config()
        dim = obs_dim(config)
        radius = config.env.observation_radius
        patch_size = (2 * radius + 1) ** 2
        field_dim = patch_size * config.field.num_channels
        food_dim = 5 * 3
        phase1_dim = 2 + field_dim + food_dim
        assert dim == phase1_dim + 1


class TestObservationShape:
    """Tests for observation output shape."""

    def test_returns_max_agents_shape(self):
        """get_observations should return (max_agents, obs_dim)."""
        config = _make_config()
        key = jax.random.PRNGKey(42)
        state = reset(key, config)
        obs = get_observations(state, config)
        assert obs.shape == (config.evolution.max_agents, obs_dim(config))

    def test_shape_with_different_max_agents(self):
        """Shape adapts to different max_agents settings."""
        config = _make_config(**{"evolution.max_agents": 16})
        key = jax.random.PRNGKey(42)
        state = reset(key, config)
        obs = get_observations(state, config)
        assert obs.shape == (16, obs_dim(config))


class TestEnergyObservation:
    """Tests for energy in observation."""

    def test_energy_normalized(self):
        """Energy component should be normalized to [0, 1]."""
        config = _make_config()
        key = jax.random.PRNGKey(42)
        state = reset(key, config)
        obs = get_observations(state, config)

        # Energy is the 3rd component (index 2) in each agent's observation
        alive_energy_obs = obs[:config.env.num_agents, 2]

        # starting_energy / max_energy = 100 / 200 = 0.5
        expected = config.evolution.starting_energy / config.evolution.max_energy
        assert jnp.allclose(alive_energy_obs, expected, atol=1e-5)

    def test_energy_at_max(self):
        """Energy at max_energy should give normalized value of 1.0."""
        config = _make_config()
        key = jax.random.PRNGKey(42)
        state = reset(key, config)

        # Set first agent's energy to max
        new_energy = state.agent_energy.at[0].set(float(config.evolution.max_energy))
        state = state.replace(agent_energy=new_energy)

        obs = get_observations(state, config)
        assert jnp.isclose(obs[0, 2], 1.0)

    def test_energy_at_zero(self):
        """Energy at 0 should give normalized value of 0.0."""
        config = _make_config()
        key = jax.random.PRNGKey(42)
        state = reset(key, config)

        # Set first agent's energy to 0 (still alive for testing)
        new_energy = state.agent_energy.at[0].set(0.0)
        state = state.replace(agent_energy=new_energy)

        obs = get_observations(state, config)
        assert jnp.isclose(obs[0, 2], 0.0)

    def test_energy_clamped_above_max(self):
        """Energy above max should clamp to 1.0."""
        config = _make_config()
        key = jax.random.PRNGKey(42)
        state = reset(key, config)

        # Set energy above max
        new_energy = state.agent_energy.at[0].set(999.0)
        state = state.replace(agent_energy=new_energy)

        obs = get_observations(state, config)
        assert jnp.isclose(obs[0, 2], 1.0)


class TestDeadAgentObservations:
    """Tests for dead agents receiving zero observations."""

    def test_dead_agents_get_zero_obs(self):
        """Dead agents (agent_alive=False) should have all-zero observations."""
        config = _make_config()
        key = jax.random.PRNGKey(42)
        state = reset(key, config)

        obs = get_observations(state, config)

        # Dead agent slots (indices num_agents to max_agents) should be all zeros
        dead_obs = obs[config.env.num_agents:]
        assert jnp.allclose(dead_obs, 0.0)

    def test_killed_agent_gets_zero_obs(self):
        """An agent that was alive but dies should get zero observations."""
        config = _make_config()
        key = jax.random.PRNGKey(42)
        state = reset(key, config)

        # Kill agent 0
        new_alive = state.agent_alive.at[0].set(False)
        state = state.replace(agent_alive=new_alive)

        obs = get_observations(state, config)
        assert jnp.allclose(obs[0], 0.0)

    def test_alive_agents_nonzero_obs(self):
        """Alive agents should have non-zero observations (at least position)."""
        config = _make_config()
        key = jax.random.PRNGKey(42)
        state = reset(key, config)

        obs = get_observations(state, config)

        # At least the first alive agent should have non-zero obs
        # (position is normalized, energy is 0.5 at start)
        alive_obs = obs[:config.env.num_agents]
        # Each alive agent should have at least some non-zero entries
        for i in range(config.env.num_agents):
            assert jnp.any(alive_obs[i] != 0.0)

    def test_mixed_alive_dead(self):
        """Test observations with a mix of alive and dead agents."""
        config = _make_config()
        key = jax.random.PRNGKey(42)
        state = reset(key, config)

        # Kill agents 1 and 3, keep 0 and 2 alive
        new_alive = state.agent_alive.at[1].set(False).at[3].set(False)
        state = state.replace(agent_alive=new_alive)

        obs = get_observations(state, config)

        # Agent 0: alive, should have non-zero obs
        assert jnp.any(obs[0] != 0.0)
        # Agent 1: dead, should be all zero
        assert jnp.allclose(obs[1], 0.0)
        # Agent 2: alive, should have non-zero obs
        assert jnp.any(obs[2] != 0.0)
        # Agent 3: dead, should be all zero
        assert jnp.allclose(obs[3], 0.0)


class TestObservationNormalization:
    """Tests for observation value ranges."""

    def test_all_obs_in_range(self):
        """All observation values should be in [-1, 1]."""
        config = _make_config()
        key = jax.random.PRNGKey(42)
        state = reset(key, config)

        obs = get_observations(state, config)

        assert jnp.all(obs >= -1.0)
        assert jnp.all(obs <= 1.0)

    def test_position_components_normalized(self):
        """Position components should be normalized to [-1, 1]."""
        config = _make_config()
        key = jax.random.PRNGKey(42)
        state = reset(key, config)

        obs = get_observations(state, config)

        # First 2 components are position
        alive_pos_obs = obs[:config.env.num_agents, :2]
        assert jnp.all(alive_pos_obs >= -1.0)
        assert jnp.all(alive_pos_obs <= 1.0)

    def test_energy_component_nonnegative(self):
        """Energy component should be in [0, 1] (not negative)."""
        config = _make_config()
        key = jax.random.PRNGKey(42)
        state = reset(key, config)

        obs = get_observations(state, config)

        # 3rd component (index 2) is energy for alive agents
        alive_energy = obs[:config.env.num_agents, 2]
        assert jnp.all(alive_energy >= 0.0)
        assert jnp.all(alive_energy <= 1.0)


class TestFoodObservationsWithAlive:
    """Tests for food observations with alive mask."""

    def test_dead_agents_no_food_obs(self):
        """Dead agents should have zero food observations."""
        config = _make_config()
        key = jax.random.PRNGKey(42)
        state = reset(key, config)

        # Kill all agents except first
        new_alive = jnp.zeros(config.evolution.max_agents, dtype=jnp.bool_)
        new_alive = new_alive.at[0].set(True)
        state = state.replace(agent_alive=new_alive)

        obs = get_observations(state, config)

        # Dead agents should have all zeros (including food section)
        d = obs_dim(config)
        radius = config.env.observation_radius
        patch_size = (2 * radius + 1) ** 2
        field_dim = patch_size * config.field.num_channels
        food_start = 3 + field_dim  # pos(2) + energy(1) + field

        dead_food_obs = obs[1:, food_start:]
        assert jnp.allclose(dead_food_obs, 0.0)


class TestObsJITCompatibility:
    """Tests for JIT compatibility."""

    def test_get_observations_jit(self):
        """get_observations should work under JIT."""
        config = _make_config()
        key = jax.random.PRNGKey(42)
        state = reset(key, config)

        @jax.jit
        def jit_obs(state):
            return get_observations(state, config)

        obs = jit_obs(state)
        assert obs.shape == (config.evolution.max_agents, obs_dim(config))

    def test_obs_dim_consistent_with_get_observations(self):
        """obs_dim should match the actual output dimension."""
        config = _make_config()
        key = jax.random.PRNGKey(42)
        state = reset(key, config)

        obs = get_observations(state, config)
        assert obs.shape[-1] == obs_dim(config)

    def test_observations_after_step(self):
        """Observations should work after environment steps."""
        config = _make_config()
        key = jax.random.PRNGKey(42)
        state = reset(key, config)

        # Take a step
        actions = jnp.zeros(config.env.num_agents, dtype=jnp.int32)
        state, _, _, _ = step(state, actions, config)

        obs = get_observations(state, config)
        assert obs.shape == (config.evolution.max_agents, obs_dim(config))
        # Values should still be in valid range
        assert jnp.all(obs >= -1.0)
        assert jnp.all(obs <= 1.0)

    def test_observations_after_death(self):
        """Observations should be correct after an agent dies."""
        config = _make_config()
        config.evolution.starting_energy = 2
        config.evolution.energy_per_step = 1
        config.evolution.food_energy = 0
        key = jax.random.PRNGKey(42)
        state = reset(key, config)

        # Step twice so energy goes to 0 and agents die
        actions = jnp.zeros(config.env.num_agents, dtype=jnp.int32)
        state, _, _, _ = step(state, actions, config)
        state, _, _, _ = step(state, actions, config)

        # All agents should be dead now
        assert jnp.sum(state.agent_alive) == 0

        obs = get_observations(state, config)
        # All dead, all zero
        assert jnp.allclose(obs, 0.0)
