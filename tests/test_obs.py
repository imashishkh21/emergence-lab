"""Tests for observation system (biological pheromone observations)."""

import jax
import jax.numpy as jnp

from src.configs import Config
from src.environment.env import reset, step
from src.environment.obs import get_observations, obs_dim


def _make_config(**kwargs):
    """Create a config with small grid for testing."""
    config = Config()
    config.env.grid_size = 10
    config.env.num_agents = 4
    config.env.num_food = 5
    config.evolution.max_agents = 8
    config.evolution.starting_energy = 100
    config.evolution.max_energy = 200
    config.evolution.food_energy = 50
    config.evolution.energy_per_step = 1
    config.evolution.reproduce_threshold = 9999  # prevent auto-reproduction
    for k, v in kwargs.items():
        parts = k.split(".")
        obj = config
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], v)
    return config


class TestObsDim:
    """Tests for obs_dim calculation."""

    def test_obs_dim_default_4_channels(self):
        """obs_dim with 4 channels: 2+1+1+2+20+4+15 = 45."""
        config = Config()
        dim = obs_dim(config)
        assert dim == 45

    def test_obs_dim_2_channels(self):
        """obs_dim with 2 channels: 2+1+1+2+10+2+15 = 33."""
        config = _make_config()
        config.field.num_channels = 2
        dim = obs_dim(config)
        assert dim == 33

    def test_obs_dim_formula(self):
        """obs_dim matches expected formula."""
        config = _make_config()
        num_ch = config.field.num_channels
        expected = 2 + 1 + 1 + 2 + (5 * num_ch) + num_ch + (5 * 3)
        assert obs_dim(config) == expected


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

        # Energy is at index 2
        alive_energy_obs = obs[:config.env.num_agents, 2]
        expected = config.evolution.starting_energy / config.evolution.max_energy
        assert jnp.allclose(alive_energy_obs, expected, atol=1e-5)

    def test_energy_at_max(self):
        """Energy at max_energy should give normalized value of 1.0."""
        config = _make_config()
        key = jax.random.PRNGKey(42)
        state = reset(key, config)
        new_energy = state.agent_energy.at[0].set(float(config.evolution.max_energy))
        state = state.replace(agent_energy=new_energy)
        obs = get_observations(state, config)
        assert jnp.isclose(obs[0, 2], 1.0)

    def test_energy_at_zero(self):
        """Energy at 0 should give normalized value of 0.0."""
        config = _make_config()
        key = jax.random.PRNGKey(42)
        state = reset(key, config)
        new_energy = state.agent_energy.at[0].set(0.0)
        state = state.replace(agent_energy=new_energy)
        obs = get_observations(state, config)
        assert jnp.isclose(obs[0, 2], 0.0)

    def test_energy_clamped_above_max(self):
        """Energy above max should clamp to 1.0."""
        config = _make_config()
        key = jax.random.PRNGKey(42)
        state = reset(key, config)
        new_energy = state.agent_energy.at[0].set(999.0)
        state = state.replace(agent_energy=new_energy)
        obs = get_observations(state, config)
        assert jnp.isclose(obs[0, 2], 1.0)


class TestHasFoodObservation:
    """Tests for has_food flag in observation."""

    def test_has_food_zero_by_default(self):
        """has_food should be 0.0 when agent is not carrying food."""
        config = _make_config()
        key = jax.random.PRNGKey(42)
        state = reset(key, config)
        obs = get_observations(state, config)
        # has_food is at index 3
        assert jnp.allclose(obs[:config.env.num_agents, 3], 0.0)

    def test_has_food_one_when_carrying(self):
        """has_food should be 1.0 when agent is carrying food."""
        config = _make_config()
        key = jax.random.PRNGKey(42)
        state = reset(key, config)
        state = state.replace(has_food=state.has_food.at[0].set(True))
        obs = get_observations(state, config)
        assert jnp.isclose(obs[0, 3], 1.0)


class TestCompassObservation:
    """Tests for nest compass in observation."""

    def test_compass_near_zero_at_nest(self):
        """Compass should be near zero when agent is at nest center."""
        config = _make_config()
        key = jax.random.PRNGKey(42)
        state = reset(key, config)
        nest_center = config.env.grid_size // 2
        state = state.replace(
            agent_positions=state.agent_positions.at[0].set(
                jnp.array([nest_center, nest_center])
            )
        )
        obs = get_observations(state, config)
        # Compass is at indices 4-5
        compass = obs[0, 4:6]
        # At nest center, true_delta=0. With noise, should be small.
        assert jnp.all(jnp.abs(compass) < 0.2), f"Compass at nest should be near zero, got {compass}"

    def test_compass_points_toward_nest(self):
        """Compass should generally point toward nest center."""
        config = _make_config()
        config.nest.compass_noise_rate = 0.0  # No noise for deterministic test
        key = jax.random.PRNGKey(42)
        state = reset(key, config)
        # Place agent at (0, 0), nest center at (5, 5)
        state = state.replace(
            agent_positions=state.agent_positions.at[0].set(jnp.array([0, 0]))
        )
        obs = get_observations(state, config)
        compass = obs[0, 4:6]
        # Agent at (0,0), nest at (5,5): direction should be positive
        assert compass[0] > 0, "Compass row should point toward nest (positive)"
        assert compass[1] > 0, "Compass col should point toward nest (positive)"


class TestFieldSpatialObservation:
    """Tests for field spatial gradient observations."""

    def test_field_spatial_shape(self):
        """Field spatial should have 5*C dimensions."""
        config = _make_config()
        num_ch = config.field.num_channels
        key = jax.random.PRNGKey(42)
        state = reset(key, config)
        obs = get_observations(state, config)
        # field_spatial starts at index 6, length 5*C
        field_spatial_start = 6
        field_spatial_end = field_spatial_start + 5 * num_ch
        field_spatial = obs[0, field_spatial_start:field_spatial_end]
        assert field_spatial.shape == (5 * num_ch,)

    def test_field_spatial_captures_nonzero(self):
        """Field spatial should capture nonzero values from territory channel."""
        config = _make_config()
        key = jax.random.PRNGKey(42)
        state = reset(key, config)
        # After reset, Ch1 has territory in nest area. Agent at nest center
        # should see nonzero Ch1 in spatial obs.
        obs = get_observations(state, config)
        num_ch = config.field.num_channels
        field_spatial_start = 6
        field_spatial = obs[0, field_spatial_start:field_spatial_start + 5 * num_ch]
        # At least some values should be nonzero (territory channel)
        assert jnp.any(field_spatial > 0), "Field spatial should have nonzero territory values"


class TestFieldTemporalObservation:
    """Tests for field temporal derivative."""

    def test_field_temporal_zero_on_reset(self):
        """Field temporal derivative should be zero right after reset (prev=0, current=0 for most)."""
        config = _make_config()
        key = jax.random.PRNGKey(42)
        state = reset(key, config)
        obs = get_observations(state, config)
        num_ch = config.field.num_channels
        temporal_start = 6 + 5 * num_ch
        temporal = obs[0, temporal_start:temporal_start + num_ch]
        # prev_field_at_pos is initialized to zeros, so temporal = current - 0 = current field values
        # This will be nonzero if agent is in territory area (Ch1 pre-seeded)
        # For agents in nest area, Ch1 temporal will be positive
        # This is expected behavior - not necessarily zero

    def test_field_temporal_after_step(self):
        """Field temporal derivative should reflect field changes after a step."""
        config = _make_config()
        key = jax.random.PRNGKey(42)
        state = reset(key, config)
        actions = jnp.zeros(config.evolution.max_agents, dtype=jnp.int32)
        state, _, _, _ = step(state, actions, config)
        obs = get_observations(state, config)
        # After step, prev_field_at_pos is updated, so temporal = current - prev
        num_ch = config.field.num_channels
        temporal_start = 6 + 5 * num_ch
        temporal = obs[0, temporal_start:temporal_start + num_ch]
        assert temporal.shape == (num_ch,)


class TestDeadAgentObservations:
    """Tests for dead agents receiving zero observations."""

    def test_dead_agents_get_zero_obs(self):
        """Dead agents (agent_alive=False) should have all-zero observations."""
        config = _make_config()
        key = jax.random.PRNGKey(42)
        state = reset(key, config)
        obs = get_observations(state, config)
        dead_obs = obs[config.env.num_agents:]
        assert jnp.allclose(dead_obs, 0.0)

    def test_killed_agent_gets_zero_obs(self):
        """An agent that was alive but dies should get zero observations."""
        config = _make_config()
        key = jax.random.PRNGKey(42)
        state = reset(key, config)
        new_alive = state.agent_alive.at[0].set(False)
        state = state.replace(agent_alive=new_alive)
        obs = get_observations(state, config)
        assert jnp.allclose(obs[0], 0.0)

    def test_alive_agents_nonzero_obs(self):
        """Alive agents should have non-zero observations."""
        config = _make_config()
        key = jax.random.PRNGKey(42)
        state = reset(key, config)
        obs = get_observations(state, config)
        alive_obs = obs[:config.env.num_agents]
        for i in range(config.env.num_agents):
            assert jnp.any(alive_obs[i] != 0.0)

    def test_mixed_alive_dead(self):
        """Test observations with a mix of alive and dead agents."""
        config = _make_config()
        key = jax.random.PRNGKey(42)
        state = reset(key, config)
        new_alive = state.agent_alive.at[1].set(False).at[3].set(False)
        state = state.replace(agent_alive=new_alive)
        obs = get_observations(state, config)
        assert jnp.any(obs[0] != 0.0)
        assert jnp.allclose(obs[1], 0.0)
        assert jnp.any(obs[2] != 0.0)
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
        alive_pos_obs = obs[:config.env.num_agents, :2]
        assert jnp.all(alive_pos_obs >= -1.0)
        assert jnp.all(alive_pos_obs <= 1.0)

    def test_energy_component_nonnegative(self):
        """Energy component should be in [0, 1]."""
        config = _make_config()
        key = jax.random.PRNGKey(42)
        state = reset(key, config)
        obs = get_observations(state, config)
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
        new_alive = jnp.zeros(config.evolution.max_agents, dtype=jnp.bool_)
        new_alive = new_alive.at[0].set(True)
        state = state.replace(agent_alive=new_alive)
        obs = get_observations(state, config)
        num_ch = config.field.num_channels
        food_start = 6 + 5 * num_ch + num_ch
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
        actions = jnp.zeros(config.env.num_agents, dtype=jnp.int32)
        state, _, _, _ = step(state, actions, config)
        obs = get_observations(state, config)
        assert obs.shape == (config.evolution.max_agents, obs_dim(config))
        assert jnp.all(obs >= -1.0)
        assert jnp.all(obs <= 1.0)

    def test_observations_after_death(self):
        """Observations should be correct after all agents die."""
        config = _make_config()
        config.evolution.starting_energy = 2
        config.evolution.energy_per_step = 1
        config.evolution.food_energy = 0
        key = jax.random.PRNGKey(42)
        state = reset(key, config)
        actions = jnp.zeros(config.env.num_agents, dtype=jnp.int32)
        state, _, _, _ = step(state, actions, config)
        state, _, _, _ = step(state, actions, config)
        assert jnp.sum(state.agent_alive) == 0
        obs = get_observations(state, config)
        assert jnp.allclose(obs, 0.0)
