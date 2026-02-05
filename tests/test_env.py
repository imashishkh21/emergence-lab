"""Tests for the environment module."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp


class TestEnvState:
    """Tests for US-007: EnvState dataclass."""
    
    def test_env_state(self):
        """Test that EnvState has required fields."""
        from src.environment.state import EnvState
        from src.configs import Config

        # Create a mock state to test structure
        config = Config()
        key = jax.random.PRNGKey(42)

        # Import after module exists
        from src.environment.state import create_env_state
        state = create_env_state(key, config)

        assert hasattr(state, 'agent_positions')
        assert hasattr(state, 'food_positions')
        assert hasattr(state, 'field_state')
        assert hasattr(state, 'step')
        assert hasattr(state, 'key')
        # Phase 2 evolution fields
        assert hasattr(state, 'agent_energy')
        assert hasattr(state, 'agent_alive')
        assert hasattr(state, 'agent_ids')
        assert hasattr(state, 'agent_parent_ids')
        assert hasattr(state, 'next_agent_id')


class TestEnvReset:
    """Tests for US-008: Environment reset."""
    
    def test_reset(self):
        """Test that reset creates valid initial state."""
        from src.environment.env import reset
        from src.configs import Config

        config = Config()
        key = jax.random.PRNGKey(42)

        state = reset(key, config)

        # Check shapes — agent_positions is now (max_agents, 2)
        assert state.agent_positions.shape == (config.evolution.max_agents, 2)
        assert state.food_positions.shape == (config.env.num_food, 2)
        assert state.step == 0

        # Check alive agent positions are within bounds
        alive = state.agent_alive
        alive_positions = state.agent_positions[:config.env.num_agents]
        assert jnp.all(alive_positions >= 0)
        assert jnp.all(alive_positions < config.env.grid_size)
        assert jnp.all(state.food_positions >= 0)
        assert jnp.all(state.food_positions < config.env.grid_size)

        # Check evolution field shapes
        assert state.agent_energy.shape == (config.evolution.max_agents,)
        assert state.agent_alive.shape == (config.evolution.max_agents,)
        assert state.agent_ids.shape == (config.evolution.max_agents,)
        assert state.agent_parent_ids.shape == (config.evolution.max_agents,)

        # Check alive mask
        assert jnp.sum(state.agent_alive) == config.env.num_agents
        assert jnp.all(state.agent_alive[:config.env.num_agents])
        assert not jnp.any(state.agent_alive[config.env.num_agents:])
    
    def test_reset_different_seeds(self):
        """Test that different seeds produce different states."""
        from src.environment.env import reset
        from src.configs import Config
        
        config = Config()
        
        state1 = reset(jax.random.PRNGKey(1), config)
        state2 = reset(jax.random.PRNGKey(2), config)
        
        # Positions should differ
        assert not jnp.allclose(state1.agent_positions, state2.agent_positions)
    
    def test_reset_no_agent_overlap(self):
        """Test that agents don't spawn on same position."""
        from src.environment.env import reset
        from src.configs import Config

        config = Config()
        config.env.num_agents = 4

        for seed in range(10):
            state = reset(jax.random.PRNGKey(seed), config)
            # Check all alive agent positions are unique
            positions = state.agent_positions[:config.env.num_agents]
            for i in range(config.env.num_agents):
                for j in range(i + 1, config.env.num_agents):
                    assert not jnp.allclose(positions[i], positions[j])

    def test_reset_field_initialized_fresh(self):
        """Test that the field is initialized correctly on reset.

        Ch1 (territory) is seeded around the nest area; other channels are zero.
        """
        from src.environment.env import reset
        from src.configs import Config

        config = Config()
        key = jax.random.PRNGKey(42)

        state = reset(key, config)

        # Field shape should match grid and channel config
        assert state.field_state.values.shape == (
            config.env.grid_size, config.env.grid_size, config.field.num_channels
        )
        # Ch0, Ch2, Ch3 should be all zeros
        assert jnp.allclose(state.field_state.values[:, :, 0], 0.0)
        if config.field.num_channels > 2:
            assert jnp.allclose(state.field_state.values[:, :, 2:], 0.0)
        # Ch1 (territory) should have nonzero values around the nest
        nest_center = config.env.grid_size // 2
        assert state.field_state.values[nest_center, nest_center, 1] == 1.0

    def test_reset_food_not_collected(self):
        """Test that no food is collected on reset."""
        from src.environment.env import reset
        from src.configs import Config

        config = Config()
        key = jax.random.PRNGKey(42)

        state = reset(key, config)

        # No food should be collected initially
        assert not jnp.any(state.food_collected)

    def test_reset_evolution_energy_values(self):
        """Test that agent energy is initialized correctly on reset."""
        from src.environment.env import reset
        from src.configs import Config

        config = Config()
        key = jax.random.PRNGKey(42)

        state = reset(key, config)

        # Alive agents get starting_energy
        alive_energy = state.agent_energy[:config.env.num_agents]
        assert jnp.all(alive_energy == config.evolution.starting_energy)

        # Dead slots have 0 energy
        dead_energy = state.agent_energy[config.env.num_agents:]
        assert jnp.all(dead_energy == 0.0)

    def test_reset_evolution_ids(self):
        """Test that agent IDs and parent IDs are initialized correctly."""
        from src.environment.env import reset
        from src.configs import Config

        config = Config()
        key = jax.random.PRNGKey(42)

        state = reset(key, config)
        num_agents = config.env.num_agents

        # Alive agents have sequential IDs [0, 1, ..., num_agents-1]
        alive_ids = state.agent_ids[:num_agents]
        expected_ids = jnp.arange(num_agents, dtype=jnp.int32)
        assert jnp.array_equal(alive_ids, expected_ids)

        # Dead slots have ID -1
        dead_ids = state.agent_ids[num_agents:]
        assert jnp.all(dead_ids == -1)

        # All parent IDs are -1 (no parents for original agents)
        assert jnp.all(state.agent_parent_ids == -1)

        # next_agent_id equals num_agents
        assert state.next_agent_id == num_agents

    def test_reset_dead_agent_positions(self):
        """Test that dead agent slots have position (0, 0)."""
        from src.environment.env import reset
        from src.configs import Config

        config = Config()
        key = jax.random.PRNGKey(42)

        state = reset(key, config)

        # Dead slots should be at (0, 0)
        dead_positions = state.agent_positions[config.env.num_agents:]
        assert jnp.all(dead_positions == 0)


class TestEnvStep:
    """Tests for US-009: Environment step."""
    
    def test_step(self):
        """Test that step updates state correctly."""
        from src.environment.env import reset, step
        from src.configs import Config
        
        config = Config()
        key = jax.random.PRNGKey(42)
        
        state = reset(key, config)
        
        # All agents stay still (action=0)
        actions = jnp.zeros((config.env.num_agents,), dtype=jnp.int32)
        
        new_state, rewards, dones, info = step(state, actions, config)
        
        # Step should increment
        assert new_state.step == 1
        
        # Rewards should be array of correct shape
        assert rewards.shape == (config.evolution.max_agents,)
        
        # Dones should be boolean
        assert dones.dtype == jnp.bool_
    
    def test_step_movement(self):
        """Test that movement actions work."""
        from src.environment.env import reset, step
        from src.configs import Config
        
        config = Config()
        config.env.grid_size = 10
        key = jax.random.PRNGKey(42)
        
        state = reset(key, config)
        initial_pos = state.agent_positions.copy()
        
        # All agents move right (action=4: right)
        actions = jnp.full((config.env.num_agents,), 4, dtype=jnp.int32)
        
        new_state, _, _, _ = step(state, actions, config)
        
        # Positions should change (unless at boundary)
        # At least some agents should have moved
        moved = jnp.any(new_state.agent_positions != initial_pos)
        assert moved
    
    def test_step_done_at_max_steps(self):
        """Test that episode ends at max_steps."""
        from src.environment.env import reset, step
        from src.configs import Config
        
        config = Config()
        config.env.max_steps = 10
        key = jax.random.PRNGKey(42)
        
        state = reset(key, config)
        actions = jnp.zeros((config.env.num_agents,), dtype=jnp.int32)
        
        # Step until max
        for i in range(10):
            state, _, done, _ = step(state, actions, config)
        
        assert done == True


class TestObservations:
    """Tests for US-010: Observation function."""
    
    def test_observations(self):
        """Test that observations have correct shape."""
        from src.environment.env import reset
        from src.environment.obs import get_observations, obs_dim
        from src.configs import Config

        config = Config()
        key = jax.random.PRNGKey(42)

        state = reset(key, config)
        obs = get_observations(state, config)

        # Should have one observation per agent slot (max_agents)
        assert obs.shape == (config.evolution.max_agents, obs_dim(config))
    
    def test_observations_normalized(self):
        """Test that observations are normalized."""
        from src.environment.env import reset
        from src.environment.obs import get_observations
        from src.configs import Config
        
        config = Config()
        key = jax.random.PRNGKey(42)
        
        state = reset(key, config)
        obs = get_observations(state, config)
        
        # Observations should be normalized
        assert jnp.all(obs >= -1.0)
        assert jnp.all(obs <= 1.0)


class TestVecEnv:
    """Tests for US-011: Vectorized environment."""
    
    def test_vec_env_reset(self):
        """Test that vectorized reset works."""
        from src.environment.vec_env import VecEnv
        from src.configs import Config
        
        config = Config()
        config.train.num_envs = 8
        
        vec_env = VecEnv(config)
        key = jax.random.PRNGKey(42)
        
        states = vec_env.reset(key)
        
        # Should have batch dimension — agent_positions is (max_agents, 2) per env
        assert states.agent_positions.shape == (8, config.evolution.max_agents, 2)
    
    def test_vec_env_step(self):
        """Test that vectorized step works."""
        from src.environment.vec_env import VecEnv
        from src.configs import Config
        
        config = Config()
        config.train.num_envs = 8
        
        vec_env = VecEnv(config)
        key = jax.random.PRNGKey(42)
        
        states = vec_env.reset(key)
        
        # Step with random actions
        actions = jax.random.randint(
            key, (8, config.env.num_agents), 0, 5
        )
        new_states, rewards, dones, info = vec_env.step(states, actions)
        
        assert rewards.shape == (8, config.evolution.max_agents)
        assert new_states.agent_positions.shape == (8, config.evolution.max_agents, 2)
    
    def test_vec_env_jit_compatible(self):
        """Test that VecEnv works with JIT."""
        from src.environment.vec_env import VecEnv
        from src.configs import Config
        
        config = Config()
        config.train.num_envs = 4
        
        vec_env = VecEnv(config)
        key = jax.random.PRNGKey(42)
        
        # JIT the step function
        @jax.jit
        def jit_step(states, actions):
            return vec_env.step(states, actions)
        
        states = vec_env.reset(key)
        actions = jnp.zeros((4, config.env.num_agents), dtype=jnp.int32)
        
        new_states, rewards, dones, info = jit_step(states, actions)

        assert new_states.step[0] == 1


class TestRender:
    """Tests for US-020: Environment rendering."""

    def test_render_frame_shape_and_type(self):
        """Test that render_frame returns a valid RGB image."""
        from src.environment.env import reset, step
        from src.environment.render import render_frame
        from src.configs import Config

        config = Config()
        key = jax.random.PRNGKey(42)

        state = reset(key, config)

        img = render_frame(state, config)

        # Must be uint8 RGB
        assert img.dtype == np.uint8
        assert img.ndim == 3
        assert img.shape[2] == 3

        # Must be at least 400x400
        assert img.shape[0] >= 400
        assert img.shape[1] >= 400

        # Must be square (grid_size * pixel_size)
        assert img.shape[0] == img.shape[1]

    def test_render_frame_with_field(self):
        """Test rendering after steps so field has values."""
        from src.environment.env import reset, step
        from src.environment.render import render_frame
        from src.configs import Config

        config = Config()
        key = jax.random.PRNGKey(42)

        state = reset(key, config)

        # Step a few times to get field activity
        actions = jnp.zeros((config.env.num_agents,), dtype=jnp.int32)
        for _ in range(5):
            state, _, _, _ = step(state, actions, config)

        img = render_frame(state, config)

        # Should still be valid
        assert img.dtype == np.uint8
        assert img.shape[0] >= 400
        assert img.shape[1] >= 400

    def test_render_frame_save_png(self, tmp_path):
        """Test that a rendered frame can be saved as PNG."""
        import imageio.v3 as iio
        from src.environment.env import reset, step
        from src.environment.render import render_frame
        from src.configs import Config

        config = Config()
        key = jax.random.PRNGKey(42)

        state = reset(key, config)

        # Take a few steps for visual interest
        actions = jnp.zeros((config.env.num_agents,), dtype=jnp.int32)
        for _ in range(3):
            state, _, _, _ = step(state, actions, config)

        img = render_frame(state, config)

        # Save as PNG
        out_path = tmp_path / "test_frame.png"
        iio.imwrite(str(out_path), img)

        assert out_path.exists()
        assert out_path.stat().st_size > 0

        # Read back and verify
        loaded = iio.imread(str(out_path))
        assert loaded.shape == img.shape


class TestFoodRespawn:
    """Tests for US-012: Food respawn with scarcity."""

    def _make_state_with_food(self, config, key, food_positions, food_collected):
        """Create a state with specific food positions and collected status."""
        from src.environment.env import reset
        state = reset(key, config)
        state = state.replace(
            food_positions=jnp.array(food_positions, dtype=jnp.int32),
            food_collected=jnp.array(food_collected, dtype=jnp.bool_),
        )
        return state

    def test_food_respawn_config_default(self):
        """Test that food_respawn_prob has correct default."""
        from src.configs import Config
        config = Config()
        assert config.env.food_respawn_prob == 0.1

    def test_food_respawn_collected_food_can_respawn(self):
        """Test that collected food can respawn at new positions."""
        from src.environment.env import reset, step
        from src.configs import Config

        config = Config()
        config.env.grid_size = 10
        config.env.num_agents = 1
        config.env.num_food = 5
        config.env.food_respawn_prob = 1.0  # Always respawn
        config.evolution.max_agents = 4
        config.evolution.food_energy = 0  # No energy gain to simplify

        key = jax.random.PRNGKey(42)
        state = reset(key, config)

        # Mark all food as collected
        state = state.replace(food_collected=jnp.ones((5,), dtype=jnp.bool_))

        actions = jnp.zeros((1,), dtype=jnp.int32)
        new_state, _, _, _ = step(state, actions, config)

        # With prob=1.0, all collected food should respawn (marked as not collected)
        assert jnp.sum(new_state.food_collected) == 0

    def test_food_respawn_uncollected_food_unchanged(self):
        """Test that uncollected food is not affected by respawn."""
        from src.environment.env import reset, step
        from src.configs import Config

        config = Config()
        config.env.grid_size = 10
        config.env.num_agents = 1
        config.env.num_food = 5
        config.env.food_respawn_prob = 1.0
        config.evolution.max_agents = 4
        config.evolution.food_energy = 0

        key = jax.random.PRNGKey(42)
        state = reset(key, config)

        # No food collected — nothing should change about collected status
        state = state.replace(food_collected=jnp.zeros((5,), dtype=jnp.bool_))
        original_positions = state.food_positions.copy()

        actions = jnp.zeros((1,), dtype=jnp.int32)
        new_state, _, _, _ = step(state, actions, config)

        # None were collected, so positions should be unchanged
        # (any food collected THIS step might respawn, but if agents are far from food, none collected)
        # Place agents far from food to ensure no collection
        state = state.replace(
            agent_positions=jnp.zeros((config.evolution.max_agents, 2), dtype=jnp.int32),
            food_positions=jnp.full((5, 2), 9, dtype=jnp.int32),
        )
        new_state, _, _, _ = step(state, actions, config)
        # Food positions should be unchanged since nothing was collected
        assert jnp.array_equal(new_state.food_positions, jnp.full((5, 2), 9, dtype=jnp.int32))

    def test_food_respawn_zero_prob_no_respawn(self):
        """Test that food_respawn_prob=0 means no respawn."""
        from src.environment.env import reset, step
        from src.configs import Config

        config = Config()
        config.env.grid_size = 10
        config.env.num_agents = 1
        config.env.num_food = 5
        config.env.food_respawn_prob = 0.0  # Never respawn
        config.evolution.max_agents = 4
        config.evolution.food_energy = 0

        key = jax.random.PRNGKey(42)
        state = reset(key, config)

        # Mark all food as collected
        state = state.replace(food_collected=jnp.ones((5,), dtype=jnp.bool_))

        actions = jnp.zeros((1,), dtype=jnp.int32)
        new_state, _, _, _ = step(state, actions, config)

        # With prob=0, no food should respawn — all still collected
        assert jnp.sum(new_state.food_collected) == 5

    def test_food_respawn_positions_within_grid(self):
        """Test that respawned food is placed within grid bounds."""
        from src.environment.env import reset, step
        from src.configs import Config

        config = Config()
        config.env.grid_size = 10
        config.env.num_agents = 1
        config.env.num_food = 20
        config.env.food_respawn_prob = 1.0
        config.evolution.max_agents = 4
        config.evolution.food_energy = 0

        key = jax.random.PRNGKey(42)
        state = reset(key, config)

        # Mark all food as collected
        state = state.replace(food_collected=jnp.ones((20,), dtype=jnp.bool_))

        actions = jnp.zeros((1,), dtype=jnp.int32)
        new_state, _, _, _ = step(state, actions, config)

        # All respawned food should be within grid
        assert jnp.all(new_state.food_positions >= 0)
        assert jnp.all(new_state.food_positions < config.env.grid_size)

    def test_food_respawn_total_food_capped(self):
        """Test that total food never exceeds num_food."""
        from src.environment.env import reset, step
        from src.configs import Config

        config = Config()
        config.env.grid_size = 10
        config.env.num_agents = 1
        config.env.num_food = 5
        config.env.food_respawn_prob = 1.0
        config.evolution.max_agents = 4
        config.evolution.food_energy = 0

        key = jax.random.PRNGKey(42)
        state = reset(key, config)

        # Run multiple steps — food count should never exceed num_food
        actions = jnp.zeros((1,), dtype=jnp.int32)
        for _ in range(10):
            state, _, _, _ = step(state, actions, config)
            total_food = state.food_positions.shape[0]
            assert total_food == config.env.num_food  # Fixed array size

    def test_food_respawn_partial_collection(self):
        """Test respawn with only some food collected."""
        from src.environment.env import reset, step
        from src.configs import Config

        config = Config()
        config.env.grid_size = 10
        config.env.num_agents = 1
        config.env.num_food = 5
        config.env.food_respawn_prob = 1.0  # Always respawn
        config.evolution.max_agents = 4
        config.evolution.food_energy = 0

        key = jax.random.PRNGKey(42)
        state = reset(key, config)

        # Only first 2 of 5 food collected
        food_collected = jnp.array([True, True, False, False, False])
        state = state.replace(food_collected=food_collected)
        original_positions = state.food_positions.copy()

        # Place agents far from remaining food to prevent collection this step
        state = state.replace(
            agent_positions=jnp.zeros((config.evolution.max_agents, 2), dtype=jnp.int32),
            food_positions=jnp.full((5, 2), 9, dtype=jnp.int32),
        )

        actions = jnp.zeros((1,), dtype=jnp.int32)
        new_state, _, _, _ = step(state, actions, config)

        # The 2 collected food items should have respawned (not collected anymore)
        # The 3 uncollected should remain uncollected
        assert jnp.sum(new_state.food_collected) == 0  # All respawned or not collected

    def test_food_respawn_jit_compatible(self):
        """Test that food respawn works with JIT compilation."""
        from src.environment.env import reset, step
        from src.configs import Config

        config = Config()
        config.env.grid_size = 10
        config.env.num_agents = 2
        config.env.num_food = 5
        config.env.food_respawn_prob = 0.5
        config.evolution.max_agents = 4

        key = jax.random.PRNGKey(42)
        state = reset(key, config)

        @jax.jit
        def jit_step(s, a):
            return step(s, a, config)

        actions = jnp.zeros((2,), dtype=jnp.int32)
        new_state, rewards, done, info = jit_step(state, actions)

        assert new_state.food_positions.shape == (5, 2)
        assert new_state.food_collected.shape == (5,)


