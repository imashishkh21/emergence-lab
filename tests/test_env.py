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


class TestEnvReset:
    """Tests for US-008: Environment reset."""
    
    def test_reset(self):
        """Test that reset creates valid initial state."""
        from src.environment.env import reset
        from src.configs import Config
        
        config = Config()
        key = jax.random.PRNGKey(42)
        
        state = reset(key, config)
        
        # Check shapes
        assert state.agent_positions.shape == (config.env.num_agents, 2)
        assert state.food_positions.shape == (config.env.num_food, 2)
        assert state.step == 0
        
        # Check positions are within bounds
        assert jnp.all(state.agent_positions >= 0)
        assert jnp.all(state.agent_positions < config.env.grid_size)
        assert jnp.all(state.food_positions >= 0)
        assert jnp.all(state.food_positions < config.env.grid_size)
    
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
            # Check all positions are unique
            positions = state.agent_positions
            for i in range(config.env.num_agents):
                for j in range(i + 1, config.env.num_agents):
                    assert not jnp.allclose(positions[i], positions[j])

    def test_reset_field_initialized_fresh(self):
        """Test that the field is initialized to zeros on reset."""
        from src.environment.env import reset
        from src.configs import Config

        config = Config()
        key = jax.random.PRNGKey(42)

        state = reset(key, config)

        # Field should be all zeros
        assert jnp.allclose(state.field_state.values, 0.0)
        # Field shape should match grid and channel config
        assert state.field_state.values.shape == (
            config.env.grid_size, config.env.grid_size, config.field.num_channels
        )

    def test_reset_food_not_collected(self):
        """Test that no food is collected on reset."""
        from src.environment.env import reset
        from src.configs import Config

        config = Config()
        key = jax.random.PRNGKey(42)

        state = reset(key, config)

        # No food should be collected initially
        assert not jnp.any(state.food_collected)


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
        assert rewards.shape == (config.env.num_agents,)
        
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
        from src.environment.obs import get_observations
        from src.configs import Config
        
        config = Config()
        key = jax.random.PRNGKey(42)
        
        state = reset(key, config)
        obs = get_observations(state, config)
        
        # Should have one observation per agent
        assert obs.shape[0] == config.env.num_agents
    
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
        
        # Should have batch dimension
        assert states.agent_positions.shape == (8, config.env.num_agents, 2)
    
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
        
        assert rewards.shape == (8, config.env.num_agents)
        assert new_states.agent_positions.shape == (8, config.env.num_agents, 2)
    
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
