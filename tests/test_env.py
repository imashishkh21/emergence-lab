"""Tests for the environment module."""

import pytest
import jax
import jax.numpy as jnp


class TestEnvState:
    """Tests for EnvState dataclass."""
    
    def test_env_state(self):
        """Test that EnvState has required fields."""
        from src.environment.state import EnvState
        from src.configs import Config
        
        config = Config()
        key = jax.random.PRNGKey(42)
        
        # This should work after state.py is implemented
        state = EnvState.create(key, config)
        
        assert hasattr(state, 'agent_positions')
        assert hasattr(state, 'food_positions')
        assert hasattr(state, 'field_state')
        assert hasattr(state, 'step')


class TestEnvReset:
    """Tests for environment reset."""
    
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


class TestEnvStep:
    """Tests for environment step."""
    
    def test_step(self):
        """Test that step updates state correctly."""
        from src.environment.env import reset, step
        from src.configs import Config
        
        config = Config()
        key = jax.random.PRNGKey(42)
        
        state = reset(key, config)
        
        # All agents move right (action=4)
        actions = jnp.full((config.env.num_agents,), 4, dtype=jnp.int32)
        
        new_state, rewards, dones, info = step(state, actions, config)
        
        # Step should increment
        assert new_state.step == 1
        
        # Rewards should be array of correct shape
        assert rewards.shape == (config.env.num_agents,)
        
        # Dones should be boolean
        assert dones.dtype == jnp.bool_


class TestObservations:
    """Tests for observation function."""
    
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
        
        # Observations should be normalized
        assert jnp.all(obs >= -1.0)
        assert jnp.all(obs <= 1.0)


class TestVecEnv:
    """Tests for vectorized environment."""
    
    def test_vec_env(self):
        """Test that vectorized env runs multiple envs in parallel."""
        from src.environment.vec_env import VecEnv
        from src.configs import Config
        
        config = Config()
        config.train.num_envs = 8
        
        vec_env = VecEnv(config)
        key = jax.random.PRNGKey(42)
        
        states = vec_env.reset(key)
        
        # Should have batch dimension
        assert states.agent_positions.shape == (8, config.env.num_agents, 2)
        
        # Step with random actions
        actions = jax.random.randint(
            key, (8, config.env.num_agents), 0, 5
        )
        new_states, rewards, dones, info = vec_env.step(states, actions)
        
        assert rewards.shape == (8, config.env.num_agents)
