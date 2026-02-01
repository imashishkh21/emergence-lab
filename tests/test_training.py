"""Tests for training infrastructure."""

import pytest
import jax
import jax.numpy as jnp


class TestGAE:
    """Tests for Generalized Advantage Estimation."""
    
    def test_gae(self):
        """Test that GAE computes advantages correctly."""
        from src.training.gae import compute_gae
        
        # Simple test case
        T = 10  # timesteps
        rewards = jnp.ones(T)
        values = jnp.zeros(T + 1)  # Include bootstrap value
        dones = jnp.zeros(T, dtype=jnp.bool_)
        
        advantages, returns = compute_gae(
            rewards, values, dones, 
            gamma=0.99, gae_lambda=0.95
        )
        
        # Check shapes
        assert advantages.shape == (T,)
        assert returns.shape == (T,)
        
        # Advantages should be positive when rewards > values
        assert jnp.all(advantages > 0)
    
    def test_gae_with_dones(self):
        """Test that GAE handles episode boundaries."""
        from src.training.gae import compute_gae
        
        T = 10
        rewards = jnp.ones(T)
        values = jnp.zeros(T + 1)
        dones = jnp.zeros(T, dtype=jnp.bool_).at[5].set(True)  # Done at step 5
        
        advantages, returns = compute_gae(
            rewards, values, dones,
            gamma=0.99, gae_lambda=0.95
        )
        
        # Should still have correct shape
        assert advantages.shape == (T,)


class TestPPOLoss:
    """Tests for PPO loss function."""
    
    def test_ppo_loss(self):
        """Test that PPO loss computes without error."""
        from src.training.ppo import ppo_loss
        from src.agents.network import ActorCritic
        
        network = ActorCritic(hidden_dims=(64, 64), num_actions=5)
        
        key = jax.random.PRNGKey(42)
        obs_dim = 64
        batch_size = 256
        
        # Initialize params
        params = network.init(key, jnp.zeros((obs_dim,)))
        
        # Create dummy batch
        batch = {
            'obs': jax.random.normal(key, (batch_size, obs_dim)),
            'actions': jax.random.randint(key, (batch_size,), 0, 5),
            'log_probs': jnp.zeros(batch_size),
            'advantages': jax.random.normal(key, (batch_size,)),
            'returns': jax.random.normal(key, (batch_size,)),
        }
        
        loss, metrics = ppo_loss(
            network, params, batch,
            clip_eps=0.2, vf_coef=0.5, ent_coef=0.01
        )
        
        # Loss should be scalar
        assert loss.shape == ()
        
        # Should have key metrics
        assert 'policy_loss' in metrics
        assert 'value_loss' in metrics
        assert 'entropy' in metrics


class TestRollout:
    """Tests for rollout collection."""
    
    def test_rollout(self):
        """Test that rollout collection works."""
        from src.training.rollout import collect_rollout, RunnerState
        from src.environment.vec_env import VecEnv
        from src.agents.network import ActorCritic
        from src.configs import Config
        
        config = Config()
        config.train.num_envs = 4
        config.train.num_steps = 16
        
        # Setup
        vec_env = VecEnv(config)
        network = ActorCritic(hidden_dims=(64, 64), num_actions=5)
        
        key = jax.random.PRNGKey(42)
        init_key, run_key = jax.random.split(key)
        
        # Initialize
        obs_dim = 64  # Example
        params = network.init(init_key, jnp.zeros((obs_dim,)))
        env_state = vec_env.reset(init_key)
        
        runner_state = RunnerState(
            params=params,
            env_state=env_state,
            key=run_key
        )
        
        # Collect rollout
        runner_state, batch = collect_rollout(
            runner_state, network, vec_env, config.train.num_steps
        )
        
        # Check batch shapes
        expected_shape = (config.train.num_steps, config.train.num_envs, config.env.num_agents)
        assert batch['rewards'].shape == expected_shape


class TestTrainStep:
    """Tests for training step."""
    
    def test_train_step(self):
        """Test that a training step runs without error."""
        from src.training.train import train_step
        from src.configs import Config
        
        config = Config()
        config.train.num_envs = 4
        config.train.num_steps = 16
        config.train.total_steps = 100  # Small for test
        
        key = jax.random.PRNGKey(42)
        
        # This is a smoke test - should not crash
        runner_state, metrics = train_step(config, key)
        
        assert 'loss' in metrics
        assert 'reward_mean' in metrics
