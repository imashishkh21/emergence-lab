"""Tests for agent neural networks."""

import pytest
import jax
import jax.numpy as jnp


class TestNetwork:
    """Tests for ActorCritic network."""
    
    def test_network_forward(self):
        """Test that network produces correct output shapes."""
        from src.agents.network import ActorCritic
        from src.configs import Config
        
        config = Config()
        
        # Create network
        network = ActorCritic(
            hidden_dims=config.agent.hidden_dims,
            num_actions=5,  # stay, up, down, left, right
        )
        
        # Initialize with dummy input
        key = jax.random.PRNGKey(42)
        obs_dim = 64  # Example observation dimension
        dummy_obs = jnp.zeros((obs_dim,))
        
        params = network.init(key, dummy_obs)
        
        # Forward pass
        logits, value = network.apply(params, dummy_obs)
        
        # Check shapes
        assert logits.shape == (5,)  # 5 actions
        assert value.shape == ()  # scalar value
    
    def test_network_batched(self):
        """Test that network works with batched inputs."""
        from src.agents.network import ActorCritic
        
        network = ActorCritic(hidden_dims=(64, 64), num_actions=5)
        
        key = jax.random.PRNGKey(42)
        obs_dim = 64
        batch_size = 32
        
        dummy_obs = jnp.zeros((batch_size, obs_dim))
        params = network.init(key, dummy_obs[0])
        
        # Batched forward pass using vmap
        batched_apply = jax.vmap(network.apply, in_axes=(None, 0))
        logits, values = batched_apply(params, dummy_obs)
        
        assert logits.shape == (32, 5)
        assert values.shape == (32,)


class TestActionSampling:
    """Tests for action sampling from policy."""
    
    def test_action_sampling(self):
        """Test that action sampling works correctly."""
        from src.agents.policy import sample_actions
        from src.agents.network import ActorCritic
        
        network = ActorCritic(hidden_dims=(64, 64), num_actions=5)
        
        key = jax.random.PRNGKey(42)
        init_key, sample_key = jax.random.split(key)
        
        obs_dim = 64
        num_envs = 8
        num_agents = 4
        
        # Initialize params
        dummy_obs = jnp.zeros((obs_dim,))
        params = network.init(init_key, dummy_obs)
        
        # Batch of observations: (num_envs, num_agents, obs_dim)
        obs = jax.random.normal(sample_key, (num_envs, num_agents, obs_dim))
        
        actions, log_probs, values, entropy = sample_actions(
            network, params, obs, sample_key
        )
        
        # Check shapes
        assert actions.shape == (num_envs, num_agents)
        assert log_probs.shape == (num_envs, num_agents)
        assert values.shape == (num_envs, num_agents)
        assert entropy.shape == (num_envs, num_agents)
        
        # Actions should be valid (0-4)
        assert jnp.all(actions >= 0)
        assert jnp.all(actions < 5)
