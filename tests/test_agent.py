"""Tests for agent neural networks."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np


class TestNetwork:
    """Tests for US-012: ActorCritic network."""
    
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
    
    def test_network_initialization(self):
        """Test that initialization follows orthogonal scheme."""
        from src.agents.network import ActorCritic
        
        network = ActorCritic(hidden_dims=(64, 64), num_actions=5)
        
        key = jax.random.PRNGKey(42)
        obs_dim = 32
        params = network.init(key, jnp.zeros((obs_dim,)))
        
        # Params should exist and be properly shaped
        assert params is not None
        
        # Check that params have reasonable magnitudes (orthogonal init)
        flat_params = jax.tree_util.tree_leaves(params)
        for p in flat_params:
            if p.ndim >= 2:  # Weight matrices
                assert jnp.abs(p).max() < 10.0  # Not exploding
    
    def test_network_different_configs(self):
        """Test network with various configurations."""
        from src.agents.network import ActorCritic
        
        key = jax.random.PRNGKey(42)
        obs_dim = 48
        
        configs = [
            (32, 32),
            (64, 64),
            (128, 128),
            (64, 64, 64),  # 3 layers
        ]
        
        for hidden_dims in configs:
            network = ActorCritic(hidden_dims=hidden_dims, num_actions=5)
            params = network.init(key, jnp.zeros((obs_dim,)))
            logits, value = network.apply(params, jnp.zeros((obs_dim,)))
            
            assert logits.shape == (5,)
            assert value.shape == ()


class TestActionSampling:
    """Tests for US-013: Action sampling from policy."""
    
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
    
    def test_action_sampling_deterministic_eval(self):
        """Test deterministic action selection for evaluation."""
        from src.agents.policy import sample_actions, get_deterministic_actions
        from src.agents.network import ActorCritic
        
        network = ActorCritic(hidden_dims=(64, 64), num_actions=5)
        
        key = jax.random.PRNGKey(42)
        obs_dim = 64
        
        params = network.init(key, jnp.zeros((obs_dim,)))
        obs = jax.random.normal(key, (4, 2, obs_dim))  # 4 envs, 2 agents
        
        # Get deterministic actions
        actions = get_deterministic_actions(network, params, obs)
        
        assert actions.shape == (4, 2)
        assert jnp.all(actions >= 0)
        assert jnp.all(actions < 5)
    
    def test_action_sampling_stochastic(self):
        """Test that stochastic sampling produces varied actions."""
        from src.agents.policy import sample_actions
        from src.agents.network import ActorCritic
        
        network = ActorCritic(hidden_dims=(64, 64), num_actions=5)
        
        key = jax.random.PRNGKey(42)
        obs_dim = 64
        
        params = network.init(key, jnp.zeros((obs_dim,)))
        obs = jax.random.normal(key, (100, 1, obs_dim))  # 100 samples
        
        # Sample multiple times with different keys
        all_actions = []
        for i in range(10):
            sample_key = jax.random.PRNGKey(i)
            actions, _, _, _ = sample_actions(network, params, obs, sample_key)
            all_actions.append(actions)
        
        all_actions = jnp.stack(all_actions)
        
        # Should have variety in actions (not all same)
        assert len(jnp.unique(all_actions)) > 1
    
    def test_action_sampling_jit_compatible(self):
        """Test that action sampling works with JIT."""
        from src.agents.policy import sample_actions
        from src.agents.network import ActorCritic
        
        network = ActorCritic(hidden_dims=(64, 64), num_actions=5)
        
        key = jax.random.PRNGKey(42)
        obs_dim = 64
        
        params = network.init(key, jnp.zeros((obs_dim,)))
        
        @jax.jit
        def jit_sample(obs, key):
            return sample_actions(network, params, obs, key)
        
        obs = jax.random.normal(key, (4, 2, obs_dim))
        actions, log_probs, values, entropy = jit_sample(obs, key)
        
        assert actions.shape == (4, 2)
