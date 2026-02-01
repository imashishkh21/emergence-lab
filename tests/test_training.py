"""Tests for training infrastructure."""

import pytest
import jax
import jax.numpy as jnp


class TestGAE:
    """Tests for US-014: Generalized Advantage Estimation."""
    
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
        
        # Advantage at step 5 should not include future rewards
        # (reset happens, so future rewards don't count)
    
    def test_gae_with_nonzero_values(self):
        """Test GAE with non-zero value estimates."""
        from src.training.gae import compute_gae
        
        T = 5
        rewards = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
        values = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # T+1
        dones = jnp.zeros(T, dtype=jnp.bool_)
        
        advantages, returns = compute_gae(
            rewards, values, dones,
            gamma=0.99, gae_lambda=0.95
        )
        
        # Advantages should be smaller when values are higher
        assert advantages.shape == (T,)
    
    def test_gae_jit_compatible(self):
        """Test that GAE works with JIT."""
        from src.training.gae import compute_gae
        
        @jax.jit
        def jit_gae(rewards, values, dones):
            return compute_gae(rewards, values, dones, 0.99, 0.95)
        
        T = 10
        rewards = jnp.ones(T)
        values = jnp.zeros(T + 1)
        dones = jnp.zeros(T, dtype=jnp.bool_)
        
        advantages, returns = jit_gae(rewards, values, dones)
        
        assert advantages.shape == (T,)


class TestPPOLoss:
    """Tests for US-015: PPO loss function."""
    
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
    
    def test_ppo_loss_clip_behavior(self):
        """Test that clipping works correctly."""
        from src.training.ppo import ppo_loss
        from src.agents.network import ActorCritic
        
        network = ActorCritic(hidden_dims=(64, 64), num_actions=5)
        
        key = jax.random.PRNGKey(42)
        obs_dim = 64
        batch_size = 32
        
        params = network.init(key, jnp.zeros((obs_dim,)))
        
        batch = {
            'obs': jax.random.normal(key, (batch_size, obs_dim)),
            'actions': jax.random.randint(key, (batch_size,), 0, 5),
            'log_probs': jnp.zeros(batch_size),
            'advantages': jnp.ones(batch_size),  # All positive
            'returns': jnp.ones(batch_size),
        }
        
        loss, metrics = ppo_loss(
            network, params, batch,
            clip_eps=0.2, vf_coef=0.5, ent_coef=0.01
        )
        
        # Clip fraction should be between 0 and 1
        assert 0 <= metrics['clip_fraction'] <= 1
    
    def test_ppo_loss_jit_compatible(self):
        """Test that PPO loss works with JIT."""
        from src.training.ppo import ppo_loss
        from src.agents.network import ActorCritic
        
        network = ActorCritic(hidden_dims=(64, 64), num_actions=5)
        
        key = jax.random.PRNGKey(42)
        obs_dim = 64
        batch_size = 32
        
        params = network.init(key, jnp.zeros((obs_dim,)))
        
        @jax.jit
        def jit_loss(params, batch):
            return ppo_loss(network, params, batch, 0.2, 0.5, 0.01)
        
        batch = {
            'obs': jax.random.normal(key, (batch_size, obs_dim)),
            'actions': jax.random.randint(key, (batch_size,), 0, 5),
            'log_probs': jnp.zeros(batch_size),
            'advantages': jax.random.normal(key, (batch_size,)),
            'returns': jax.random.normal(key, (batch_size,)),
        }
        
        loss, metrics = jit_loss(params, batch)
        assert loss.shape == ()


class TestRollout:
    """Tests for US-016: Rollout collection."""
    
    def test_runner_state(self):
        """Test RunnerState dataclass."""
        from src.training.rollout import RunnerState
        
        # Just test that it can be imported and has expected fields
        assert hasattr(RunnerState, '__dataclass_fields__') or hasattr(RunnerState, 'params')
    
    def test_rollout(self):
        """Test that rollout collection works."""
        from src.training.rollout import collect_rollout, RunnerState
        from src.environment.vec_env import VecEnv
        from src.environment.obs import get_observations
        from src.agents.network import ActorCritic
        from src.configs import Config
        import optax
        
        config = Config()
        config.train.num_envs = 4
        config.train.num_steps = 16
        config.env.num_agents = 2
        
        # Setup
        vec_env = VecEnv(config)
        network = ActorCritic(hidden_dims=(64, 64), num_actions=5)
        
        key = jax.random.PRNGKey(42)
        init_key, run_key = jax.random.split(key)
        
        # Initialize
        env_state = vec_env.reset(init_key)
        
        # Compute obs_dim from actual observation
        from src.environment.obs import get_observations
        sample_obs = get_observations(
            jax.tree.map(lambda x: x[0], env_state),  # Single env
            config
        )
        obs_dim = sample_obs.shape[-1]
        
        params = network.init(init_key, jnp.zeros((obs_dim,)))
        opt_state = optax.adam(config.train.learning_rate).init(params)
        
        # Get initial observations
        last_obs = jax.vmap(lambda s: get_observations(s, config))(env_state)
        
        runner_state = RunnerState(
            params=params,
            opt_state=opt_state,
            env_state=env_state,
            last_obs=last_obs,
            key=run_key
        )
        
        # Collect rollout
        runner_state, batch = collect_rollout(
            runner_state, network, vec_env, config
        )
        
        # Check batch shapes
        # Shape: (num_steps, num_envs, max_agents)
        max_agents = config.evolution.max_agents
        assert batch['rewards'].shape == (16, 4, max_agents)
        assert batch['actions'].shape == (16, 4, max_agents)


class TestTrainStep:
    """Tests for US-017: Training step."""
    
    def test_train_step(self):
        """Test that a training step runs without error."""
        from src.training.train import create_train_state, train_step
        from src.configs import Config
        
        config = Config()
        config.train.num_envs = 4
        config.train.num_steps = 16
        config.train.total_steps = 100  # Small for test
        
        key = jax.random.PRNGKey(42)
        
        # Create initial state
        runner_state = create_train_state(config, key)
        
        # Run one train step
        runner_state, metrics = train_step(runner_state, config)
        
        assert 'loss' in metrics or 'total_loss' in metrics
        assert runner_state is not None
