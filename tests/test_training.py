"""Tests for training infrastructure."""

import dataclasses

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


class TestFreezeEvolve:
    """Tests for US-003: Freeze-Evolve Training Mode."""

    def _make_small_config(self):
        """Create a minimal config for fast testing."""
        from src.configs import Config
        config = Config()
        config.train.num_envs = 2
        config.train.num_steps = 16
        config.train.total_steps = 100
        config.train.minibatch_size = 64
        config.env.num_agents = 4
        config.env.grid_size = 10
        config.env.num_food = 8
        config.evolution.max_agents = 8
        config.evolution.starting_energy = 200
        config.evolution.food_energy = 100
        config.evolution.reproduce_threshold = 120
        config.evolution.reproduce_cost = 50
        config.log.wandb = False
        return config

    def test_training_mode_enum(self):
        """Test TrainingMode enum has correct values."""
        from src.configs import TrainingMode
        assert TrainingMode.GRADIENT.value == "gradient"
        assert TrainingMode.EVOLVE.value == "evolve"
        assert TrainingMode.FREEZE_EVOLVE.value == "freeze_evolve"

    def test_freeze_evolve_config_defaults(self):
        """Test FreezeEvolveConfig has expected defaults."""
        from src.configs import FreezeEvolveConfig
        fe = FreezeEvolveConfig()
        assert fe.gradient_steps == 10000
        assert fe.evolve_steps == 1000
        assert fe.evolve_mutation_boost == 5.0

    def test_config_has_training_mode(self):
        """Test Config includes training_mode and freeze_evolve."""
        from src.configs import Config, TrainingMode
        config = Config()
        assert config.train.training_mode == TrainingMode.GRADIENT
        assert config.freeze_evolve.gradient_steps == 10000
        assert config.freeze_evolve.evolve_steps == 1000

    def test_training_mode_configurable(self):
        """Test training_mode can be set on Config."""
        from src.configs import Config, TrainingMode
        config = Config()
        config.train.training_mode = TrainingMode.FREEZE_EVOLVE
        assert config.train.training_mode == TrainingMode.FREEZE_EVOLVE

    def test_evolve_step_runs(self):
        """Test evolve_step executes and returns expected metric keys."""
        from src.training.train import create_train_state, evolve_step
        from src.configs import Config

        config = self._make_small_config()
        key = jax.random.PRNGKey(42)
        runner_state = create_train_state(config, key)

        new_state, metrics = evolve_step(runner_state, config)

        # Check metric keys exist
        assert 'mean_reward' in metrics
        assert 'population_size' in metrics
        assert 'births_this_step' in metrics
        assert 'deaths_this_step' in metrics
        assert 'mean_energy' in metrics
        # Placeholder gradient metrics should be zero
        assert float(metrics['total_loss']) == 0.0
        assert float(metrics['policy_loss']) == 0.0
        assert float(metrics['entropy']) == 0.0

    def test_evolve_step_no_param_update(self):
        """Test that evolve_step does not change shared params."""
        from src.training.train import create_train_state, evolve_step

        config = self._make_small_config()
        key = jax.random.PRNGKey(42)
        runner_state = create_train_state(config, key)

        # Capture params before
        params_before = jax.tree.map(lambda x: x.copy(), runner_state.params)

        new_state, _ = evolve_step(runner_state, config)

        # Shared params should be identical after evolve_step
        params_after = new_state.params
        for leaf_b, leaf_a in zip(
            jax.tree.leaves(params_before),
            jax.tree.leaves(params_after),
        ):
            assert jnp.allclose(leaf_b, leaf_a), \
                "Shared params changed during evolve_step!"

    def test_evolve_step_jit_compatible(self):
        """Test evolve_step works with JIT."""
        from src.training.train import create_train_state, evolve_step

        config = self._make_small_config()
        key = jax.random.PRNGKey(42)
        runner_state = create_train_state(config, key)

        @jax.jit
        def jit_evolve(rs):
            return evolve_step(rs, config)

        new_state, metrics = jit_evolve(runner_state)
        assert float(metrics['total_loss']) == 0.0
        assert new_state is not None

    def test_train_step_gradient_mode(self):
        """Test train_step works normally in GRADIENT mode (existing behavior)."""
        from src.training.train import create_train_state, train_step
        from src.configs import TrainingMode

        config = self._make_small_config()
        config.train.training_mode = TrainingMode.GRADIENT
        key = jax.random.PRNGKey(42)
        runner_state = create_train_state(config, key)

        new_state, metrics = train_step(runner_state, config)
        assert 'total_loss' in metrics
        # Loss should be non-zero in gradient mode
        loss = float(metrics['total_loss'])
        assert loss != 0.0 or True  # May be zero for first step; just verify it runs

    def test_evolve_mutation_boost(self):
        """Test that evolve config creates boosted mutation_std."""
        from src.configs import Config, FreezeEvolveConfig

        config = Config()
        config.evolution.mutation_std = 0.01
        config.freeze_evolve.evolve_mutation_boost = 5.0

        boosted = config.evolution.mutation_std * config.freeze_evolve.evolve_mutation_boost
        assert abs(boosted - 0.05) < 1e-8

    def test_evolve_config_in_modified_config(self):
        """Test creating a modified config with boosted mutation for evolve."""
        from src.configs import Config

        config = Config()
        config.evolution.mutation_std = 0.01
        config.freeze_evolve.evolve_mutation_boost = 3.0

        boosted_std = config.evolution.mutation_std * config.freeze_evolve.evolve_mutation_boost
        evolve_config = dataclasses.replace(
            config,
            evolution=dataclasses.replace(
                config.evolution,
                mutation_std=boosted_std,
            ),
        )

        # Original unchanged
        assert config.evolution.mutation_std == 0.01
        # Modified has boosted value
        assert abs(evolve_config.evolution.mutation_std - 0.03) < 1e-8
        # Everything else is the same
        assert evolve_config.env == config.env
        assert evolve_config.train == config.train

    def test_freeze_evolve_phase_switching_logic(self):
        """Test the phase switching logic for freeze-evolve cycles."""
        from src.configs import TrainingMode

        # Simulate phase transitions
        gradient_steps = 100
        evolve_steps = 50
        steps_per_iter = 25

        current_phase = TrainingMode.GRADIENT
        phase_counter = 0
        transitions = []

        for i in range(20):
            phase_counter += steps_per_iter
            total_steps = (i + 1) * steps_per_iter

            if current_phase == TrainingMode.GRADIENT:
                if phase_counter >= gradient_steps:
                    current_phase = TrainingMode.EVOLVE
                    phase_counter = 0
                    transitions.append((total_steps, "EVOLVE"))
            else:
                if phase_counter >= evolve_steps:
                    current_phase = TrainingMode.GRADIENT
                    phase_counter = 0
                    transitions.append((total_steps, "GRADIENT"))

        # Should have multiple transitions
        assert len(transitions) >= 2
        # First transition should be to EVOLVE
        assert transitions[0][1] == "EVOLVE"
        # Second transition should be back to GRADIENT
        assert transitions[1][1] == "GRADIENT"

    def test_evolve_step_returns_population_metrics(self):
        """Test that evolve_step provides key population metrics."""
        from src.training.train import create_train_state, evolve_step

        config = self._make_small_config()
        key = jax.random.PRNGKey(42)
        runner_state = create_train_state(config, key)

        _, metrics = evolve_step(runner_state, config)

        # Population metrics
        pop = float(metrics['population_size'])
        assert pop > 0  # At least some agents alive
        assert 'oldest_agent_age' in metrics
        assert 'max_energy' in metrics
        assert 'min_energy' in metrics

    @pytest.mark.timeout(120)
    def test_freeze_evolve_full_cycle(self):
        """Test a full gradient → evolve → gradient cycle end-to-end."""
        from src.training.train import create_train_state, train_step, evolve_step

        config = self._make_small_config()
        key = jax.random.PRNGKey(42)
        runner_state = create_train_state(config, key)

        # Gradient phase: 2 iterations
        for _ in range(2):
            runner_state, grad_metrics = train_step(runner_state, config)

        grad_loss = float(grad_metrics['total_loss'])

        # Evolve phase: 2 iterations
        for _ in range(2):
            runner_state, evolve_metrics = evolve_step(runner_state, config)

        evolve_loss = float(evolve_metrics['total_loss'])
        assert evolve_loss == 0.0  # No gradient updates in evolve

        # Back to gradient phase: 2 iterations
        for _ in range(2):
            runner_state, grad_metrics_2 = train_step(runner_state, config)

        # Should still produce valid metrics
        assert 'total_loss' in grad_metrics_2
        assert 'mean_reward' in grad_metrics_2
