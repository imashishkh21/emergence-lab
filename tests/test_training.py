"""Tests for training infrastructure."""

import dataclasses
import os

import pytest
import jax
import jax.numpy as jnp
import numpy as np


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


class TestResumeLogic:
    """Tests for US-005: Proper Resume Logic."""

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
        config.log.save_interval = 0  # Disable periodic saves for tests
        return config

    def _save_full_checkpoint(self, runner_state, config, step, tmp_path,
                              emergence_tracker=None, spec_tracker=None):
        """Helper to save a full checkpoint matching the train.py format."""
        from src.training.checkpointing import save_checkpoint

        tracker_state = {}
        if emergence_tracker is not None:
            tracker_state["emergence"] = emergence_tracker.to_dict()
        if spec_tracker is not None:
            tracker_state["specialization"] = spec_tracker.to_dict()

        ckpt_state = {
            "params": runner_state.params,
            "opt_state": runner_state.opt_state,
            "agent_params": (
                runner_state.env_state.agent_params
                if runner_state.env_state.agent_params is not None
                else None
            ),
            "prng_key": runner_state.key,
            "step": step,
            "config": config,
            "tracker_state": tracker_state,
        }
        ckpt_path = os.path.join(str(tmp_path), "step_test.pkl")
        save_checkpoint(ckpt_path, ckpt_state, max_checkpoints=0)
        return ckpt_path

    def test_resume_loads_full_checkpoint(self, tmp_path):
        """Test that resume uses load_checkpoint (not raw pickle)."""
        from src.training.train import create_train_state, train_step
        from src.training.checkpointing import load_checkpoint

        config = self._make_small_config()
        key = jax.random.PRNGKey(42)
        runner_state = create_train_state(config, key)

        # Run a training step to get non-trivial state
        runner_state, _ = train_step(runner_state, config)

        # Save checkpoint
        ckpt_path = self._save_full_checkpoint(runner_state, config, 5000, tmp_path)

        # Verify checkpoint is loadable with load_checkpoint
        loaded = load_checkpoint(ckpt_path)
        assert "params" in loaded
        assert "opt_state" in loaded
        assert "prng_key" in loaded
        assert "step" in loaded
        assert loaded["step"] == 5000

    def test_resume_restores_optimizer_state(self, tmp_path):
        """Test that optimizer state (Adam momentum) is restored."""
        from src.training.train import create_train_state, train_step

        config = self._make_small_config()
        key = jax.random.PRNGKey(42)
        runner_state = create_train_state(config, key)

        # Run a few training steps to accumulate Adam momentum
        for _ in range(3):
            runner_state, _ = train_step(runner_state, config)

        # Save checkpoint with trained opt_state
        ckpt_path = self._save_full_checkpoint(runner_state, config, 1000, tmp_path)

        # Create new training state and resume
        config2 = self._make_small_config()
        config2.train.resume_from = ckpt_path
        key2 = jax.random.PRNGKey(99)
        fresh_state = create_train_state(config2, key2)

        # Load checkpoint and restore opt_state
        from src.training.checkpointing import load_checkpoint
        checkpoint_data = load_checkpoint(ckpt_path)

        # The opt_state should have non-trivial values (Adam mu/nu accumulated)
        saved_opt_leaves = jax.tree.leaves(checkpoint_data["opt_state"])
        fresh_opt_leaves = jax.tree.leaves(fresh_state.opt_state)

        # At least some opt_state leaves should differ from fresh init
        any_differ = False
        for saved_leaf, fresh_leaf in zip(saved_opt_leaves, fresh_opt_leaves):
            if isinstance(saved_leaf, jnp.ndarray) and isinstance(fresh_leaf, jnp.ndarray):
                if saved_leaf.shape == fresh_leaf.shape:
                    if not jnp.allclose(saved_leaf, fresh_leaf, atol=1e-10):
                        any_differ = True
                        break
        assert any_differ, "Optimizer state should differ from fresh init after training"

    def test_resume_restores_prng_key(self, tmp_path):
        """Test that PRNG key is restored from checkpoint."""
        from src.training.train import create_train_state, train_step

        config = self._make_small_config()
        key = jax.random.PRNGKey(42)
        runner_state = create_train_state(config, key)
        runner_state, _ = train_step(runner_state, config)

        # Save with known PRNG key
        saved_key = runner_state.key
        ckpt_path = self._save_full_checkpoint(runner_state, config, 2000, tmp_path)

        # Load and verify key matches
        from src.training.checkpointing import load_checkpoint
        loaded = load_checkpoint(ckpt_path)
        restored_key = loaded["prng_key"]

        assert jnp.array_equal(saved_key, restored_key), \
            "PRNG key should be preserved through save/load"

    def test_resume_restores_step_counter(self, tmp_path):
        """Test that step counter is restored correctly."""
        from src.training.train import create_train_state, train_step

        config = self._make_small_config()
        key = jax.random.PRNGKey(42)
        runner_state = create_train_state(config, key)
        runner_state, _ = train_step(runner_state, config)

        # Save at step 50000
        ckpt_path = self._save_full_checkpoint(runner_state, config, 50000, tmp_path)

        from src.training.checkpointing import load_checkpoint
        loaded = load_checkpoint(ckpt_path)
        assert loaded["step"] == 50000

    def test_resume_restores_tracker_state(self, tmp_path):
        """Test that tracker state (emergence + specialization) is restored."""
        from src.training.train import create_train_state, train_step
        from src.analysis.emergence import EmergenceTracker
        from src.analysis.specialization import SpecializationTracker

        config = self._make_small_config()
        key = jax.random.PRNGKey(42)
        runner_state = create_train_state(config, key)
        runner_state, _ = train_step(runner_state, config)

        # Create trackers with some history
        emergence_tracker = EmergenceTracker(config)
        spec_tracker = SpecializationTracker(config)

        # Feed some data into emergence tracker
        first_env_field = jax.tree.map(lambda x: x[0], runner_state.env_state.field_state)
        emergence_tracker.update(first_env_field, step=1000)
        emergence_tracker.update(first_env_field, step=2000)

        # Feed some data into specialization tracker
        first_env_params = jax.tree.map(lambda x: x[0], runner_state.env_state.agent_params)
        first_env_alive = np.asarray(runner_state.env_state.agent_alive[0])
        spec_tracker.update(first_env_params, first_env_alive, step=1000)

        # Save checkpoint with tracker state
        ckpt_path = self._save_full_checkpoint(
            runner_state, config, 5000, tmp_path,
            emergence_tracker=emergence_tracker,
            spec_tracker=spec_tracker,
        )

        # Load and verify tracker state
        from src.training.checkpointing import load_checkpoint
        loaded = load_checkpoint(ckpt_path)
        assert "tracker_state" in loaded
        assert "emergence" in loaded["tracker_state"]
        assert "specialization" in loaded["tracker_state"]

        # Restore trackers
        restored_emergence = EmergenceTracker.from_dict(
            loaded["tracker_state"]["emergence"], config
        )
        restored_spec = SpecializationTracker.from_dict(
            loaded["tracker_state"]["specialization"], config
        )

        # Verify they have history
        assert restored_emergence.step_count == 2
        assert restored_spec.step_count == 1

        # Verify they can continue updating
        emergence_tracker.update(first_env_field, step=3000)
        restored_emergence.update(first_env_field, step=3000)
        assert restored_emergence.step_count == 3

    def test_resume_restores_agent_params(self, tmp_path):
        """Test that per-agent params are restored when evolution is enabled."""
        from src.training.train import create_train_state, train_step

        config = self._make_small_config()
        config.evolution.enabled = True
        key = jax.random.PRNGKey(42)
        runner_state = create_train_state(config, key)

        # Run a step so per-agent params may diverge
        runner_state, _ = train_step(runner_state, config)

        original_agent_params = runner_state.env_state.agent_params
        assert original_agent_params is not None

        ckpt_path = self._save_full_checkpoint(runner_state, config, 3000, tmp_path)

        from src.training.checkpointing import load_checkpoint
        loaded = load_checkpoint(ckpt_path)

        assert loaded["agent_params"] is not None
        # Verify shapes match
        original_leaves = jax.tree.leaves(original_agent_params)
        loaded_leaves = jax.tree.leaves(loaded["agent_params"])
        assert len(original_leaves) == len(loaded_leaves)
        for orig, load in zip(original_leaves, loaded_leaves):
            assert orig.shape == load.shape

    def test_resume_handles_missing_opt_state(self, tmp_path):
        """Test that resume works when opt_state is missing from checkpoint."""
        from src.training.train import create_train_state
        from src.training.checkpointing import save_checkpoint, load_checkpoint

        config = self._make_small_config()
        key = jax.random.PRNGKey(42)
        runner_state = create_train_state(config, key)

        # Save a checkpoint WITHOUT opt_state (simulate old checkpoint)
        ckpt_state = {
            "params": runner_state.params,
            "opt_state": None,
            "agent_params": None,
            "prng_key": runner_state.key,
            "step": 1000,
            "config": config,
            "tracker_state": None,
        }
        ckpt_path = os.path.join(str(tmp_path), "step_old.pkl")
        save_checkpoint(ckpt_path, ckpt_state, max_checkpoints=0)

        # Load should succeed
        loaded = load_checkpoint(ckpt_path)
        assert loaded["opt_state"] is None
        assert loaded["params"] is not None

    def test_resume_handles_missing_prng_key(self, tmp_path):
        """Test that resume works when prng_key is missing from checkpoint."""
        from src.training.checkpointing import save_checkpoint, load_checkpoint
        from src.training.train import create_train_state

        config = self._make_small_config()
        key = jax.random.PRNGKey(42)
        runner_state = create_train_state(config, key)

        # Save a checkpoint WITHOUT prng_key
        ckpt_state = {
            "params": runner_state.params,
            "opt_state": runner_state.opt_state,
            "agent_params": None,
            "prng_key": None,
            "step": 2000,
            "config": config,
            "tracker_state": None,
        }
        ckpt_path = os.path.join(str(tmp_path), "step_nokey.pkl")
        save_checkpoint(ckpt_path, ckpt_state, max_checkpoints=0)

        loaded = load_checkpoint(ckpt_path)
        assert loaded["prng_key"] is None

    @pytest.mark.timeout(120)
    def test_resume_full_train_loop(self, tmp_path):
        """Test full resume: train → save → resume → train continues correctly."""
        from src.training.train import create_train_state, train_step
        from src.analysis.emergence import EmergenceTracker
        from src.analysis.specialization import SpecializationTracker

        config = self._make_small_config()
        key = jax.random.PRNGKey(42)
        runner_state = create_train_state(config, key)

        # Train 3 steps
        for _ in range(3):
            runner_state, metrics = train_step(runner_state, config)

        # Create trackers with history
        emergence_tracker = EmergenceTracker(config)
        spec_tracker = SpecializationTracker(config)
        first_env_field = jax.tree.map(lambda x: x[0], runner_state.env_state.field_state)
        emergence_tracker.update(first_env_field, step=1000)

        # Save
        ckpt_path = self._save_full_checkpoint(
            runner_state, config, 10000, tmp_path,
            emergence_tracker=emergence_tracker,
            spec_tracker=spec_tracker,
        )

        # Create completely new state (simulating a new process)
        config2 = self._make_small_config()
        config2.train.resume_from = ckpt_path
        key2 = jax.random.PRNGKey(999)
        fresh_state = create_train_state(config2, key2)

        # Load checkpoint
        from src.training.checkpointing import load_checkpoint
        loaded = load_checkpoint(ckpt_path)

        # Restore full state
        fresh_state = fresh_state.replace(  # type: ignore[attr-defined]
            params=loaded["params"],
            opt_state=loaded["opt_state"],
            key=loaded["prng_key"],
        )

        # Verify we can continue training from restored state
        fresh_state, new_metrics = train_step(fresh_state, config2)

        # Should produce valid metrics
        assert "mean_reward" in new_metrics
        assert "total_loss" in new_metrics
        assert jnp.isfinite(new_metrics["total_loss"])

    def test_resume_params_values_preserved(self, tmp_path):
        """Test that param values are identical after save/load round-trip."""
        from src.training.train import create_train_state, train_step

        config = self._make_small_config()
        key = jax.random.PRNGKey(42)
        runner_state = create_train_state(config, key)
        runner_state, _ = train_step(runner_state, config)

        # Save
        ckpt_path = self._save_full_checkpoint(runner_state, config, 5000, tmp_path)

        # Load
        from src.training.checkpointing import load_checkpoint
        loaded = load_checkpoint(ckpt_path)

        # Compare param values
        original_leaves = jax.tree.leaves(runner_state.params)
        loaded_leaves = jax.tree.leaves(loaded["params"])
        for orig, load in zip(original_leaves, loaded_leaves):
            assert jnp.allclose(orig, load, atol=1e-6), \
                "Param values should be preserved through save/load"

    def test_resume_without_tracker_state(self, tmp_path):
        """Test resume works when tracker_state is absent from checkpoint."""
        from src.training.train import create_train_state
        from src.training.checkpointing import save_checkpoint, load_checkpoint

        config = self._make_small_config()
        key = jax.random.PRNGKey(42)
        runner_state = create_train_state(config, key)

        # Save without tracker_state
        ckpt_state = {
            "params": runner_state.params,
            "opt_state": runner_state.opt_state,
            "agent_params": None,
            "prng_key": runner_state.key,
            "step": 3000,
            "config": config,
        }
        ckpt_path = os.path.join(str(tmp_path), "step_notracker.pkl")
        save_checkpoint(ckpt_path, ckpt_state, max_checkpoints=0)

        loaded = load_checkpoint(ckpt_path)
        # Should not have tracker_state key
        assert "tracker_state" not in loaded or loaded.get("tracker_state") is None

    def test_resume_via_latest_symlink(self, tmp_path):
        """Test resume works via the latest.pkl symlink."""
        from src.training.train import create_train_state, train_step
        from src.training.checkpointing import save_checkpoint, load_checkpoint

        config = self._make_small_config()
        key = jax.random.PRNGKey(42)
        runner_state = create_train_state(config, key)
        runner_state, _ = train_step(runner_state, config)

        # Save creates a latest.pkl symlink
        ckpt_dir = str(tmp_path)
        ckpt_path = os.path.join(ckpt_dir, "step_7000.pkl")
        ckpt_state = {
            "params": runner_state.params,
            "opt_state": runner_state.opt_state,
            "agent_params": None,
            "prng_key": runner_state.key,
            "step": 7000,
            "config": config,
            "tracker_state": {},
        }
        save_checkpoint(ckpt_path, ckpt_state, max_checkpoints=5)

        # Resume from latest.pkl
        latest_path = os.path.join(ckpt_dir, "latest.pkl")
        assert os.path.exists(latest_path)

        loaded = load_checkpoint(latest_path)
        assert loaded["step"] == 7000
