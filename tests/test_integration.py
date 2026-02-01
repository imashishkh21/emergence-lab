"""Integration tests for US-027: Full pipeline test.

Tests the complete pipeline: init → train → evaluate → render → analyze.
"""

import os
import tempfile

import pytest
import jax
import jax.numpy as jnp
import numpy as np


def _small_config():
    """Create a small config suitable for fast integration testing."""
    from src.configs import Config

    config = Config()
    config.train.num_envs = 4
    config.train.num_steps = 32
    config.train.total_steps = 1000
    config.train.num_epochs = 2
    config.train.minibatch_size = 64
    config.env.num_agents = 4
    config.env.grid_size = 10
    config.env.num_food = 5
    config.env.max_steps = 20
    config.field.num_channels = 2
    config.agent.hidden_dims = (32, 32)
    config.log.wandb = False
    return config


@pytest.mark.timeout(120)  # 2 minute timeout
class TestIntegration:
    """Full pipeline integration tests."""

    def test_full_pipeline(self):
        """Test: init → train 1000 steps → evaluate → render → analyze.

        This is the primary integration test covering the full pipeline:
        1. Initialize training state
        2. Train for ~1000 env steps
        3. Evaluate with deterministic policy
        4. Render frames
        5. Analyze field metrics (entropy, structure, mutual information)
        6. Run emergence tracker
        7. Run ablation test
        """
        from src.configs import Config
        from src.training.train import create_train_state, train_step
        from src.environment.env import reset, step
        from src.environment.obs import get_observations
        from src.environment.render import render_frame
        from src.agents.network import ActorCritic
        from src.agents.policy import get_deterministic_actions
        from src.analysis.field_metrics import field_entropy, field_structure, field_food_mi
        from src.analysis.emergence import EmergenceTracker
        from src.analysis.ablation import ablation_test
        from src.utils.video import record_episode

        config = _small_config()
        key = jax.random.PRNGKey(42)

        # ---- 1. Initialize training state ----
        runner_state = create_train_state(config, key)
        assert runner_state is not None
        assert runner_state.params is not None

        # ---- 2. Train for some steps ----
        steps_per_iter = config.train.num_envs * config.train.num_steps * config.env.num_agents
        num_updates = max(config.train.total_steps // steps_per_iter, 3)
        # Run at least a few updates but cap to keep test fast
        num_updates = min(num_updates, 10)

        all_metrics = []
        for i in range(num_updates):
            runner_state, metrics = train_step(runner_state, config)
            all_metrics.append(metrics)

        # Training should complete without NaN/Inf
        for i, m in enumerate(all_metrics):
            for name, value in m.items():
                if isinstance(value, (float, jnp.ndarray)):
                    assert not jnp.isnan(value).any(), f"NaN in {name} at update {i}"
                    assert not jnp.isinf(value).any(), f"Inf in {name} at update {i}"

        # Verify expected metric keys exist
        expected_keys = {"total_loss", "policy_loss", "value_loss", "entropy",
                         "mean_reward", "mean_value"}
        assert expected_keys.issubset(set(all_metrics[-1].keys()))

        # ---- 3. Evaluate with deterministic policy ----
        network = ActorCritic(
            hidden_dims=tuple(config.agent.hidden_dims),
            num_actions=6,
        )
        eval_key = jax.random.PRNGKey(99)
        eval_state = reset(eval_key, config)

        total_eval_reward = 0.0
        for t in range(config.env.max_steps):
            obs = get_observations(eval_state, config)
            obs_batched = obs[None, :, :]
            actions = get_deterministic_actions(network, runner_state.params, obs_batched)
            actions = actions[0]
            eval_state, rewards, done, info = step(eval_state, actions, config)
            total_eval_reward += float(jnp.sum(rewards))
            if bool(done):
                break

        # Reward is a finite number
        assert np.isfinite(total_eval_reward)

        # ---- 4. Render frames ----
        # Render current evaluation state
        frame = render_frame(eval_state, config)
        assert frame.ndim == 3
        assert frame.shape[2] == 3  # RGB
        assert frame.shape[0] >= 100  # at least 100px height
        assert frame.shape[1] >= 100  # at least 100px width
        assert frame.dtype == np.uint8

        # Record a short video episode
        frames = record_episode(network, runner_state.params, config, jax.random.PRNGKey(7))
        assert len(frames) > 0
        assert all(f.shape == frames[0].shape for f in frames)

        # ---- 5. Analyze field metrics ----
        field_state = eval_state.field_state
        entropy = field_entropy(field_state)
        structure = field_structure(field_state)
        mi = field_food_mi(field_state, eval_state.food_positions)

        assert entropy.shape == ()
        assert structure.shape == ()
        assert mi.shape == ()
        assert float(entropy) >= 0.0
        assert 0.0 <= float(structure) <= 1.0
        assert float(mi) >= 0.0

        # ---- 6. Emergence tracker ----
        tracker = EmergenceTracker(config)
        # Feed it field states from the training run
        first_env_field = jax.tree.map(lambda x: x[0], runner_state.env_state.field_state)
        tracker.update(first_env_field, step=0)

        em_metrics = tracker.get_metrics()
        assert "emergence/entropy" in em_metrics
        assert "emergence/structure" in em_metrics
        assert "emergence/num_events" in em_metrics

        summary = tracker.get_summary()
        assert "total_updates" in summary
        assert summary["total_updates"] == 1

        # ---- 7. Ablation test (small: 2 episodes) ----
        results = ablation_test(
            network=network,
            params=runner_state.params,
            config=config,
            num_episodes=2,
            seed=42,
        )
        assert "normal" in results
        assert "zeroed" in results
        assert "random" in results
        for cond_name, result in results.items():
            assert np.isfinite(result.mean_reward), f"Non-finite reward in {cond_name}"
            assert len(result.episode_rewards) == 2

        print(f"\nIntegration test passed!")
        print(f"  Training updates: {num_updates}")
        print(f"  Eval reward: {total_eval_reward:.2f}")
        print(f"  Field entropy: {float(entropy):.4f}")
        print(f"  Field structure: {float(structure):.4f}")
        print(f"  Field-food MI: {float(mi):.4f}")
        print(f"  Frames recorded: {len(frames)}")
        print(f"  Ablation: normal={results['normal'].mean_reward:.2f}, "
              f"zeroed={results['zeroed'].mean_reward:.2f}, "
              f"random={results['random'].mean_reward:.2f}")

    def test_training_stability(self):
        """Test that training produces no NaN/Inf values over multiple updates."""
        from src.training.train import create_train_state, train_step

        config = _small_config()
        key = jax.random.PRNGKey(123)
        runner_state = create_train_state(config, key)

        # Run several training steps
        for i in range(5):
            runner_state, metrics = train_step(runner_state, config)

            # Check for NaN/Inf in all metrics
            for name, value in metrics.items():
                if isinstance(value, (float, jnp.ndarray)):
                    assert not jnp.isnan(value).any(), f"NaN in {name} at step {i}"
                    assert not jnp.isinf(value).any(), f"Inf in {name} at step {i}"

    def test_video_save(self):
        """Test that video recording and saving works end-to-end."""
        from src.agents.network import ActorCritic
        from src.training.train import create_train_state
        from src.utils.video import record_episode, save_video

        config = _small_config()
        key = jax.random.PRNGKey(77)
        runner_state = create_train_state(config, key)

        network = ActorCritic(
            hidden_dims=tuple(config.agent.hidden_dims),
            num_actions=6,
        )

        frames = record_episode(network, runner_state.params, config, jax.random.PRNGKey(0))
        assert len(frames) > 0

        # Save as MP4 to a temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "test_episode.mp4")
            save_video(frames, video_path, fps=10)
            assert os.path.exists(video_path)
            assert os.path.getsize(video_path) > 0
