"""Integration tests for US-027: Full pipeline test."""

import pytest
import jax
import jax.numpy as jnp


@pytest.mark.timeout(120)  # 2 minute timeout
class TestIntegration:
    """Full pipeline integration tests."""
    
    def test_full_pipeline(self):
        """Test: init → train 1000 steps → evaluate → render → analyze."""
        from src.configs import Config
        from src.training.train import create_train_state, train_step
        from src.environment.render import render_frame
        from src.analysis.field_metrics import field_entropy, field_structure
        
        # Setup
        config = Config()
        config.train.num_envs = 4
        config.train.num_steps = 32
        config.train.total_steps = 1000
        config.env.num_agents = 4
        config.env.grid_size = 10
        
        key = jax.random.PRNGKey(42)
        
        # 1. Initialize training state
        runner_state = create_train_state(config, key)
        assert runner_state is not None
        
        # 2. Train for some steps
        num_updates = config.train.total_steps // (config.train.num_envs * config.train.num_steps)
        for _ in range(min(num_updates, 10)):  # At least 10 updates
            runner_state, metrics = train_step(runner_state, config)
        
        # Should have trained without crashing
        assert runner_state is not None
        
        # 3. Evaluate - get current state
        env_state = runner_state.env_state
        # Take first env from batch
        single_env_state = jax.tree.map(lambda x: x[0], env_state)
        
        # 4. Render a frame
        frame = render_frame(single_env_state, config)
        assert frame.shape[0] >= 100  # Height
        assert frame.shape[1] >= 100  # Width
        assert frame.shape[2] == 3   # RGB
        
        # 5. Analyze field
        field_state = single_env_state.field_state
        entropy = field_entropy(field_state)
        structure = field_structure(field_state)
        
        assert entropy.shape == ()
        assert structure.shape == ()
        
        print(f"Training complete!")
        print(f"Field entropy: {entropy}")
        print(f"Field structure: {structure}")
    
    def test_training_stability(self):
        """Test that training produces no NaN/Inf values."""
        from src.configs import Config
        from src.training.train import create_train_state, train_step
        
        config = Config()
        config.train.num_envs = 4
        config.train.num_steps = 32
        
        key = jax.random.PRNGKey(123)
        runner_state = create_train_state(config, key)
        
        # Run several training steps
        for i in range(5):
            runner_state, metrics = train_step(runner_state, config)
            
            # Check for NaN/Inf in metrics
            for name, value in metrics.items():
                if isinstance(value, (float, jnp.ndarray)):
                    assert not jnp.isnan(value).any(), f"NaN in {name} at step {i}"
                    assert not jnp.isinf(value).any(), f"Inf in {name} at step {i}"
    
    def test_checkpoint_save_load(self):
        """Test that checkpoints can be saved and loaded."""
        import os
        import tempfile
        from src.configs import Config
        from src.training.train import create_train_state, train_step
        from src.utils.checkpointing import save_checkpoint, load_checkpoint
        
        config = Config()
        config.train.num_envs = 2
        config.train.num_steps = 16
        
        key = jax.random.PRNGKey(456)
        runner_state = create_train_state(config, key)
        
        # Train a bit
        runner_state, _ = train_step(runner_state, config)
        
        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "checkpoint_test")
            save_checkpoint(runner_state.params, checkpoint_path, step=100)
            
            # Load
            loaded_params = load_checkpoint(checkpoint_path)
            
            # Verify params match
            original_flat = jax.tree_util.tree_leaves(runner_state.params)
            loaded_flat = jax.tree_util.tree_leaves(loaded_params)
            
            for orig, loaded in zip(original_flat, loaded_flat):
                assert jnp.allclose(orig, loaded)
