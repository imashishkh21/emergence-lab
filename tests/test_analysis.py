"""Tests for analysis module."""

import pytest
import jax
import jax.numpy as jnp


class TestFieldMetrics:
    """Tests for US-022: Field analysis metrics."""
    
    def test_field_entropy(self):
        """Test that field_entropy computes spatial entropy."""
        from src.analysis.field_metrics import field_entropy
        from src.field.field import create_field
        
        # Uniform field should have high entropy
        field_uniform = create_field(20, 20, 1)
        values_uniform = jnp.ones((20, 20, 1)) * 0.5
        field_uniform = field_uniform.replace(values=values_uniform)
        
        # Sparse field should have lower entropy
        field_sparse = create_field(20, 20, 1)
        values_sparse = jnp.zeros((20, 20, 1)).at[10, 10, 0].set(1.0)
        field_sparse = field_sparse.replace(values=values_sparse)
        
        entropy_uniform = field_entropy(field_uniform)
        entropy_sparse = field_entropy(field_sparse)
        
        # Both should be scalars
        assert entropy_uniform.shape == ()
        assert entropy_sparse.shape == ()
    
    def test_field_structure(self):
        """Test that field_structure measures autocorrelation."""
        from src.analysis.field_metrics import field_structure
        from src.field.field import create_field
        
        # Random field should have low structure
        key = jax.random.PRNGKey(42)
        field_random = create_field(20, 20, 1)
        values_random = jax.random.uniform(key, (20, 20, 1))
        field_random = field_random.replace(values=values_random)
        
        # Smooth field should have high structure
        field_smooth = create_field(20, 20, 1)
        x = jnp.linspace(0, 1, 20)
        y = jnp.linspace(0, 1, 20)
        xx, yy = jnp.meshgrid(x, y)
        values_smooth = jnp.sin(xx * 3.14159)[:, :, None]
        field_smooth = field_smooth.replace(values=values_smooth)
        
        structure_random = field_structure(field_random)
        structure_smooth = field_structure(field_smooth)
        
        # Smooth should have more structure than random
        assert structure_smooth > structure_random
    
    def test_field_food_mi(self):
        """Test mutual information between field and food positions."""
        from src.analysis.field_metrics import field_food_mi
        from src.field.field import create_field
        
        # Field with signal at food positions
        field = create_field(20, 20, 1)
        food_positions = jnp.array([[5, 5], [15, 15]])
        
        # Write signal at food locations
        values = jnp.zeros((20, 20, 1))
        values = values.at[5, 5, 0].set(1.0)
        values = values.at[15, 15, 0].set(1.0)
        field = field.replace(values=values)
        
        mi = field_food_mi(field, food_positions)
        
        # Should be a scalar
        assert mi.shape == ()
        # Should be non-negative
        assert mi >= 0
    
    def test_metrics_jit_compatible(self):
        """Test that all metrics work with JIT."""
        from src.analysis.field_metrics import field_entropy, field_structure
        from src.field.field import create_field
        
        field = create_field(20, 20, 4)
        values = jax.random.uniform(jax.random.PRNGKey(0), (20, 20, 4))
        field = field.replace(values=values)
        
        jit_entropy = jax.jit(field_entropy)
        jit_structure = jax.jit(field_structure)
        
        e = jit_entropy(field)
        s = jit_structure(field)
        
        assert e.shape == ()
        assert s.shape == ()


class TestAblation:
    """Tests for US-023: Ablation test."""
    
    def test_ablation_interface(self):
        """Test that ablation_test function exists with correct signature."""
        from src.analysis.ablation import ablation_test
        
        # Just check it's callable with expected args
        import inspect
        sig = inspect.signature(ablation_test)
        params = list(sig.parameters.keys())
        
        assert 'network' in params or 'policy' in params
        assert 'config' in params


class TestEmergenceTracker:
    """Tests for US-024: Emergence detection."""
    
    def test_emergence_tracker_creation(self):
        """Test EmergenceTracker can be created."""
        from src.analysis.emergence import EmergenceTracker
        from src.configs import Config
        
        config = Config()
        tracker = EmergenceTracker(config)
        
        assert tracker is not None
    
    def test_emergence_tracker_update(self):
        """Test EmergenceTracker can record metrics."""
        from src.analysis.emergence import EmergenceTracker
        from src.field.field import create_field
        from src.configs import Config
        
        config = Config()
        tracker = EmergenceTracker(config)
        
        # Create a sample field
        field = create_field(20, 20, 4)
        values = jax.random.uniform(jax.random.PRNGKey(0), (20, 20, 4))
        field = field.replace(values=values)
        
        # Update tracker
        tracker.update(field, step=0)
        tracker.update(field, step=100)
        
        # Should have recorded something
        assert len(tracker.history) >= 2 or tracker.step_count >= 2
