"""Tests for the shared field module."""

import pytest
import jax
import jax.numpy as jnp


class TestFieldState:
    """Tests for FieldState dataclass."""
    
    def test_field_state_creation(self):
        """Test that FieldState can be created with correct shape."""
        # Import here to allow module to be created first
        from src.field.field import FieldState, create_field
        
        field = create_field(height=20, width=20, channels=4)
        
        assert isinstance(field, FieldState)
        assert field.values.shape == (20, 20, 4)
        assert field.values.dtype == jnp.float32


class TestFieldDynamics:
    """Tests for field diffusion and decay."""
    
    def test_diffusion(self):
        """Test that diffusion spreads values."""
        from src.field.field import create_field
        from src.field.dynamics import diffuse
        
        # Create field with a spike in the center
        field = create_field(20, 20, 1)
        values = field.values.at[10, 10, 0].set(1.0)
        field = field.replace(values=values)
        
        # Apply diffusion
        diffused = diffuse(field, rate=0.5)
        
        # Center should decrease, neighbors should increase
        assert diffused.values[10, 10, 0] < 1.0
        assert diffused.values[10, 11, 0] > 0.0
        assert diffused.values[9, 10, 0] > 0.0
    
    def test_decay(self):
        """Test that decay reduces values."""
        from src.field.field import create_field
        from src.field.dynamics import decay
        
        # Create field with values
        field = create_field(20, 20, 1)
        values = jnp.ones((20, 20, 1))
        field = field.replace(values=values)
        
        # Apply decay
        decayed = decay(field, rate=0.1)
        
        # All values should be reduced
        assert jnp.allclose(decayed.values, 0.9)
    
    def test_step_field(self):
        """Test combined diffusion + decay step."""
        from src.field.field import create_field
        from src.field.dynamics import step_field
        
        field = create_field(20, 20, 4)
        values = jax.random.uniform(jax.random.PRNGKey(0), (20, 20, 4))
        field = field.replace(values=values)
        
        stepped = step_field(field, diffusion_rate=0.1, decay_rate=0.05)
        
        # Should have same shape
        assert stepped.values.shape == (20, 20, 4)
        # Total energy should decrease (due to decay)
        assert stepped.values.sum() < values.sum()


class TestFieldOps:
    """Tests for field read/write operations."""
    
    def test_read_write(self):
        """Test that agents can read and write to field."""
        from src.field.field import create_field
        from src.field.ops import read_local, write_local
        
        field = create_field(20, 20, 4)
        
        # Agent positions (batch of 3 agents)
        positions = jnp.array([[5, 5], [10, 10], [15, 15]])
        
        # Write values at agent positions
        write_values = jnp.ones((3, 4))  # 3 agents, 4 channels
        field = write_local(field, positions, write_values)
        
        # Read values at agent positions
        read_values = read_local(field, positions, radius=2)
        
        # Should have read something non-zero at write positions
        assert read_values.shape[0] == 3  # 3 agents
        assert jnp.any(read_values > 0)
