"""Tests for the shared field module."""

import pytest
import jax
import jax.numpy as jnp


class TestFieldState:
    """Tests for US-004: FieldState dataclass."""
    
    def test_field_state_creation(self):
        """Test that FieldState can be created with correct shape."""
        from src.field.field import FieldState, create_field
        
        field = create_field(height=20, width=20, channels=4)
        
        assert isinstance(field, FieldState)
        assert field.values.shape == (20, 20, 4)
        assert field.values.dtype == jnp.float32
    
    def test_field_different_sizes(self):
        """Test field creation with different dimensions."""
        from src.field.field import create_field
        
        field1 = create_field(10, 10, 2)
        assert field1.values.shape == (10, 10, 2)
        
        field2 = create_field(32, 32, 8)
        assert field2.values.shape == (32, 32, 8)


class TestFieldDynamics:
    """Tests for US-005: Field diffusion and decay."""
    
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
    
    def test_diffusion_conserves_mass_approximately(self):
        """Test that diffusion approximately conserves total mass."""
        from src.field.field import create_field
        from src.field.dynamics import diffuse
        
        field = create_field(20, 20, 1)
        values = jax.random.uniform(jax.random.PRNGKey(0), (20, 20, 1))
        field = field.replace(values=values)
        
        original_sum = field.values.sum()
        diffused = diffuse(field, rate=0.3)
        diffused_sum = diffused.values.sum()
        
        # Should be approximately equal (some loss at boundaries)
        assert jnp.abs(original_sum - diffused_sum) < original_sum * 0.1
    
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
        
        # All values should be reduced by exactly 10%
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
    
    def test_step_field_jit_compatible(self):
        """Test that step_field works with JIT."""
        from src.field.field import create_field
        from src.field.dynamics import step_field
        
        field = create_field(20, 20, 4)
        values = jax.random.uniform(jax.random.PRNGKey(0), (20, 20, 4))
        field = field.replace(values=values)
        
        # JIT compile and run
        jit_step = jax.jit(lambda f: step_field(f, 0.1, 0.05))
        stepped = jit_step(field)
        
        assert stepped.values.shape == (20, 20, 4)


class TestFieldOps:
    """Tests for US-006: Field read/write operations."""
    
    def test_write_local(self):
        """Test writing to field at agent positions."""
        from src.field.field import create_field
        from src.field.ops import write_local
        
        field = create_field(20, 20, 4)
        
        # Agent positions (batch of 3 agents)
        positions = jnp.array([[5, 5], [10, 10], [15, 15]])
        
        # Write values at agent positions
        write_values = jnp.ones((3, 4))  # 3 agents, 4 channels
        field = write_local(field, positions, write_values)
        
        # Check values were written
        assert jnp.allclose(field.values[5, 5], 1.0)
        assert jnp.allclose(field.values[10, 10], 1.0)
        assert jnp.allclose(field.values[15, 15], 1.0)
        # Other positions should be zero
        assert jnp.allclose(field.values[0, 0], 0.0)
    
    def test_read_local(self):
        """Test reading local field values."""
        from src.field.field import create_field
        from src.field.ops import read_local, write_local
        
        field = create_field(20, 20, 4)
        
        # Write a value (within default cap of 1.0)
        positions = jnp.array([[10, 10]])
        write_values = jnp.ones((1, 4)) * 0.5
        field = write_local(field, positions, write_values)

        # Read at same position
        read_values = read_local(field, positions, radius=0)

        assert jnp.allclose(read_values[0], 0.5)
    
    def test_read_local_with_radius(self):
        """Test reading local field values with neighborhood."""
        from src.field.field import create_field
        from src.field.ops import read_local, write_local
        
        field = create_field(20, 20, 2)
        
        # Write at center
        positions = jnp.array([[10, 10]])
        write_values = jnp.ones((1, 2)) * 10.0
        field = write_local(field, positions, write_values)
        
        # Read with radius 2 (5x5 neighborhood)
        read_values = read_local(field, positions, radius=2)
        
        # Should be a flattened neighborhood
        # Shape: (1, (2*radius+1)^2 * channels) = (1, 25 * 2) = (1, 50)
        assert read_values.shape[0] == 1
        assert read_values.shape[1] > 0  # Has neighborhood data
    
    def test_read_write_batched(self):
        """Test that read/write work with batched positions."""
        from src.field.field import create_field
        from src.field.ops import read_local, write_local
        
        field = create_field(20, 20, 4)
        
        # 8 agent positions
        positions = jnp.array([
            [2, 2], [5, 5], [8, 8], [10, 10],
            [12, 12], [14, 14], [16, 16], [18, 18]
        ])
        
        # Write different values for each agent
        write_values = jnp.arange(8 * 4).reshape(8, 4).astype(jnp.float32)
        field = write_local(field, positions, write_values)
        
        # Read back
        read_values = read_local(field, positions, radius=0)
        
        assert read_values.shape[0] == 8
