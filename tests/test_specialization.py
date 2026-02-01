"""Tests for specialization detection module."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np


class TestWeightDivergence:
    """US-001: Weight divergence metric tests."""

    def test_identical_agents_zero_divergence(self):
        """Agents with identical weights should have zero divergence."""
        from src.analysis.specialization import compute_weight_divergence

        # Create per-agent params where all agents are identical
        key = jax.random.PRNGKey(42)
        num_agents = 4
        single_weights = jax.random.normal(key, (64, 32))
        # Tile to all agents (all identical)
        params = {"layer": {"kernel": jnp.tile(single_weights[None], (num_agents, 1, 1))}}
        alive = np.ones(num_agents, dtype=bool)

        result = compute_weight_divergence(params, alive)

        assert result["mean_divergence"] == pytest.approx(0.0, abs=1e-6)
        assert result["max_divergence"] == pytest.approx(0.0, abs=1e-6)
        assert result["divergence_matrix"].shape == (num_agents, num_agents)

    def test_different_agents_positive_divergence(self):
        """Agents with different weights should have positive divergence."""
        from src.analysis.specialization import compute_weight_divergence

        num_agents = 4
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, num_agents)
        # Each agent gets independent random weights
        kernels = jnp.stack([jax.random.normal(k, (64, 32)) for k in keys])
        params = {"layer": {"kernel": kernels}}
        alive = np.ones(num_agents, dtype=bool)

        result = compute_weight_divergence(params, alive)

        assert result["mean_divergence"] > 0.0
        assert result["max_divergence"] >= result["mean_divergence"]
        assert result["divergence_matrix"].shape == (num_agents, num_agents)

    def test_divergence_matrix_symmetric(self):
        """Divergence matrix should be symmetric with zero diagonal."""
        from src.analysis.specialization import compute_weight_divergence

        num_agents = 3
        key = jax.random.PRNGKey(1)
        keys = jax.random.split(key, num_agents)
        kernels = jnp.stack([jax.random.normal(k, (32, 16)) for k in keys])
        params = {"dense": {"kernel": kernels}}
        alive = np.ones(num_agents, dtype=bool)

        result = compute_weight_divergence(params, alive)
        matrix = result["divergence_matrix"]

        # Symmetric
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-6)
        # Zero diagonal
        np.testing.assert_allclose(np.diag(matrix), 0.0, atol=1e-6)

    def test_alive_mask_filters_agents(self):
        """Only alive agents should appear in the divergence matrix."""
        from src.analysis.specialization import compute_weight_divergence

        num_agents = 5
        key = jax.random.PRNGKey(2)
        keys = jax.random.split(key, num_agents)
        kernels = jnp.stack([jax.random.normal(k, (16, 8)) for k in keys])
        params = {"layer": {"kernel": kernels}}
        alive = np.array([True, False, True, False, True])

        result = compute_weight_divergence(params, alive)

        # Only 3 alive agents
        assert result["divergence_matrix"].shape == (3, 3)
        np.testing.assert_array_equal(result["agent_indices"], [0, 2, 4])

    def test_single_alive_agent(self):
        """With only one alive agent, divergence should be zero."""
        from src.analysis.specialization import compute_weight_divergence

        num_agents = 4
        key = jax.random.PRNGKey(3)
        kernels = jax.random.normal(key, (num_agents, 32, 16))
        params = {"layer": {"kernel": kernels}}
        alive = np.array([False, True, False, False])

        result = compute_weight_divergence(params, alive)

        assert result["mean_divergence"] == 0.0
        assert result["max_divergence"] == 0.0

    def test_no_alive_agents(self):
        """With no alive agents, divergence should be zero."""
        from src.analysis.specialization import compute_weight_divergence

        num_agents = 3
        key = jax.random.PRNGKey(4)
        kernels = jax.random.normal(key, (num_agents, 16, 8))
        params = {"layer": {"kernel": kernels}}
        alive = np.zeros(num_agents, dtype=bool)

        result = compute_weight_divergence(params, alive)

        assert result["mean_divergence"] == 0.0
        assert result["max_divergence"] == 0.0

    def test_none_alive_mask_uses_all(self):
        """Passing alive_mask=None should use all agents."""
        from src.analysis.specialization import compute_weight_divergence

        num_agents = 3
        key = jax.random.PRNGKey(5)
        keys = jax.random.split(key, num_agents)
        kernels = jnp.stack([jax.random.normal(k, (32, 16)) for k in keys])
        params = {"dense": {"kernel": kernels}}

        result = compute_weight_divergence(params, alive_mask=None)

        assert result["divergence_matrix"].shape == (num_agents, num_agents)
        assert len(result["agent_indices"]) == num_agents

    def test_multi_leaf_params(self):
        """Divergence works with multi-leaf param pytrees (realistic network)."""
        from src.analysis.specialization import compute_weight_divergence

        num_agents = 3
        key = jax.random.PRNGKey(6)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        # Simulate realistic ActorCritic params with multiple layers
        params = {
            "params": {
                "Dense_0": {
                    "kernel": jax.random.normal(k1, (num_agents, 32, 64)),
                    "bias": jax.random.normal(k2, (num_agents, 64)),
                },
                "Dense_1": {
                    "kernel": jax.random.normal(k3, (num_agents, 64, 5)),
                    "bias": jax.random.normal(k4, (num_agents, 5)),
                },
            }
        }
        alive = np.ones(num_agents, dtype=bool)

        result = compute_weight_divergence(params, alive)

        assert result["mean_divergence"] > 0.0
        assert result["divergence_matrix"].shape == (num_agents, num_agents)

    def test_cosine_distance_range(self):
        """Cosine distance should be in [0, 2] range."""
        from src.analysis.specialization import compute_weight_divergence

        num_agents = 4
        key = jax.random.PRNGKey(7)
        keys = jax.random.split(key, num_agents)
        kernels = jnp.stack([jax.random.normal(k, (64, 32)) for k in keys])
        params = {"layer": {"kernel": kernels}}
        alive = np.ones(num_agents, dtype=bool)

        result = compute_weight_divergence(params, alive)

        assert result["mean_divergence"] >= 0.0
        assert result["max_divergence"] <= 2.0

    def test_flatten_agent_params(self):
        """Test that flatten_agent_params returns correct shape."""
        from src.analysis.specialization import flatten_agent_params

        num_agents = 3
        key = jax.random.PRNGKey(8)
        k1, k2 = jax.random.split(key)

        params = {
            "layer1": {"kernel": jax.random.normal(k1, (num_agents, 10, 5))},
            "layer2": {"kernel": jax.random.normal(k2, (num_agents, 5, 3))},
        }

        flat = flatten_agent_params(params, agent_idx=0)

        # Should be 10*5 + 5*3 = 65 elements
        assert flat.shape == (65,)
        assert isinstance(flat, np.ndarray)
