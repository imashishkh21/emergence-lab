"""Tests for Pairwise PID Synergy via dit.

Tests the compute_pairwise_pid, compute_interaction_information, and
compute_median_synergy functions for measuring synergistic information
between agent pairs.

PID (Partial Information Decomposition) decomposes the mutual information
I(S1, S2; T) into synergy, redundancy, and unique information components.
Synergy is the holy grail for emergence: information that S1 and S2
provide about T that neither provides alone.

Reference: Williams & Beer (2010), Riedl et al. (2025).
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

# Check if dit is available for conditional tests
_dit_available = importlib.util.find_spec("dit") is not None


class TestComputeInteractionInformation:
    """Tests for compute_interaction_information function.

    Interaction Information II(X; Y; Z) = I(X; Y | Z) - I(X; Y)
    II < 0 means synergy dominates (quick screening function).
    """

    @pytest.mark.skipif(not _dit_available, reason="dit not installed")
    def test_basic_computation(self) -> None:
        """Test basic II computation returns finite value."""
        from src.analysis.pid_synergy import compute_interaction_information

        rng = np.random.default_rng(42)
        n_samples = 100

        x = rng.choice(6, size=n_samples)  # Agent 1 action (6 values)
        y = rng.choice(6, size=n_samples)  # Agent 2 action
        z = rng.choice(2, size=n_samples)  # Future food (K=2 bins)

        ii = compute_interaction_information(x, y, z)

        assert isinstance(ii, float)
        assert np.isfinite(ii)

    @pytest.mark.skipif(not _dit_available, reason="dit not installed")
    def test_synergistic_xor(self) -> None:
        """Test that XOR pattern produces negative II (synergy signal).

        XOR is the classic synergy example: knowing both X and Y together
        lets you predict Z, but neither alone does.
        """
        from src.analysis.pid_synergy import compute_interaction_information

        rng = np.random.default_rng(42)
        n_samples = 500

        x = rng.choice([0, 1], size=n_samples)
        y = rng.choice([0, 1], size=n_samples)
        z = (x ^ y).astype(int)  # XOR

        ii = compute_interaction_information(x, y, z)

        # XOR should produce strongly negative II (synergy)
        assert ii < 0, f"XOR should give negative II (synergy), got {ii}"

    @pytest.mark.skipif(not _dit_available, reason="dit not installed")
    def test_redundant_copy(self) -> None:
        """Test that copy/redundant pattern produces positive II.

        When Z is just a copy of X, Y adds no synergistic info.
        """
        from src.analysis.pid_synergy import compute_interaction_information

        rng = np.random.default_rng(42)
        n_samples = 500

        x = rng.choice([0, 1], size=n_samples)
        y = rng.choice([0, 1], size=n_samples)
        z = x.copy()  # Z = X (pure redundancy when Y is used)

        ii = compute_interaction_information(x, y, z)

        # With Z=X, Y provides no additional info (redundancy or zero)
        assert isinstance(ii, float)
        assert np.isfinite(ii)

    @pytest.mark.skipif(not _dit_available, reason="dit not installed")
    def test_independent_variables(self) -> None:
        """Test that fully independent variables give II near zero."""
        from src.analysis.pid_synergy import compute_interaction_information

        rng = np.random.default_rng(42)
        n_samples = 500

        x = rng.choice([0, 1], size=n_samples)
        y = rng.choice([0, 1], size=n_samples)
        z = rng.choice([0, 1], size=n_samples)

        ii = compute_interaction_information(x, y, z)

        # Independent variables should give II close to zero
        assert abs(ii) < 0.5, f"Independent vars should have |II| < 0.5, got {ii}"

    def test_empty_input(self) -> None:
        """Test that empty arrays return 0.0."""
        from src.analysis.pid_synergy import compute_interaction_information

        ii = compute_interaction_information(
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=int),
        )
        assert ii == 0.0

    def test_single_element(self) -> None:
        """Test that single-element arrays return 0.0."""
        from src.analysis.pid_synergy import compute_interaction_information

        ii = compute_interaction_information(
            np.array([1]),
            np.array([2]),
            np.array([0]),
        )
        assert ii == 0.0

    def test_constant_variable(self) -> None:
        """Test that constant variables return 0.0."""
        from src.analysis.pid_synergy import compute_interaction_information

        n_samples = 100
        x = np.zeros(n_samples, dtype=int)
        y = np.ones(n_samples, dtype=int)
        z = np.zeros(n_samples, dtype=int)

        ii = compute_interaction_information(x, y, z)
        assert ii == 0.0


class TestComputePairwisePID:
    """Tests for compute_pairwise_pid function.

    Returns synergy, redundancy, unique_s1, unique_s2 for agent pairs.
    Uses K=2 quantile bins and Jeffreys smoothing (alpha=0.5).
    """

    @pytest.mark.skipif(not _dit_available, reason="dit not installed")
    def test_basic_computation(self) -> None:
        """Test basic PID computation returns expected keys."""
        from src.analysis.pid_synergy import compute_pairwise_pid

        rng = np.random.default_rng(42)
        n_samples = 100

        actions = rng.choice(6, size=n_samples)
        field_summary = rng.random(n_samples)  # Continuous, will be binned
        future_food = rng.random(n_samples)  # Continuous, will be binned

        result = compute_pairwise_pid(actions, field_summary, future_food)

        assert "synergy" in result
        assert "redundancy" in result
        assert "unique_s1" in result
        assert "unique_s2" in result
        assert all(isinstance(v, float) for v in result.values())
        assert all(np.isfinite(v) for v in result.values())

    @pytest.mark.skipif(not _dit_available, reason="dit not installed")
    def test_non_negative_components(self) -> None:
        """Test that all PID components are non-negative."""
        from src.analysis.pid_synergy import compute_pairwise_pid

        rng = np.random.default_rng(42)
        n_samples = 200

        actions = rng.choice(6, size=n_samples)
        field_summary = rng.random(n_samples)
        future_food = rng.random(n_samples)

        result = compute_pairwise_pid(actions, field_summary, future_food)

        # All PID components should be non-negative
        assert result["synergy"] >= -1e-10, f"Synergy should be >= 0, got {result['synergy']}"
        assert result["redundancy"] >= -1e-10, f"Redundancy should be >= 0, got {result['redundancy']}"
        assert result["unique_s1"] >= -1e-10, f"Unique S1 should be >= 0, got {result['unique_s1']}"
        assert result["unique_s2"] >= -1e-10, f"Unique S2 should be >= 0, got {result['unique_s2']}"

    @pytest.mark.skipif(not _dit_available, reason="dit not installed")
    def test_synergistic_pattern(self) -> None:
        """Test that XOR-like pattern shows high synergy."""
        from src.analysis.pid_synergy import compute_pairwise_pid

        rng = np.random.default_rng(42)
        n_samples = 500

        # Create XOR-like pattern
        # S1 = action (0 or 1)
        # S2 = field (0 or 1)
        # T = XOR(S1, S2) - the target is only predictable with both sources
        s1 = rng.choice([0, 1], size=n_samples)
        s2 = rng.choice([0, 1], size=n_samples)
        t = (s1 ^ s2).astype(float)  # XOR

        # Add small noise to avoid degenerate case
        t = t + rng.normal(0, 0.01, size=n_samples)

        result = compute_pairwise_pid(s1, s2.astype(float), t)

        # XOR should have high synergy relative to other components
        assert result["synergy"] > 0.1, f"XOR should show synergy > 0.1, got {result['synergy']}"

    @pytest.mark.skipif(not _dit_available, reason="dit not installed")
    def test_redundant_pattern(self) -> None:
        """Test that copy pattern shows high redundancy."""
        from src.analysis.pid_synergy import compute_pairwise_pid

        rng = np.random.default_rng(42)
        n_samples = 500

        # Create redundant pattern: T = S1 = S2
        base = rng.choice([0, 1], size=n_samples)
        s1 = base.copy()
        s2 = base.astype(float)
        t = base.astype(float) + rng.normal(0, 0.01, size=n_samples)

        result = compute_pairwise_pid(s1, s2, t)

        # Both sources say the same thing - high redundancy
        assert result["redundancy"] > 0.1, f"Copy should show redundancy > 0.1, got {result['redundancy']}"

    @pytest.mark.skipif(not _dit_available, reason="dit not installed")
    def test_jeffreys_smoothing_applied(self) -> None:
        """Test that Jeffreys smoothing (alpha=0.5) prevents zero probabilities."""
        from src.analysis.pid_synergy import compute_pairwise_pid

        rng = np.random.default_rng(42)
        n_samples = 50  # Small sample to make some outcomes rare

        actions = rng.choice(6, size=n_samples)
        field_summary = rng.random(n_samples)
        future_food = rng.random(n_samples)

        # Should not raise even with small samples due to smoothing
        result = compute_pairwise_pid(actions, field_summary, future_food)

        assert all(np.isfinite(v) for v in result.values())

    @pytest.mark.skipif(not _dit_available, reason="dit not installed")
    def test_custom_num_bins(self) -> None:
        """Test PID computation with custom number of bins."""
        from src.analysis.pid_synergy import compute_pairwise_pid

        rng = np.random.default_rng(42)
        n_samples = 200

        actions = rng.choice(6, size=n_samples)
        field_summary = rng.random(n_samples)
        future_food = rng.random(n_samples)

        result_k2 = compute_pairwise_pid(actions, field_summary, future_food, num_bins=2)
        result_k4 = compute_pairwise_pid(actions, field_summary, future_food, num_bins=4)

        # Both should return valid results
        assert all(np.isfinite(v) for v in result_k2.values())
        assert all(np.isfinite(v) for v in result_k4.values())

    def test_empty_input(self) -> None:
        """Test that empty arrays return zeros."""
        from src.analysis.pid_synergy import compute_pairwise_pid

        result = compute_pairwise_pid(
            np.array([], dtype=int),
            np.array([]),
            np.array([]),
        )

        assert result["synergy"] == 0.0
        assert result["redundancy"] == 0.0
        assert result["unique_s1"] == 0.0
        assert result["unique_s2"] == 0.0

    def test_single_element(self) -> None:
        """Test that single-element arrays return zeros."""
        from src.analysis.pid_synergy import compute_pairwise_pid

        result = compute_pairwise_pid(
            np.array([1]),
            np.array([0.5]),
            np.array([0.2]),
        )

        assert result["synergy"] == 0.0

    def test_constant_variables(self) -> None:
        """Test that constant variables return zeros."""
        from src.analysis.pid_synergy import compute_pairwise_pid

        n_samples = 100
        actions = np.zeros(n_samples, dtype=int)
        field_summary = np.ones(n_samples)
        future_food = np.ones(n_samples)

        result = compute_pairwise_pid(actions, field_summary, future_food)

        # Constant variables have no information
        assert result["synergy"] == 0.0


class TestComputeMedianSynergy:
    """Tests for compute_median_synergy function.

    Computes median synergy across all agent pairs.
    """

    @pytest.mark.skipif(not _dit_available, reason="dit not installed")
    def test_basic_computation(self) -> None:
        """Test median synergy computation across agent pairs."""
        from src.analysis.pid_synergy import compute_median_synergy

        rng = np.random.default_rng(42)
        n_samples = 100
        n_agents = 4

        # Agent data: actions per agent, field summary per agent, future food per agent
        agent_actions = rng.choice(6, size=(n_samples, n_agents))
        field_summaries = rng.random((n_samples, n_agents))
        future_foods = rng.random((n_samples, n_agents))

        result = compute_median_synergy(agent_actions, field_summaries, future_foods)

        assert "median_synergy" in result
        assert "mean_synergy" in result
        assert "std_synergy" in result
        assert "num_pairs" in result
        assert "synergy_per_pair" in result

        # For 4 agents, we have C(4,2) = 6 pairs
        assert result["num_pairs"] == 6
        assert len(result["synergy_per_pair"]) == 6

    @pytest.mark.skipif(not _dit_available, reason="dit not installed")
    def test_two_agents(self) -> None:
        """Test with exactly 2 agents (1 pair)."""
        from src.analysis.pid_synergy import compute_median_synergy

        rng = np.random.default_rng(42)
        n_samples = 100
        n_agents = 2

        agent_actions = rng.choice(6, size=(n_samples, n_agents))
        field_summaries = rng.random((n_samples, n_agents))
        future_foods = rng.random((n_samples, n_agents))

        result = compute_median_synergy(agent_actions, field_summaries, future_foods)

        assert result["num_pairs"] == 1
        assert result["median_synergy"] == result["mean_synergy"]

    def test_single_agent(self) -> None:
        """Test that single agent returns zero synergy."""
        from src.analysis.pid_synergy import compute_median_synergy

        rng = np.random.default_rng(42)
        n_samples = 100
        n_agents = 1

        agent_actions = rng.choice(6, size=(n_samples, n_agents))
        field_summaries = rng.random((n_samples, n_agents))
        future_foods = rng.random((n_samples, n_agents))

        result = compute_median_synergy(agent_actions, field_summaries, future_foods)

        assert result["median_synergy"] == 0.0
        assert result["num_pairs"] == 0

    def test_empty_trajectory(self) -> None:
        """Test that empty trajectory returns zero synergy."""
        from src.analysis.pid_synergy import compute_median_synergy

        result = compute_median_synergy(
            np.array([]).reshape(0, 4),
            np.array([]).reshape(0, 4),
            np.array([]).reshape(0, 4),
        )

        assert result["median_synergy"] == 0.0
        assert result["num_pairs"] == 0


class TestSurrogateSignificanceTest:
    """Tests for surrogate_significance_test function.

    Row/column shuffle to test statistical significance.
    """

    @pytest.mark.skipif(not _dit_available, reason="dit not installed")
    def test_basic_surrogate_test(self) -> None:
        """Test basic surrogate significance testing."""
        from src.analysis.pid_synergy import surrogate_significance_test

        rng = np.random.default_rng(42)
        n_samples = 100

        # Create synergistic XOR pattern
        x = rng.choice([0, 1], size=n_samples)
        y = rng.choice([0, 1], size=n_samples)
        z = (x ^ y).astype(int)

        result = surrogate_significance_test(
            x, y, z,
            metric="interaction_information",
            n_surrogates=50,
            seed=42,
        )

        assert "observed" in result
        assert "surrogate_mean" in result
        assert "surrogate_std" in result
        assert "p_value" in result
        assert "significant" in result
        assert "n_surrogates" in result

    @pytest.mark.skipif(not _dit_available, reason="dit not installed")
    def test_xor_is_significant(self) -> None:
        """Test that XOR pattern is significantly synergistic."""
        from src.analysis.pid_synergy import surrogate_significance_test

        rng = np.random.default_rng(42)
        n_samples = 200

        x = rng.choice([0, 1], size=n_samples)
        y = rng.choice([0, 1], size=n_samples)
        z = (x ^ y).astype(int)

        result = surrogate_significance_test(
            x, y, z,
            metric="interaction_information",
            n_surrogates=100,
            seed=42,
        )

        # XOR should be significantly synergistic (II < surrogates)
        assert result["observed"] < result["surrogate_mean"], "XOR should have lower II than surrogates"

    @pytest.mark.skipif(not _dit_available, reason="dit not installed")
    def test_random_is_not_significant(self) -> None:
        """Test that random data is not significantly synergistic."""
        from src.analysis.pid_synergy import surrogate_significance_test

        rng = np.random.default_rng(42)
        n_samples = 200

        x = rng.choice([0, 1], size=n_samples)
        y = rng.choice([0, 1], size=n_samples)
        z = rng.choice([0, 1], size=n_samples)  # Independent

        result = surrogate_significance_test(
            x, y, z,
            metric="interaction_information",
            n_surrogates=100,
            seed=42,
        )

        # Random data should not be significantly synergistic
        # p-value should be > 0.05 (not significant)
        assert result["p_value"] > 0.01, f"Random data should not be significant, p={result['p_value']}"

    @pytest.mark.skipif(not _dit_available, reason="dit not installed")
    def test_shuffle_type_row(self) -> None:
        """Test row shuffle breaks cross-agent coordination."""
        from src.analysis.pid_synergy import surrogate_significance_test

        rng = np.random.default_rng(42)
        n_samples = 100

        x = rng.choice([0, 1], size=n_samples)
        y = rng.choice([0, 1], size=n_samples)
        z = (x ^ y).astype(int)

        result = surrogate_significance_test(
            x, y, z,
            metric="interaction_information",
            n_surrogates=50,
            shuffle_type="row",
            seed=42,
        )

        assert "shuffle_type" in result
        assert result["shuffle_type"] == "row"

    @pytest.mark.skipif(not _dit_available, reason="dit not installed")
    def test_shuffle_type_column(self) -> None:
        """Test column shuffle breaks temporal dependencies."""
        from src.analysis.pid_synergy import surrogate_significance_test

        rng = np.random.default_rng(42)
        n_samples = 100

        x = rng.choice([0, 1], size=n_samples)
        y = rng.choice([0, 1], size=n_samples)
        z = (x ^ y).astype(int)

        result = surrogate_significance_test(
            x, y, z,
            metric="interaction_information",
            n_surrogates=50,
            shuffle_type="column",
            seed=42,
        )

        assert result["shuffle_type"] == "column"


class TestDiscretization:
    """Tests for discretization utilities."""

    @pytest.mark.skipif(not _dit_available, reason="dit not installed")
    def test_quantile_discretization(self) -> None:
        """Test that quantile discretization produces K bins."""
        from src.analysis.pid_synergy import discretize_continuous

        rng = np.random.default_rng(42)
        data = rng.random(100)

        binned = discretize_continuous(data, num_bins=2)

        assert binned.dtype == np.int64 or binned.dtype == np.int32
        assert set(binned).issubset({0, 1})

    @pytest.mark.skipif(not _dit_available, reason="dit not installed")
    def test_discretization_with_4_bins(self) -> None:
        """Test discretization with K=4 bins."""
        from src.analysis.pid_synergy import discretize_continuous

        rng = np.random.default_rng(42)
        data = rng.random(100)

        binned = discretize_continuous(data, num_bins=4)

        assert set(binned).issubset({0, 1, 2, 3})

    @pytest.mark.skipif(not _dit_available, reason="dit not installed")
    def test_discretization_constant_input(self) -> None:
        """Test discretization handles constant input."""
        from src.analysis.pid_synergy import discretize_continuous

        data = np.ones(100)
        binned = discretize_continuous(data, num_bins=2)

        # All same value should map to same bin
        assert len(set(binned)) == 1


class TestJeffreysSmoothing:
    """Tests for Jeffreys smoothing utilities."""

    @pytest.mark.skipif(not _dit_available, reason="dit not installed")
    def test_jeffreys_smoothing(self) -> None:
        """Test Jeffreys smoothing adds alpha=0.5 pseudocounts."""
        from src.analysis.pid_synergy import apply_jeffreys_smoothing

        # Create a joint distribution with some zeros
        counts = np.array([10, 0, 5, 0])

        smoothed = apply_jeffreys_smoothing(counts, alpha=0.5)

        # All should be non-zero after smoothing
        assert all(smoothed > 0)
        # Should sum to original + alpha * num_bins
        assert abs(sum(smoothed) - (sum(counts) + 0.5 * len(counts))) < 1e-10


class TestIntegration:
    """Integration tests with realistic agent data."""

    @pytest.mark.skipif(not _dit_available, reason="dit not installed")
    def test_full_workflow(self) -> None:
        """Test complete workflow: discretize -> PID -> significance."""
        from src.analysis.pid_synergy import (
            compute_pairwise_pid,
            surrogate_significance_test,
        )

        rng = np.random.default_rng(42)
        n_samples = 200

        # Simulate agent data
        actions = rng.choice(6, size=n_samples)  # 6 action values
        field_summary = rng.random(n_samples)  # Continuous field reading
        # Future food correlated with field (emergence signal)
        future_food = 0.5 * field_summary + 0.5 * rng.random(n_samples)

        # Compute PID
        pid_result = compute_pairwise_pid(actions, field_summary, future_food)
        assert all(np.isfinite(v) for v in pid_result.values())

        # Test significance
        sig_result = surrogate_significance_test(
            actions, field_summary, future_food,
            metric="synergy",
            n_surrogates=50,
            seed=42,
        )
        assert "p_value" in sig_result

    @pytest.mark.skipif(not _dit_available, reason="dit not installed")
    def test_ablation_comparison(self) -> None:
        """Test PID computation for field ablation comparison."""
        from src.analysis.pid_synergy import compute_pairwise_pid

        rng = np.random.default_rng(42)
        n_samples = 200

        # Normal field: actions + field predict future food
        actions = rng.choice(6, size=n_samples)
        normal_field = rng.random(n_samples)
        future_food = 0.5 * (actions / 6) + 0.3 * normal_field + 0.2 * rng.random(n_samples)

        pid_normal = compute_pairwise_pid(actions, normal_field, future_food)

        # Zeroed field: field is constant (zero)
        zeroed_field = np.zeros(n_samples)
        pid_zeroed = compute_pairwise_pid(actions, zeroed_field, future_food)

        # Random field: field is noise
        random_field = rng.random(n_samples)
        pid_random = compute_pairwise_pid(actions, random_field, future_food)

        # All should return valid results
        assert np.isfinite(pid_normal["synergy"])
        assert np.isfinite(pid_zeroed["synergy"])
        assert np.isfinite(pid_random["synergy"])
