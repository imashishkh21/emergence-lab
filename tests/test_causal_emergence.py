"""Tests for causal emergence metrics.

Tests the Effective Information (EI) and Rosas' Psi metrics for measuring
causal emergence at the macro scale.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.causal_emergence import (
    CausalEmergenceEvent,
    CausalEmergenceTracker,
    build_tpm,
    compute_causal_emergence_from_trajectory,
    compute_effective_information,
    compute_mutual_information_discrete,
    compute_rosas_psi,
    compute_tpm_entropy,
    compute_windowed_causal_emergence,
    discretize_to_bins,
    extract_macro_variables,
)


class TestDiscretizeToBins:
    """Tests for discretize_to_bins function."""

    def test_basic_quantile_binning(self) -> None:
        """Test basic quantile binning."""
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
        bins = discretize_to_bins(values, num_bins=4, method="quantile")

        assert len(bins) == 8
        assert bins.min() >= 0
        assert bins.max() <= 3  # 4 bins = indices 0-3
        assert bins.dtype == np.int32

    def test_basic_uniform_binning(self) -> None:
        """Test basic uniform binning."""
        values = np.array([0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
        bins = discretize_to_bins(values, num_bins=4, method="uniform")

        assert len(bins) == 5
        assert bins.min() >= 0
        assert bins.max() <= 3

    def test_constant_values(self) -> None:
        """Test handling of constant values."""
        values = np.ones(100, dtype=np.float64)
        bins = discretize_to_bins(values, num_bins=4)

        assert np.all(bins == 0)  # All should map to same bin

    def test_nan_handling(self) -> None:
        """Test NaN values are replaced with median."""
        values = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        bins = discretize_to_bins(values, num_bins=2)

        assert len(bins) == 5
        assert np.all(bins >= 0)
        # NaN should be replaced, not cause errors

    def test_unknown_method(self) -> None:
        """Test that unknown method raises error."""
        values = np.array([1, 2, 3, 4])
        with pytest.raises(ValueError, match="Unknown binning method"):
            discretize_to_bins(values, method="unknown")


class TestBuildTPM:
    """Tests for build_tpm function."""

    def test_basic_tpm(self) -> None:
        """Test basic TPM construction."""
        states_t = np.array([0, 0, 1, 1, 0])
        states_t1 = np.array([0, 1, 1, 0, 0])
        tpm = build_tpm(states_t, states_t1, num_states=2)

        assert tpm.shape == (2, 2)
        # Rows should sum to ~1 (with smoothing)
        assert np.allclose(tpm.sum(axis=1), 1.0, atol=1e-6)

    def test_tpm_probabilities(self) -> None:
        """Test that TPM probabilities are correct."""
        # Deterministic transitions: 0->1, 1->0
        states_t = np.array([0, 0, 0, 1, 1, 1])
        states_t1 = np.array([1, 1, 1, 0, 0, 0])
        tpm = build_tpm(states_t, states_t1, num_states=2, smoothing=0)

        # State 0 should transition to 1 with high probability
        assert tpm[0, 1] > tpm[0, 0]
        # State 1 should transition to 0 with high probability
        assert tpm[1, 0] > tpm[1, 1]

    def test_tpm_with_smoothing(self) -> None:
        """Test that smoothing prevents zero probabilities."""
        states_t = np.array([0, 0, 0])
        states_t1 = np.array([0, 0, 0])
        tpm = build_tpm(states_t, states_t1, num_states=2, smoothing=1e-10)

        # All entries should be positive
        assert np.all(tpm > 0)

    def test_tpm_infer_num_states(self) -> None:
        """Test automatic inference of num_states."""
        states_t = np.array([0, 1, 2])
        states_t1 = np.array([1, 2, 0])
        tpm = build_tpm(states_t, states_t1)

        assert tpm.shape == (3, 3)


class TestComputeTPMEntropy:
    """Tests for compute_tpm_entropy function."""

    def test_deterministic_tpm(self) -> None:
        """Test entropy of deterministic TPM (should be near 0)."""
        # Each state deterministically transitions to a specific state
        tpm = np.array([[1.0, 0.0], [0.0, 1.0]])
        entropy = compute_tpm_entropy(tpm)

        assert entropy < 0.1  # Should be very low

    def test_uniform_tpm(self) -> None:
        """Test entropy of uniform TPM (should be maximal)."""
        # Each state transitions uniformly to all states
        tpm = np.array([[0.5, 0.5], [0.5, 0.5]])
        entropy = compute_tpm_entropy(tpm)

        # Maximum entropy for 2 states is 1 bit
        assert entropy > 0.9

    def test_intermediate_tpm(self) -> None:
        """Test entropy of intermediate TPM."""
        tpm = np.array([[0.8, 0.2], [0.3, 0.7]])
        entropy = compute_tpm_entropy(tpm)

        assert 0 < entropy < 1


class TestComputeEffectiveInformation:
    """Tests for compute_effective_information function."""

    def test_basic_ei(self) -> None:
        """Test basic EI computation."""
        # Simple 2x2 TPMs
        micro_tpm = np.array([[0.7, 0.3], [0.4, 0.6]])
        macro_tpm = np.array([[0.9, 0.1], [0.2, 0.8]])

        result = compute_effective_information(micro_tpm, macro_tpm)

        assert "micro_ei" in result
        assert "macro_ei" in result
        assert "ei_gap" in result
        assert "micro_entropy" in result
        assert "macro_entropy" in result

        # EI should be non-negative
        assert result["micro_ei"] >= 0
        assert result["macro_ei"] >= 0

        # EI gap should match difference
        assert np.isclose(
            result["ei_gap"],
            result["macro_ei"] - result["micro_ei"],
            atol=1e-6,
        )

    def test_ei_with_more_deterministic_macro(self) -> None:
        """Test that more deterministic macro gives positive EI gap."""
        # Micro: noisy transitions
        micro_tpm = np.array([[0.5, 0.5], [0.5, 0.5]])

        # Macro: deterministic transitions
        macro_tpm = np.array([[0.99, 0.01], [0.01, 0.99]])

        result = compute_effective_information(micro_tpm, macro_tpm)

        # More deterministic macro should have higher EI
        assert result["macro_ei"] > result["micro_ei"]
        assert result["ei_gap"] > 0

    def test_ei_single_state(self) -> None:
        """Test EI with single-state TPMs."""
        micro_tpm = np.array([[1.0]])
        macro_tpm = np.array([[1.0]])

        result = compute_effective_information(micro_tpm, macro_tpm)

        # With only 1 state, EI should be 0 (log2(1) = 0)
        assert result["micro_ei"] == 0.0
        assert result["macro_ei"] == 0.0


class TestComputeMutualInformationDiscrete:
    """Tests for compute_mutual_information_discrete function."""

    def test_identical_variables(self) -> None:
        """Test MI of identical variables (should be maximal)."""
        x = np.array([0, 1, 0, 1, 0, 1])
        y = np.array([0, 1, 0, 1, 0, 1])

        mi = compute_mutual_information_discrete(x, y)

        assert mi > 0.9  # Should be close to 1 bit for binary

    def test_independent_variables(self) -> None:
        """Test MI of independent variables (should be near 0)."""
        rng = np.random.default_rng(42)
        x = rng.integers(0, 4, size=1000)
        y = rng.integers(0, 4, size=1000)

        mi = compute_mutual_information_discrete(x, y)

        assert mi < 0.1  # Should be close to 0

    def test_mi_non_negative(self) -> None:
        """Test that MI is always non-negative."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            x = rng.integers(0, 3, size=100)
            y = rng.integers(0, 3, size=100)
            mi = compute_mutual_information_discrete(x, y)
            assert mi >= 0

    def test_mismatched_lengths(self) -> None:
        """Test that mismatched lengths raise error."""
        x = np.array([0, 1, 2])
        y = np.array([0, 1])

        with pytest.raises(ValueError, match="same length"):
            compute_mutual_information_discrete(x, y)


class TestComputeRosasPsi:
    """Tests for compute_rosas_psi function."""

    def test_basic_psi(self) -> None:
        """Test basic Psi computation."""
        rng = np.random.default_rng(42)
        n_samples = 100

        # Macro variable: population count (simulated)
        macro_t = rng.integers(5, 10, size=n_samples).astype(np.float64)
        macro_t1 = macro_t + rng.choice([-1, 0, 1], size=n_samples)
        macro_t1 = np.clip(macro_t1, 1, 15)

        # Micro variables: individual agent states
        micro_t = rng.integers(0, 2, size=(n_samples, 5)).astype(np.float64)

        result = compute_rosas_psi(macro_t, macro_t1, micro_t)

        assert "psi" in result
        assert "macro_mi" in result
        assert "sum_micro_mi" in result
        assert "n_micro" in result

        assert result["n_micro"] == 5
        assert np.isfinite(result["psi"])

    def test_psi_calculation(self) -> None:
        """Test that Psi = macro_mi - sum_micro_mi."""
        rng = np.random.default_rng(42)
        n_samples = 100

        macro_t = rng.standard_normal(n_samples)
        macro_t1 = macro_t + rng.normal(0, 0.1, n_samples)
        micro_t = rng.standard_normal((n_samples, 4))

        result = compute_rosas_psi(macro_t, macro_t1, micro_t)

        expected_psi = result["macro_mi"] - result["sum_micro_mi"]
        assert np.isclose(result["psi"], expected_psi, atol=1e-6)

    def test_psi_1d_micro(self) -> None:
        """Test Psi with 1D micro variables."""
        rng = np.random.default_rng(42)
        n_samples = 100

        macro_t = rng.standard_normal(n_samples)
        macro_t1 = rng.standard_normal(n_samples)
        micro_t = rng.standard_normal(n_samples)  # 1D

        result = compute_rosas_psi(macro_t, macro_t1, micro_t)

        assert result["n_micro"] == 1

    def test_psi_mismatched_lengths(self) -> None:
        """Test that mismatched lengths raise error."""
        macro_t = np.array([1.0, 2.0, 3.0])
        macro_t1 = np.array([1.0, 2.0])  # Different length
        micro_t = np.array([[1.0], [2.0], [3.0]])

        with pytest.raises(ValueError, match="same length"):
            compute_rosas_psi(macro_t, macro_t1, micro_t)

    def test_psi_too_few_samples(self) -> None:
        """Test Psi with too few samples returns zeros."""
        macro_t = np.array([1.0, 2.0, 3.0])
        macro_t1 = np.array([2.0, 3.0, 4.0])
        micro_t = np.array([[1.0], [2.0], [3.0]])

        result = compute_rosas_psi(macro_t, macro_t1, micro_t)

        assert result["psi"] == 0.0


class TestExtractMacroVariables:
    """Tests for extract_macro_variables function."""

    def test_extract_population_count(self) -> None:
        """Test extraction of population count."""
        trajectory = {
            "alive_mask": np.array([
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ])
        }

        macros = extract_macro_variables(trajectory)

        assert "population_count" in macros
        np.testing.assert_array_equal(macros["population_count"], [2, 3, 4])

    def test_extract_mean_field_intensity(self) -> None:
        """Test extraction of mean field intensity."""
        # Create field with different mean intensities per timestep
        field = np.zeros((3, 4, 4, 2))
        field[0, :, :, :] = 1.0
        field[1, :, :, :] = 2.0
        field[2, :, :, :] = 3.0
        trajectory = {"field": field}

        macros = extract_macro_variables(trajectory)

        assert "mean_field_intensity" in macros
        assert len(macros["mean_field_intensity"]) == 3
        # Check that values increase
        np.testing.assert_array_almost_equal(
            macros["mean_field_intensity"], [1.0, 2.0, 3.0]
        )

    def test_extract_spatial_dispersion(self) -> None:
        """Test extraction of spatial dispersion."""
        trajectory = {
            "positions": np.array([
                [[0, 0], [1, 1], [2, 2], [3, 3]],
                [[0, 0], [0, 0], [0, 0], [0, 0]],  # No dispersion
            ], dtype=np.float64),
            "alive_mask": np.ones((2, 4), dtype=bool),
        }

        macros = extract_macro_variables(trajectory)

        assert "spatial_dispersion" in macros
        # Second timestep should have lower dispersion
        assert macros["spatial_dispersion"][1] < macros["spatial_dispersion"][0]

    def test_extract_total_food_collected(self) -> None:
        """Test extraction of cumulative food collection."""
        trajectory = {
            "rewards": np.array([
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 1, 0, 0],
            ], dtype=np.float64)
        }

        macros = extract_macro_variables(trajectory)

        assert "total_food_collected" in macros
        expected = [2, 4, 6]  # Cumulative sum
        np.testing.assert_array_equal(macros["total_food_collected"], expected)

    def test_extract_field_entropy(self) -> None:
        """Test extraction of field entropy."""
        rng = np.random.default_rng(42)
        trajectory = {
            "field": rng.standard_normal((3, 4, 4, 2))
        }

        macros = extract_macro_variables(trajectory)

        assert "field_entropy" in macros
        assert len(macros["field_entropy"]) == 3
        assert np.all(macros["field_entropy"] >= 0)

    def test_empty_trajectory(self) -> None:
        """Test with empty trajectory returns empty dict."""
        macros = extract_macro_variables({})
        assert macros == {}


class TestComputeCausalEmergenceFromTrajectory:
    """Tests for compute_causal_emergence_from_trajectory function."""

    def test_basic_computation(self) -> None:
        """Test basic causal emergence computation from trajectory."""
        rng = np.random.default_rng(42)
        T = 100

        trajectory = {
            "alive_mask": rng.choice([True, False], size=(T, 8), p=[0.8, 0.2]),
            "field": rng.standard_normal((T, 4, 4, 2)),
            "positions": rng.integers(0, 10, size=(T, 8, 2)).astype(np.float64),
            "rewards": rng.integers(0, 2, size=(T, 8)).astype(np.float64),
        }

        result = compute_causal_emergence_from_trajectory(
            trajectory, macro_name="population_count"
        )

        assert "macro_name" in result
        assert "ei_gap" in result
        assert "psi" in result
        assert "causal_emergence" in result

        assert result["macro_name"] == "population_count"
        assert isinstance(result["causal_emergence"], bool)

    def test_different_macro_variables(self) -> None:
        """Test computation with different macro variables."""
        rng = np.random.default_rng(42)
        T = 100

        trajectory = {
            "alive_mask": rng.choice([True, False], size=(T, 8), p=[0.8, 0.2]),
            "field": rng.standard_normal((T, 4, 4, 2)),
            "positions": rng.integers(0, 10, size=(T, 8, 2)).astype(np.float64),
            "rewards": rng.integers(0, 2, size=(T, 8)).astype(np.float64),
        }

        for macro_name in [
            "population_count",
            "mean_field_intensity",
            "spatial_dispersion",
            "total_food_collected",
            "field_entropy",
        ]:
            result = compute_causal_emergence_from_trajectory(
                trajectory, macro_name=macro_name
            )
            assert result["macro_name"] == macro_name

    def test_unavailable_macro(self) -> None:
        """Test error when macro variable not available."""
        trajectory = {"field": np.ones((10, 4, 4, 2))}

        with pytest.raises(ValueError, match="not available"):
            compute_causal_emergence_from_trajectory(
                trajectory, macro_name="population_count"
            )

    def test_window_size(self) -> None:
        """Test windowed analysis."""
        rng = np.random.default_rng(42)
        T = 200

        trajectory = {
            "alive_mask": rng.choice([True, False], size=(T, 8), p=[0.8, 0.2]),
        }

        result = compute_causal_emergence_from_trajectory(
            trajectory, macro_name="population_count", window_size=50
        )

        # Should complete without error
        assert "ei_gap" in result

    def test_too_short_trajectory(self) -> None:
        """Test with trajectory too short for analysis."""
        trajectory = {
            "alive_mask": np.ones((2, 4), dtype=bool),
        }

        result = compute_causal_emergence_from_trajectory(
            trajectory, macro_name="population_count"
        )

        assert result["ei_gap"] == 0.0
        assert result["psi"] == 0.0
        assert result["causal_emergence"] is False


class TestComputeWindowedCausalEmergence:
    """Tests for compute_windowed_causal_emergence function."""

    def test_basic_windowed(self) -> None:
        """Test basic windowed analysis."""
        rng = np.random.default_rng(42)
        T = 500

        trajectory = {
            "alive_mask": rng.choice([True, False], size=(T, 8), p=[0.8, 0.2]),
        }

        results = compute_windowed_causal_emergence(
            trajectory,
            macro_name="population_count",
            window_size=100,
            overlap=0.5,
        )

        assert len(results) > 0
        for r in results:
            assert "window_start" in r
            assert "window_end" in r
            assert "ei_gap" in r
            assert "psi" in r

    def test_no_overlap(self) -> None:
        """Test with zero overlap."""
        rng = np.random.default_rng(42)
        T = 400

        trajectory = {
            "alive_mask": rng.choice([True, False], size=(T, 8), p=[0.8, 0.2]),
        }

        results = compute_windowed_causal_emergence(
            trajectory,
            window_size=100,
            overlap=0.0,
        )

        # Should have 4 non-overlapping windows
        assert len(results) == 4
        assert results[0]["window_start"] == 0
        assert results[1]["window_start"] == 100

    def test_trajectory_too_short(self) -> None:
        """Test with trajectory shorter than window."""
        trajectory = {
            "alive_mask": np.ones((50, 4), dtype=bool),
        }

        results = compute_windowed_causal_emergence(
            trajectory, window_size=100
        )

        assert results == []


class TestCausalEmergenceEvent:
    """Tests for CausalEmergenceEvent dataclass."""

    def test_event_creation(self) -> None:
        """Test creating a CausalEmergenceEvent."""
        event = CausalEmergenceEvent(
            step=1000,
            metric_name="ei_gap",
            old_value=0.1,
            new_value=0.5,
            z_score=4.2,
        )

        assert event.step == 1000
        assert event.metric_name == "ei_gap"
        assert event.old_value == 0.1
        assert event.new_value == 0.5
        assert event.z_score == 4.2

    def test_event_str_increase(self) -> None:
        """Test string representation for increasing metric."""
        event = CausalEmergenceEvent(
            step=1000,
            metric_name="ei_gap",
            old_value=0.1,
            new_value=0.5,
            z_score=4.2,
        )

        s = str(event)
        assert "step 1000" in s
        assert "ei_gap" in s
        assert "0.1000" in s
        assert "0.5000" in s
        assert "increase" in s
        assert "z=4.20" in s

    def test_event_str_decrease(self) -> None:
        """Test string representation for decreasing metric."""
        event = CausalEmergenceEvent(
            step=1000,
            metric_name="psi",
            old_value=0.5,
            new_value=0.1,
            z_score=3.5,
        )

        s = str(event)
        assert "decrease" in s


class TestCausalEmergenceTracker:
    """Tests for CausalEmergenceTracker class."""

    def test_initialization(self) -> None:
        """Test tracker initialization."""
        tracker = CausalEmergenceTracker(
            window_size=10, z_threshold=2.5, num_bins=8
        )

        assert tracker.window_size == 10
        assert tracker.z_threshold == 2.5
        assert tracker.num_bins == 8
        assert len(tracker.history["ei_gap"]) == 0
        assert len(tracker.history["psi"]) == 0
        assert len(tracker.steps) == 0
        assert len(tracker.events) == 0
        assert tracker.step_count == 0

    def test_update_basic(self) -> None:
        """Test basic update functionality."""
        tracker = CausalEmergenceTracker()
        rng = np.random.default_rng(42)

        trajectory = {
            "alive_mask": rng.choice([True, False], size=(100, 8), p=[0.8, 0.2]),
        }

        events = tracker.update(trajectory, step=0)

        assert tracker.step_count == 1
        assert len(tracker.steps) == 1
        assert len(tracker.history["ei_gap"]) == 1
        assert len(tracker.history["psi"]) == 1
        # First update shouldn't trigger event
        assert len(events) == 0

    def test_update_multiple(self) -> None:
        """Test multiple updates."""
        tracker = CausalEmergenceTracker(window_size=5)
        rng = np.random.default_rng(42)

        for i in range(10):
            trajectory = {
                "alive_mask": rng.choice(
                    [True, False], size=(100, 8), p=[0.8, 0.2]
                ),
            }
            tracker.update(trajectory, step=i * 1000)

        assert tracker.step_count == 10
        assert len(tracker.history["ei_gap"]) == 10
        assert tracker.steps == list(range(0, 10000, 1000))

    def test_event_detection(self) -> None:
        """Test that significant changes trigger events."""
        tracker = CausalEmergenceTracker(window_size=5, z_threshold=2.0)
        rng = np.random.default_rng(42)

        # Build up stable history
        for i in range(10):
            trajectory = {
                "alive_mask": np.ones((100, 8), dtype=bool),  # Stable population
            }
            tracker.update(trajectory, step=i * 1000)

        # Inject drastic change
        trajectory = {
            "alive_mask": rng.choice([True, False], size=(100, 8), p=[0.3, 0.7]),
        }
        events = tracker.update(trajectory, step=10000)

        # Should detect some events
        assert isinstance(events, list)
        for event in events:
            assert isinstance(event, CausalEmergenceEvent)

    def test_get_metrics_empty(self) -> None:
        """Test get_metrics with no updates."""
        tracker = CausalEmergenceTracker()
        metrics = tracker.get_metrics()

        assert "causal_emergence/num_events" in metrics
        assert metrics["causal_emergence/num_events"] == 0.0

    def test_get_metrics_after_updates(self) -> None:
        """Test get_metrics after some updates."""
        tracker = CausalEmergenceTracker()
        rng = np.random.default_rng(42)

        for i in range(5):
            trajectory = {
                "alive_mask": rng.choice(
                    [True, False], size=(100, 8), p=[0.8, 0.2]
                ),
            }
            tracker.update(trajectory, step=i * 1000)

        metrics = tracker.get_metrics()

        assert "causal_emergence/ei_gap" in metrics
        assert "causal_emergence/psi" in metrics
        assert "causal_emergence/macro_ei" in metrics
        assert "causal_emergence/micro_ei" in metrics
        assert "causal_emergence/emergence_ratio" in metrics
        assert "causal_emergence/num_events" in metrics

    def test_get_summary_empty(self) -> None:
        """Test get_summary with no updates."""
        tracker = CausalEmergenceTracker()
        summary = tracker.get_summary()

        assert summary["total_updates"] == 0
        assert summary["total_events"] == 0
        assert summary["events"] == []

    def test_get_summary_after_updates(self) -> None:
        """Test get_summary after some updates."""
        tracker = CausalEmergenceTracker()
        rng = np.random.default_rng(42)

        for i in range(5):
            trajectory = {
                "alive_mask": rng.choice(
                    [True, False], size=(100, 8), p=[0.8, 0.2]
                ),
            }
            tracker.update(trajectory, step=i * 1000)

        summary = tracker.get_summary()

        assert summary["total_updates"] == 5
        assert "ei_gap_final" in summary
        assert "ei_gap_mean" in summary
        assert "ei_gap_std" in summary
        assert "psi_final" in summary
        assert "positive_ei_gap_fraction" in summary
        assert "positive_psi_fraction" in summary
        assert "emergence_fraction" in summary
        assert "max_ei_gap" in summary
        assert "max_psi" in summary

    def test_emergence_ratio_tracking(self) -> None:
        """Test that emergence ratio is correctly tracked."""
        tracker = CausalEmergenceTracker()
        rng = np.random.default_rng(42)

        for i in range(10):
            trajectory = {
                "alive_mask": rng.choice(
                    [True, False], size=(100, 8), p=[0.8, 0.2]
                ),
            }
            tracker.update(trajectory, step=i * 1000)

        summary = tracker.get_summary()

        # Emergence fraction should be between 0 and 1
        assert 0.0 <= summary["emergence_fraction"] <= 1.0


class TestCausalEmergenceIntegration:
    """Integration tests for causal emergence with realistic data."""

    def test_with_evolving_population(self) -> None:
        """Test with population that evolves over time."""
        rng = np.random.default_rng(42)
        T = 200

        # Simulate population dynamics: start low, grow, then stabilize
        population_prob = np.concatenate([
            np.linspace(0.3, 0.9, T // 2),
            np.ones(T - T // 2) * 0.9,
        ])

        alive_mask = np.array([
            rng.random(8) < p for p in population_prob
        ])

        trajectory = {
            "alive_mask": alive_mask,
            "field": rng.standard_normal((T, 4, 4, 2)),
            "positions": rng.integers(0, 10, size=(T, 8, 2)).astype(np.float64),
        }

        result = compute_causal_emergence_from_trajectory(
            trajectory, macro_name="population_count"
        )

        assert np.isfinite(result["ei_gap"])
        assert np.isfinite(result["psi"])

    def test_windowed_analysis_over_training(self) -> None:
        """Test windowed analysis simulating a training run."""
        rng = np.random.default_rng(42)
        T = 1000

        trajectory = {
            "alive_mask": rng.choice([True, False], size=(T, 8), p=[0.8, 0.2]),
            "field": rng.standard_normal((T, 4, 4, 2)),
            "rewards": rng.integers(0, 2, size=(T, 8)).astype(np.float64),
        }

        # Simulate 1M-step windows at smaller scale (100-step windows)
        results = compute_windowed_causal_emergence(
            trajectory,
            macro_name="population_count",
            window_size=100,
            overlap=0.5,  # 50% overlap as per PRD
        )

        assert len(results) > 5
        # Check temporal ordering
        for i in range(len(results) - 1):
            assert results[i]["window_start"] < results[i + 1]["window_start"]

    def test_multiple_macro_variables_comparison(self) -> None:
        """Test comparing causal emergence across different macro variables."""
        rng = np.random.default_rng(42)
        T = 100

        trajectory = {
            "alive_mask": rng.choice([True, False], size=(T, 8), p=[0.8, 0.2]),
            "field": rng.standard_normal((T, 4, 4, 2)),
            "positions": rng.integers(0, 10, size=(T, 8, 2)).astype(np.float64),
            "rewards": rng.integers(0, 2, size=(T, 8)).astype(np.float64),
        }

        macro_vars = [
            "population_count",
            "mean_field_intensity",
            "spatial_dispersion",
            "total_food_collected",
            "field_entropy",
        ]

        results = {}
        for macro in macro_vars:
            results[macro] = compute_causal_emergence_from_trajectory(
                trajectory, macro_name=macro
            )

        # All should complete without error
        for macro in macro_vars:
            assert np.isfinite(results[macro]["ei_gap"])
            assert np.isfinite(results[macro]["psi"])

    def test_tracker_across_training(self) -> None:
        """Test tracker usage pattern during training."""
        tracker = CausalEmergenceTracker(window_size=5, z_threshold=2.0)
        rng = np.random.default_rng(42)

        all_events = []
        for i in range(20):
            trajectory = {
                "alive_mask": rng.choice(
                    [True, False], size=(100, 8), p=[0.8, 0.2]
                ),
            }
            events = tracker.update(trajectory, step=i * 10000)
            all_events.extend(events)

            # Log metrics at each step
            metrics = tracker.get_metrics()
            assert "causal_emergence/ei_gap" in metrics

        summary = tracker.get_summary()
        assert summary["total_updates"] == 20
        assert len(summary["events"]) == len(all_events)
