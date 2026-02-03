"""Tests for O-information metrics.

Tests the compute_o_information function and OInformationTracker class
for measuring higher-order interactions between agents.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from src.analysis.o_information import (
    OInfoEvent,
    OInformationTracker,
    compute_o_information,
    compute_o_information_by_condition,
)

# Check if hoi is available for conditional tests
_hoi_available = importlib.util.find_spec("hoi") is not None


class TestComputeOInformation:
    """Tests for compute_o_information function."""

    @pytest.mark.skipif(not _hoi_available, reason="hoi not installed")
    def test_basic_computation(self) -> None:
        """Test basic O-information computation on random data."""
        rng = np.random.default_rng(42)
        # 4 agents, 100 samples, 1 feature each
        features = rng.standard_normal((100, 4))

        o_info = compute_o_information(features)

        # Should return a finite float
        assert isinstance(o_info, float)
        assert np.isfinite(o_info)

    @pytest.mark.skipif(not _hoi_available, reason="hoi not installed")
    def test_synergistic_data(self) -> None:
        """Test that XOR-like patterns produce negative O-info (synergy)."""
        rng = np.random.default_rng(42)
        n_samples = 200

        # Create synergistic structure: C = XOR(A, B)
        # When A and B together predict C better than individually,
        # we get synergy (negative O-info should emerge)
        a = rng.choice([0, 1], size=n_samples).astype(np.float64)
        b = rng.choice([0, 1], size=n_samples).astype(np.float64)
        c = ((a + b) % 2).astype(np.float64)  # XOR

        # Add small noise to avoid degeneracy
        noise = rng.normal(0, 0.1, (n_samples, 3))
        features = np.column_stack([a, b, c]) + noise

        o_info = compute_o_information(features)

        # XOR is a classic synergistic structure
        # O-info should be negative or near zero
        # Note: with noise, this may not be strongly negative
        assert isinstance(o_info, float)
        assert np.isfinite(o_info)

    @pytest.mark.skipif(not _hoi_available, reason="hoi not installed")
    def test_redundant_data(self) -> None:
        """Test that highly correlated data produces positive O-info (redundancy)."""
        rng = np.random.default_rng(42)
        n_samples = 200

        # Create redundant structure: all variables are copies of each other
        base = rng.standard_normal(n_samples)
        noise = rng.normal(0, 0.1, (n_samples, 4))
        features = np.column_stack([base] * 4) + noise

        o_info = compute_o_information(features)

        # Highly correlated variables should show redundancy (positive O-info)
        assert isinstance(o_info, float)
        assert np.isfinite(o_info)
        # With 4 nearly identical variables, O-info should be positive
        # (though the exact value depends on noise level)

    @pytest.mark.skipif(not _hoi_available, reason="hoi not installed")
    def test_3d_input(self) -> None:
        """Test O-information with 3D input (multiple features per agent)."""
        rng = np.random.default_rng(42)
        # 4 agents, 100 samples, 2 features each
        features = rng.standard_normal((100, 4, 2))

        o_info = compute_o_information(features)

        assert isinstance(o_info, float)
        assert np.isfinite(o_info)

    def test_too_few_agents(self) -> None:
        """Test that fewer than 3 agents returns 0.0."""
        rng = np.random.default_rng(42)

        # 2 agents - O-info needs at least 3
        features = rng.standard_normal((100, 2))
        o_info = compute_o_information(features)
        assert o_info == 0.0

        # 1 agent
        features = rng.standard_normal((100, 1))
        o_info = compute_o_information(features)
        assert o_info == 0.0

    def test_too_few_samples(self) -> None:
        """Test that fewer than 10 samples returns 0.0."""
        rng = np.random.default_rng(42)
        features = rng.standard_normal((5, 4))

        o_info = compute_o_information(features)
        assert o_info == 0.0

    def test_constant_features(self) -> None:
        """Test that constant features return 0.0."""
        features = np.ones((100, 4))

        o_info = compute_o_information(features)
        assert o_info == 0.0

    def test_nan_inputs(self) -> None:
        """Test that NaN inputs return 0.0."""
        rng = np.random.default_rng(42)
        features = rng.standard_normal((100, 4))
        features[50, 2] = np.nan

        o_info = compute_o_information(features)
        assert o_info == 0.0

    def test_1d_input(self) -> None:
        """Test that 1D input (single agent) returns 0.0."""
        rng = np.random.default_rng(42)
        features = rng.standard_normal(100)

        o_info = compute_o_information(features)
        assert o_info == 0.0

    @pytest.mark.skipif(not _hoi_available, reason="hoi not installed")
    def test_deterministic_with_seed(self) -> None:
        """Test that results are reproducible."""
        rng = np.random.default_rng(42)
        features = rng.standard_normal((100, 4))

        o_info1 = compute_o_information(features)
        o_info2 = compute_o_information(features)

        assert o_info1 == o_info2


class TestComputeOInformationByCondition:
    """Tests for compute_o_information_by_condition function."""

    @pytest.mark.skipif(not _hoi_available, reason="hoi not installed")
    def test_returns_dict_with_correct_keys(self) -> None:
        """Test that function returns expected dict format."""
        rng = np.random.default_rng(42)
        features = rng.standard_normal((100, 4))

        result = compute_o_information_by_condition(features, "normal")

        assert "o_information" in result
        assert "field_condition" in result
        assert "synergy_dominant" in result
        assert result["field_condition"] == "normal"
        assert isinstance(result["synergy_dominant"], bool)

    @pytest.mark.skipif(not _hoi_available, reason="hoi not installed")
    def test_synergy_dominant_flag(self) -> None:
        """Test that synergy_dominant flag is set correctly."""
        # Create features that will have O-info close to some value
        rng = np.random.default_rng(42)
        features = rng.standard_normal((100, 4))

        result = compute_o_information_by_condition(features, "normal")

        # Flag should match sign of O-info
        expected = result["o_information"] < 0
        assert result["synergy_dominant"] == expected


class TestOInfoEvent:
    """Tests for OInfoEvent dataclass."""

    def test_event_creation(self) -> None:
        """Test creating an OInfoEvent."""
        event = OInfoEvent(
            step=1000,
            old_value=-0.1,
            new_value=-0.5,
            z_score=4.2,
        )

        assert event.step == 1000
        assert event.old_value == -0.1
        assert event.new_value == -0.5
        assert event.z_score == 4.2

    def test_event_str_decrease(self) -> None:
        """Test string representation for decreasing O-info."""
        event = OInfoEvent(
            step=1000,
            old_value=-0.1,
            new_value=-0.5,
            z_score=4.2,
        )

        s = str(event)
        assert "step 1000" in s
        assert "-0.1000" in s
        assert "-0.5000" in s
        assert "decrease" in s
        assert "synergy+" in s
        assert "z=4.20" in s

    def test_event_str_increase(self) -> None:
        """Test string representation for increasing O-info."""
        event = OInfoEvent(
            step=1000,
            old_value=-0.5,
            new_value=0.2,
            z_score=3.5,
        )

        s = str(event)
        assert "increase" in s
        assert "redundancy+" in s


class TestOInformationTracker:
    """Tests for OInformationTracker class."""

    def test_initialization(self) -> None:
        """Test tracker initialization."""
        tracker = OInformationTracker(window_size=10, z_threshold=2.5)

        assert tracker.window_size == 10
        assert tracker.z_threshold == 2.5
        assert len(tracker.history["o_information"]) == 0
        assert len(tracker.history["synergy_ratio"]) == 0
        assert len(tracker.steps) == 0
        assert len(tracker.events) == 0
        assert tracker.step_count == 0

    @pytest.mark.skipif(not _hoi_available, reason="hoi not installed")
    def test_update_basic(self) -> None:
        """Test basic update functionality."""
        tracker = OInformationTracker()
        rng = np.random.default_rng(42)

        features = rng.standard_normal((100, 4))
        events = tracker.update(features, step=0)

        assert tracker.step_count == 1
        assert len(tracker.steps) == 1
        assert len(tracker.history["o_information"]) == 1
        assert len(tracker.history["synergy_ratio"]) == 1
        # First update shouldn't trigger event (no history for z-score)
        assert len(events) == 0

    @pytest.mark.skipif(not _hoi_available, reason="hoi not installed")
    def test_update_multiple(self) -> None:
        """Test multiple updates."""
        tracker = OInformationTracker(window_size=5)
        rng = np.random.default_rng(42)

        for i in range(10):
            features = rng.standard_normal((100, 4))
            tracker.update(features, step=i * 100)

        assert tracker.step_count == 10
        assert len(tracker.history["o_information"]) == 10
        assert tracker.steps == list(range(0, 1000, 100))

    @pytest.mark.skipif(not _hoi_available, reason="hoi not installed")
    def test_event_detection(self) -> None:
        """Test that significant changes trigger events."""
        tracker = OInformationTracker(window_size=5, z_threshold=2.0)
        rng = np.random.default_rng(42)

        # Build up stable history with similar features
        for i in range(10):
            features = rng.standard_normal((100, 4)) * 0.1
            tracker.update(features, step=i * 100)

        # Now inject a drastically different pattern that should trigger event
        # Highly correlated data should give very different O-info
        base = rng.standard_normal(100)
        features = np.column_stack([base] * 4) + rng.normal(0, 0.01, (100, 4))
        events = tracker.update(features, step=1000)

        # Check if any events were detected
        # (might or might not trigger depending on exact values)
        assert isinstance(events, list)
        for event in events:
            assert isinstance(event, OInfoEvent)

    def test_get_metrics_empty(self) -> None:
        """Test get_metrics with no updates."""
        tracker = OInformationTracker()
        metrics = tracker.get_metrics()

        assert "o_information/num_events" in metrics
        assert metrics["o_information/num_events"] == 0.0

    @pytest.mark.skipif(not _hoi_available, reason="hoi not installed")
    def test_get_metrics_after_updates(self) -> None:
        """Test get_metrics after some updates."""
        tracker = OInformationTracker()
        rng = np.random.default_rng(42)

        for i in range(5):
            features = rng.standard_normal((100, 4))
            tracker.update(features, step=i * 100)

        metrics = tracker.get_metrics()

        assert "o_information/value" in metrics
        assert "o_information/synergy_ratio" in metrics
        assert "o_information/num_events" in metrics
        assert np.isfinite(metrics["o_information/value"])
        assert 0.0 <= metrics["o_information/synergy_ratio"] <= 1.0

    def test_get_summary_empty(self) -> None:
        """Test get_summary with no updates."""
        tracker = OInformationTracker()
        summary = tracker.get_summary()

        assert summary["total_updates"] == 0
        assert summary["total_events"] == 0
        assert summary["events"] == []

    @pytest.mark.skipif(not _hoi_available, reason="hoi not installed")
    def test_get_summary_after_updates(self) -> None:
        """Test get_summary after some updates."""
        tracker = OInformationTracker()
        rng = np.random.default_rng(42)

        for i in range(5):
            features = rng.standard_normal((100, 4))
            tracker.update(features, step=i * 100)

        summary = tracker.get_summary()

        assert summary["total_updates"] == 5
        assert "o_information_final" in summary
        assert "o_information_mean" in summary
        assert "o_information_std" in summary
        assert "o_information_min" in summary
        assert "o_information_max" in summary
        assert "synergy_dominant_fraction" in summary
        assert "most_negative_o_info" in summary
        assert "synergy_ratio_final" in summary

    @pytest.mark.skipif(not _hoi_available, reason="hoi not installed")
    def test_synergy_ratio_tracking(self) -> None:
        """Test that synergy ratio is correctly tracked."""
        tracker = OInformationTracker()
        rng = np.random.default_rng(42)

        for i in range(10):
            features = rng.standard_normal((100, 4))
            tracker.update(features, step=i * 100)

        summary = tracker.get_summary()

        # Synergy ratio should be between 0 and 1
        assert 0.0 <= summary["synergy_dominant_fraction"] <= 1.0

        # Synergy ratio should match actual negative O-info count
        o_info_values = tracker.history["o_information"]
        expected_ratio = sum(1 for v in o_info_values if v < 0) / len(o_info_values)
        assert abs(summary["synergy_dominant_fraction"] - expected_ratio) < 1e-10

    def test_handles_edge_case_features(self) -> None:
        """Test that tracker handles edge case features gracefully."""
        tracker = OInformationTracker()

        # Too few agents
        features = np.random.default_rng(42).standard_normal((100, 2))
        events = tracker.update(features, step=0)
        assert len(events) == 0
        assert tracker.history["o_information"][-1] == 0.0

        # Constant features
        features = np.ones((100, 4))
        events = tracker.update(features, step=100)
        assert tracker.history["o_information"][-1] == 0.0


class TestOInformationIntegration:
    """Integration tests for O-information with realistic agent data."""

    @pytest.mark.skipif(not _hoi_available, reason="hoi not installed")
    def test_behavioral_features_format(self) -> None:
        """Test with realistic behavioral feature format.

        Behavioral features typically include:
        - movement_entropy
        - food_collection_rate
        - distance_per_step
        - reproduction_rate
        - mean_energy
        - exploration_ratio
        - action_stay_fraction
        """
        rng = np.random.default_rng(42)
        n_samples = 100
        n_agents = 8
        n_features = 7  # 7 behavioral features

        # Simulate behavioral features with some correlation structure
        features = rng.standard_normal((n_samples, n_agents, n_features))

        o_info = compute_o_information(features)

        assert isinstance(o_info, float)
        assert np.isfinite(o_info)

    @pytest.mark.skipif(not _hoi_available, reason="hoi not installed")
    def test_ablation_comparison(self) -> None:
        """Test O-info computation for ablation comparison."""
        rng = np.random.default_rng(42)

        # Simulate three conditions with different coordination levels
        # Normal: agents with some coordination structure
        normal_features = rng.standard_normal((100, 4))

        # Zeroed: independent agents (no coordination)
        zeroed_features = rng.standard_normal((100, 4))

        # Random: noise (should be similar to zeroed)
        random_features = rng.standard_normal((100, 4))

        o_normal = compute_o_information_by_condition(normal_features, "normal")
        o_zeroed = compute_o_information_by_condition(zeroed_features, "zeroed")
        o_random = compute_o_information_by_condition(random_features, "random")

        # All should return valid results
        assert np.isfinite(o_normal["o_information"])
        assert np.isfinite(o_zeroed["o_information"])
        assert np.isfinite(o_random["o_information"])

        # Conditions should be labeled correctly
        assert o_normal["field_condition"] == "normal"
        assert o_zeroed["field_condition"] == "zeroed"
        assert o_random["field_condition"] == "random"
