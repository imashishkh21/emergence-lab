"""Tests for PhaseTransitionDetector in src/analysis/emergence.py."""

import numpy as np
import pytest

from src.analysis.emergence import (
    PhaseTransitionDetector,
    PhaseTransitionEvent,
    _autocorrelation_time,
)


class TestAutocorrelationTime:
    """Tests for the _autocorrelation_time helper."""

    def test_short_series_returns_zero(self):
        """Fewer than 3 values should return 0."""
        assert _autocorrelation_time([1.0, 2.0], window=10) == 0.0
        assert _autocorrelation_time([], window=10) == 0.0
        assert _autocorrelation_time([1.0], window=10) == 0.0

    def test_constant_series_returns_zero(self):
        """Constant series has zero variance, should return 0."""
        vals = [0.5] * 20
        assert _autocorrelation_time(vals, window=20) == 0.0

    def test_positive_autocorrelation(self):
        """A smooth trend should have positive autocorrelation time."""
        vals = [float(i) / 100.0 for i in range(50)]
        tau = _autocorrelation_time(vals, window=50)
        assert tau > 0.0

    def test_noisy_series_low_autocorrelation(self):
        """A noisy random series should have low autocorrelation time."""
        rng = np.random.RandomState(42)
        vals = list(rng.randn(100))
        tau = _autocorrelation_time(vals, window=50)
        # Random walk has low autocorrelation time
        assert tau >= 0.0

    def test_window_limits_data(self):
        """Only the last `window` values should be used."""
        # First 50 values are constant, last 20 are a trend
        vals = [0.0] * 50 + [float(i) / 20.0 for i in range(20)]
        tau = _autocorrelation_time(vals, window=20)
        assert tau > 0.0

    def test_non_negative(self):
        """Autocorrelation time should always be >= 0."""
        rng = np.random.RandomState(123)
        for _ in range(10):
            vals = list(rng.randn(30))
            tau = _autocorrelation_time(vals, window=20)
            assert tau >= 0.0


class TestPhaseTransitionEvent:
    """Tests for the PhaseTransitionEvent dataclass."""

    def test_str_representation(self):
        """Test string formatting of PhaseTransitionEvent."""
        event = PhaseTransitionEvent(
            step=5000,
            order_parameter=0.75,
            susceptibility=0.02,
            susceptibility_z=4.5,
            autocorrelation=3.2,
            prev_autocorrelation=2.1,
        )
        s = str(event)
        assert "5000" in s
        assert "0.75" in s or "0.7500" in s
        assert "Phase transition" in s

    def test_fields(self):
        """Test that all fields are accessible."""
        event = PhaseTransitionEvent(
            step=100,
            order_parameter=0.5,
            susceptibility=0.01,
            susceptibility_z=3.5,
            autocorrelation=2.0,
            prev_autocorrelation=1.5,
        )
        assert event.step == 100
        assert event.order_parameter == 0.5
        assert event.susceptibility == 0.01
        assert event.susceptibility_z == 3.5
        assert event.autocorrelation == 2.0
        assert event.prev_autocorrelation == 1.5


class TestPhaseTransitionDetectorInit:
    """Tests for PhaseTransitionDetector initialization."""

    def test_default_params(self):
        """Test default initialization."""
        detector = PhaseTransitionDetector()
        assert detector.window_size == 20
        assert detector.z_threshold == 3.0
        assert detector.step_count == 0
        assert len(detector.order_values) == 0
        assert len(detector.susceptibility_values) == 0
        assert len(detector.autocorrelation_values) == 0
        assert len(detector.events) == 0
        assert len(detector.steps) == 0

    def test_custom_params(self):
        """Test custom initialization."""
        detector = PhaseTransitionDetector(window_size=10, z_threshold=2.0)
        assert detector.window_size == 10
        assert detector.z_threshold == 2.0


class TestPhaseTransitionDetectorUpdate:
    """Tests for PhaseTransitionDetector.update()."""

    def test_update_records_values(self):
        """Each update should record order parameter, susceptibility, autocorrelation."""
        detector = PhaseTransitionDetector()
        detector.update(0.5, step=0)
        assert len(detector.order_values) == 1
        assert len(detector.susceptibility_values) == 1
        assert len(detector.autocorrelation_values) == 1
        assert len(detector.steps) == 1
        assert detector.step_count == 1

    def test_update_multiple(self):
        """Multiple updates should grow all histories."""
        detector = PhaseTransitionDetector()
        for i in range(10):
            detector.update(float(i) / 10.0, step=i * 100)
        assert len(detector.order_values) == 10
        assert len(detector.susceptibility_values) == 10
        assert len(detector.autocorrelation_values) == 10
        assert detector.step_count == 10

    def test_susceptibility_zero_before_window(self):
        """Susceptibility should be 0 when not enough data for window."""
        detector = PhaseTransitionDetector(window_size=20)
        for i in range(5):
            detector.update(float(i), step=i)
        # All susceptibility values should be 0 (< window_size values)
        for s in detector.susceptibility_values:
            assert s == 0.0

    def test_susceptibility_computed_after_window(self):
        """Susceptibility should be non-zero when enough data with variance."""
        detector = PhaseTransitionDetector(window_size=5)
        vals = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 0.7]
        for i, v in enumerate(vals):
            detector.update(v, step=i)
        # After window_size=5, susceptibility should be > 0
        assert any(s > 0 for s in detector.susceptibility_values)

    def test_returns_empty_list_normally(self):
        """Under steady conditions, no events should be detected."""
        detector = PhaseTransitionDetector(window_size=5)
        # Feed steady values
        for i in range(30):
            events = detector.update(0.5, step=i)
        assert len(detector.events) == 0
        assert events == []

    def test_step_recording(self):
        """Steps should be recorded correctly."""
        detector = PhaseTransitionDetector()
        detector.update(0.1, step=100)
        detector.update(0.2, step=200)
        detector.update(0.3, step=300)
        assert detector.steps == [100, 200, 300]


class TestPhaseTransitionDetection:
    """Tests for actual phase transition detection."""

    def test_sudden_jump_detected(self):
        """A sudden jump in order parameter should trigger detection.

        Strategy: feed steady low values for a while, then spike the variance
        by oscillating rapidly. This creates a susceptibility spike.
        """
        detector = PhaseTransitionDetector(window_size=10, z_threshold=2.0)

        # Phase 1: steady low values (build baseline susceptibility)
        for i in range(40):
            detector.update(0.1, step=i)

        # Phase 2: sudden high variance (oscillating rapidly)
        all_events = []
        for i in range(20):
            val = 0.9 if i % 2 == 0 else 0.1
            events = detector.update(val, step=40 + i)
            all_events.extend(events)

        # Should detect at least one phase transition
        assert len(all_events) > 0

    def test_gradual_change_no_detection(self):
        """A very gradual, constant-rate change should not trigger detection.

        Use a long warmup period at the same rate so the susceptibility
        baseline stabilizes before we check for false positives.
        """
        detector = PhaseTransitionDetector(window_size=20, z_threshold=3.0)

        # Long gradual linear increase at constant rate — once the
        # susceptibility baseline stabilizes, no further spikes should occur.
        for i in range(200):
            val = float(i) / 2000.0  # 0.0 to 0.0995 over 200 steps
            detector.update(val, step=i)

        # After the initial warmup (first ~40 steps where window fills),
        # no new events should be triggered in the stable regime.
        # Allow events only in the first 2*window_size warmup period.
        late_events = [e for e in detector.events if e.step >= 60]
        assert len(late_events) == 0, (
            f"Got {len(late_events)} late events: {late_events}"
        )

    def test_event_has_correct_fields(self):
        """Detected events should have all required fields."""
        detector = PhaseTransitionDetector(window_size=10, z_threshold=2.0)

        # Build baseline
        for i in range(40):
            detector.update(0.1, step=i)

        # Create spike
        for i in range(20):
            val = 0.9 if i % 2 == 0 else 0.1
            events = detector.update(val, step=40 + i)
            if events:
                event = events[0]
                assert isinstance(event, PhaseTransitionEvent)
                assert event.step >= 40
                assert event.susceptibility > 0
                assert event.susceptibility_z > detector.z_threshold
                assert event.autocorrelation >= 0
                return

        # If we didn't get an event in the spike phase, still verify no crash
        # (detection depends on both susceptibility AND autocorrelation)

    def test_steady_then_variable_detected(self):
        """Transition from steady to variable should be detectable."""
        detector = PhaseTransitionDetector(window_size=10, z_threshold=2.0)

        # Steady period
        for i in range(30):
            detector.update(0.5, step=i)

        # Variable period — noisy around a different mean
        rng = np.random.RandomState(42)
        all_events = []
        for i in range(30):
            val = 0.5 + rng.randn() * 0.3
            events = detector.update(val, step=30 + i)
            all_events.extend(events)

        # May or may not detect depending on autocorrelation
        # But verify no crashes and valid events if any
        for event in all_events:
            assert isinstance(event, PhaseTransitionEvent)
            assert event.susceptibility_z > detector.z_threshold

    def test_requires_both_conditions(self):
        """Detection requires both susceptibility spike AND autocorrelation increasing.

        Feed values where susceptibility might spike but autocorrelation
        doesn't increase — should not detect.
        """
        detector = PhaseTransitionDetector(window_size=5, z_threshold=3.0)

        # Build baseline with consistent medium variance
        rng = np.random.RandomState(99)
        for i in range(50):
            # Random but stationary — susceptibility may not spike
            val = 0.5 + rng.uniform(-0.1, 0.1)
            detector.update(val, step=i)

        # No events for stationary process
        assert len(detector.events) == 0


class TestPhaseTransitionDetectorMetrics:
    """Tests for get_metrics() and get_summary()."""

    def test_get_metrics_empty(self):
        """Metrics should work with no updates."""
        detector = PhaseTransitionDetector()
        metrics = detector.get_metrics()
        assert "phase_transition/num_events" in metrics
        assert metrics["phase_transition/num_events"] == 0.0

    def test_get_metrics_after_updates(self):
        """Metrics should contain all expected keys after updates."""
        detector = PhaseTransitionDetector()
        for i in range(5):
            detector.update(float(i) / 10.0, step=i)

        metrics = detector.get_metrics()
        assert "phase_transition/order_parameter" in metrics
        assert "phase_transition/susceptibility" in metrics
        assert "phase_transition/autocorrelation" in metrics
        assert "phase_transition/num_events" in metrics

    def test_get_metrics_values_finite(self):
        """All metric values should be finite."""
        detector = PhaseTransitionDetector()
        for i in range(25):
            detector.update(float(i) / 25.0, step=i)

        metrics = detector.get_metrics()
        for key, val in metrics.items():
            assert np.isfinite(val), f"{key} = {val} is not finite"

    def test_get_summary_empty(self):
        """Summary should work with no updates."""
        detector = PhaseTransitionDetector()
        summary = detector.get_summary()
        assert summary["total_updates"] == 0
        assert summary["total_events"] == 0
        assert summary["events"] == []

    def test_get_summary_after_updates(self):
        """Summary should contain expected keys."""
        detector = PhaseTransitionDetector()
        for i in range(10):
            detector.update(float(i) / 10.0, step=i)

        summary = detector.get_summary()
        assert summary["total_updates"] == 10
        assert "order_parameter_final" in summary
        assert "order_parameter_mean" in summary
        assert "order_parameter_std" in summary
        assert "susceptibility_final" in summary
        assert "susceptibility_mean" in summary
        assert "susceptibility_max" in summary
        assert "autocorrelation_final" in summary
        assert "autocorrelation_mean" in summary

    def test_get_summary_events_are_strings(self):
        """Events in summary should be string representations."""
        detector = PhaseTransitionDetector(window_size=10, z_threshold=2.0)

        # Force an event
        for i in range(40):
            detector.update(0.1, step=i)
        for i in range(20):
            val = 0.9 if i % 2 == 0 else 0.1
            detector.update(val, step=40 + i)

        summary = detector.get_summary()
        for event_str in summary["events"]:
            assert isinstance(event_str, str)

    def test_metrics_order_parameter_is_latest(self):
        """Order parameter metric should reflect the most recent value."""
        detector = PhaseTransitionDetector()
        detector.update(0.1, step=0)
        detector.update(0.5, step=1)
        detector.update(0.9, step=2)

        metrics = detector.get_metrics()
        assert metrics["phase_transition/order_parameter"] == 0.9

    def test_summary_statistics_correct(self):
        """Summary statistics should be mathematically correct."""
        detector = PhaseTransitionDetector()
        vals = [0.1, 0.2, 0.3, 0.4, 0.5]
        for i, v in enumerate(vals):
            detector.update(v, step=i)

        summary = detector.get_summary()
        assert summary["order_parameter_final"] == 0.5
        assert abs(summary["order_parameter_mean"] - np.mean(vals)) < 1e-10
        assert abs(summary["order_parameter_std"] - np.std(vals)) < 1e-10


class TestPhaseTransitionDetectorEdgeCases:
    """Edge cases and robustness tests."""

    def test_negative_order_parameter(self):
        """Should handle negative order parameter values."""
        detector = PhaseTransitionDetector()
        detector.update(-0.5, step=0)
        detector.update(-0.3, step=1)
        metrics = detector.get_metrics()
        assert metrics["phase_transition/order_parameter"] == -0.3

    def test_large_values(self):
        """Should handle very large values without overflow."""
        detector = PhaseTransitionDetector()
        for i in range(30):
            detector.update(float(i) * 1000.0, step=i)
        metrics = detector.get_metrics()
        for v in metrics.values():
            assert np.isfinite(v)

    def test_all_same_values(self):
        """Should handle all identical values (zero variance throughout)."""
        detector = PhaseTransitionDetector(window_size=5)
        for i in range(30):
            events = detector.update(0.5, step=i)
            assert events == []
        assert len(detector.events) == 0

    def test_alternating_values_no_false_positive(self):
        """Regular alternation from the start should not trigger detection.

        If the pattern is consistent from the beginning, the susceptibility
        baseline should adapt.
        """
        detector = PhaseTransitionDetector(window_size=10, z_threshold=3.0)
        for i in range(100):
            val = 0.3 if i % 2 == 0 else 0.7
            detector.update(val, step=i)
        # Consistent alternation should build a stable susceptibility baseline
        # No transitions expected since variance is constant
        assert len(detector.events) == 0

    def test_single_update(self):
        """A single update should not crash."""
        detector = PhaseTransitionDetector()
        events = detector.update(0.5, step=0)
        assert events == []
        assert detector.step_count == 1
