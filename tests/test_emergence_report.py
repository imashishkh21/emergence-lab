"""Tests for emergence_report.py — unified emergence metrics module."""

import json
from datetime import datetime

import numpy as np
import pytest

from src.analysis.emergence_report import (
    EmergenceReport,
    MetricResult,
    _create_zero_metric,
    compute_all_emergence_metrics,
    compute_windowed_metrics,
    print_emergence_report,
    report_from_json,
    report_to_json,
)


class TestMetricResult:
    """Tests for MetricResult dataclass."""

    def test_create_basic(self):
        """Test basic creation with just value."""
        result = MetricResult(value=0.5)
        assert result.value == 0.5
        assert result.p_value is None
        assert result.significant is False
        assert result.ci_lower is None
        assert result.ci_upper is None

    def test_create_with_significance(self):
        """Test creation with significance info."""
        result = MetricResult(
            value=-0.3,
            p_value=0.01,
            significant=True,
            ci_lower=-0.5,
            ci_upper=-0.1,
            surrogate_mean=0.0,
            surrogate_std=0.1,
        )
        assert result.value == -0.3
        assert result.p_value == 0.01
        assert result.significant is True
        assert result.ci_lower == -0.5
        assert result.ci_upper == -0.1
        assert result.surrogate_mean == 0.0
        assert result.surrogate_std == 0.1

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = MetricResult(value=1.5, p_value=0.05, significant=True)
        d = result.to_dict()
        assert d["value"] == 1.5
        assert d["p_value"] == 0.05
        assert d["significant"] is True
        assert "ci_lower" in d
        assert "ci_upper" in d

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "value": 2.0,
            "p_value": 0.03,
            "significant": True,
            "ci_lower": 1.5,
            "ci_upper": 2.5,
            "surrogate_mean": 1.0,
            "surrogate_std": 0.2,
        }
        result = MetricResult.from_dict(d)
        assert result.value == 2.0
        assert result.p_value == 0.03
        assert result.significant is True

    def test_roundtrip(self):
        """Test dict roundtrip preserves values."""
        original = MetricResult(
            value=-0.5,
            p_value=0.001,
            significant=True,
            ci_lower=-0.7,
            ci_upper=-0.3,
        )
        d = original.to_dict()
        restored = MetricResult.from_dict(d)
        assert restored.value == original.value
        assert restored.p_value == original.p_value
        assert restored.significant == original.significant


class TestEmergenceReport:
    """Tests for EmergenceReport dataclass."""

    def _create_sample_report(self) -> EmergenceReport:
        """Create a sample report for testing."""
        return EmergenceReport(
            o_information=MetricResult(value=-0.3, p_value=0.02, significant=True),
            median_pid_synergy=MetricResult(value=0.15),
            causal_emergence_ei_gap=MetricResult(value=0.5),
            rosas_psi=MetricResult(value=0.2),
            mean_transfer_entropy=MetricResult(value=0.1),
            specialization_score=MetricResult(value=0.6),
            division_of_labor=MetricResult(value=0.45),
            checkpoint_path="/path/to/checkpoint.pkl",
            n_agents=8,
            n_timesteps=1000,
            timestamp="2026-02-03T12:00:00",
        )

    def test_create_basic(self):
        """Test basic report creation."""
        report = self._create_sample_report()
        assert report.o_information.value == -0.3
        assert report.median_pid_synergy.value == 0.15
        assert report.n_agents == 8
        assert report.n_timesteps == 1000

    def test_to_dict(self):
        """Test conversion to dictionary."""
        report = self._create_sample_report()
        d = report.to_dict()

        assert "o_information" in d
        assert "median_pid_synergy" in d
        assert "checkpoint_path" in d
        assert d["n_agents"] == 8
        assert isinstance(d["o_information"], dict)

    def test_from_dict(self):
        """Test creation from dictionary."""
        report = self._create_sample_report()
        d = report.to_dict()
        restored = EmergenceReport.from_dict(d)

        assert restored.o_information.value == report.o_information.value
        assert restored.n_agents == report.n_agents
        assert restored.checkpoint_path == report.checkpoint_path

    def test_roundtrip(self):
        """Test dict roundtrip."""
        original = self._create_sample_report()
        original.windowed_results = [
            {"window_start": 0, "window_end": 500, "metrics": {"o_info": -0.2}}
        ]
        d = original.to_dict()
        restored = EmergenceReport.from_dict(d)

        assert restored.o_information.value == original.o_information.value
        assert len(restored.windowed_results) == 1

    def test_windowed_results_default(self):
        """Test windowed_results defaults to empty list."""
        report = self._create_sample_report()
        assert report.windowed_results == []


class TestCreateZeroMetric:
    """Tests for _create_zero_metric helper."""

    def test_creates_zero_value(self):
        """Test creates metric with zero value."""
        metric = _create_zero_metric()
        assert metric.value == 0.0

    def test_no_significance(self):
        """Test creates metric with no significance info."""
        metric = _create_zero_metric()
        assert metric.p_value is None
        assert metric.significant is False
        assert metric.ci_lower is None
        assert metric.ci_upper is None


def _create_synthetic_trajectory(
    n_timesteps: int = 100,
    n_agents: int = 8,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Create synthetic trajectory for testing."""
    rng = np.random.default_rng(seed)

    return {
        "actions": rng.integers(0, 6, size=(n_timesteps, n_agents)),
        "positions": rng.integers(0, 20, size=(n_timesteps, n_agents, 2)),
        "rewards": rng.random((n_timesteps, n_agents)).astype(np.float32),
        "alive_mask": np.ones((n_timesteps, n_agents), dtype=bool),
        "energy": rng.random((n_timesteps, n_agents)).astype(np.float32) * 100,
        "field": rng.random((n_timesteps, 20, 20, 4)).astype(np.float32),
    }


class TestComputeAllEmergenceMetrics:
    """Tests for compute_all_emergence_metrics function."""

    def test_returns_report(self):
        """Test returns EmergenceReport instance."""
        trajectory = _create_synthetic_trajectory()
        report = compute_all_emergence_metrics(
            trajectory, run_surrogates=False
        )
        assert isinstance(report, EmergenceReport)

    def test_all_metrics_present(self):
        """Test all 7 metrics are computed."""
        trajectory = _create_synthetic_trajectory()
        report = compute_all_emergence_metrics(
            trajectory, run_surrogates=False
        )

        # Check all metrics are MetricResult instances
        assert isinstance(report.o_information, MetricResult)
        assert isinstance(report.median_pid_synergy, MetricResult)
        assert isinstance(report.causal_emergence_ei_gap, MetricResult)
        assert isinstance(report.rosas_psi, MetricResult)
        assert isinstance(report.mean_transfer_entropy, MetricResult)
        assert isinstance(report.specialization_score, MetricResult)
        assert isinstance(report.division_of_labor, MetricResult)

    def test_metadata_populated(self):
        """Test metadata is populated."""
        trajectory = _create_synthetic_trajectory(n_timesteps=50, n_agents=4)
        report = compute_all_emergence_metrics(
            trajectory,
            run_surrogates=False,
            checkpoint_path="/test/path.pkl",
        )

        assert report.n_agents == 4
        assert report.n_timesteps == 50
        assert report.checkpoint_path == "/test/path.pkl"
        assert report.timestamp  # Non-empty timestamp

    def test_empty_trajectory(self):
        """Test handles empty trajectory gracefully."""
        trajectory = {}
        report = compute_all_emergence_metrics(
            trajectory, run_surrogates=False
        )
        assert report.n_agents == 0
        assert report.n_timesteps == 0

    def test_single_agent(self):
        """Test handles single agent case."""
        trajectory = _create_synthetic_trajectory(n_agents=1)
        report = compute_all_emergence_metrics(
            trajectory, run_surrogates=False
        )
        # Should not crash, metrics may be zero
        assert isinstance(report, EmergenceReport)

    def test_short_trajectory(self):
        """Test handles very short trajectory."""
        trajectory = _create_synthetic_trajectory(n_timesteps=5)
        report = compute_all_emergence_metrics(
            trajectory, run_surrogates=False
        )
        assert isinstance(report, EmergenceReport)

    def test_with_surrogates(self):
        """Test with surrogate testing enabled."""
        trajectory = _create_synthetic_trajectory(n_timesteps=50)
        report = compute_all_emergence_metrics(
            trajectory,
            run_surrogates=True,
            n_surrogates=10,  # Small number for speed
        )

        # O-information should have surrogate info
        # (if hoi library is available)
        assert isinstance(report.o_information, MetricResult)


class TestComputeWindowedMetrics:
    """Tests for compute_windowed_metrics function."""

    def test_returns_list(self):
        """Test returns list of window results."""
        trajectory = _create_synthetic_trajectory(n_timesteps=200)
        results = compute_windowed_metrics(
            trajectory,
            window_size=100,
            overlap=0.5,
            run_surrogates=False,
        )
        assert isinstance(results, list)

    def test_correct_window_count(self):
        """Test correct number of windows."""
        trajectory = _create_synthetic_trajectory(n_timesteps=200)

        # Window size 100, overlap 0.5 -> step 50
        # Windows: 0-100, 50-150, 100-200 = 3 windows
        results = compute_windowed_metrics(
            trajectory,
            window_size=100,
            overlap=0.5,
            run_surrogates=False,
        )
        assert len(results) == 3

    def test_window_boundaries(self):
        """Test window start/end are correct."""
        trajectory = _create_synthetic_trajectory(n_timesteps=200)
        results = compute_windowed_metrics(
            trajectory,
            window_size=100,
            overlap=0.5,
            run_surrogates=False,
        )

        assert results[0]["window_start"] == 0
        assert results[0]["window_end"] == 100
        assert results[1]["window_start"] == 50
        assert results[1]["window_end"] == 150

    def test_metrics_in_each_window(self):
        """Test all metrics present in each window."""
        trajectory = _create_synthetic_trajectory(n_timesteps=200)
        results = compute_windowed_metrics(
            trajectory,
            window_size=100,
            overlap=0.5,
            run_surrogates=False,
        )

        for window in results:
            assert "metrics" in window
            metrics = window["metrics"]
            assert "o_information" in metrics
            assert "mean_transfer_entropy" in metrics
            assert "specialization_score" in metrics

    def test_trajectory_too_short(self):
        """Test returns empty list if trajectory too short."""
        trajectory = _create_synthetic_trajectory(n_timesteps=50)
        results = compute_windowed_metrics(
            trajectory,
            window_size=100,
            overlap=0.5,
        )
        assert results == []

    def test_no_overlap(self):
        """Test with no overlap between windows."""
        trajectory = _create_synthetic_trajectory(n_timesteps=200)
        results = compute_windowed_metrics(
            trajectory,
            window_size=100,
            overlap=0.0,
            run_surrogates=False,
        )
        # 200 / 100 = 2 windows with no overlap
        assert len(results) == 2
        assert results[0]["window_start"] == 0
        assert results[1]["window_start"] == 100


class TestJSONSerialization:
    """Tests for JSON serialization functions."""

    def _create_sample_report(self) -> EmergenceReport:
        """Create sample report."""
        return EmergenceReport(
            o_information=MetricResult(value=-0.3, p_value=0.02, significant=True),
            median_pid_synergy=MetricResult(value=0.15),
            causal_emergence_ei_gap=MetricResult(value=0.5),
            rosas_psi=MetricResult(value=0.2),
            mean_transfer_entropy=MetricResult(value=0.1),
            specialization_score=MetricResult(value=0.6),
            division_of_labor=MetricResult(value=0.45),
            checkpoint_path="/path/to/checkpoint.pkl",
            n_agents=8,
            n_timesteps=1000,
            timestamp="2026-02-03T12:00:00",
        )

    def test_report_to_json_valid(self):
        """Test report_to_json produces valid JSON."""
        report = self._create_sample_report()
        json_str = report_to_json(report)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_report_from_json(self):
        """Test report_from_json restores report."""
        report = self._create_sample_report()
        json_str = report_to_json(report)
        restored = report_from_json(json_str)

        assert restored.o_information.value == report.o_information.value
        assert restored.n_agents == report.n_agents

    def test_json_roundtrip(self):
        """Test full JSON roundtrip."""
        original = self._create_sample_report()
        original.windowed_results = [
            {
                "window_start": 0,
                "window_end": 500,
                "metrics": {"o_information": -0.2, "mean_te": 0.1},
            }
        ]

        json_str = report_to_json(original)
        restored = report_from_json(json_str)

        assert restored.o_information.value == original.o_information.value
        assert restored.o_information.p_value == original.o_information.p_value
        assert restored.n_agents == original.n_agents
        assert len(restored.windowed_results) == 1

    def test_json_pretty_printed(self):
        """Test JSON is pretty-printed with indentation."""
        report = self._create_sample_report()
        json_str = report_to_json(report)

        # Should have indentation (newlines and spaces)
        assert "\n" in json_str
        assert "  " in json_str  # Two-space indent


class TestPrintEmergenceReport:
    """Tests for print_emergence_report function."""

    def test_prints_without_error(self, capsys):
        """Test printing doesn't raise errors."""
        report = EmergenceReport(
            o_information=MetricResult(value=-0.3, p_value=0.02, significant=True),
            median_pid_synergy=MetricResult(value=0.15),
            causal_emergence_ei_gap=MetricResult(value=0.5),
            rosas_psi=MetricResult(value=0.2),
            mean_transfer_entropy=MetricResult(value=0.1),
            specialization_score=MetricResult(value=0.6),
            division_of_labor=MetricResult(value=0.45),
            n_agents=8,
            n_timesteps=1000,
        )

        print_emergence_report(report)
        captured = capsys.readouterr()

        assert "EMERGENCE METRICS REPORT" in captured.out
        assert "O-Information" in captured.out
        assert "SIGNIFICANT" in captured.out

    def test_prints_windowed_info(self, capsys):
        """Test prints windowed analysis info."""
        report = EmergenceReport(
            o_information=MetricResult(value=-0.3),
            median_pid_synergy=MetricResult(value=0.15),
            causal_emergence_ei_gap=MetricResult(value=0.5),
            rosas_psi=MetricResult(value=0.2),
            mean_transfer_entropy=MetricResult(value=0.1),
            specialization_score=MetricResult(value=0.6),
            division_of_labor=MetricResult(value=0.45),
            windowed_results=[
                {"window_start": 0, "window_end": 500, "metrics": {}},
                {"window_start": 250, "window_end": 750, "metrics": {}},
            ],
        )

        print_emergence_report(report)
        captured = capsys.readouterr()

        assert "Windowed Analysis" in captured.out
        assert "2 windows" in captured.out


class TestIntegration:
    """Integration tests for full pipeline."""

    def test_full_pipeline(self):
        """Test full pipeline: compute -> serialize -> deserialize."""
        # Create trajectory
        trajectory = _create_synthetic_trajectory(n_timesteps=100)

        # Compute metrics
        report = compute_all_emergence_metrics(
            trajectory,
            run_surrogates=False,
            checkpoint_path="/test/checkpoint.pkl",
        )

        # Serialize and deserialize
        json_str = report_to_json(report)
        restored = report_from_json(json_str)

        # Verify restoration
        assert restored.n_agents == report.n_agents
        assert restored.n_timesteps == report.n_timesteps
        assert restored.o_information.value == report.o_information.value

    def test_pipeline_with_windowed(self):
        """Test pipeline with windowed analysis."""
        trajectory = _create_synthetic_trajectory(n_timesteps=200)

        # Compute base metrics
        report = compute_all_emergence_metrics(
            trajectory, run_surrogates=False
        )

        # Add windowed results
        windowed = compute_windowed_metrics(
            trajectory,
            window_size=100,
            overlap=0.5,
            run_surrogates=False,
        )
        report.windowed_results = windowed

        # Serialize and deserialize
        json_str = report_to_json(report)
        restored = report_from_json(json_str)

        assert len(restored.windowed_results) == len(report.windowed_results)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_constant_data(self):
        """Test handles constant data gracefully."""
        trajectory = {
            "actions": np.zeros((100, 8), dtype=np.int32),
            "rewards": np.zeros((100, 8), dtype=np.float32),
            "positions": np.zeros((100, 8, 2), dtype=np.int32),
            "alive_mask": np.ones((100, 8), dtype=bool),
            "energy": np.ones((100, 8), dtype=np.float32) * 100,
        }

        report = compute_all_emergence_metrics(
            trajectory, run_surrogates=False
        )
        assert isinstance(report, EmergenceReport)

    def test_partial_trajectory(self):
        """Test handles trajectory with missing keys."""
        trajectory = {
            "actions": np.random.randint(0, 6, size=(100, 8)),
            "rewards": np.random.random((100, 8)).astype(np.float32),
            # Missing: positions, alive_mask, energy
        }

        report = compute_all_emergence_metrics(
            trajectory, run_surrogates=False
        )
        assert isinstance(report, EmergenceReport)

    def test_nan_in_data(self):
        """Test handles NaN in data gracefully."""
        trajectory = _create_synthetic_trajectory()
        trajectory["rewards"][0, 0] = np.nan

        report = compute_all_emergence_metrics(
            trajectory, run_surrogates=False
        )
        assert isinstance(report, EmergenceReport)

    def test_inf_in_data(self):
        """Test handles infinity in data gracefully."""
        trajectory = _create_synthetic_trajectory()
        trajectory["rewards"][0, 0] = np.inf

        report = compute_all_emergence_metrics(
            trajectory, run_surrogates=False
        )
        assert isinstance(report, EmergenceReport)

    def test_two_agents(self):
        """Test with exactly 2 agents (edge case for O-info)."""
        trajectory = _create_synthetic_trajectory(n_agents=2)
        report = compute_all_emergence_metrics(
            trajectory, run_surrogates=False
        )
        # O-info requires 3+ agents, should return 0
        assert report.o_information.value == 0.0


class TestWindowEdgeCases:
    """Tests for windowed metrics edge cases."""

    def test_window_larger_than_trajectory(self):
        """Test window size larger than trajectory length."""
        trajectory = _create_synthetic_trajectory(n_timesteps=100)
        windows = compute_windowed_metrics(
            trajectory,
            window_size=200,  # Larger than trajectory
            overlap=0.5,
            run_surrogates=False,
        )
        # Should return empty list (no complete windows)
        assert len(windows) == 0

    def test_window_equals_trajectory_length(self):
        """Test window exactly equals trajectory length."""
        trajectory = _create_synthetic_trajectory(n_timesteps=100)
        windows = compute_windowed_metrics(
            trajectory,
            window_size=100,  # Exact match
            overlap=0.5,
            run_surrogates=False,
        )
        # Should return exactly one window
        assert len(windows) == 1
        assert windows[0]["window_start"] == 0
        assert windows[0]["window_end"] == 100

    def test_zero_overlap(self):
        """Test windowed metrics with 0% overlap."""
        trajectory = _create_synthetic_trajectory(n_timesteps=200)
        windows = compute_windowed_metrics(
            trajectory,
            window_size=100,
            overlap=0.0,  # No overlap
            run_surrogates=False,
        )
        # Should get 2 non-overlapping windows
        assert len(windows) == 2
        assert windows[0]["window_start"] == 0
        assert windows[0]["window_end"] == 100
        assert windows[1]["window_start"] == 100
        assert windows[1]["window_end"] == 200

    def test_high_overlap(self):
        """Test windowed metrics with 90% overlap."""
        trajectory = _create_synthetic_trajectory(n_timesteps=150)
        windows = compute_windowed_metrics(
            trajectory,
            window_size=100,
            overlap=0.9,  # 90% overlap, step of 10
            run_surrogates=False,
        )
        # Step = 100 * (1 - 0.9) = 10, should get 6 windows
        # (0-100, 10-110, 20-120, 30-130, 40-140, 50-150)
        assert len(windows) == 6

    def test_window_result_structure(self):
        """Test windowed result contains expected keys."""
        trajectory = _create_synthetic_trajectory(n_timesteps=200)
        windows = compute_windowed_metrics(
            trajectory,
            window_size=100,
            overlap=0.5,
            run_surrogates=False,
        )
        assert len(windows) > 0
        window = windows[0]
        # Check required keys
        assert "window_start" in window
        assert "window_end" in window
        assert "metrics" in window
        # Check metrics structure
        metrics = window["metrics"]
        assert "o_information" in metrics
        assert "mean_transfer_entropy" in metrics
        assert "specialization_score" in metrics

    def test_small_window_size(self):
        """Test with very small window size."""
        trajectory = _create_synthetic_trajectory(n_timesteps=100)
        windows = compute_windowed_metrics(
            trajectory,
            window_size=20,  # Small window
            overlap=0.5,
            run_surrogates=False,
        )
        # With step=10, should get ~9 windows
        assert len(windows) >= 5