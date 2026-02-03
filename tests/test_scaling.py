"""Tests for the scaling analysis module.

Tests superlinear scaling analysis including:
- ScalingResult and ScalingAnalysis dataclasses
- compute_per_agent_efficiency() function
- fit_power_law() function
- aggregate_scaling_results() function
- Power law fits with various scaling types
"""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.analysis.scaling import (
    ScalingAnalysis,
    ScalingResult,
    aggregate_scaling_results,
    compare_scaling_analyses,
    compute_per_agent_efficiency,
    compute_r_squared,
    fit_power_law,
    print_scaling_analysis,
    print_scaling_comparison,
)


class TestScalingResult:
    """Tests for ScalingResult dataclass."""

    def test_create_scaling_result(self):
        """Should be able to create a ScalingResult."""
        result = ScalingResult(
            n_agents=8,
            field_condition="normal",
            total_food=100.0,
            per_agent_food=12.5,
            efficiency=1.25,
            total_reward=150.0,
            per_agent_reward=18.75,
            seed=42,
            episode_foods=[90.0, 100.0, 110.0],
        )

        assert result.n_agents == 8
        assert result.field_condition == "normal"
        assert result.total_food == 100.0
        assert result.per_agent_food == 12.5
        assert result.efficiency == 1.25

    def test_default_values(self):
        """Default values should be sensible."""
        result = ScalingResult(
            n_agents=4,
            field_condition="zeroed",
            total_food=50.0,
            per_agent_food=12.5,
            efficiency=1.0,
        )

        assert result.total_reward == 0.0
        assert result.per_agent_reward == 0.0
        assert result.seed == 0
        assert result.episode_foods == []

    def test_serializable(self):
        """ScalingResult should be picklable."""
        result = ScalingResult(
            n_agents=8,
            field_condition="normal",
            total_food=100.0,
            per_agent_food=12.5,
            efficiency=1.25,
            episode_foods=[90.0, 100.0, 110.0],
        )

        serialized = pickle.dumps(result)
        deserialized = pickle.loads(serialized)

        assert deserialized.n_agents == result.n_agents
        assert deserialized.field_condition == result.field_condition
        assert deserialized.total_food == result.total_food


class TestScalingAnalysis:
    """Tests for ScalingAnalysis dataclass."""

    def test_create_scaling_analysis(self):
        """Should be able to create a ScalingAnalysis."""
        analysis = ScalingAnalysis(
            field_condition="normal",
            n_agents_list=[1, 2, 4, 8],
            solo_food=10.0,
            alpha=1.2,
            alpha_ci_lower=1.1,
            alpha_ci_upper=1.3,
            c=2.3,
            r_squared=0.95,
        )

        assert analysis.field_condition == "normal"
        assert analysis.alpha == 1.2
        assert analysis.is_superlinear() is True
        assert analysis.is_sublinear() is False

    def test_is_superlinear(self):
        """is_superlinear should return True when alpha > 1.0."""
        analysis = ScalingAnalysis(
            field_condition="normal",
            alpha=1.1,
        )
        assert analysis.is_superlinear() is True

        analysis_linear = ScalingAnalysis(
            field_condition="zeroed",
            alpha=1.0,
        )
        assert analysis_linear.is_superlinear() is False

    def test_is_sublinear(self):
        """is_sublinear should return True when alpha < 1.0."""
        analysis = ScalingAnalysis(
            field_condition="no_field",
            alpha=0.9,
        )
        assert analysis.is_sublinear() is True

        analysis_linear = ScalingAnalysis(
            field_condition="zeroed",
            alpha=1.0,
        )
        assert analysis_linear.is_sublinear() is False

    def test_str_representation(self):
        """__str__ should return readable representation."""
        analysis = ScalingAnalysis(
            field_condition="normal",
            n_agents_list=[1, 2, 4],
            alpha=1.15,
            alpha_ci_lower=1.05,
            alpha_ci_upper=1.25,
            r_squared=0.98,
        )

        result_str = str(analysis)

        assert "normal" in result_str
        assert "1.15" in result_str
        assert "superlinear" in result_str

    def test_serializable(self):
        """ScalingAnalysis should be picklable."""
        analysis = ScalingAnalysis(
            field_condition="normal",
            n_agents_list=[1, 2, 4, 8],
            solo_food=10.0,
            mean_foods=[10.0, 22.0, 48.0, 100.0],
            alpha=1.2,
        )

        serialized = pickle.dumps(analysis)
        deserialized = pickle.loads(serialized)

        assert deserialized.field_condition == analysis.field_condition
        assert deserialized.alpha == analysis.alpha


class TestComputePerAgentEfficiency:
    """Tests for compute_per_agent_efficiency function."""

    def test_efficiency_equal_to_solo(self):
        """Efficiency should be 1.0 when agents perform same as solo."""
        # 8 agents, each collecting 10 food on average
        efficiency = compute_per_agent_efficiency(
            total_food=80.0,  # 8 * 10
            n_agents=8,
            solo_food=10.0,
        )
        assert efficiency == 1.0

    def test_superlinear_efficiency(self):
        """Efficiency > 1.0 indicates superlinear scaling."""
        # 8 agents collecting more per-agent than solo
        efficiency = compute_per_agent_efficiency(
            total_food=100.0,  # More than 8 * 10 = 80
            n_agents=8,
            solo_food=10.0,
        )
        assert efficiency == 1.25

    def test_sublinear_efficiency(self):
        """Efficiency < 1.0 indicates sublinear scaling."""
        # 8 agents collecting less per-agent than solo (crowding)
        efficiency = compute_per_agent_efficiency(
            total_food=60.0,  # Less than 8 * 10 = 80
            n_agents=8,
            solo_food=10.0,
        )
        assert efficiency == 0.75

    def test_single_agent_efficiency(self):
        """Single agent should have efficiency = total_food / solo_food."""
        efficiency = compute_per_agent_efficiency(
            total_food=10.0,
            n_agents=1,
            solo_food=10.0,
        )
        assert efficiency == 1.0

    def test_zero_n_agents_raises(self):
        """Zero agents should raise ValueError."""
        with pytest.raises(ValueError, match="n_agents must be positive"):
            compute_per_agent_efficiency(10.0, 0, 10.0)

    def test_negative_n_agents_raises(self):
        """Negative agents should raise ValueError."""
        with pytest.raises(ValueError, match="n_agents must be positive"):
            compute_per_agent_efficiency(10.0, -1, 10.0)

    def test_zero_solo_food_raises(self):
        """Zero solo_food should raise ValueError."""
        with pytest.raises(ValueError, match="solo_food must be positive"):
            compute_per_agent_efficiency(10.0, 8, 0.0)

    def test_negative_solo_food_raises(self):
        """Negative solo_food should raise ValueError."""
        with pytest.raises(ValueError, match="solo_food must be positive"):
            compute_per_agent_efficiency(10.0, 8, -5.0)


class TestFitPowerLaw:
    """Tests for fit_power_law function."""

    def test_linear_scaling(self):
        """Linear scaling (alpha = 1.0) should be detected."""
        n_agents = [1, 2, 4, 8, 16]
        total_food = [10, 20, 40, 80, 160]  # F = 10 * N^1.0

        alpha, (ci_lower, ci_upper), c = fit_power_law(n_agents, total_food, seed=42)

        assert abs(alpha - 1.0) < 0.01  # Should be very close to 1.0
        assert ci_lower <= 1.0 <= ci_upper

    def test_superlinear_scaling(self):
        """Superlinear scaling (alpha > 1.0) should be detected."""
        n_agents = [1, 2, 4, 8, 16]
        # F = 10 * N^1.2 (superlinear)
        total_food = [10 * (n ** 1.2) for n in n_agents]

        alpha, (ci_lower, ci_upper), c = fit_power_law(n_agents, total_food, seed=42)

        assert alpha > 1.0
        assert abs(alpha - 1.2) < 0.05

    def test_sublinear_scaling(self):
        """Sublinear scaling (alpha < 1.0) should be detected."""
        n_agents = [1, 2, 4, 8, 16]
        # F = 10 * N^0.8 (sublinear)
        total_food = [10 * (n ** 0.8) for n in n_agents]

        alpha, (ci_lower, ci_upper), c = fit_power_law(n_agents, total_food, seed=42)

        assert alpha < 1.0
        assert abs(alpha - 0.8) < 0.05

    def test_bootstrap_ci(self):
        """Bootstrap CI should have sensible bounds."""
        n_agents = [1, 2, 4, 8, 16, 32]
        total_food = [10 * (n ** 1.1) for n in n_agents]

        alpha, (ci_lower, ci_upper), c = fit_power_law(
            n_agents, total_food, bootstrap_n=1000, seed=42
        )

        assert ci_lower < alpha < ci_upper
        assert ci_upper - ci_lower < 0.5  # CI shouldn't be too wide

    def test_minimum_two_points(self):
        """Should require at least 2 data points."""
        with pytest.raises(ValueError, match="at least 2 data points"):
            fit_power_law([1], [10])

    def test_mismatched_lengths(self):
        """Should raise if lists have different lengths."""
        with pytest.raises(ValueError, match="same length"):
            fit_power_law([1, 2, 4], [10, 20])

    def test_non_positive_n_raises(self):
        """Should raise if n_agents contains non-positive values."""
        with pytest.raises(ValueError, match="positive"):
            fit_power_law([0, 2, 4], [10, 20, 40])

    def test_non_positive_food_raises(self):
        """Should raise if total_food contains non-positive values."""
        with pytest.raises(ValueError, match="positive"):
            fit_power_law([1, 2, 4], [0, 20, 40])

    def test_reproducible_with_seed(self):
        """Same seed should give same results."""
        n_agents = [1, 2, 4, 8]
        total_food = [10, 25, 60, 150]

        alpha1, ci1, c1 = fit_power_law(n_agents, total_food, seed=42)
        alpha2, ci2, c2 = fit_power_law(n_agents, total_food, seed=42)

        assert alpha1 == alpha2
        assert ci1 == ci2
        assert c1 == c2


class TestComputeRSquared:
    """Tests for compute_r_squared function."""

    def test_perfect_fit(self):
        """Perfect power law fit should have R² = 1.0."""
        n_agents = [1, 2, 4, 8]
        total_food = [10 * (n ** 1.2) for n in n_agents]

        alpha, _, c = fit_power_law(n_agents, total_food, seed=42)
        r_squared = compute_r_squared(n_agents, total_food, alpha, c)

        assert r_squared > 0.99

    def test_noisy_fit(self):
        """Noisy data should have lower R²."""
        n_agents = [1, 2, 4, 8, 16]
        # Add noise to perfect power law
        rng = np.random.default_rng(42)
        total_food = [10 * (n ** 1.1) * (1 + rng.uniform(-0.2, 0.2)) for n in n_agents]

        alpha, _, c = fit_power_law(n_agents, total_food, seed=42)
        r_squared = compute_r_squared(n_agents, total_food, alpha, c)

        assert 0.7 < r_squared < 1.0


class TestAggregateScalingResults:
    """Tests for aggregate_scaling_results function."""

    def test_basic_aggregation(self):
        """Should correctly aggregate results across seeds."""
        results = [
            ScalingResult(n_agents=1, field_condition="normal", total_food=10.0,
                         per_agent_food=10.0, efficiency=1.0),
            ScalingResult(n_agents=1, field_condition="normal", total_food=12.0,
                         per_agent_food=12.0, efficiency=1.0),
            ScalingResult(n_agents=2, field_condition="normal", total_food=25.0,
                         per_agent_food=12.5, efficiency=1.25),
            ScalingResult(n_agents=2, field_condition="normal", total_food=27.0,
                         per_agent_food=13.5, efficiency=1.35),
        ]

        analysis = aggregate_scaling_results(results, [1, 2])

        assert analysis.field_condition == "normal"
        assert len(analysis.mean_foods) == 2
        assert analysis.mean_foods[0] == 11.0  # Mean of 10, 12
        assert analysis.mean_foods[1] == 26.0  # Mean of 25, 27

    def test_computes_solo_baseline(self):
        """Should compute solo baseline from N=1 results."""
        results = [
            ScalingResult(n_agents=1, field_condition="normal", total_food=10.0,
                         per_agent_food=10.0, efficiency=1.0),
            ScalingResult(n_agents=4, field_condition="normal", total_food=50.0,
                         per_agent_food=12.5, efficiency=1.0),
        ]

        analysis = aggregate_scaling_results(results, [1, 4])

        assert analysis.solo_food == 10.0

    def test_estimates_solo_from_smallest_n(self):
        """Should estimate solo from smallest N if N=1 missing."""
        results = [
            ScalingResult(n_agents=2, field_condition="normal", total_food=20.0,
                         per_agent_food=10.0, efficiency=1.0),
            ScalingResult(n_agents=4, field_condition="normal", total_food=50.0,
                         per_agent_food=12.5, efficiency=1.0),
        ]

        analysis = aggregate_scaling_results(results, [2, 4])

        assert analysis.solo_food == 10.0  # 20.0 / 2

    def test_override_solo_food(self):
        """Should use provided solo_food override."""
        results = [
            ScalingResult(n_agents=4, field_condition="normal", total_food=50.0,
                         per_agent_food=12.5, efficiency=1.0),
        ]

        analysis = aggregate_scaling_results(results, [4], solo_food=8.0)

        assert analysis.solo_food == 8.0

    def test_empty_results_raises(self):
        """Should raise on empty results list."""
        with pytest.raises(ValueError, match="empty"):
            aggregate_scaling_results([], [1, 2, 4])

    def test_inconsistent_conditions_raises(self):
        """Should raise if results have different field conditions."""
        results = [
            ScalingResult(n_agents=1, field_condition="normal", total_food=10.0,
                         per_agent_food=10.0, efficiency=1.0),
            ScalingResult(n_agents=1, field_condition="zeroed", total_food=8.0,
                         per_agent_food=8.0, efficiency=1.0),
        ]

        with pytest.raises(ValueError, match="same field_condition"):
            aggregate_scaling_results(results, [1])

    def test_fits_power_law(self):
        """Should fit power law with sufficient data."""
        results = []
        for n in [1, 2, 4, 8]:
            for seed in range(3):
                food = 10 * (n ** 1.15) * (1 + (seed - 1) * 0.05)
                results.append(ScalingResult(
                    n_agents=n,
                    field_condition="normal",
                    total_food=food,
                    per_agent_food=food / n,
                    efficiency=food / (n * 10),
                    seed=seed,
                ))

        analysis = aggregate_scaling_results(results, [1, 2, 4, 8])

        assert analysis.alpha > 1.0  # Should detect superlinear
        assert analysis.r_squared > 0.8


class TestCompareScalingAnalyses:
    """Tests for compare_scaling_analyses function."""

    def test_ranks_by_alpha(self):
        """Should rank conditions by alpha (descending)."""
        analyses = {
            "normal": ScalingAnalysis(field_condition="normal", alpha=1.2),
            "zeroed": ScalingAnalysis(field_condition="zeroed", alpha=1.0),
            "no_field": ScalingAnalysis(field_condition="no_field", alpha=0.8),
        }

        comparison = compare_scaling_analyses(analyses)

        assert len(comparison["rankings"]) == 3
        assert comparison["rankings"][0]["condition"] == "normal"
        assert comparison["rankings"][1]["condition"] == "zeroed"
        assert comparison["rankings"][2]["condition"] == "no_field"

    def test_identifies_superlinear(self):
        """Should identify superlinear conditions."""
        analyses = {
            "normal": ScalingAnalysis(field_condition="normal", alpha=1.2),
            "zeroed": ScalingAnalysis(field_condition="zeroed", alpha=0.95),
        }

        comparison = compare_scaling_analyses(analyses)

        assert "normal" in comparison["superlinear_conditions"]
        assert "zeroed" not in comparison["superlinear_conditions"]

    def test_identifies_best_condition(self):
        """Should identify best condition."""
        analyses = {
            "normal": ScalingAnalysis(field_condition="normal", alpha=1.2),
            "zeroed": ScalingAnalysis(field_condition="zeroed", alpha=1.0),
        }

        comparison = compare_scaling_analyses(analyses)

        assert comparison["best_condition"] == "normal"

    def test_empty_analyses(self):
        """Should handle empty analyses dict."""
        comparison = compare_scaling_analyses({})

        assert comparison["rankings"] == []
        assert comparison["superlinear_conditions"] == []
        assert comparison["best_condition"] is None


class TestPrintFunctions:
    """Tests for print functions (should not raise)."""

    def test_print_scaling_analysis_runs(self, capsys):
        """print_scaling_analysis should run without error."""
        analysis = ScalingAnalysis(
            field_condition="normal",
            n_agents_list=[1, 2, 4, 8],
            solo_food=10.0,
            mean_foods=[10.0, 22.0, 48.0, 100.0],
            std_foods=[1.0, 2.0, 4.0, 8.0],
            mean_efficiencies=[1.0, 1.1, 1.2, 1.25],
            std_efficiencies=[0.1, 0.1, 0.1, 0.1],
            per_agent_foods=[10.0, 11.0, 12.0, 12.5],
            alpha=1.2,
            alpha_ci_lower=1.1,
            alpha_ci_upper=1.3,
            c=2.3,
            r_squared=0.95,
        )

        # Should not raise
        print_scaling_analysis(analysis)

        captured = capsys.readouterr()
        assert "normal" in captured.out
        assert "SUPERLINEAR" in captured.out

    def test_print_scaling_comparison_runs(self, capsys):
        """print_scaling_comparison should run without error."""
        analyses = {
            "normal": ScalingAnalysis(
                field_condition="normal",
                alpha=1.2,
                alpha_ci_lower=1.1,
                alpha_ci_upper=1.3,
                r_squared=0.95,
            ),
            "zeroed": ScalingAnalysis(
                field_condition="zeroed",
                alpha=1.0,
                alpha_ci_lower=0.95,
                alpha_ci_upper=1.05,
                r_squared=0.90,
            ),
        }

        # Should not raise
        print_scaling_comparison(analyses)

        captured = capsys.readouterr()
        assert "SCALING COMPARISON" in captured.out
        assert "normal" in captured.out
        assert "zeroed" in captured.out


class TestIntegration:
    """Integration tests with environment."""

    def test_scaling_result_from_dict_conversion(self):
        """ScalingResult should be convertible to/from dict."""
        from dataclasses import asdict

        result = ScalingResult(
            n_agents=8,
            field_condition="normal",
            total_food=100.0,
            per_agent_food=12.5,
            efficiency=1.25,
            episode_foods=[90.0, 100.0, 110.0],
        )

        result_dict = asdict(result)

        assert isinstance(result_dict, dict)
        assert result_dict["n_agents"] == 8
        assert result_dict["field_condition"] == "normal"
        assert result_dict["total_food"] == 100.0

    def test_scaling_analysis_from_dict_conversion(self):
        """ScalingAnalysis should be convertible to/from dict."""
        from dataclasses import asdict

        analysis = ScalingAnalysis(
            field_condition="normal",
            n_agents_list=[1, 2, 4],
            alpha=1.15,
            r_squared=0.98,
        )

        analysis_dict = asdict(analysis)

        assert isinstance(analysis_dict, dict)
        assert analysis_dict["field_condition"] == "normal"
        assert analysis_dict["alpha"] == 1.15

    def test_save_and_load_results(self):
        """Should be able to save and load results via pickle."""
        from dataclasses import asdict

        analysis = ScalingAnalysis(
            field_condition="normal",
            n_agents_list=[1, 2, 4, 8],
            solo_food=10.0,
            mean_foods=[10.0, 22.0, 48.0, 100.0],
            alpha=1.2,
            alpha_ci_lower=1.1,
            alpha_ci_upper=1.3,
            c=2.3,
            r_squared=0.95,
        )

        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            pickle.dump(asdict(analysis), f)
            temp_path = f.name

        with open(temp_path, 'rb') as f:
            loaded_dict = pickle.load(f)

        Path(temp_path).unlink()

        assert loaded_dict["field_condition"] == "normal"
        assert loaded_dict["alpha"] == 1.2
        assert loaded_dict["n_agents_list"] == [1, 2, 4, 8]


class TestEdgeCases:
    """Edge case tests."""

    def test_two_point_fit(self):
        """Fit should work with exactly 2 points."""
        alpha, (ci_lower, ci_upper), c = fit_power_law([1, 2], [10, 25], seed=42)

        # Should compute alpha (log(25/10) / log(2) ≈ 1.32)
        expected_alpha = np.log(25 / 10) / np.log(2)
        assert abs(alpha - expected_alpha) < 0.01

    def test_large_n_agents(self):
        """Should handle large agent counts."""
        n_agents = [1, 10, 100, 1000]
        total_food = [10 * (n ** 1.1) for n in n_agents]

        alpha, (ci_lower, ci_upper), c = fit_power_law(n_agents, total_food, seed=42)

        assert abs(alpha - 1.1) < 0.05

    def test_aggregate_single_seed(self):
        """Should work with single seed per N."""
        results = [
            ScalingResult(n_agents=1, field_condition="normal", total_food=10.0,
                         per_agent_food=10.0, efficiency=1.0),
            ScalingResult(n_agents=2, field_condition="normal", total_food=22.0,
                         per_agent_food=11.0, efficiency=1.1),
        ]

        analysis = aggregate_scaling_results(results, [1, 2])

        assert analysis.mean_foods[0] == 10.0
        assert analysis.std_foods[0] == 0.0  # Single value has 0 std
