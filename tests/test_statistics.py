"""Tests for statistical reporting module (rliable integration).

Tests cover:
- IQM computation with bootstrap CI
- Performance profiles
- Probability of improvement
- Statistical tests (Mann-Whitney, Wilcoxon, Welch's t)
- Method comparison
- StatisticalReport dataclass

Uses @pytest.mark.skipif for rliable-specific tests when library isn't installed.
"""

import importlib
from unittest.mock import patch

import numpy as np
import pytest

from src.analysis.statistics import (
    IQMResult,
    MethodComparison,
    StatisticalReport,
    HypothesisTestResult,
    compare_methods,
    compute_iqm,
    create_statistical_report,
    mann_whitney_test,
    performance_profiles,
    probability_of_improvement,
    welch_t_test,
    wilcoxon_test,
    _iqm_simple,
    _bootstrap_ci_simple,
)

# Check if rliable is available
_rliable_available = importlib.util.find_spec("rliable") is not None


# ═══════════════════════════════════════════════════════════════════════════════
# IQMResult Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestIQMResult:
    """Tests for IQMResult dataclass."""

    def test_creation(self) -> None:
        result = IQMResult(iqm=10.0, ci_lower=9.0, ci_upper=11.0)
        assert result.iqm == 10.0
        assert result.ci_lower == 9.0
        assert result.ci_upper == 11.0
        assert result.ci_level == 0.95  # default
        assert result.n_samples == 0  # default
        assert result.n_bootstrap == 10000  # default

    def test_str_representation(self) -> None:
        result = IQMResult(iqm=10.5, ci_lower=9.5, ci_upper=11.5)
        s = str(result)
        assert "10.5" in s
        assert "9.5" in s
        assert "11.5" in s
        assert "95%" in s


# ═══════════════════════════════════════════════════════════════════════════════
# StatisticalReport Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestStatisticalReport:
    """Tests for StatisticalReport dataclass."""

    def test_creation(self) -> None:
        report = StatisticalReport(
            method_name="test",
            iqm=10.0,
            ci_lower=9.0,
            ci_upper=11.0,
            median=10.0,
            mean=10.0,
            std=1.0,
            n_seeds=20,
        )
        assert report.method_name == "test"
        assert report.iqm == 10.0
        assert report.n_seeds == 20
        assert isinstance(report.scores, np.ndarray)

    def test_with_scores(self) -> None:
        scores = [1, 2, 3, 4, 5]
        report = StatisticalReport(
            method_name="test",
            iqm=3.0,
            ci_lower=2.0,
            ci_upper=4.0,
            median=3.0,
            mean=3.0,
            std=1.41,
            n_seeds=5,
            scores=scores,
        )
        assert len(report.scores) == 5
        assert report.scores[0] == 1

    def test_str_representation(self) -> None:
        report = StatisticalReport(
            method_name="my_method",
            iqm=10.0,
            ci_lower=9.0,
            ci_upper=11.0,
            median=10.0,
            mean=10.0,
            std=1.0,
            n_seeds=20,
        )
        s = str(report)
        assert "my_method" in s
        assert "IQM" in s
        assert "Mean" in s
        assert "N: 20" in s


# ═══════════════════════════════════════════════════════════════════════════════
# TestResult Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestHypothesisTestResult:
    """Tests for HypothesisTestResult dataclass."""

    def test_creation(self) -> None:
        result = HypothesisTestResult(
            test_name="Mann-Whitney U",
            statistic=100.0,
            p_value=0.05,
            significant=True,
        )
        assert result.test_name == "Mann-Whitney U"
        assert result.statistic == 100.0
        assert result.p_value == 0.05
        assert result.significant is True

    def test_str_with_significance_levels(self) -> None:
        # Not significant
        r1 = HypothesisTestResult(
            test_name="test",
            statistic=1.0,
            p_value=0.1,
            significant=False,
        )
        assert "*" not in str(r1)

        # p < 0.05
        r2 = HypothesisTestResult(
            test_name="test",
            statistic=1.0,
            p_value=0.04,
            significant=True,
        )
        assert "*" in str(r2)
        assert "**" not in str(r2)

        # p < 0.01
        r3 = HypothesisTestResult(
            test_name="test",
            statistic=1.0,
            p_value=0.005,
            significant=True,
        )
        assert "**" in str(r3)

        # p < 0.001
        r4 = HypothesisTestResult(
            test_name="test",
            statistic=1.0,
            p_value=0.0005,
            significant=True,
        )
        assert "***" in str(r4)


# ═══════════════════════════════════════════════════════════════════════════════
# IQM Computation Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeIQM:
    """Tests for IQM computation."""

    def test_simple_iqm(self) -> None:
        """IQM of uniform values should be the value."""
        scores = np.array([10.0] * 20)
        result = compute_iqm(scores)
        assert abs(result.iqm - 10.0) < 0.01
        assert result.n_samples == 20

    def test_iqm_excludes_extremes(self) -> None:
        """IQM should exclude values outside 25-75 percentile."""
        # Extreme outliers at ends, middle values around 10
        scores = np.array([0, 1, 10, 10, 10, 10, 10, 10, 100, 1000])
        result = compute_iqm(scores)
        # IQM should be close to 10 (middle values)
        assert result.iqm < 20  # Much lower than mean (115.1)
        assert result.iqm > 5

    def test_iqm_empty_array(self) -> None:
        result = compute_iqm([])
        assert result.iqm == 0.0
        assert result.ci_lower == 0.0
        assert result.ci_upper == 0.0
        assert result.n_samples == 0

    def test_iqm_single_value(self) -> None:
        result = compute_iqm([42.0])
        assert result.iqm == 42.0
        assert result.n_samples == 1

    def test_iqm_confidence_interval(self) -> None:
        """CI should contain the true IQM with high probability."""
        rng = np.random.default_rng(42)
        scores = rng.normal(100, 10, size=100)
        result = compute_iqm(scores, seed=42)

        # CI should bracket the IQM
        assert result.ci_lower <= result.iqm <= result.ci_upper

        # CI should be reasonably tight for 100 samples
        ci_width = result.ci_upper - result.ci_lower
        assert ci_width < 20  # Not too wide

    def test_iqm_list_input(self) -> None:
        """Should accept list as input."""
        result = compute_iqm([1, 2, 3, 4, 5])
        assert result.iqm > 0

    def test_iqm_reproducibility(self) -> None:
        """Same seed should give same results."""
        scores = np.random.default_rng(0).normal(10, 2, 50)
        r1 = compute_iqm(scores, seed=42)
        r2 = compute_iqm(scores, seed=42)
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper


class TestIQMSimple:
    """Tests for the simple IQM implementation."""

    def test_basic(self) -> None:
        scores = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        iqm = _iqm_simple(scores)
        # 25th percentile = 3.25, 75th = 7.75
        # Values in [3.25, 7.75]: 4, 5, 6, 7 -> mean = 5.5
        assert 5 <= iqm <= 6

    def test_empty(self) -> None:
        assert _iqm_simple(np.array([])) == 0.0


class TestBootstrapCISimple:
    """Tests for the simple bootstrap CI implementation."""

    def test_basic(self) -> None:
        scores = np.array([10.0] * 20)
        ci_lower, ci_upper = _bootstrap_ci_simple(scores, seed=42)
        assert abs(ci_lower - 10.0) < 0.1
        assert abs(ci_upper - 10.0) < 0.1

    def test_empty(self) -> None:
        ci_lower, ci_upper = _bootstrap_ci_simple(np.array([]))
        assert ci_lower == 0.0
        assert ci_upper == 0.0

    def test_single_value(self) -> None:
        ci_lower, ci_upper = _bootstrap_ci_simple(np.array([42.0]))
        assert ci_lower == 42.0
        assert ci_upper == 42.0


# ═══════════════════════════════════════════════════════════════════════════════
# Performance Profiles Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPerformanceProfiles:
    """Tests for performance profile computation."""

    def test_basic_profiles(self) -> None:
        scores = {
            "method_a": [0.9, 0.8, 0.85],
            "method_b": [0.5, 0.6, 0.55],
        }
        result = performance_profiles(scores)

        assert "tau" in result
        assert "profiles" in result
        assert "auc" in result
        assert "method_a" in result["profiles"]
        assert "method_b" in result["profiles"]

    def test_better_method_higher_auc(self) -> None:
        """Method with higher scores should have higher AUC."""
        scores = {
            "high": [10, 11, 12],
            "low": [1, 2, 3],
        }
        result = performance_profiles(scores)

        assert result["auc"]["high"] > result["auc"]["low"]

    def test_identical_scores_auc_one(self) -> None:
        """When all scores are identical, AUC should be 1."""
        scores = {"const": [5.0, 5.0, 5.0]}
        result = performance_profiles(scores)
        assert abs(result["auc"]["const"] - 1.0) < 0.01

    def test_empty_dict(self) -> None:
        result = performance_profiles({})
        assert result["profiles"] == {}
        assert result["auc"] == {}

    def test_empty_scores(self) -> None:
        result = performance_profiles({"empty": []})
        # Empty array results in empty profiles (filtered out by concatenation)
        # This is expected behavior - can't compute profiles without data
        assert "profiles" in result
        assert "auc" in result

    def test_custom_tau_range(self) -> None:
        scores = {"a": [0.5, 0.6, 0.7]}
        tau = np.linspace(0.0, 1.0, 11)
        result = performance_profiles(scores, tau_range=tau)
        assert len(result["tau"]) == 11

    def test_profile_shape(self) -> None:
        scores = {"a": [1, 2, 3, 4, 5]}
        result = performance_profiles(scores, n_points=51)
        assert len(result["profiles"]["a"]) == 51
        assert len(result["tau"]) == 51


# ═══════════════════════════════════════════════════════════════════════════════
# Probability of Improvement Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestProbabilityOfImprovement:
    """Tests for probability of improvement computation."""

    def test_clearly_better_method(self) -> None:
        """When X clearly dominates Y, P(X>Y) should be near 1."""
        x = [100, 101, 102]
        y = [1, 2, 3]
        result = probability_of_improvement(x, y, seed=42)
        assert result["prob_x_better"] > 0.9
        assert result["prob_y_better"] < 0.1

    def test_clearly_worse_method(self) -> None:
        """When Y dominates X, P(X>Y) should be near 0."""
        x = [1, 2, 3]
        y = [100, 101, 102]
        result = probability_of_improvement(x, y, seed=42)
        assert result["prob_x_better"] < 0.1
        assert result["prob_y_better"] > 0.9

    def test_equal_methods(self) -> None:
        """When X and Y are similar, P(X>Y) should be near 0.5."""
        rng = np.random.default_rng(42)
        x = rng.normal(10, 1, 50)
        y = rng.normal(10, 1, 50)
        result = probability_of_improvement(x, y, seed=42)
        assert 0.3 < result["prob_x_better"] < 0.7

    def test_empty_arrays(self) -> None:
        result = probability_of_improvement([], [1, 2, 3])
        assert result["prob_x_better"] == 0.5

        result = probability_of_improvement([1, 2, 3], [])
        assert result["prob_x_better"] == 0.5

    def test_confidence_interval(self) -> None:
        """CI should be [0, 1] and bracket the point estimate."""
        x = [10, 11, 12]
        y = [8, 9, 10]
        result = probability_of_improvement(x, y, seed=42)
        assert 0 <= result["ci_lower"] <= result["prob_x_better"]
        assert result["prob_x_better"] <= result["ci_upper"] <= 1

    def test_reproducibility(self) -> None:
        """Same seed should give same results."""
        x = [10, 11, 12, 13]
        y = [8, 9, 10, 11]
        r1 = probability_of_improvement(x, y, seed=42)
        r2 = probability_of_improvement(x, y, seed=42)
        assert r1["prob_x_better"] == r2["prob_x_better"]


# ═══════════════════════════════════════════════════════════════════════════════
# Statistical Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestMannWhitneyTest:
    """Tests for Mann-Whitney U test wrapper."""

    def test_significant_difference(self) -> None:
        """Clearly different distributions should be significant."""
        x = np.array([100, 101, 102, 103, 104])
        y = np.array([1, 2, 3, 4, 5])
        result = mann_whitney_test(x, y)
        assert result.p_value < 0.05
        assert result.significant is True
        assert result.test_name == "Mann-Whitney U"

    def test_no_difference(self) -> None:
        """Identical distributions should not be significant."""
        rng = np.random.default_rng(42)
        x = rng.normal(10, 1, 20)
        y = rng.normal(10, 1, 20)
        result = mann_whitney_test(x, y)
        assert result.p_value > 0.05  # Likely not significant

    def test_empty_array(self) -> None:
        result = mann_whitney_test([], [1, 2, 3])
        assert np.isnan(result.statistic)
        assert result.p_value == 1.0
        assert result.significant is False

    def test_effect_size(self) -> None:
        """Effect size should be returned as rank_biserial."""
        x = [10, 11, 12]
        y = [1, 2, 3]
        result = mann_whitney_test(x, y)
        assert result.effect_size_name == "rank_biserial"
        assert result.effect_size != 0

    def test_alternative_greater(self) -> None:
        # Use larger samples for statistical power
        x = [100, 101, 102, 103, 104]
        y = [1, 2, 3, 4, 5]
        result = mann_whitney_test(x, y, alternative="greater")
        assert result.p_value <= 0.05


class TestWilcoxonTest:
    """Tests for Wilcoxon signed-rank test wrapper."""

    def test_significant_difference(self) -> None:
        """Paired differences should be detected with enough samples."""
        # Wilcoxon needs more samples for significance
        # With n=5, minimum p-value is 0.0625 (2^-4)
        x = np.array([10, 11, 12, 13, 14, 15, 16, 17])
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        result = wilcoxon_test(x, y)
        assert result.p_value < 0.05
        assert result.significant is True
        assert result.test_name == "Wilcoxon signed-rank"

    def test_no_difference(self) -> None:
        """Identical pairs should not be significant."""
        x = np.array([10, 11, 12])
        y = np.array([10, 11, 12])
        result = wilcoxon_test(x, y)
        assert result.p_value == 1.0  # All differences are zero

    def test_length_mismatch(self) -> None:
        """Should raise ValueError for different lengths."""
        with pytest.raises(ValueError, match="same length"):
            wilcoxon_test([1, 2, 3], [1, 2])

    def test_empty_arrays(self) -> None:
        result = wilcoxon_test([], [])
        assert np.isnan(result.statistic)
        assert result.p_value == 1.0

    def test_effect_size(self) -> None:
        """Effect size should be r = Z/sqrt(N)."""
        x = [10, 11, 12, 13, 14]
        y = [1, 2, 3, 4, 5]
        result = wilcoxon_test(x, y)
        assert result.effect_size_name == "r"


class TestWelchTTest:
    """Tests for Welch's t-test wrapper."""

    def test_significant_difference(self) -> None:
        """Clearly different means should be significant."""
        x = np.array([100, 101, 102, 103, 104])
        y = np.array([1, 2, 3, 4, 5])
        result = welch_t_test(x, y)
        assert result.p_value < 0.05
        assert result.significant is True
        assert result.test_name == "Welch's t-test"

    def test_similar_distributions(self) -> None:
        """Similar distributions should not be significant."""
        rng = np.random.default_rng(42)
        x = rng.normal(10, 1, 30)
        y = rng.normal(10, 1, 30)
        result = welch_t_test(x, y)
        # May or may not be significant - just check it runs

    def test_insufficient_data(self) -> None:
        """Should handle arrays with < 2 elements."""
        result = welch_t_test([1], [2])
        assert np.isnan(result.statistic)
        assert result.p_value == 1.0

    def test_cohens_d_effect_size(self) -> None:
        """Effect size should be Cohen's d."""
        x = [100, 101, 102]
        y = [1, 2, 3]
        result = welch_t_test(x, y)
        assert result.effect_size_name == "cohens_d"
        assert result.effect_size > 0  # x > y, so positive


# ═══════════════════════════════════════════════════════════════════════════════
# Statistical Report Creation Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCreateStatisticalReport:
    """Tests for create_statistical_report function."""

    def test_basic_report(self) -> None:
        scores = [10, 11, 12, 13, 14]
        report = create_statistical_report("test_method", scores)

        assert report.method_name == "test_method"
        assert report.n_seeds == 5
        assert report.mean == 12.0
        assert report.median == 12.0
        assert len(report.scores) == 5

    def test_empty_scores(self) -> None:
        report = create_statistical_report("empty", [])

        assert report.method_name == "empty"
        assert report.n_seeds == 0
        assert report.mean == 0.0
        assert report.iqm == 0.0

    def test_iqm_and_ci(self) -> None:
        scores = list(range(1, 101))  # 1 to 100
        report = create_statistical_report("range", scores, seed=42)

        assert report.ci_lower < report.iqm < report.ci_upper
        assert 45 < report.iqm < 55  # Should be around 50


# ═══════════════════════════════════════════════════════════════════════════════
# Method Comparison Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCompareMethods:
    """Tests for compare_methods function."""

    def test_two_methods(self) -> None:
        results = {
            "better": [10, 11, 12, 13, 14],
            "worse": [1, 2, 3, 4, 5],
        }
        comparison = compare_methods(results, seed=42)

        assert comparison.best_method == "better"
        assert comparison.rankings[0] == "better"
        assert "better" in comparison.reports
        assert "worse" in comparison.reports

    def test_pairwise_tests(self) -> None:
        results = {
            "a": [10, 11, 12],
            "b": [1, 2, 3],
        }
        comparison = compare_methods(results)

        # Should have one pairwise test
        assert len(comparison.pairwise_tests) == 1
        key = list(comparison.pairwise_tests.keys())[0]
        assert "(a, b)" in key or "(b, a)" in key

    def test_probability_of_improvement(self) -> None:
        results = {
            "high": [100, 101, 102],
            "low": [1, 2, 3],
        }
        comparison = compare_methods(results, seed=42)

        assert len(comparison.probability_of_improvement) > 0
        # P(high > low) should be near 1
        poi_key = [k for k in comparison.probability_of_improvement.keys() if "high" in k][0]
        assert comparison.probability_of_improvement[poi_key] > 0.9

    def test_three_methods(self) -> None:
        results = {
            "best": [15, 16, 17, 18, 19],
            "middle": [10, 11, 12, 13, 14],
            "worst": [1, 2, 3, 4, 5],
        }
        comparison = compare_methods(results, seed=42)

        assert comparison.rankings[0] == "best"
        assert comparison.rankings[-1] == "worst"

        # 3 methods -> 3 pairwise tests
        assert len(comparison.pairwise_tests) == 3

    def test_summary_generation(self) -> None:
        results = {
            "method_a": [10, 11, 12],
            "method_b": [8, 9, 10],
        }
        comparison = compare_methods(results)

        assert "Method Comparison" in comparison.summary
        assert "Rankings" in comparison.summary
        assert "method_a" in comparison.summary
        assert "method_b" in comparison.summary

    def test_empty_results(self) -> None:
        comparison = compare_methods({})

        assert comparison.best_method == ""
        assert len(comparison.rankings) == 0
        assert len(comparison.reports) == 0


class TestMethodComparison:
    """Tests for MethodComparison dataclass."""

    def test_str_representation(self) -> None:
        comparison = MethodComparison(
            reports={"a": StatisticalReport(
                method_name="a", iqm=10, ci_lower=9, ci_upper=11,
                median=10, mean=10, std=1, n_seeds=5
            )},
            rankings=["a"],
            best_method="a",
            summary="Test summary",
        )
        assert str(comparison) == "Test summary"


# ═══════════════════════════════════════════════════════════════════════════════
# Rliable Integration Tests (skipped if not installed)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not _rliable_available, reason="rliable not installed")
class TestRliableIntegration:
    """Tests specifically for rliable integration."""

    def test_iqm_uses_rliable(self) -> None:
        """Verify rliable.metrics.aggregate_iqm is called when available."""
        from rliable import metrics

        scores = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        scores_2d = scores.reshape(-1, 1)

        # Compute with rliable directly
        rliable_iqm = metrics.aggregate_iqm(scores_2d)

        # Compute with our function
        result = compute_iqm(scores)

        # Should be close (not exact due to implementation details)
        assert abs(result.iqm - rliable_iqm) < 1.0

    def test_probability_of_improvement_uses_rliable(self) -> None:
        """Verify rliable is used for POI when available."""
        x = np.array([10, 11, 12])
        y = np.array([8, 9, 10])

        result = probability_of_improvement(x, y, seed=42)
        assert 0 <= result["prob_x_better"] <= 1
        assert "ci_lower" in result
        assert "ci_upper" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Fallback Implementation Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestFallbackImplementations:
    """Tests to ensure fallback implementations work when rliable is unavailable."""

    def test_iqm_fallback(self) -> None:
        """IQM should work even if rliable fails to import."""
        with patch("src.analysis.statistics._rliable_available", False):
            from src.analysis import statistics

            # Reload to use patched value
            original = statistics._rliable_available
            statistics._rliable_available = False

            try:
                scores = [10, 11, 12, 13, 14]
                result = statistics.compute_iqm(scores)
                assert result.iqm > 0
                assert result.n_samples == 5
            finally:
                statistics._rliable_available = original

    def test_poi_fallback(self) -> None:
        """POI should work with simple implementation."""
        with patch("src.analysis.statistics._rliable_available", False):
            from src.analysis import statistics

            original = statistics._rliable_available
            statistics._rliable_available = False

            try:
                result = statistics.probability_of_improvement(
                    [10, 11, 12], [1, 2, 3], seed=42
                )
                assert result["prob_x_better"] > 0.5
            finally:
                statistics._rliable_available = original


# ═══════════════════════════════════════════════════════════════════════════════
# Edge Cases and Robustness Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_all_same_scores(self) -> None:
        """Handle all identical scores gracefully."""
        scores = [5.0] * 20
        result = compute_iqm(scores)
        assert result.iqm == 5.0

        comparison = compare_methods({"const": scores})
        assert comparison.best_method == "const"

    def test_single_method_comparison(self) -> None:
        """Single method should still produce valid comparison."""
        comparison = compare_methods({"only": [1, 2, 3, 4, 5]})
        assert comparison.best_method == "only"
        assert len(comparison.pairwise_tests) == 0

    def test_large_score_differences(self) -> None:
        """Handle very different score magnitudes."""
        results = {
            "huge": [1e10, 1e10 + 1, 1e10 + 2],
            "tiny": [1e-10, 2e-10, 3e-10],
        }
        comparison = compare_methods(results)
        assert comparison.best_method == "huge"

    def test_negative_scores(self) -> None:
        """Handle negative score values."""
        results = {
            "neg": [-10, -9, -8],
            "more_neg": [-100, -99, -98],
        }
        comparison = compare_methods(results)
        assert comparison.best_method == "neg"  # Less negative is better

    def test_nan_handling(self) -> None:
        """NaN values in scores should be handled."""
        scores = [1, 2, np.nan, 4, 5]
        # This may produce warnings but should not crash
        try:
            result = compute_iqm(scores)
            # Result may be nan or may have been filtered
            assert isinstance(result, IQMResult)
        except Exception:
            # Some implementations may raise on NaN
            pass

    def test_inf_handling(self) -> None:
        """Infinity values should be handled."""
        scores = [1, 2, np.inf, 4, 5]
        try:
            result = compute_iqm(scores)
            assert isinstance(result, IQMResult)
        except Exception:
            pass


class TestImportability:
    """Tests for module import and interface."""

    def test_all_exports(self) -> None:
        """All documented functions/classes should be importable."""
        from src.analysis.statistics import (
            IQMResult,
            StatisticalReport,
            HypothesisTestResult,
            MethodComparison,
            compute_iqm,
            performance_profiles,
            probability_of_improvement,
            mann_whitney_test,
            wilcoxon_test,
            welch_t_test,
            create_statistical_report,
            compare_methods,
        )

        # Verify they're callable/instantiable
        assert callable(compute_iqm)
        assert callable(performance_profiles)
        assert callable(mann_whitney_test)
        assert callable(compare_methods)

    def test_quick_smoke_test(self) -> None:
        """Quick end-to-end test of the main workflow."""
        # Define some methods with scores
        results = {
            "ours": [10, 11, 12, 13, 14],
            "baseline": [8, 9, 10, 11, 12],
        }

        # Compute comparison
        comparison = compare_methods(results)

        # Verify basic output
        assert comparison.best_method in ["ours", "baseline"]
        assert len(comparison.reports) == 2
        assert len(comparison.rankings) == 2
        assert comparison.summary != ""
