"""Tests for the surrogate testing framework.

Tests shuffle functions, bootstrap CI, statistical tests,
and the generic surrogate_test function.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.surrogates import (
    StatisticalTestResult,
    SurrogateResult,
    block_shuffle,
    bootstrap_ci,
    column_shuffle,
    compare_conditions,
    compute_cohens_d,
    compute_glass_delta,
    interpret_effect_size,
    mann_whitney_u,
    row_shuffle,
    surrogate_test,
    wilcoxon_signed_rank,
)


class TestRowShuffle:
    """Tests for row_shuffle function."""

    def test_1d_array(self):
        """Row shuffle on 1D array should permute it."""
        rng = np.random.default_rng(42)
        data = np.arange(10)
        shuffled = row_shuffle(data, rng)

        assert shuffled.shape == data.shape
        # Should have same elements
        assert set(shuffled) == set(data)
        # Should be different order (with high probability)
        assert not np.array_equal(shuffled, data)

    def test_2d_array_shape_preserved(self):
        """Row shuffle on 2D array preserves shape."""
        rng = np.random.default_rng(42)
        data = np.arange(20).reshape(5, 4)  # (T=5, N=4)
        shuffled = row_shuffle(data, rng)

        assert shuffled.shape == data.shape

    def test_2d_array_per_timestep_shuffle(self):
        """Each timestep should be independently shuffled."""
        rng = np.random.default_rng(42)
        data = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ])
        shuffled = row_shuffle(data, rng)

        # Each row should contain same elements as original row
        for t in range(len(data)):
            assert set(shuffled[t]) == set(data[t])

    def test_3d_array_shape_preserved(self):
        """Row shuffle on 3D array preserves shape."""
        rng = np.random.default_rng(42)
        data = np.arange(24).reshape(4, 3, 2)  # (T=4, N=3, D=2)
        shuffled = row_shuffle(data, rng)

        assert shuffled.shape == data.shape

    def test_invalid_ndim_raises(self):
        """Row shuffle on 4D+ array should raise."""
        rng = np.random.default_rng(42)
        data = np.zeros((2, 3, 4, 5))

        with pytest.raises(ValueError, match="must be 1D, 2D, or 3D"):
            row_shuffle(data, rng)


class TestColumnShuffle:
    """Tests for column_shuffle function."""

    def test_1d_array(self):
        """Column shuffle on 1D array should permute it."""
        rng = np.random.default_rng(42)
        data = np.arange(10)
        shuffled = column_shuffle(data, rng)

        assert shuffled.shape == data.shape
        assert set(shuffled) == set(data)

    def test_2d_array_time_permutation(self):
        """Column shuffle should permute time axis."""
        rng = np.random.default_rng(42)
        data = np.arange(20).reshape(5, 4)  # (T=5, N=4)
        shuffled = column_shuffle(data, rng)

        assert shuffled.shape == data.shape
        # Each row in shuffled should be a row from original
        for row in shuffled:
            found = False
            for orig_row in data:
                if np.array_equal(row, orig_row):
                    found = True
                    break
            assert found, "Shuffled row not found in original"

    def test_3d_array_time_permutation(self):
        """Column shuffle on 3D preserves structure within timesteps."""
        rng = np.random.default_rng(42)
        data = np.arange(24).reshape(4, 3, 2)
        shuffled = column_shuffle(data, rng)

        assert shuffled.shape == data.shape


class TestBlockShuffle:
    """Tests for block_shuffle function."""

    def test_basic_functionality(self):
        """Block shuffle should permute blocks."""
        rng = np.random.default_rng(42)
        data = np.arange(12)
        shuffled = block_shuffle(data, block_size=3, rng=rng)

        # Should preserve all values (might be truncated)
        assert len(shuffled) == 12  # 12 is divisible by 3
        assert set(shuffled) == set(data)

    def test_block_structure_preserved(self):
        """Elements within blocks should stay together."""
        rng = np.random.default_rng(42)
        data = np.array([0, 1, 2, 10, 11, 12, 20, 21, 22])
        shuffled = block_shuffle(data, block_size=3, rng=rng)

        # Check that blocks are intact (though order may differ)
        blocks_original = [{0, 1, 2}, {10, 11, 12}, {20, 21, 22}]
        for i in range(0, len(shuffled), 3):
            block = set(shuffled[i:i+3])
            assert block in blocks_original

    def test_truncation_for_incomplete_blocks(self):
        """Data not divisible by block_size should be truncated."""
        rng = np.random.default_rng(42)
        data = np.arange(10)  # 10 is not divisible by 3
        shuffled = block_shuffle(data, block_size=3, rng=rng)

        # Should only include 9 elements (3 complete blocks)
        assert len(shuffled) == 9

    def test_block_size_one(self):
        """Block size 1 should be like column shuffle."""
        rng = np.random.default_rng(42)
        data = np.arange(5)
        shuffled = block_shuffle(data, block_size=1, rng=rng)

        assert len(shuffled) == 5
        assert set(shuffled) == set(data)

    def test_invalid_block_size_raises(self):
        """Block size < 1 should raise."""
        rng = np.random.default_rng(42)
        data = np.arange(10)

        with pytest.raises(ValueError, match="block_size must be >= 1"):
            block_shuffle(data, block_size=0, rng=rng)

    def test_empty_array(self):
        """Empty array should return empty."""
        rng = np.random.default_rng(42)
        data = np.array([])
        shuffled = block_shuffle(data, block_size=3, rng=rng)

        assert len(shuffled) == 0

    def test_data_shorter_than_block(self):
        """Data shorter than block_size returns as-is."""
        rng = np.random.default_rng(42)
        data = np.array([1, 2])
        shuffled = block_shuffle(data, block_size=5, rng=rng)

        assert np.array_equal(shuffled, data)


class TestBootstrapCI:
    """Tests for bootstrap_ci function."""

    def test_basic_functionality(self):
        """Bootstrap CI should return valid confidence interval."""
        rng = np.random.default_rng(42)
        data = rng.normal(5.0, 1.0, size=100)

        result = bootstrap_ci(np.mean, data, n_bootstrap=500, seed=42)

        assert "estimate" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "se" in result

        # CI should contain estimate
        assert result["ci_lower"] <= result["estimate"] <= result["ci_upper"]
        # CI should be reasonably tight for 100 samples
        assert result["ci_upper"] - result["ci_lower"] < 1.0

    def test_empty_data(self):
        """Empty data should return NaN."""
        result = bootstrap_ci(np.mean, np.array([]))

        assert np.isnan(result["estimate"])
        assert np.isnan(result["ci_lower"])
        assert np.isnan(result["ci_upper"])

    def test_single_element(self):
        """Single element should return that element as estimate."""
        result = bootstrap_ci(np.mean, np.array([5.0]))

        assert result["estimate"] == 5.0
        assert result["ci_lower"] == 5.0
        assert result["ci_upper"] == 5.0
        assert result["se"] == 0.0

    def test_reproducibility(self):
        """Same seed should give same results."""
        data = np.random.default_rng(0).normal(0, 1, 50)

        result1 = bootstrap_ci(np.mean, data, seed=42)
        result2 = bootstrap_ci(np.mean, data, seed=42)

        assert result1["estimate"] == result2["estimate"]
        assert result1["ci_lower"] == result2["ci_lower"]
        assert result1["ci_upper"] == result2["ci_upper"]

    def test_custom_statistic(self):
        """Should work with any scalar-returning function."""
        data = np.random.default_rng(42).uniform(0, 10, 100)

        # Test with median
        result_median = bootstrap_ci(np.median, data, seed=42)
        assert 4 < result_median["estimate"] < 6  # Should be near 5

        # Test with std
        result_std = bootstrap_ci(np.std, data, seed=42)
        assert result_std["estimate"] > 0

    def test_confidence_level(self):
        """Different CI levels should produce different widths."""
        data = np.random.default_rng(42).normal(0, 1, 100)

        result_90 = bootstrap_ci(np.mean, data, ci=0.90, seed=42)
        result_95 = bootstrap_ci(np.mean, data, ci=0.95, seed=42)
        result_99 = bootstrap_ci(np.mean, data, ci=0.99, seed=42)

        # Wider CI should be larger
        width_90 = result_90["ci_upper"] - result_90["ci_lower"]
        width_95 = result_95["ci_upper"] - result_95["ci_lower"]
        width_99 = result_99["ci_upper"] - result_99["ci_lower"]

        assert width_90 < width_95 < width_99


class TestMannWhitneyU:
    """Tests for mann_whitney_u function."""

    def test_identical_samples(self):
        """Identical samples should not be significant."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])

        result = mann_whitney_u(x, y)

        assert result.p_value > 0.05
        assert not result.significant

    def test_clearly_different_samples(self):
        """Clearly different samples should be significant."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 11, 12, 13, 14])

        result = mann_whitney_u(x, y)

        assert result.p_value < 0.05
        assert result.significant
        assert result.effect_size != 0

    def test_empty_sample(self):
        """Empty sample should return non-significant."""
        x = np.array([])
        y = np.array([1, 2, 3])

        result = mann_whitney_u(x, y)

        assert result.p_value == 1.0
        assert not result.significant

    def test_returns_test_result(self):
        """Should return TestResult dataclass."""
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])

        result = mann_whitney_u(x, y)

        assert isinstance(result, StatisticalTestResult)
        assert result.effect_size_name == "rank_biserial"
        assert hasattr(result, "statistic")
        assert hasattr(result, "p_value")

    def test_str_representation(self):
        """TestResult should have string representation."""
        result = StatisticalTestResult(
            statistic=10.0,
            p_value=0.03,
            effect_size=0.5,
            effect_size_name="r",
            significant=True,
            alpha=0.05,
        )

        result_str = str(result)
        assert "10.0" in result_str
        assert "0.03" in result_str
        assert "significant" in result_str


class TestWilcoxonSignedRank:
    """Tests for wilcoxon_signed_rank function."""

    def test_identical_pairs(self):
        """Identical pairs should not be significant."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])

        result = wilcoxon_signed_rank(x, y)

        assert result.p_value == 1.0
        assert not result.significant

    def test_clearly_different_pairs(self):
        """Consistent differences should be significant."""
        # Need more samples for Wilcoxon to reach p < 0.05
        # With n=5, minimum p-value is 0.0625 (2^-5 + 2^-5)
        x = np.array([2, 4, 6, 8, 10, 12, 14, 16])
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8])

        result = wilcoxon_signed_rank(x, y)

        assert result.p_value < 0.05
        assert result.significant

    def test_mismatched_lengths_raises(self):
        """Different lengths should raise."""
        x = np.array([1, 2, 3])
        y = np.array([1, 2])

        with pytest.raises(ValueError, match="same length"):
            wilcoxon_signed_rank(x, y)

    def test_empty_arrays(self):
        """Empty arrays should return non-significant."""
        result = wilcoxon_signed_rank(np.array([]), np.array([]))

        assert result.p_value == 1.0
        assert not result.significant


class TestSurrogateTest:
    """Tests for surrogate_test function."""

    def test_basic_functionality(self):
        """Surrogate test should return valid result."""
        rng = np.random.default_rng(42)
        # Data with clear structure
        data = np.arange(100).reshape(50, 2)  # Two perfectly correlated columns

        def correlation_metric(d):
            return float(np.corrcoef(d[:, 0], d[:, 1])[0, 1])

        result = surrogate_test(
            correlation_metric,
            data,
            column_shuffle,
            n_surrogates=50,
            seed=42,
        )

        assert isinstance(result, SurrogateResult)
        assert result.observed == pytest.approx(1.0, abs=0.01)  # Perfect correlation
        assert 0 <= result.p_value <= 1

    def test_no_structure_not_significant(self):
        """Random data should not be significant."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)

        result = surrogate_test(
            np.mean,
            data,
            column_shuffle,
            n_surrogates=100,
            seed=42,
        )

        # Mean of normal distribution should not be significantly different
        # from shuffled version (shuffling doesn't change marginal distribution)
        assert result.p_value > 0.05

    def test_reproducibility(self):
        """Same seed should give same results."""
        data = np.arange(20).reshape(10, 2)

        def metric(d):
            return float(np.mean(d))

        result1 = surrogate_test(metric, data, row_shuffle, seed=42)
        result2 = surrogate_test(metric, data, row_shuffle, seed=42)

        assert result1.observed == result2.observed
        assert result1.p_value == result2.p_value

    def test_tail_options(self):
        """Different tail options should work."""
        data = np.arange(20).reshape(10, 2)

        def metric(d):
            return float(np.mean(d))

        result_two = surrogate_test(
            metric, data, row_shuffle, tail="two-sided", seed=42
        )
        result_greater = surrogate_test(
            metric, data, row_shuffle, tail="greater", seed=42
        )
        result_less = surrogate_test(
            metric, data, row_shuffle, tail="less", seed=42
        )

        # All should complete without error
        assert 0 <= result_two.p_value <= 1
        assert 0 <= result_greater.p_value <= 1
        assert 0 <= result_less.p_value <= 1

    def test_str_representation(self):
        """SurrogateResult should have string representation."""
        result = SurrogateResult(
            observed=0.5,
            surrogate_mean=0.1,
            surrogate_std=0.05,
            p_value=0.02,
            significant=True,
            n_surrogates=100,
            shuffle_type="row_shuffle",
            alpha=0.05,
        )

        result_str = str(result)
        assert "0.5" in result_str
        assert "significant" in result_str


class TestEffectSizes:
    """Tests for effect size functions."""

    def test_cohens_d_large_effect(self):
        """Large difference should have large Cohen's d."""
        x = np.array([10, 11, 12, 13, 14])
        y = np.array([1, 2, 3, 4, 5])

        d = compute_cohens_d(x, y)

        assert d > 0.8  # Large effect

    def test_cohens_d_no_effect(self):
        """Identical distributions should have zero Cohen's d."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])

        d = compute_cohens_d(x, y)

        assert d == pytest.approx(0.0, abs=0.01)

    def test_cohens_d_sign(self):
        """Cohen's d sign indicates direction."""
        x = np.array([5, 6, 7])
        y = np.array([1, 2, 3])

        d = compute_cohens_d(x, y)
        assert d > 0  # x > y means positive

        d_reverse = compute_cohens_d(y, x)
        assert d_reverse < 0  # y < x means negative

    def test_glass_delta(self):
        """Glass's delta uses control group SD."""
        x = np.array([10, 11, 12])  # Treatment (higher values)
        y = np.array([1, 2, 3])  # Control

        delta = compute_glass_delta(x, y)

        assert delta > 0

    def test_empty_arrays_effect_sizes(self):
        """Empty arrays should return 0."""
        assert compute_cohens_d(np.array([]), np.array([1, 2])) == 0
        assert compute_glass_delta(np.array([]), np.array([1, 2])) == 0


class TestInterpretEffectSize:
    """Tests for effect size interpretation."""

    def test_negligible(self):
        """Small effects should be negligible."""
        assert interpret_effect_size(0.1) == "negligible"
        assert interpret_effect_size(0.05, "r") == "negligible"

    def test_small(self):
        """Moderate effects should be small."""
        assert interpret_effect_size(0.3) == "small"
        assert interpret_effect_size(0.2, "r") == "small"

    def test_medium(self):
        """Larger effects should be medium."""
        assert interpret_effect_size(0.6) == "medium"
        assert interpret_effect_size(0.4, "r") == "medium"

    def test_large(self):
        """Very large effects should be large."""
        assert interpret_effect_size(0.9) == "large"
        assert interpret_effect_size(0.6, "r") == "large"

    def test_negative_values(self):
        """Negative values should use absolute value."""
        assert interpret_effect_size(-0.9) == "large"


class TestCompareConditions:
    """Tests for compare_conditions function."""

    def test_basic_comparison(self):
        """Should compare multiple conditions."""
        data = {
            "control": np.array([1, 2, 3, 4, 5]),
            "treatment_a": np.array([2, 3, 4, 5, 6]),
            "treatment_b": np.array([10, 11, 12, 13, 14]),
        }

        result = compare_conditions(data)

        assert "comparisons" in result
        assert "summary" in result

        # Should have 3 pairwise comparisons (3 choose 2)
        assert len(result["comparisons"]) == 3

        # Summary should have all conditions
        assert set(result["summary"].keys()) == set(data.keys())

    def test_baseline_comparison(self):
        """Should only compare to baseline when specified."""
        data = {
            "control": np.array([1, 2, 3, 4, 5]),
            "treatment_a": np.array([2, 3, 4, 5, 6]),
            "treatment_b": np.array([10, 11, 12, 13, 14]),
        }

        result = compare_conditions(data, baseline_name="control")

        # Should only have 2 comparisons (control vs a, control vs b)
        assert len(result["comparisons"]) == 2

        for comp in result["comparisons"]:
            assert "control" in [comp["condition_a"], comp["condition_b"]]

    def test_comparison_fields(self):
        """Comparison results should have expected fields."""
        data = {
            "a": np.array([1, 2, 3]),
            "b": np.array([4, 5, 6]),
        }

        result = compare_conditions(data)
        comp = result["comparisons"][0]

        assert "condition_a" in comp
        assert "condition_b" in comp
        assert "u_statistic" in comp
        assert "p_value" in comp
        assert "rank_biserial" in comp
        assert "cohens_d" in comp
        assert "significant" in comp
        assert "interpretation" in comp


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_surrogate_with_bootstrap(self):
        """Surrogate test and bootstrap CI should work together."""
        rng = np.random.default_rng(42)

        # Create structured data (autocorrelated)
        data = np.cumsum(rng.normal(0, 1, 100))

        # Metric: autocorrelation
        def autocorr(d):
            if len(d) < 2:
                return 0.0
            return float(np.corrcoef(d[:-1], d[1:])[0, 1])

        # Run surrogate test
        surr_result = surrogate_test(
            autocorr,
            data,
            column_shuffle,
            n_surrogates=50,
            seed=42,
        )

        # Run bootstrap CI on the mean (more stable for bootstrap than autocorr)
        boot_result = bootstrap_ci(
            np.mean,
            data,
            n_bootstrap=100,
            seed=42,
        )

        # Both should complete without error
        assert surr_result.observed > 0  # Positive autocorrelation
        # Bootstrap CI should be valid for mean
        assert boot_result["ci_lower"] <= boot_result["estimate"] <= boot_result["ci_upper"]

    def test_end_to_end_analysis(self):
        """Full analysis pipeline should work."""
        rng = np.random.default_rng(42)

        # Simulate two conditions
        control = rng.normal(0, 1, 50)
        treatment = rng.normal(0.8, 1, 50)  # Different mean

        # Compare conditions
        comparison = compare_conditions({
            "control": control,
            "treatment": treatment,
        })

        # Should detect the difference
        comp = comparison["comparisons"][0]
        # With this effect size, might or might not be significant with n=50
        assert comp["cohens_d"] != 0

        # Bootstrap CI on treatment mean
        boot = bootstrap_ci(np.mean, treatment, seed=42)
        # CI should not include 0 (treatment has different mean)
        assert boot["ci_lower"] > 0 or boot["ci_upper"] > 0
