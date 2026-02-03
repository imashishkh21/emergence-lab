"""Surrogate Testing Framework for Statistical Significance.

Provides reusable surrogate testing functions for all emergence metrics.
Surrogate methods break specific types of structure in the data to test
whether observed patterns are statistically significant:

  - Row shuffle: breaks cross-agent coordination (shuffle across agents)
  - Column shuffle: breaks temporal dependencies (shuffle across time)
  - Block shuffle: preserves short-range structure, breaks long-range

Also provides:
  - Bootstrap confidence intervals (BCa method)
  - Mann-Whitney U test wrapper with effect size
  - Wilcoxon signed-rank test wrapper with effect size
  - Generic surrogate_test function for any metric

Uses only numpy/scipy — no Phase 5 dependencies required.

Reference: Theiler et al. (1992), "Testing for nonlinearity in time series"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy import stats


def row_shuffle(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle data across agents (rows) to break spatial coordination.

    For a (T, N) or (T, N, D) array, shuffles the N (agent) dimension
    independently for each time step. This breaks any coordination
    patterns between agents while preserving individual agent statistics.

    Args:
        data: Array with shape (T, N) or (T, N, D) where T=time, N=agents.
        rng: NumPy random generator.

    Returns:
        Shuffled array with same shape. Each time step has agents reordered.
    """
    data = np.asarray(data)

    if data.ndim == 1:
        # Single sequence — just permute it
        return rng.permutation(data)

    if data.ndim == 2:
        # (T, N) — shuffle agent axis for each timestep
        shuffled = np.empty_like(data)
        for t in range(len(data)):
            shuffled[t] = rng.permutation(data[t])
        return shuffled

    if data.ndim == 3:
        # (T, N, D) — shuffle agent axis for each timestep
        shuffled = np.empty_like(data)
        for t in range(len(data)):
            perm_idx = rng.permutation(data.shape[1])
            shuffled[t] = data[t, perm_idx]
        return shuffled

    raise ValueError(f"data must be 1D, 2D, or 3D, got shape {data.shape}")


def column_shuffle(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle data across time (columns) to break temporal dependencies.

    For a (T, N) or (T, N, D) array, shuffles the T (time) dimension.
    This breaks temporal correlations while preserving the
    instantaneous distribution of values across agents.

    Args:
        data: Array with shape (T, N) or (T, N, D).
        rng: NumPy random generator.

    Returns:
        Shuffled array with same shape. Time order is randomized.
    """
    data = np.asarray(data)

    if data.ndim == 1:
        return rng.permutation(data)

    # Permute along the time axis (axis 0)
    perm_idx = rng.permutation(len(data))
    return data[perm_idx]


def block_shuffle(
    data: np.ndarray,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Shuffle temporal blocks to preserve short-range, break long-range structure.

    Divides the time series into non-overlapping blocks, then shuffles
    the order of blocks. This preserves autocorrelation within blocks
    while randomizing the sequence of blocks.

    Args:
        data: Array with shape (T, ...). First axis is time.
        block_size: Size of each block. Must be >= 1.
        rng: NumPy random generator.

    Returns:
        Shuffled array with same shape (may be slightly shorter if T
        is not divisible by block_size).
    """
    data = np.asarray(data)

    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")

    n_timesteps = len(data)

    if n_timesteps == 0:
        return np.array(data.copy())

    # Number of complete blocks
    n_blocks = n_timesteps // block_size

    if n_blocks == 0:
        # Data shorter than block_size, return as-is
        return np.array(data.copy())

    # Truncate to complete blocks only
    truncated_len = n_blocks * block_size
    truncated_data = data[:truncated_len]

    # Reshape to (n_blocks, block_size, ...)
    new_shape = (n_blocks, block_size) + truncated_data.shape[1:]
    blocks = truncated_data.reshape(new_shape)

    # Shuffle blocks
    perm_idx = rng.permutation(n_blocks)
    shuffled_blocks = blocks[perm_idx]

    # Reshape back to original format
    return shuffled_blocks.reshape(truncated_data.shape)


def bootstrap_ci(
    statistic_fn: Callable[[np.ndarray], float],
    data: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int | None = None,
) -> dict[str, float]:
    """Compute bootstrap confidence interval using BCa method.

    Uses the bias-corrected and accelerated (BCa) bootstrap method
    for more accurate confidence intervals, especially for skewed
    distributions or small samples.

    Args:
        statistic_fn: Function that takes data and returns a scalar statistic.
        data: 1D array of data points.
        n_bootstrap: Number of bootstrap samples.
        ci: Confidence level (default 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys:
        - estimate: point estimate from original data
        - ci_lower: lower bound of confidence interval
        - ci_upper: upper bound of confidence interval
        - se: standard error estimated from bootstrap distribution
        - ci_level: the confidence level used
    """
    data = np.asarray(data).ravel()
    n = len(data)

    if n == 0:
        return {
            "estimate": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "se": np.nan,
            "ci_level": ci,
        }

    if n == 1:
        estimate = float(statistic_fn(data))
        return {
            "estimate": estimate,
            "ci_lower": estimate,
            "ci_upper": estimate,
            "se": 0.0,
            "ci_level": ci,
        }

    rng = np.random.default_rng(seed)

    # Original estimate
    estimate = float(statistic_fn(data))

    # Generate bootstrap samples and compute statistics
    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        boot_sample = rng.choice(data, size=n, replace=True)
        boot_stats[i] = statistic_fn(boot_sample)

    # Standard error from bootstrap distribution
    se = float(np.std(boot_stats, ddof=1))

    # BCa method for confidence intervals
    # Step 1: Bias correction (z0)
    # Fraction of bootstrap statistics less than original estimate
    prop_below = np.mean(boot_stats < estimate)
    # Avoid edge cases
    prop_below = np.clip(prop_below, 1e-10, 1 - 1e-10)
    z0 = stats.norm.ppf(prop_below)

    # Step 2: Acceleration (a) via jackknife
    jackknife_stats = np.empty(n)
    for i in range(n):
        # Leave-one-out sample
        jack_sample = np.delete(data, i)
        jackknife_stats[i] = statistic_fn(jack_sample)

    jack_mean = np.mean(jackknife_stats)
    jack_diff = jack_mean - jackknife_stats
    numerator = np.sum(jack_diff**3)
    denominator = 6 * (np.sum(jack_diff**2) ** 1.5)

    if abs(denominator) < 1e-10:
        a = 0.0
    else:
        a = numerator / denominator

    # Step 3: Compute adjusted percentiles
    alpha_lower = (1 - ci) / 2
    alpha_upper = 1 - alpha_lower

    z_lower = stats.norm.ppf(alpha_lower)
    z_upper = stats.norm.ppf(alpha_upper)

    # BCa adjusted percentiles
    def bca_percentile(z_alpha: float) -> float:
        numerator = z0 + z_alpha
        denominator = 1 - a * numerator
        if abs(denominator) < 1e-10:
            return z_alpha  # Fall back to standard percentile
        adjusted_z = z0 + numerator / denominator
        return float(stats.norm.cdf(adjusted_z))

    p_lower = bca_percentile(z_lower)
    p_upper = bca_percentile(z_upper)

    # Clip to valid range
    p_lower = np.clip(p_lower, 0.0, 1.0)
    p_upper = np.clip(p_upper, 0.0, 1.0)

    ci_lower = float(np.percentile(boot_stats, p_lower * 100))
    ci_upper = float(np.percentile(boot_stats, p_upper * 100))

    return {
        "estimate": estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": se,
        "ci_level": ci,
    }


@dataclass
class StatisticalTestResult:
    """Result of a statistical hypothesis test."""

    statistic: float
    p_value: float
    effect_size: float
    effect_size_name: str
    significant: bool
    alpha: float

    def __str__(self) -> str:
        sig_str = "significant" if self.significant else "not significant"
        return (
            f"stat={self.statistic:.4f}, p={self.p_value:.4f} ({sig_str}), "
            f"{self.effect_size_name}={self.effect_size:.4f}"
        )


def mann_whitney_u(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> StatisticalTestResult:
    """Mann-Whitney U test wrapper with effect size.

    Non-parametric test for comparing two independent samples.
    Returns rank-biserial correlation as effect size.

    Args:
        x: First sample (1D array).
        y: Second sample (1D array).
        alpha: Significance level.
        alternative: "two-sided", "less", or "greater".

    Returns:
        TestResult with U statistic, p-value, and rank-biserial effect size.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    n_x, n_y = len(x), len(y)

    if n_x == 0 or n_y == 0:
        return StatisticalTestResult(
            statistic=np.nan,
            p_value=1.0,
            effect_size=0.0,
            effect_size_name="rank_biserial",
            significant=False,
            alpha=alpha,
        )

    # Run Mann-Whitney U test
    result = stats.mannwhitneyu(x, y, alternative=alternative)
    u_stat = float(result.statistic)
    p_value = float(result.pvalue)

    # Compute rank-biserial correlation as effect size
    # r = 1 - (2U)/(n1*n2)
    # Ranges from -1 to 1, where 0 means no effect
    max_u = n_x * n_y
    if max_u > 0:
        rank_biserial = 1 - (2 * u_stat) / max_u
    else:
        rank_biserial = 0.0

    return StatisticalTestResult(
        statistic=u_stat,
        p_value=p_value,
        effect_size=rank_biserial,
        effect_size_name="rank_biserial",
        significant=p_value < alpha,
        alpha=alpha,
    )


def wilcoxon_signed_rank(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> StatisticalTestResult:
    """Wilcoxon signed-rank test wrapper with effect size.

    Non-parametric test for comparing two paired samples.
    Returns rank-biserial correlation (matched-pairs) as effect size.

    Args:
        x: First sample (1D array).
        y: Second sample (1D array), paired with x.
        alpha: Significance level.
        alternative: "two-sided", "less", or "greater".

    Returns:
        TestResult with W statistic, p-value, and effect size (r = Z/sqrt(N)).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if len(x) != len(y):
        raise ValueError(f"x and y must have same length, got {len(x)} and {len(y)}")

    n = len(x)

    if n == 0:
        return StatisticalTestResult(
            statistic=np.nan,
            p_value=1.0,
            effect_size=0.0,
            effect_size_name="r",
            significant=False,
            alpha=alpha,
        )

    # Compute differences
    d = x - y

    # Remove zeros (ties at zero)
    d_nonzero = d[d != 0]
    n_eff = len(d_nonzero)

    if n_eff == 0:
        # All differences are zero
        return StatisticalTestResult(
            statistic=0.0,
            p_value=1.0,
            effect_size=0.0,
            effect_size_name="r",
            significant=False,
            alpha=alpha,
        )

    # Run Wilcoxon signed-rank test
    result = stats.wilcoxon(d_nonzero, alternative=alternative)
    w_stat = float(result.statistic)
    p_value = float(result.pvalue)

    # Effect size: r = Z / sqrt(N)
    # Approximate Z from W using normal approximation
    # W ~ N(mean, var) where mean = n(n+1)/4, var = n(n+1)(2n+1)/24
    mean_w = n_eff * (n_eff + 1) / 4
    var_w = n_eff * (n_eff + 1) * (2 * n_eff + 1) / 24

    if var_w > 0:
        z_approx = (w_stat - mean_w) / np.sqrt(var_w)
        r_effect = z_approx / np.sqrt(n_eff)
    else:
        r_effect = 0.0

    return StatisticalTestResult(
        statistic=w_stat,
        p_value=p_value,
        effect_size=r_effect,
        effect_size_name="r",
        significant=p_value < alpha,
        alpha=alpha,
    )


@dataclass
class SurrogateResult:
    """Result of a surrogate significance test."""

    observed: float
    surrogate_mean: float
    surrogate_std: float
    p_value: float
    significant: bool
    n_surrogates: int
    shuffle_type: str
    alpha: float

    def __str__(self) -> str:
        sig_str = "significant" if self.significant else "not significant"
        return (
            f"observed={self.observed:.4f}, "
            f"surrogate={self.surrogate_mean:.4f}±{self.surrogate_std:.4f}, "
            f"p={self.p_value:.4f} ({sig_str})"
        )


def surrogate_test(
    metric_fn: Callable[[np.ndarray], float],
    real_data: np.ndarray,
    shuffle_fn: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    n_surrogates: int = 100,
    seed: int | None = None,
    alpha: float = 0.05,
    tail: str = "two-sided",
) -> SurrogateResult:
    """Generic surrogate test for statistical significance.

    Generates surrogate data using the provided shuffle function,
    computes the metric on each surrogate, and compares the observed
    metric to the surrogate distribution.

    Args:
        metric_fn: Function that takes data and returns a scalar metric.
        real_data: Original data array.
        shuffle_fn: Function that takes (data, rng) and returns shuffled data.
        n_surrogates: Number of surrogate samples to generate.
        seed: Random seed for reproducibility.
        alpha: Significance level.
        tail: "two-sided", "greater" (observed > surrogates),
              or "less" (observed < surrogates).

    Returns:
        SurrogateResult with observed value, surrogate distribution stats,
        p-value, and significance flag.
    """
    real_data = np.asarray(real_data)
    rng = np.random.default_rng(seed)

    # Compute observed metric
    observed = float(metric_fn(real_data))

    # Generate surrogates and compute metrics
    surrogate_values = np.empty(n_surrogates)
    for i in range(n_surrogates):
        shuffled_data = shuffle_fn(real_data, rng)
        surrogate_values[i] = metric_fn(shuffled_data)

    surrogate_mean = float(np.mean(surrogate_values))
    surrogate_std = float(np.std(surrogate_values))

    # Compute p-value based on tail
    if tail == "greater":
        # Test if observed is significantly greater than surrogates
        n_extreme = np.sum(surrogate_values >= observed)
    elif tail == "less":
        # Test if observed is significantly less than surrogates
        n_extreme = np.sum(surrogate_values <= observed)
    else:  # two-sided
        # Test if observed is significantly different from surrogates
        # Count surrogates that are at least as extreme (either direction)
        observed_dev = abs(observed - surrogate_mean)
        surrogate_dev = np.abs(surrogate_values - surrogate_mean)
        n_extreme = np.sum(surrogate_dev >= observed_dev)

    p_value = float(n_extreme + 1) / (n_surrogates + 1)  # +1 for observed

    # Get shuffle type name from function
    shuffle_type = getattr(shuffle_fn, "__name__", "custom")

    return SurrogateResult(
        observed=observed,
        surrogate_mean=surrogate_mean,
        surrogate_std=surrogate_std,
        p_value=p_value,
        significant=p_value < alpha,
        n_surrogates=n_surrogates,
        shuffle_type=shuffle_type,
        alpha=alpha,
    )


def compute_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d effect size for two independent samples.

    Args:
        x: First sample.
        y: Second sample.

    Returns:
        Cohen's d (standardized mean difference).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    n_x, n_y = len(x), len(y)

    if n_x == 0 or n_y == 0:
        return 0.0

    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)

    # Pooled standard deviation
    if n_x + n_y <= 2:
        return 0.0

    pooled_var = ((n_x - 1) * var_x + (n_y - 1) * var_y) / (n_x + n_y - 2)
    pooled_std = np.sqrt(pooled_var)

    if pooled_std < 1e-10:
        return 0.0

    return float((mean_x - mean_y) / pooled_std)


def compute_glass_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Glass's delta effect size.

    Uses the standard deviation of the control group (y) as the denominator.

    Args:
        x: Treatment/experimental sample.
        y: Control sample (used for standardization).

    Returns:
        Glass's delta.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if len(x) == 0 or len(y) == 0:
        return 0.0

    mean_x, mean_y = np.mean(x), np.mean(y)
    std_y = np.std(y, ddof=1)

    if std_y < 1e-10:
        return 0.0

    return float((mean_x - mean_y) / std_y)


def interpret_effect_size(effect: float, metric: str = "cohens_d") -> str:
    """Interpret effect size using conventional thresholds.

    Args:
        effect: Effect size value.
        metric: Type of effect size ("cohens_d", "r", "rank_biserial").

    Returns:
        String interpretation: "negligible", "small", "medium", or "large".
    """
    effect = abs(effect)

    if metric in ("cohens_d", "glass_delta"):
        # Cohen's conventions
        if effect < 0.2:
            return "negligible"
        elif effect < 0.5:
            return "small"
        elif effect < 0.8:
            return "medium"
        else:
            return "large"
    elif metric in ("r", "rank_biserial"):
        # Correlation-based conventions
        if effect < 0.1:
            return "negligible"
        elif effect < 0.3:
            return "small"
        elif effect < 0.5:
            return "medium"
        else:
            return "large"
    else:
        # Default to Cohen's conventions
        if effect < 0.2:
            return "negligible"
        elif effect < 0.5:
            return "small"
        elif effect < 0.8:
            return "medium"
        else:
            return "large"


def compare_conditions(
    condition_data: dict[str, np.ndarray],
    baseline_name: str | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compare multiple conditions with statistical tests.

    Performs pairwise Mann-Whitney U tests between all conditions
    and returns a comparison table.

    Args:
        condition_data: Dict mapping condition names to 1D arrays of values.
        baseline_name: If provided, only compare other conditions to this baseline.
        alpha: Significance level.

    Returns:
        Dict with keys:
        - comparisons: list of dicts with pairwise comparison results
        - summary: overall statistics per condition
    """
    conditions = list(condition_data.keys())
    n_conditions = len(conditions)

    comparisons = []
    summary: dict[str, dict[str, float]] = {}

    # Summary stats for each condition
    for name, data in condition_data.items():
        data_arr = np.asarray(data).ravel()
        summary[name] = {
            "mean": float(np.mean(data_arr)) if len(data_arr) > 0 else np.nan,
            "std": float(np.std(data_arr)) if len(data_arr) > 0 else np.nan,
            "median": float(np.median(data_arr)) if len(data_arr) > 0 else np.nan,
            "n": len(data_arr),
        }

    # Pairwise comparisons
    for i in range(n_conditions):
        for j in range(i + 1, n_conditions):
            name_a, name_b = conditions[i], conditions[j]

            # Skip if baseline specified and neither is baseline
            if baseline_name is not None:
                if name_a != baseline_name and name_b != baseline_name:
                    continue

            data_a = np.asarray(condition_data[name_a]).ravel()
            data_b = np.asarray(condition_data[name_b]).ravel()

            result = mann_whitney_u(data_a, data_b, alpha=alpha)
            cohens_d = compute_cohens_d(data_a, data_b)

            comparisons.append({
                "condition_a": name_a,
                "condition_b": name_b,
                "u_statistic": result.statistic,
                "p_value": result.p_value,
                "rank_biserial": result.effect_size,
                "cohens_d": cohens_d,
                "significant": result.significant,
                "interpretation": interpret_effect_size(cohens_d),
            })

    return {
        "comparisons": comparisons,
        "summary": summary,
    }
