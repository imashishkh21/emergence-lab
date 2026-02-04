"""Statistical reporting module using rliable for IQM, CIs, and comparisons.

This module provides statistical analysis tools following the methodology from
Agarwal et al. (2021), "Deep RL at the Edge of the Statistical Precipice".

Features:
  - IQM (Interquartile Mean) with bootstrap confidence intervals
  - Performance profiles (CDFs of normalized scores)
  - Probability of improvement comparisons
  - Wrapped scipy statistical tests (Mann-Whitney, Wilcoxon, Welch's t)
  - Pairwise method comparison tables

The rliable library is an optional dependency. Functions fall back to simple
implementations when rliable is not installed.

Example usage:
    >>> from src.analysis.statistics import compute_iqm, compare_methods
    >>> scores = {"ours": [10, 12, 11], "baseline": [8, 9, 7]}
    >>> result = compute_iqm(scores["ours"])
    >>> print(f"IQM: {result.iqm:.2f} ({result.ci_lower:.2f}, {result.ci_upper:.2f})")
    >>> comparison = compare_methods(scores)
    >>> print(comparison.summary)

Reference: Agarwal et al. (2021), "Deep Reinforcement Learning at the Edge
of the Statistical Precipice", NeurIPS.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

# Check if rliable is available
_rliable_available = importlib.util.find_spec("rliable") is not None


@dataclass
class IQMResult:
    """Result of IQM computation with bootstrap confidence interval.

    Attributes:
        iqm: Interquartile mean (25th-75th percentile mean).
        ci_lower: Lower bound of 95% confidence interval.
        ci_upper: Upper bound of 95% confidence interval.
        ci_level: Confidence level used (default 0.95).
        n_samples: Number of data points used.
        n_bootstrap: Number of bootstrap samples used.
    """

    iqm: float
    ci_lower: float
    ci_upper: float
    ci_level: float = 0.95
    n_samples: int = 0
    n_bootstrap: int = 10000

    def __str__(self) -> str:
        return (
            f"IQM: {self.iqm:.4f} "
            f"[{self.ci_level * 100:.0f}% CI: ({self.ci_lower:.4f}, {self.ci_upper:.4f})]"
        )


@dataclass
class StatisticalReport:
    """Comprehensive statistical report for a method's performance.

    Contains summary statistics, confidence intervals, and raw data
    for downstream analysis. Compatible with rliable conventions.

    Attributes:
        method_name: Name of the method.
        iqm: Interquartile mean of scores.
        ci_lower: Lower 95% CI bound for IQM.
        ci_upper: Upper 95% CI bound for IQM.
        median: Median score.
        mean: Mean score.
        std: Standard deviation of scores.
        n_seeds: Number of seeds/runs.
        scores: Raw score array (for paired comparisons).
    """

    method_name: str
    iqm: float
    ci_lower: float
    ci_upper: float
    median: float
    mean: float
    std: float
    n_seeds: int
    scores: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self) -> None:
        # Ensure scores is a numpy array
        if not isinstance(self.scores, np.ndarray):
            self.scores = np.asarray(self.scores)

    def __str__(self) -> str:
        return (
            f"StatisticalReport({self.method_name})\n"
            f"  IQM: {self.iqm:.4f} [{self.ci_lower:.4f}, {self.ci_upper:.4f}]\n"
            f"  Mean: {self.mean:.4f} ± {self.std:.4f}\n"
            f"  Median: {self.median:.4f}\n"
            f"  N: {self.n_seeds}"
        )


@dataclass
class HypothesisTestResult:
    """Result of a statistical hypothesis test.

    Generic container for test results from Mann-Whitney, Wilcoxon,
    or Welch's t-test. Includes effect size when applicable.

    Attributes:
        test_name: Name of the test used.
        statistic: Test statistic value.
        p_value: p-value (probability of null hypothesis).
        significant: Whether result is significant at alpha level.
        alpha: Significance threshold used.
        effect_size: Effect size measure (depends on test).
        effect_size_name: Name of effect size measure.
        method_a: Name of first method being compared.
        method_b: Name of second method being compared.
    """

    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float = 0.05
    effect_size: float = 0.0
    effect_size_name: str = ""
    method_a: str = ""
    method_b: str = ""

    def __str__(self) -> str:
        sig = "***" if self.p_value < 0.001 else (
            "**" if self.p_value < 0.01 else (
                "*" if self.p_value < 0.05 else ""
            )
        )
        result = (
            f"{self.test_name}: stat={self.statistic:.4f}, p={self.p_value:.4f}{sig}"
        )
        if self.effect_size_name:
            result += f", {self.effect_size_name}={self.effect_size:.4f}"
        return result


@dataclass
class MethodComparison:
    """Comprehensive comparison of multiple methods.

    Aggregates statistical reports, pairwise tests, and rankings.

    Attributes:
        reports: Dict mapping method names to StatisticalReports.
        pairwise_tests: Dict of pairwise HypothesisTestResults keyed by "(a, b)".
        rankings: List of method names sorted by IQM (best first).
        best_method: Name of method with highest IQM.
        probability_of_improvement: Dict of P(X > Y) for each pair.
        summary: Human-readable summary string.
    """

    reports: dict[str, StatisticalReport] = field(default_factory=dict)
    pairwise_tests: dict[str, HypothesisTestResult] = field(default_factory=dict)
    rankings: list[str] = field(default_factory=list)
    best_method: str = ""
    probability_of_improvement: dict[str, float] = field(default_factory=dict)
    summary: str = ""

    def __str__(self) -> str:
        return self.summary if self.summary else f"MethodComparison({list(self.reports.keys())})"


# ═══════════════════════════════════════════════════════════════════════════════
# IQM and Bootstrap CI
# ═══════════════════════════════════════════════════════════════════════════════


def _iqm_simple(scores: np.ndarray) -> float:
    """Compute IQM without rliable (simple implementation).

    Args:
        scores: 1D array of scores.

    Returns:
        Interquartile mean.
    """
    if len(scores) == 0:
        return 0.0

    q25 = np.percentile(scores, 25)
    q75 = np.percentile(scores, 75)
    mask = (scores >= q25) & (scores <= q75)
    iq_scores = scores[mask]

    if len(iq_scores) == 0:
        return float(np.median(scores))

    return float(np.mean(iq_scores))


def _bootstrap_ci_simple(
    scores: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float]:
    """Bootstrap CI without rliable (simple implementation).

    Args:
        scores: 1D array of scores.
        n_bootstrap: Number of bootstrap samples.
        ci: Confidence level.
        seed: Random seed.

    Returns:
        Tuple of (ci_lower, ci_upper).
    """
    if len(scores) == 0:
        return (0.0, 0.0)

    if len(scores) == 1:
        return (float(scores[0]), float(scores[0]))

    rng = np.random.default_rng(seed)
    boot_iqms = np.empty(n_bootstrap)
    n = len(scores)

    for i in range(n_bootstrap):
        boot_sample = rng.choice(scores, size=n, replace=True)
        boot_iqms[i] = _iqm_simple(boot_sample)

    alpha = (1 - ci) / 2
    ci_lower = float(np.percentile(boot_iqms, alpha * 100))
    ci_upper = float(np.percentile(boot_iqms, (1 - alpha) * 100))

    return (ci_lower, ci_upper)


def compute_iqm(
    scores: np.ndarray | list[float],
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int | None = None,
) -> IQMResult:
    """Compute Interquartile Mean with bootstrap confidence interval.

    Uses rliable.metrics.aggregate_iqm when available, falls back to
    simple implementation otherwise.

    Args:
        scores: 1D array or list of scores.
        n_bootstrap: Number of bootstrap samples for CI estimation.
        ci_level: Confidence level (default 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        IQMResult with IQM value and confidence interval.

    Example:
        >>> result = compute_iqm([10, 12, 11, 13, 9, 14, 8, 15])
        >>> print(result)
        IQM: 11.5000 [95% CI: (10.0000, 13.0000)]
    """
    scores = np.asarray(scores).ravel()

    if len(scores) == 0:
        return IQMResult(
            iqm=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            ci_level=ci_level,
            n_samples=0,
            n_bootstrap=n_bootstrap,
        )

    if _rliable_available:
        try:
            from rliable import metrics

            # rliable expects shape (n_runs, n_tasks) for aggregate functions
            # For single task, reshape to (n_runs, 1)
            scores_2d = scores.reshape(-1, 1)

            # Compute IQM
            iqm_value = float(metrics.aggregate_iqm(scores_2d))

            # Bootstrap CI - use our own implementation since rliable.library
            # has compatibility issues with newer versions of arch
            ci_lower, ci_upper = _bootstrap_ci_simple(
                scores, n_bootstrap=n_bootstrap, ci=ci_level, seed=seed
            )

            return IQMResult(
                iqm=iqm_value,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                ci_level=ci_level,
                n_samples=len(scores),
                n_bootstrap=n_bootstrap,
            )
        except Exception:
            # Fall through to simple implementation
            pass

    # Simple implementation
    iqm_value = _iqm_simple(scores)
    ci_lower, ci_upper = _bootstrap_ci_simple(
        scores, n_bootstrap=n_bootstrap, ci=ci_level, seed=seed
    )

    return IQMResult(
        iqm=iqm_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        n_samples=len(scores),
        n_bootstrap=n_bootstrap,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Performance Profiles
# ═══════════════════════════════════════════════════════════════════════════════


def performance_profiles(
    score_dict: dict[str, np.ndarray | list[float]],
    tau_range: np.ndarray | None = None,
    n_points: int = 101,
) -> dict[str, Any]:
    """Compute performance profiles (CDFs) for each method.

    Performance profiles show the fraction of runs achieving at least
    a given normalized score threshold. Following Agarwal et al. (2021).

    Args:
        score_dict: Dict mapping method names to score arrays.
        tau_range: Array of threshold values (0 to 1). If None, uses
            linspace(0, 1, n_points).
        n_points: Number of points in tau_range if not provided.

    Returns:
        Dict with:
        - tau: Threshold values
        - profiles: Dict mapping method names to CDF arrays
        - auc: Dict mapping method names to area under CDF

    Example:
        >>> scores = {"ours": [0.9, 0.8, 0.7], "baseline": [0.5, 0.6, 0.4]}
        >>> result = performance_profiles(scores)
        >>> print(result["auc"])
        {'ours': 0.8, 'baseline': 0.5}
    """
    if tau_range is None:
        tau_range = np.linspace(0.0, 1.0, n_points)
    else:
        tau_range = np.asarray(tau_range)

    # Convert all scores to numpy arrays
    arrays: dict[str, np.ndarray] = {}
    for name, scores in score_dict.items():
        arrays[name] = np.asarray(scores).ravel()

    if len(arrays) == 0:
        return {"tau": tau_range, "profiles": {}, "auc": {}}

    # Find global min and max for normalization
    all_scores = np.concatenate(list(arrays.values()))
    if len(all_scores) == 0:
        return {"tau": tau_range, "profiles": {}, "auc": {}}

    score_min = float(np.min(all_scores))
    score_max = float(np.max(all_scores))
    score_range = score_max - score_min

    if score_range < 1e-10:
        # All scores are the same
        const_profiles = {name: np.ones_like(tau_range) for name in arrays}
        const_auc = {name: 1.0 for name in arrays}
        return {"tau": tau_range, "profiles": const_profiles, "auc": const_auc}

    # Normalize scores to [0, 1]
    normalized: dict[str, np.ndarray] = {}
    for name, arr_scores in arrays.items():
        normalized[name] = (arr_scores - score_min) / score_range

    # Compute CDFs (fraction of runs >= tau)
    profiles: dict[str, np.ndarray] = {}
    auc: dict[str, float] = {}

    for name, norm_scores in normalized.items():
        n_runs = len(norm_scores)
        if n_runs == 0:
            profiles[name] = np.zeros_like(tau_range)
            auc[name] = 0.0
            continue

        # For each tau, count fraction of scores >= tau
        cdf = np.array([np.mean(norm_scores >= t) for t in tau_range])
        profiles[name] = cdf

        # Area under CDF (trapezoidal integration)
        auc[name] = float(np.trapezoid(cdf, tau_range))

    return {"tau": tau_range, "profiles": profiles, "auc": auc}


# ═══════════════════════════════════════════════════════════════════════════════
# Probability of Improvement
# ═══════════════════════════════════════════════════════════════════════════════


def probability_of_improvement(
    scores_x: np.ndarray | list[float],
    scores_y: np.ndarray | list[float],
    n_bootstrap: int = 2000,
    seed: int | None = None,
) -> dict[str, float]:
    """Compute probability that method X outperforms method Y.

    P(X > Y) is estimated by bootstrap sampling from each distribution
    and counting the fraction where X > Y. Following Agarwal et al. (2021).

    Args:
        scores_x: Scores from method X.
        scores_y: Scores from method Y.
        n_bootstrap: Number of bootstrap samples for estimation.
        seed: Random seed for reproducibility.

    Returns:
        Dict with:
        - prob_x_better: P(X > Y)
        - prob_y_better: P(Y > X)
        - prob_tie: P(X = Y) (within numerical tolerance)
        - ci_lower: Lower 95% CI for P(X > Y)
        - ci_upper: Upper 95% CI for P(X > Y)

    Example:
        >>> prob = probability_of_improvement([10, 11, 12], [8, 9, 10])
        >>> print(f"P(X > Y) = {prob['prob_x_better']:.2f}")
        P(X > Y) = 0.85
    """
    scores_x = np.asarray(scores_x).ravel()
    scores_y = np.asarray(scores_y).ravel()

    n_x, n_y = len(scores_x), len(scores_y)

    if n_x == 0 or n_y == 0:
        return {
            "prob_x_better": 0.5,
            "prob_y_better": 0.5,
            "prob_tie": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 1.0,
        }

    if _rliable_available:
        try:
            from rliable import metrics

            # rliable expects (n_runs, n_tasks) shape
            scores_x_2d = scores_x.reshape(-1, 1)
            scores_y_2d = scores_y.reshape(-1, 1)

            # Use rliable's probability_of_improvement
            prob = metrics.probability_of_improvement(
                scores_x_2d, scores_y_2d
            )
            prob_x_better = float(prob)

            # Bootstrap CI for the probability
            rng = np.random.default_rng(seed)
            boot_probs = np.empty(n_bootstrap)

            for i in range(n_bootstrap):
                boot_x = rng.choice(scores_x, size=n_x, replace=True)
                boot_y = rng.choice(scores_y, size=n_y, replace=True)
                boot_x_2d = boot_x.reshape(-1, 1)
                boot_y_2d = boot_y.reshape(-1, 1)
                boot_probs[i] = metrics.probability_of_improvement(
                    boot_x_2d, boot_y_2d
                )

            ci_lower = float(np.percentile(boot_probs, 2.5))
            ci_upper = float(np.percentile(boot_probs, 97.5))

            return {
                "prob_x_better": prob_x_better,
                "prob_y_better": 1.0 - prob_x_better,
                "prob_tie": 0.0,  # rliable doesn't track ties
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        except Exception:
            # Fall through to simple implementation
            pass

    # Simple implementation: pairwise comparison
    rng = np.random.default_rng(seed)

    # Point estimate via U-statistic
    n_x_better = 0
    n_y_better = 0
    n_tie = 0
    total = n_x * n_y

    for x in scores_x:
        for y in scores_y:
            if x > y + 1e-10:
                n_x_better += 1
            elif y > x + 1e-10:
                n_y_better += 1
            else:
                n_tie += 1

    prob_x_better = n_x_better / total
    prob_y_better = n_y_better / total
    prob_tie = n_tie / total

    # Bootstrap CI
    boot_probs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        boot_x = rng.choice(scores_x, size=n_x, replace=True)
        boot_y = rng.choice(scores_y, size=n_y, replace=True)

        # Count X > Y in bootstrap sample
        count = sum(1 for x in boot_x for y in boot_y if x > y + 1e-10)
        boot_probs[i] = count / (n_x * n_y)

    ci_lower = float(np.percentile(boot_probs, 2.5))
    ci_upper = float(np.percentile(boot_probs, 97.5))

    return {
        "prob_x_better": prob_x_better,
        "prob_y_better": prob_y_better,
        "prob_tie": prob_tie,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Statistical Tests
# ═══════════════════════════════════════════════════════════════════════════════


def mann_whitney_test(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> HypothesisTestResult:
    """Mann-Whitney U test for comparing two independent samples.

    Non-parametric test that doesn't assume normality. Returns rank-biserial
    correlation as effect size.

    Args:
        x: Scores from first method.
        y: Scores from second method.
        alpha: Significance level.
        alternative: "two-sided", "less", or "greater".

    Returns:
        HypothesisTestResult with U statistic, p-value, and rank-biserial effect size.

    Example:
        >>> result = mann_whitney_test([10, 11, 12], [8, 9, 10])
        >>> print(result)
        Mann-Whitney U: stat=7.5000, p=0.1266, rank_biserial=0.6667
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    n_x, n_y = len(x), len(y)

    if n_x == 0 or n_y == 0:
        return HypothesisTestResult(
            test_name="Mann-Whitney U",
            statistic=np.nan,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            effect_size=0.0,
            effect_size_name="rank_biserial",
        )

    # Run test
    result = stats.mannwhitneyu(x, y, alternative=alternative)
    u_stat = float(result.statistic)
    p_value = float(result.pvalue)

    # Rank-biserial correlation: r = 2U/(n1*n2) - 1
    # Positive r means x tends to be larger than y.
    max_u = n_x * n_y
    rank_biserial = (2 * u_stat) / max_u - 1 if max_u > 0 else 0.0

    return HypothesisTestResult(
        test_name="Mann-Whitney U",
        statistic=u_stat,
        p_value=p_value,
        significant=p_value < alpha,
        alpha=alpha,
        effect_size=rank_biserial,
        effect_size_name="rank_biserial",
    )


def wilcoxon_test(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> HypothesisTestResult:
    """Wilcoxon signed-rank test for paired samples.

    Non-parametric test for comparing paired samples. Returns effect size
    r = Z / sqrt(N).

    Args:
        x: Scores from first method (paired with y).
        y: Scores from second method (paired with x).
        alpha: Significance level.
        alternative: "two-sided", "less", or "greater".

    Returns:
        HypothesisTestResult with W statistic, p-value, and r effect size.

    Raises:
        ValueError: If x and y have different lengths.

    Example:
        >>> result = wilcoxon_test([10, 11, 12], [8, 9, 10])
        >>> print(result)
        Wilcoxon signed-rank: stat=6.0000, p=0.1088, r=0.8660
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if len(x) != len(y):
        raise ValueError(f"x and y must have same length, got {len(x)} and {len(y)}")

    n = len(x)

    if n == 0:
        return HypothesisTestResult(
            test_name="Wilcoxon signed-rank",
            statistic=np.nan,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            effect_size=0.0,
            effect_size_name="r",
        )

    # Compute differences
    d = x - y

    # Remove zeros
    d_nonzero = d[d != 0]
    n_eff = len(d_nonzero)

    if n_eff == 0:
        return HypothesisTestResult(
            test_name="Wilcoxon signed-rank",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            effect_size=0.0,
            effect_size_name="r",
        )

    # Run test
    result = stats.wilcoxon(d_nonzero, alternative=alternative)
    w_stat = float(result.statistic)
    p_value = float(result.pvalue)

    # Effect size: r = Z / sqrt(N)
    # Approximate Z from normal approximation
    mean_w = n_eff * (n_eff + 1) / 4
    var_w = n_eff * (n_eff + 1) * (2 * n_eff + 1) / 24

    if var_w > 0:
        z_approx = (w_stat - mean_w) / np.sqrt(var_w)
        r_effect = z_approx / np.sqrt(n_eff)
    else:
        r_effect = 0.0

    return HypothesisTestResult(
        test_name="Wilcoxon signed-rank",
        statistic=w_stat,
        p_value=p_value,
        significant=p_value < alpha,
        alpha=alpha,
        effect_size=float(r_effect),
        effect_size_name="r",
    )


def welch_t_test(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> HypothesisTestResult:
    """Welch's t-test for comparing means with unequal variances.

    Parametric test that assumes approximate normality but allows unequal
    variances. Returns Cohen's d as effect size.

    Args:
        x: Scores from first method.
        y: Scores from second method.
        alpha: Significance level.
        alternative: "two-sided", "less", or "greater".

    Returns:
        HypothesisTestResult with t statistic, p-value, and Cohen's d effect size.

    Example:
        >>> result = welch_t_test([10, 11, 12], [8, 9, 10])
        >>> print(result)
        Welch's t-test: stat=1.7321, p=0.1583, cohens_d=1.4142
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    n_x, n_y = len(x), len(y)

    if n_x < 2 or n_y < 2:
        return HypothesisTestResult(
            test_name="Welch's t-test",
            statistic=np.nan,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            effect_size=0.0,
            effect_size_name="cohens_d",
        )

    # Run Welch's t-test
    result = stats.ttest_ind(x, y, equal_var=False, alternative=alternative)
    t_stat = float(result.statistic)
    p_value = float(result.pvalue)

    # Cohen's d effect size
    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)

    # Pooled standard deviation
    pooled_var = ((n_x - 1) * var_x + (n_y - 1) * var_y) / (n_x + n_y - 2)
    pooled_std = np.sqrt(pooled_var)

    if pooled_std > 1e-10:
        cohens_d = (mean_x - mean_y) / pooled_std
    else:
        cohens_d = 0.0

    return HypothesisTestResult(
        test_name="Welch's t-test",
        statistic=t_stat,
        p_value=p_value,
        significant=p_value < alpha,
        alpha=alpha,
        effect_size=float(cohens_d),
        effect_size_name="cohens_d",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Method Comparison
# ═══════════════════════════════════════════════════════════════════════════════


def create_statistical_report(
    method_name: str,
    scores: np.ndarray | list[float],
    n_bootstrap: int = 10000,
    seed: int | None = None,
) -> StatisticalReport:
    """Create a comprehensive statistical report for a method.

    Args:
        method_name: Name of the method.
        scores: Array of scores from multiple runs/seeds.
        n_bootstrap: Number of bootstrap samples for CI.
        seed: Random seed for reproducibility.

    Returns:
        StatisticalReport with summary statistics.

    Example:
        >>> report = create_statistical_report("ours", [10, 11, 12, 13, 14])
        >>> print(report)
        StatisticalReport(ours)
          IQM: 12.0000 [11.0000, 13.0000]
          Mean: 12.0000 ± 1.5811
          Median: 12.0000
          N: 5
    """
    scores = np.asarray(scores).ravel()

    if len(scores) == 0:
        return StatisticalReport(
            method_name=method_name,
            iqm=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            median=0.0,
            mean=0.0,
            std=0.0,
            n_seeds=0,
            scores=scores,
        )

    # Compute IQM with CI
    iqm_result = compute_iqm(scores, n_bootstrap=n_bootstrap, seed=seed)

    return StatisticalReport(
        method_name=method_name,
        iqm=iqm_result.iqm,
        ci_lower=iqm_result.ci_lower,
        ci_upper=iqm_result.ci_upper,
        median=float(np.median(scores)),
        mean=float(np.mean(scores)),
        std=float(np.std(scores)),
        n_seeds=len(scores),
        scores=scores,
    )


def compare_methods(
    results_dict: dict[str, np.ndarray | list[float]],
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
    seed: int | None = None,
    paired: bool = False,
) -> MethodComparison:
    """Compare multiple methods with comprehensive statistics.

    Computes IQM + CI for each method, runs pairwise statistical tests,
    and generates rankings. Uses Mann-Whitney (unpaired) by default.
    Set paired=True for Wilcoxon signed-rank (requires same sample sizes).

    Args:
        results_dict: Dict mapping method names to score arrays.
        alpha: Significance level for tests.
        n_bootstrap: Number of bootstrap samples.
        seed: Random seed for reproducibility.
        paired: If True, use Wilcoxon signed-rank (paired test) when
            sample sizes match. If False (default), always use
            Mann-Whitney U (unpaired test).

    Returns:
        MethodComparison with reports, tests, and rankings.

    Example:
        >>> results = {
        ...     "ours": [10, 11, 12, 13],
        ...     "baseline": [8, 9, 10, 11],
        ... }
        >>> comparison = compare_methods(results)
        >>> print(comparison.best_method)
        ours
        >>> print(comparison)
        Method Comparison
        ================
        Rankings (by IQM):
        1. ours: 11.50 [10.00, 13.00]
        2. baseline: 9.50 [8.00, 11.00]
        ...
    """
    method_names = list(results_dict.keys())
    n_methods = len(method_names)

    # Create statistical reports for each method
    reports: dict[str, StatisticalReport] = {}
    for name, scores in results_dict.items():
        reports[name] = create_statistical_report(
            name, scores, n_bootstrap=n_bootstrap, seed=seed
        )

    # Rank by IQM
    sorted_methods = sorted(method_names, key=lambda m: reports[m].iqm, reverse=True)
    best_method = sorted_methods[0] if sorted_methods else ""

    # Pairwise tests and probability of improvement
    pairwise_tests: dict[str, HypothesisTestResult] = {}
    poi: dict[str, float] = {}

    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            name_a, name_b = method_names[i], method_names[j]
            scores_a = reports[name_a].scores
            scores_b = reports[name_b].scores

            # Use Wilcoxon only when explicitly paired and same length
            if paired and len(scores_a) == len(scores_b) and len(scores_a) > 0:
                test_result = wilcoxon_test(scores_a, scores_b, alpha=alpha)
            else:
                test_result = mann_whitney_test(scores_a, scores_b, alpha=alpha)

            test_result.method_a = name_a
            test_result.method_b = name_b
            pairwise_tests[f"({name_a}, {name_b})"] = test_result

            # Probability of improvement
            poi_result = probability_of_improvement(
                scores_a, scores_b, n_bootstrap=min(n_bootstrap, 2000), seed=seed
            )
            poi[f"P({name_a} > {name_b})"] = poi_result["prob_x_better"]

    # Generate summary string
    lines = ["Method Comparison", "=" * 40]

    lines.append("\nRankings (by IQM):")
    for i, name in enumerate(sorted_methods, 1):
        r = reports[name]
        lines.append(
            f"  {i}. {name}: {r.iqm:.4f} [{r.ci_lower:.4f}, {r.ci_upper:.4f}]"
        )

    if pairwise_tests:
        lines.append("\nPairwise Tests:")
        for key, test in pairwise_tests.items():
            sig = "*" if test.significant else ""
            lines.append(f"  {key}: p={test.p_value:.4f}{sig}")

    if poi:
        lines.append("\nProbability of Improvement:")
        for key, prob in poi.items():
            lines.append(f"  {key}: {prob:.4f}")

    summary = "\n".join(lines)

    return MethodComparison(
        reports=reports,
        pairwise_tests=pairwise_tests,
        rankings=sorted_methods,
        best_method=best_method,
        probability_of_improvement=poi,
        summary=summary,
    )
