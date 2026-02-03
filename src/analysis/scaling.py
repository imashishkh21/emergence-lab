"""Superlinear scaling analysis for emergence experiments.

This module tests whether agents achieve superlinear scaling — where adding
more agents increases per-agent efficiency due to field-mediated coordination.

Key metrics:
    - Per-agent efficiency: F_total(N) / (N * F_solo)
    - Power law fit: log(F_total) = alpha * log(N) + c
        - alpha > 1.0 => superlinear (field helps more with more agents)
        - alpha = 1.0 => linear (no coordination benefit)
        - alpha < 1.0 => sublinear (crowding hurts)

Reference: Hamann (2018), "Swarm Robotics: A Formal Approach"

Example usage:
    >>> from src.analysis.scaling import compute_per_agent_efficiency, fit_power_law
    >>>
    >>> # Test scaling with different agent counts
    >>> n_agents_list = [1, 2, 4, 8, 16, 32]
    >>> total_food_list = [10, 25, 60, 150, 400, 1000]  # Example superlinear data
    >>> solo_food = 10  # Food collected by single agent
    >>>
    >>> efficiency = compute_per_agent_efficiency(150, 8, solo_food)
    >>> print(f"Efficiency with 8 agents: {efficiency:.2f}x")
    >>>
    >>> alpha, alpha_ci, c = fit_power_law(n_agents_list, total_food_list)
    >>> print(f"Power law exponent: {alpha:.3f} (superlinear: {alpha > 1.0})")
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class ScalingResult:
    """Results from a scaling experiment at a specific agent count.

    Attributes:
        n_agents: Number of agents in this experiment.
        field_condition: Field condition used ("normal", "zeroed", "no_field").
        total_food: Total food collected across all agents.
        per_agent_food: Food collected divided by number of agents.
        efficiency: Per-agent efficiency relative to solo baseline.
        total_reward: Total reward across all agents.
        per_agent_reward: Reward divided by number of agents.
        seed: Random seed used for this result.
        episode_foods: Per-episode food collected (for variance analysis).
    """

    n_agents: int
    field_condition: str
    total_food: float
    per_agent_food: float
    efficiency: float
    total_reward: float = 0.0
    per_agent_reward: float = 0.0
    seed: int = 0
    episode_foods: list[float] = field(default_factory=list)


@dataclass
class ScalingAnalysis:
    """Aggregated results from a full scaling experiment.

    Attributes:
        field_condition: Field condition used.
        n_agents_list: List of agent counts tested.
        solo_food: Baseline food collected by single agent.

        # Per-N statistics (lists of length len(n_agents_list))
        mean_foods: Mean total food at each N.
        std_foods: Std of total food at each N.
        mean_efficiencies: Mean efficiency at each N.
        std_efficiencies: Std of efficiency at each N.
        per_agent_foods: Mean per-agent food at each N.

        # Power law fit
        alpha: Power law exponent.
        alpha_ci_lower: Lower bound of 95% CI for alpha.
        alpha_ci_upper: Upper bound of 95% CI for alpha.
        c: Power law constant (log-space intercept).
        r_squared: R-squared of the fit.

        # Raw results
        results: All ScalingResult objects.
    """

    field_condition: str
    n_agents_list: list[int] = field(default_factory=list)
    solo_food: float = 0.0

    # Per-N statistics
    mean_foods: list[float] = field(default_factory=list)
    std_foods: list[float] = field(default_factory=list)
    mean_efficiencies: list[float] = field(default_factory=list)
    std_efficiencies: list[float] = field(default_factory=list)
    per_agent_foods: list[float] = field(default_factory=list)

    # Power law fit
    alpha: float = 0.0
    alpha_ci_lower: float = 0.0
    alpha_ci_upper: float = 0.0
    c: float = 0.0
    r_squared: float = 0.0

    # Raw results
    results: list[ScalingResult] = field(default_factory=list)

    def is_superlinear(self) -> bool:
        """Check if scaling is superlinear (alpha > 1.0)."""
        return self.alpha > 1.0

    def is_sublinear(self) -> bool:
        """Check if scaling is sublinear (alpha < 1.0)."""
        return self.alpha < 1.0

    def __str__(self) -> str:
        scaling_type = "superlinear" if self.is_superlinear() else (
            "sublinear" if self.is_sublinear() else "linear"
        )
        return (
            f"ScalingAnalysis({self.field_condition})\n"
            f"  Agent counts: {self.n_agents_list}\n"
            f"  Power law: alpha = {self.alpha:.3f} [{self.alpha_ci_lower:.3f}, {self.alpha_ci_upper:.3f}]\n"
            f"  Scaling type: {scaling_type}\n"
            f"  R-squared: {self.r_squared:.3f}"
        )


def compute_per_agent_efficiency(
    total_food: float,
    n_agents: int,
    solo_food: float,
) -> float:
    """Compute per-agent efficiency relative to solo baseline.

    Efficiency = F_total(N) / (N * F_solo)

    This measures how much more effective agents are when working together
    compared to working independently.

    Args:
        total_food: Total food collected by N agents working together.
        n_agents: Number of agents.
        solo_food: Baseline food collected by a single agent working alone.

    Returns:
        Efficiency ratio. Values > 1.0 indicate coordination benefit,
        < 1.0 indicates crowding cost.

    Raises:
        ValueError: If n_agents or solo_food is <= 0.
    """
    if n_agents <= 0:
        raise ValueError(f"n_agents must be positive, got {n_agents}")
    if solo_food <= 0:
        raise ValueError(f"solo_food must be positive, got {solo_food}")

    expected_independent = n_agents * solo_food
    return total_food / expected_independent


def fit_power_law(
    n_agents_list: list[int],
    total_food_list: list[float],
    bootstrap_n: int = 1000,
    seed: int | None = None,
) -> tuple[float, tuple[float, float], float]:
    """Fit power law: log(F_total) = alpha * log(N) + c.

    Uses linear regression in log-log space. The exponent alpha indicates:
        - alpha > 1.0: superlinear scaling (coordination benefit)
        - alpha = 1.0: linear scaling (no coordination effect)
        - alpha < 1.0: sublinear scaling (crowding/competition)

    Args:
        n_agents_list: List of agent counts (must be positive integers).
        total_food_list: List of corresponding total food values (must be positive).
        bootstrap_n: Number of bootstrap samples for CI estimation.
        seed: Random seed for bootstrap sampling.

    Returns:
        Tuple of (alpha, (ci_lower, ci_upper), c) where:
            - alpha: Power law exponent
            - (ci_lower, ci_upper): 95% bootstrap confidence interval for alpha
            - c: Log-space intercept

    Raises:
        ValueError: If lists have different lengths, fewer than 2 points,
                   or contain non-positive values.
    """
    if len(n_agents_list) != len(total_food_list):
        raise ValueError(
            f"Lists must have same length: {len(n_agents_list)} vs {len(total_food_list)}"
        )

    if len(n_agents_list) < 2:
        raise ValueError("Need at least 2 data points for fitting")

    n_agents_arr = np.array(n_agents_list, dtype=np.float64)
    total_food_arr = np.array(total_food_list, dtype=np.float64)

    if np.any(n_agents_arr <= 0):
        raise ValueError("All n_agents values must be positive")
    if np.any(total_food_arr <= 0):
        raise ValueError("All total_food values must be positive")

    # Log-log regression
    log_n = np.log(n_agents_arr)
    log_food = np.log(total_food_arr)

    # Fit using linear regression
    slope, intercept, _, _, _ = stats.linregress(log_n, log_food)
    alpha = float(slope)
    c = float(intercept)

    # Bootstrap CI for alpha
    rng = np.random.default_rng(seed)
    n_points = len(n_agents_list)
    boot_alphas_list: list[float] = []

    for i in range(bootstrap_n):
        indices = rng.choice(n_points, size=n_points, replace=True)
        boot_log_n = log_n[indices]
        boot_log_food = log_food[indices]

        # Skip if all x values are identical (can't fit regression)
        if np.all(boot_log_n == boot_log_n[0]):
            continue

        try:
            boot_slope, _, _, _, _ = stats.linregress(boot_log_n, boot_log_food)
            boot_alphas_list.append(boot_slope)
        except ValueError:
            # Skip failed regressions
            continue

    # Handle case where no bootstrap samples worked
    if len(boot_alphas_list) == 0:
        ci_lower = alpha
        ci_upper = alpha
    else:
        boot_alphas = np.array(boot_alphas_list)
        ci_lower = float(np.percentile(boot_alphas, 2.5))
        ci_upper = float(np.percentile(boot_alphas, 97.5))

    return alpha, (ci_lower, ci_upper), c


def compute_r_squared(
    n_agents_list: list[int],
    total_food_list: list[float],
    alpha: float,
    c: float,
) -> float:
    """Compute R-squared for power law fit.

    Args:
        n_agents_list: List of agent counts.
        total_food_list: List of total food values.
        alpha: Fitted power law exponent.
        c: Fitted log-space intercept.

    Returns:
        R-squared value (0 to 1, higher is better fit).
    """
    log_n = np.log(np.array(n_agents_list, dtype=np.float64))
    log_food = np.log(np.array(total_food_list, dtype=np.float64))

    # Predicted values
    log_food_pred = alpha * log_n + c

    # R-squared
    ss_res = np.sum((log_food - log_food_pred) ** 2)
    ss_tot = np.sum((log_food - np.mean(log_food)) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return float(1 - ss_res / ss_tot)


def aggregate_scaling_results(
    results: list[ScalingResult],
    n_agents_list: list[int],
    solo_food: float | None = None,
) -> ScalingAnalysis:
    """Aggregate scaling results across seeds for a single field condition.

    Groups results by n_agents, computes statistics, and fits power law.

    Args:
        results: List of ScalingResult objects (can be from multiple seeds).
        n_agents_list: Canonical list of agent counts to aggregate.
        solo_food: Override for solo baseline. If None, uses results from N=1.

    Returns:
        ScalingAnalysis with aggregated statistics and power law fit.

    Raises:
        ValueError: If results list is empty or contains inconsistent conditions.
    """
    if len(results) == 0:
        raise ValueError("results list cannot be empty")

    # Check all results have same field condition
    conditions = set(r.field_condition for r in results)
    if len(conditions) > 1:
        raise ValueError(f"All results must have same field_condition, got: {conditions}")
    field_condition = results[0].field_condition

    # Group results by n_agents
    by_n_agents: dict[int, list[ScalingResult]] = {n: [] for n in n_agents_list}
    for r in results:
        if r.n_agents in by_n_agents:
            by_n_agents[r.n_agents].append(r)

    # Compute solo baseline
    if solo_food is None:
        if 1 in by_n_agents and len(by_n_agents[1]) > 0:
            solo_food = float(np.mean([r.total_food for r in by_n_agents[1]]))
        else:
            # Estimate from smallest N
            min_n = min(n for n in n_agents_list if len(by_n_agents[n]) > 0)
            min_n_results = by_n_agents[min_n]
            solo_food = float(np.mean([r.total_food for r in min_n_results])) / min_n

    # Compute per-N statistics
    mean_foods: list[float] = []
    std_foods: list[float] = []
    mean_efficiencies: list[float] = []
    std_efficiencies: list[float] = []
    per_agent_foods: list[float] = []

    for n in n_agents_list:
        n_results = by_n_agents[n]
        if len(n_results) == 0:
            mean_foods.append(0.0)
            std_foods.append(0.0)
            mean_efficiencies.append(0.0)
            std_efficiencies.append(0.0)
            per_agent_foods.append(0.0)
            continue

        foods = np.array([r.total_food for r in n_results])
        effs = np.array([r.efficiency for r in n_results])

        mean_foods.append(float(np.mean(foods)))
        std_foods.append(float(np.std(foods)))
        mean_efficiencies.append(float(np.mean(effs)))
        std_efficiencies.append(float(np.std(effs)))
        per_agent_foods.append(float(np.mean(foods)) / n)

    # Fit power law (only on N values with data)
    valid_n = [n for n in n_agents_list if len(by_n_agents[n]) > 0]
    valid_foods = [mean_foods[n_agents_list.index(n)] for n in valid_n]

    if len(valid_n) >= 2 and all(f > 0 for f in valid_foods):
        alpha, (ci_lower, ci_upper), c = fit_power_law(valid_n, valid_foods)
        r_squared = compute_r_squared(valid_n, valid_foods, alpha, c)
    else:
        alpha = 1.0
        ci_lower = 1.0
        ci_upper = 1.0
        c = 0.0
        r_squared = 0.0

    return ScalingAnalysis(
        field_condition=field_condition,
        n_agents_list=n_agents_list,
        solo_food=solo_food,
        mean_foods=mean_foods,
        std_foods=std_foods,
        mean_efficiencies=mean_efficiencies,
        std_efficiencies=std_efficiencies,
        per_agent_foods=per_agent_foods,
        alpha=alpha,
        alpha_ci_lower=ci_lower,
        alpha_ci_upper=ci_upper,
        c=c,
        r_squared=r_squared,
        results=results,
    )


def print_scaling_analysis(analysis: ScalingAnalysis) -> None:
    """Pretty-print scaling analysis results."""
    print("=" * 70)
    print(f"Scaling Analysis: {analysis.field_condition}")
    print("=" * 70)

    # Print per-N statistics
    print(f"{'N':>4} {'Mean Food':>12} {'Std':>10} {'Per-Agent':>12} {'Efficiency':>12}")
    print("-" * 70)
    for i, n in enumerate(analysis.n_agents_list):
        mean_food = analysis.mean_foods[i]
        std_food = analysis.std_foods[i]
        per_agent = analysis.per_agent_foods[i]
        efficiency = analysis.mean_efficiencies[i]
        print(f"{n:>4} {mean_food:>12.2f} {std_food:>10.2f} {per_agent:>12.2f} {efficiency:>12.2f}x")

    print("-" * 70)

    # Power law summary
    scaling_type = "SUPERLINEAR" if analysis.is_superlinear() else (
        "SUBLINEAR" if analysis.is_sublinear() else "LINEAR"
    )
    print(f"\nPower Law Fit: F = {np.exp(analysis.c):.2f} * N^{analysis.alpha:.3f}")
    print(f"  Alpha: {analysis.alpha:.3f} [{analysis.alpha_ci_lower:.3f}, {analysis.alpha_ci_upper:.3f}]")
    print(f"  R-squared: {analysis.r_squared:.3f}")
    print(f"  Scaling type: {scaling_type}")

    if analysis.is_superlinear():
        print("  -> Field-mediated coordination provides BENEFIT with more agents")
    elif analysis.is_sublinear():
        print("  -> Crowding/competition causes COST with more agents")
    else:
        print("  -> No coordination effect detected")

    print("=" * 70)


def compare_scaling_analyses(
    analyses: dict[str, ScalingAnalysis],
) -> dict[str, Any]:
    """Compare scaling analyses across field conditions.

    Args:
        analyses: Dict mapping field_condition to ScalingAnalysis.

    Returns:
        Dict with comparison summary including:
            - rankings: Conditions ranked by alpha
            - alpha_comparison: Table of alpha values
            - superlinear_conditions: Conditions with alpha > 1.0
            - best_condition: Condition with highest alpha
    """
    if len(analyses) == 0:
        return {
            "rankings": [],
            "alpha_comparison": {},
            "superlinear_conditions": [],
            "best_condition": None,
        }

    # Rank by alpha
    sorted_conditions = sorted(
        analyses.keys(),
        key=lambda c: analyses[c].alpha,
        reverse=True,
    )

    rankings = [
        {
            "rank": i + 1,
            "condition": c,
            "alpha": analyses[c].alpha,
            "ci": (analyses[c].alpha_ci_lower, analyses[c].alpha_ci_upper),
            "r_squared": analyses[c].r_squared,
        }
        for i, c in enumerate(sorted_conditions)
    ]

    alpha_comparison = {c: a.alpha for c, a in analyses.items()}
    superlinear_conditions = [c for c, a in analyses.items() if a.is_superlinear()]
    best_condition = sorted_conditions[0] if sorted_conditions else None

    return {
        "rankings": rankings,
        "alpha_comparison": alpha_comparison,
        "superlinear_conditions": superlinear_conditions,
        "best_condition": best_condition,
    }


def print_scaling_comparison(analyses: dict[str, ScalingAnalysis]) -> None:
    """Print comparison of scaling analyses across conditions."""
    comparison = compare_scaling_analyses(analyses)

    print("\n" + "=" * 70)
    print("SCALING COMPARISON ACROSS FIELD CONDITIONS")
    print("=" * 70)

    print(f"{'Rank':>4} {'Condition':>12} {'Alpha':>10} {'CI':>24} {'R²':>8}")
    print("-" * 70)

    for r in comparison["rankings"]:
        ci_str = f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]"
        superlinear_marker = "*" if r["alpha"] > 1.0 else ""
        print(
            f"{r['rank']:>4} {r['condition']:>12} {r['alpha']:>9.3f}{superlinear_marker} "
            f"{ci_str:>24} {r['r_squared']:>8.3f}"
        )

    print("-" * 70)
    print("* indicates superlinear scaling (alpha > 1.0)")

    if comparison["superlinear_conditions"]:
        print(f"\nSuperlinear conditions: {comparison['superlinear_conditions']}")
    else:
        print("\nNo conditions showed superlinear scaling")

    if comparison["best_condition"]:
        best = comparison["best_condition"]
        best_alpha = analyses[best].alpha
        print(f"Best condition: {best} (alpha = {best_alpha:.3f})")

        # Compare to no_field if available
        if "no_field" in analyses and best != "no_field":
            no_field_alpha = analyses["no_field"].alpha
            benefit = best_alpha - no_field_alpha
            print(f"  Field benefit over no_field: {benefit:+.3f}")

    print("=" * 70)
