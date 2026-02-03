"""Experiment runner for multi-seed experiments.

Provides infrastructure for running experiments with multiple seeds,
paired seed support for reduced variance, and standardized results
compatible with statistical analysis via rliable.

Example usage:
    >>> from src.experiments.runner import ExperimentConfig, run_experiment
    >>> from src.baselines.ippo import ippo_config, evaluate_ippo
    >>>
    >>> config = ExperimentConfig(
    ...     method_name="ippo",
    ...     n_seeds=20,
    ...     env_config_name="standard",
    ...     paired_seeds=True,
    ... )
    >>>
    >>> def method_fn(seed: int) -> dict:
    ...     # Run evaluation with this seed
    ...     ...
    ...     return {"total_reward": ..., "food_collected": ...}
    >>>
    >>> result = run_experiment(config, method_fn)
    >>> print(f"IQM reward: {result.iqm:.2f}")
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass
class ExperimentConfig:
    """Configuration for a multi-seed experiment.

    Attributes:
        method_name: Name of the method being evaluated (e.g., "ippo", "mappo").
        n_seeds: Number of seeds to run. Default 20 per DR-4 gold standard.
        env_config_name: Environment configuration preset ("standard",
            "hidden_resources", "food_scarcity").
        paired_seeds: If True, use identical seeds across methods for reduced
            variance in comparisons. Seeds are deterministically generated
            from seed_offset.
        seed_offset: Base offset for seed generation. Seeds will be
            [seed_offset, seed_offset + 1, ..., seed_offset + n_seeds - 1].
        n_episodes: Number of episodes per seed for evaluation.
        save_per_seed_results: Whether to save individual seed results
            (increases storage but enables detailed analysis).
    """

    method_name: str
    n_seeds: int = 20
    env_config_name: str = "standard"
    paired_seeds: bool = True
    seed_offset: int = 0
    n_episodes: int = 10
    save_per_seed_results: bool = True


@dataclass
class ExperimentResult:
    """Aggregated results from a multi-seed experiment.

    Stores both aggregate statistics and per-seed results for statistical
    analysis. Compatible with rliable for IQM/bootstrap CI computation.

    Attributes:
        method_name: Name of the method.
        env_config_name: Environment configuration used.
        n_seeds: Number of seeds run.

        # Per-seed results (lists of length n_seeds)
        seed_list: List of seeds used.
        per_seed_rewards: Total reward per seed.
        per_seed_food: Food collected per seed.
        per_seed_population: Final population per seed.
        per_seed_results: Full result dict per seed (if save_per_seed_results=True).

        # Aggregate statistics
        mean_reward: Mean of per_seed_rewards.
        std_reward: Std of per_seed_rewards.
        median_reward: Median of per_seed_rewards.
        iqm_reward: Interquartile mean (25-75 percentile) of rewards.
        ci_lower: Lower bound of 95% bootstrap CI for mean.
        ci_upper: Upper bound of 95% bootstrap CI for mean.

        mean_food: Mean of per_seed_food.
        std_food: Std of per_seed_food.

        mean_population: Mean final population.

        # Metadata
        config: The ExperimentConfig used.
    """

    method_name: str
    env_config_name: str
    n_seeds: int

    # Per-seed results
    seed_list: list[int] = field(default_factory=list)
    per_seed_rewards: list[float] = field(default_factory=list)
    per_seed_food: list[float] = field(default_factory=list)
    per_seed_population: list[int] = field(default_factory=list)
    per_seed_results: list[dict[str, Any]] = field(default_factory=list)

    # Aggregate statistics
    mean_reward: float = 0.0
    std_reward: float = 0.0
    median_reward: float = 0.0
    iqm_reward: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0

    mean_food: float = 0.0
    std_food: float = 0.0

    mean_population: float = 0.0

    # Metadata
    config: ExperimentConfig | None = None

    def __str__(self) -> str:
        return (
            f"ExperimentResult({self.method_name}, {self.env_config_name}, "
            f"n_seeds={self.n_seeds})\n"
            f"  Reward: {self.mean_reward:.2f} ± {self.std_reward:.2f} "
            f"[IQM: {self.iqm_reward:.2f}, CI: ({self.ci_lower:.2f}, {self.ci_upper:.2f})]\n"
            f"  Food: {self.mean_food:.2f} ± {self.std_food:.2f}\n"
            f"  Population: {self.mean_population:.2f}"
        )


def compute_iqm(values: np.ndarray) -> float:
    """Compute Interquartile Mean (25th to 75th percentile).

    Args:
        values: 1D array of values.

    Returns:
        Mean of values between 25th and 75th percentile.
    """
    if len(values) == 0:
        return 0.0

    q25 = np.percentile(values, 25)
    q75 = np.percentile(values, 75)

    # Select values in the interquartile range
    mask = (values >= q25) & (values <= q75)
    iq_values = values[mask]

    if len(iq_values) == 0:
        return float(np.median(values))

    return float(np.mean(iq_values))


def bootstrap_ci_simple(
    values: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean.

    Args:
        values: 1D array of values.
        n_bootstrap: Number of bootstrap samples.
        ci: Confidence level (default 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (ci_lower, ci_upper).
    """
    if len(values) == 0:
        return (0.0, 0.0)

    if len(values) == 1:
        return (float(values[0]), float(values[0]))

    rng = np.random.default_rng(seed)

    boot_means = np.empty(n_bootstrap)
    n = len(values)

    for i in range(n_bootstrap):
        boot_sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = np.mean(boot_sample)

    alpha = (1 - ci) / 2
    ci_lower = float(np.percentile(boot_means, alpha * 100))
    ci_upper = float(np.percentile(boot_means, (1 - alpha) * 100))

    return (ci_lower, ci_upper)


def run_experiment(
    experiment_config: ExperimentConfig,
    method_fn: Callable[[int], dict[str, Any]],
    seed_offset: int | None = None,
    verbose: bool = False,
) -> ExperimentResult:
    """Run a multi-seed experiment.

    Runs the provided method function for each seed, aggregates results,
    and computes statistics.

    Args:
        experiment_config: Configuration for the experiment.
        method_fn: Function that takes a seed (int) and returns a result dict.
            The dict must contain at least:
            - "total_reward": float
            - "food_collected": float
            - "final_population": int (or float)
            Optionally:
            - "per_agent_rewards": list[float]
        seed_offset: Override for config's seed_offset. If provided, uses this
            instead of config.seed_offset. Useful for paired experiments.
        verbose: If True, print progress during experiment.

    Returns:
        ExperimentResult with per-seed and aggregate statistics.
    """
    # Determine seeds to use
    offset = seed_offset if seed_offset is not None else experiment_config.seed_offset
    seeds = list(range(offset, offset + experiment_config.n_seeds))

    # Storage for results
    per_seed_rewards: list[float] = []
    per_seed_food: list[float] = []
    per_seed_population: list[int] = []
    per_seed_results: list[dict[str, Any]] = []

    # Run experiments
    for i, seed in enumerate(seeds):
        if verbose:
            print(
                f"[{experiment_config.method_name}] Running seed {seed} "
                f"({i + 1}/{experiment_config.n_seeds})..."
            )

        result = method_fn(seed)

        # Extract required fields
        per_seed_rewards.append(float(result["total_reward"]))
        per_seed_food.append(float(result["food_collected"]))
        per_seed_population.append(int(result["final_population"]))

        if experiment_config.save_per_seed_results:
            per_seed_results.append(result)

    # Convert to numpy for statistics
    rewards_array = np.array(per_seed_rewards)
    food_array = np.array(per_seed_food)
    pop_array = np.array(per_seed_population)

    # Compute aggregate statistics
    mean_reward = float(np.mean(rewards_array))
    std_reward = float(np.std(rewards_array))
    median_reward = float(np.median(rewards_array))
    iqm_reward = compute_iqm(rewards_array)
    ci_lower, ci_upper = bootstrap_ci_simple(rewards_array, seed=42)

    mean_food = float(np.mean(food_array))
    std_food = float(np.std(food_array))

    mean_population = float(np.mean(pop_array))

    return ExperimentResult(
        method_name=experiment_config.method_name,
        env_config_name=experiment_config.env_config_name,
        n_seeds=experiment_config.n_seeds,
        seed_list=seeds,
        per_seed_rewards=per_seed_rewards,
        per_seed_food=per_seed_food,
        per_seed_population=per_seed_population,
        per_seed_results=per_seed_results,
        mean_reward=mean_reward,
        std_reward=std_reward,
        median_reward=median_reward,
        iqm_reward=iqm_reward,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        mean_food=mean_food,
        std_food=std_food,
        mean_population=mean_population,
        config=experiment_config,
    )


def save_experiment_result(result: ExperimentResult, path: str | Path) -> None:
    """Save ExperimentResult to a pickle file.

    Args:
        result: ExperimentResult to save.
        path: File path (will be created if doesn't exist).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(result, f)


def load_experiment_result(path: str | Path) -> ExperimentResult:
    """Load ExperimentResult from a pickle file.

    Args:
        path: Path to pickle file.

    Returns:
        Loaded ExperimentResult.

    Raises:
        FileNotFoundError: If path doesn't exist.
    """
    path = Path(path)

    with open(path, "rb") as f:
        result = pickle.load(f)

    if not isinstance(result, ExperimentResult):
        raise TypeError(f"Expected ExperimentResult, got {type(result)}")

    return result


def run_paired_experiment(
    experiment_configs: list[ExperimentConfig],
    method_fns: list[Callable[[int], dict[str, Any]]],
    seed_offset: int = 0,
    verbose: bool = False,
) -> list[ExperimentResult]:
    """Run multiple methods with paired seeds for reduced variance.

    Uses the same seeds across all methods to enable paired statistical
    comparisons (e.g., Wilcoxon signed-rank test).

    Args:
        experiment_configs: List of ExperimentConfigs (one per method).
            All should have paired_seeds=True and same n_seeds.
        method_fns: List of method functions corresponding to configs.
        seed_offset: Base seed offset for all methods.
        verbose: If True, print progress.

    Returns:
        List of ExperimentResults (one per method).

    Raises:
        ValueError: If configs have different n_seeds values.
    """
    if len(experiment_configs) != len(method_fns):
        raise ValueError(
            f"Number of configs ({len(experiment_configs)}) must match "
            f"number of method_fns ({len(method_fns)})"
        )

    if len(experiment_configs) == 0:
        return []

    # Verify all configs have same n_seeds
    n_seeds = experiment_configs[0].n_seeds
    for i, config in enumerate(experiment_configs):
        if config.n_seeds != n_seeds:
            raise ValueError(
                f"All configs must have same n_seeds. Config 0 has {n_seeds}, "
                f"config {i} has {config.n_seeds}"
            )

    # Run each method with same seed offset
    results = []
    for config, method_fn in zip(experiment_configs, method_fns):
        result = run_experiment(
            config, method_fn, seed_offset=seed_offset, verbose=verbose
        )
        results.append(result)

    return results


def compare_experiment_results(
    results: list[ExperimentResult],
) -> dict[str, Any]:
    """Compare multiple experiment results with summary statistics.

    Args:
        results: List of ExperimentResults to compare.

    Returns:
        Dict with comparison summary including:
        - rankings: Methods ranked by IQM reward
        - pairwise_diffs: Mean difference between each pair
        - best_method: Name of method with highest IQM
    """
    if len(results) == 0:
        return {"rankings": [], "pairwise_diffs": {}, "best_method": None}

    # Rank by IQM reward
    sorted_results = sorted(results, key=lambda r: r.iqm_reward, reverse=True)
    rankings = [
        {
            "rank": i + 1,
            "method": r.method_name,
            "iqm_reward": r.iqm_reward,
            "mean_reward": r.mean_reward,
            "ci": (r.ci_lower, r.ci_upper),
        }
        for i, r in enumerate(sorted_results)
    ]

    # Pairwise differences
    pairwise_diffs: dict[str, dict[str, float]] = {}
    for r1 in results:
        pairwise_diffs[r1.method_name] = {}
        for r2 in results:
            if r1.method_name != r2.method_name:
                diff = r1.mean_reward - r2.mean_reward
                pairwise_diffs[r1.method_name][r2.method_name] = diff

    best_method = sorted_results[0].method_name if sorted_results else None

    return {
        "rankings": rankings,
        "pairwise_diffs": pairwise_diffs,
        "best_method": best_method,
    }
