#!/usr/bin/env python
"""Generate a comprehensive multi-seed analysis report.

Loads checkpoints from all seeds (up to 30), runs analysis on each, and
produces aggregate statistics with confidence intervals.

Usage:
    python scripts/generate_multi_seed_report.py --checkpoint-dir checkpoints/parallel
    python scripts/generate_multi_seed_report.py --checkpoint-dir /path/to/checkpoints --output-dir reports/multi_seed
    python scripts/generate_multi_seed_report.py --checkpoint-dir checkpoints/field_on --compare-dir checkpoints/field_off

Example directory structure expected:
    checkpoints/parallel/
        batch_0/
            seed_0/latest.pkl
            seed_1/latest.pkl
            ...
        batch_1/
            seed_5/latest.pkl
            ...
    OR
    checkpoints/parallel/
        seed_0/latest.pkl
        seed_1/latest.pkl
        ...
"""

import argparse
import glob
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from src.agents.network import ActorCritic
from src.analysis.ablation import specialization_ablation_test
from src.analysis.specialization import (
    compute_weight_divergence,
    extract_behavior_features,
    find_optimal_clusters,
    specialization_score,
)
from src.analysis.trajectory import record_episode
from src.configs import Config
from src.environment.obs import obs_dim


@dataclass
class SeedResult:
    """Analysis results for a single seed."""
    seed_id: int
    step: int
    weight_divergence_mean: float
    weight_divergence_max: float
    specialization_score: float
    silhouette_score: float
    optimal_k: int
    num_alive: int
    mean_reward: float | None = None
    food_collected: float | None = None


@dataclass
class AggregateStats:
    """Aggregate statistics across seeds."""
    metric_name: str
    mean: float
    std: float
    ci_low: float
    ci_high: float
    min_val: float
    max_val: float
    n_samples: int


def compute_confidence_interval(
    values: list[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute confidence interval for a list of values.

    Args:
        values: List of numeric values.
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        Tuple of (lower bound, upper bound).
    """
    if len(values) < 2:
        return (values[0] if values else 0.0, values[0] if values else 0.0)

    arr = np.array(values)
    mean = np.mean(arr)
    sem = stats.sem(arr)
    ci = stats.t.interval(confidence, len(arr) - 1, loc=mean, scale=sem)
    return (float(ci[0]), float(ci[1]))


def load_checkpoint(path: str) -> dict[str, Any]:
    """Load a checkpoint file."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Convert numpy back to JAX
    def to_jax(x: Any) -> Any:
        if isinstance(x, np.ndarray):
            return jnp.array(x)
        return x

    return jax.tree_util.tree_map(to_jax, data)


def find_all_seed_checkpoints(checkpoint_dir: str) -> list[tuple[int, str]]:
    """Find all seed checkpoints in a directory.

    Supports both flat structure (seed_N/latest.pkl) and batch structure
    (batch_N/seed_N/latest.pkl).

    Args:
        checkpoint_dir: Base checkpoint directory.

    Returns:
        List of (seed_id, checkpoint_path) tuples, sorted by seed_id.
    """
    checkpoints = []

    # Try batch structure first
    batch_dirs = sorted(glob.glob(os.path.join(checkpoint_dir, "batch_*")))
    if batch_dirs:
        for batch_dir in batch_dirs:
            seed_dirs = glob.glob(os.path.join(batch_dir, "seed_*"))
            for seed_dir in seed_dirs:
                latest = os.path.join(seed_dir, "latest.pkl")
                if os.path.exists(latest):
                    # Extract seed ID from directory name
                    seed_id = int(os.path.basename(seed_dir).split("_")[1])
                    checkpoints.append((seed_id, latest))

    # Also try flat structure
    flat_seed_dirs = glob.glob(os.path.join(checkpoint_dir, "seed_*"))
    for seed_dir in flat_seed_dirs:
        if os.path.isdir(seed_dir):
            latest = os.path.join(seed_dir, "latest.pkl")
            if os.path.exists(latest):
                seed_id = int(os.path.basename(seed_dir).split("_")[1])
                # Avoid duplicates
                if not any(s == seed_id for s, _ in checkpoints):
                    checkpoints.append((seed_id, latest))

    return sorted(checkpoints, key=lambda x: x[0])


def analyze_seed(
    seed_id: int,
    checkpoint_path: str,
    config: Config,
    num_episodes: int = 3,
) -> SeedResult:
    """Analyze a single seed checkpoint.

    Args:
        seed_id: The seed ID.
        checkpoint_path: Path to the checkpoint file.
        config: Master configuration.
        num_episodes: Number of episodes for trajectory recording.

    Returns:
        SeedResult with analysis metrics.
    """
    print(f"  Analyzing seed {seed_id}...")

    # Load checkpoint
    data = load_checkpoint(checkpoint_path)
    params = data["params"]
    agent_params = data.get("agent_params")
    step = data.get("step", 0)

    # Create network
    network = ActorCritic(
        hidden_dims=tuple(config.agent.hidden_dims),
        num_actions=config.agent.num_actions,
    )

    # Verify params
    observation_dim = obs_dim(config)
    dummy_obs = jnp.zeros((observation_dim,))
    network.apply(params, dummy_obs)

    # Initialize result with defaults
    result = SeedResult(
        seed_id=seed_id,
        step=step,
        weight_divergence_mean=0.0,
        weight_divergence_max=0.0,
        specialization_score=0.0,
        silhouette_score=0.0,
        optimal_k=1,
        num_alive=0,
    )

    if agent_params is None:
        print(f"    WARNING: No per-agent params found for seed {seed_id}")
        return result

    # Create alive mask (assume all agents in checkpoint are alive)
    max_agents = config.evolution.max_agents
    alive_mask = np.ones(max_agents, dtype=bool)

    # Weight divergence
    div_result = compute_weight_divergence(agent_params, alive_mask)
    result.weight_divergence_mean = div_result["mean_divergence"]
    result.weight_divergence_max = div_result["max_divergence"]
    result.num_alive = len(div_result["agent_indices"])

    # Record trajectories and extract features
    all_features = []
    for ep in range(num_episodes):
        ep_key = jax.random.PRNGKey(seed_id * 1000 + ep)
        trajectory = record_episode(network, params, config, ep_key)
        features = extract_behavior_features(trajectory)
        all_features.append(features)

    avg_features = np.mean(all_features, axis=0)

    # Specialization score
    spec_result = specialization_score(
        avg_features,
        agent_params=agent_params,
        alive_mask=alive_mask,
    )
    result.specialization_score = spec_result["score"]
    result.silhouette_score = spec_result["silhouette_component"]
    result.optimal_k = spec_result["optimal_k"]

    print(f"    Weight divergence: {result.weight_divergence_mean:.6f}")
    print(f"    Specialization score: {result.specialization_score:.4f}")

    return result


def generate_report(
    seed_results: list[SeedResult],
    output_dir: Path,
    condition_name: str = "field_on",
    comparison_results: list[SeedResult] | None = None,
    comparison_name: str = "field_off",
) -> str:
    """Generate a markdown report from seed results.

    Args:
        seed_results: List of results from each seed.
        output_dir: Output directory for report and figures.
        condition_name: Name of the condition being analyzed.
        comparison_results: Optional results from comparison condition.
        comparison_name: Name of the comparison condition.

    Returns:
        The generated report text.
    """
    lines = []

    lines.append("# Multi-Seed Analysis Report")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Summary statistics
    lines.append("## Summary Statistics")
    lines.append("")

    n_seeds = len(seed_results)
    lines.append(f"**Seeds analyzed**: {n_seeds}")
    lines.append("")

    # Compute aggregate stats for each metric
    metrics = {
        "Weight Divergence (mean)": [r.weight_divergence_mean for r in seed_results],
        "Weight Divergence (max)": [r.weight_divergence_max for r in seed_results],
        "Specialization Score": [r.specialization_score for r in seed_results],
        "Silhouette Score": [r.silhouette_score for r in seed_results],
        "Optimal Clusters (k)": [float(r.optimal_k) for r in seed_results],
        "Alive Agents": [float(r.num_alive) for r in seed_results],
    }

    lines.append("| Metric | Mean | Std | 95% CI | Min | Max |")
    lines.append("|--------|------|-----|--------|-----|-----|")

    aggregate_stats = {}
    for metric_name, values in metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        ci_low, ci_high = compute_confidence_interval(values)
        min_val = np.min(values)
        max_val = np.max(values)

        aggregate_stats[metric_name] = AggregateStats(
            metric_name=metric_name,
            mean=mean,
            std=std,
            ci_low=ci_low,
            ci_high=ci_high,
            min_val=min_val,
            max_val=max_val,
            n_samples=n_seeds,
        )

        lines.append(
            f"| {metric_name} | {mean:.4f} | {std:.4f} | "
            f"[{ci_low:.4f}, {ci_high:.4f}] | {min_val:.4f} | {max_val:.4f} |"
        )

    lines.append("")

    # Per-seed results table
    lines.append("## Per-Seed Results")
    lines.append("")
    lines.append("| Seed | Step | Weight Div | Spec Score | Silhouette | k | Alive |")
    lines.append("|------|------|------------|------------|------------|---|-------|")

    for r in sorted(seed_results, key=lambda x: x.seed_id):
        lines.append(
            f"| {r.seed_id} | {r.step:,} | {r.weight_divergence_mean:.6f} | "
            f"{r.specialization_score:.4f} | {r.silhouette_score:.4f} | "
            f"{r.optimal_k} | {r.num_alive} |"
        )

    lines.append("")

    # Comparison section (if provided)
    if comparison_results:
        lines.append("## Condition Comparison")
        lines.append("")
        lines.append(f"Comparing **{condition_name}** (N={len(seed_results)}) vs "
                     f"**{comparison_name}** (N={len(comparison_results)})")
        lines.append("")

        comp_metrics = {
            "Weight Divergence": (
                [r.weight_divergence_mean for r in seed_results],
                [r.weight_divergence_mean for r in comparison_results],
            ),
            "Specialization Score": (
                [r.specialization_score for r in seed_results],
                [r.specialization_score for r in comparison_results],
            ),
            "Silhouette Score": (
                [r.silhouette_score for r in seed_results],
                [r.silhouette_score for r in comparison_results],
            ),
        }

        lines.append(f"| Metric | {condition_name} | {comparison_name} | Difference | p-value |")
        lines.append("|--------|-----------------|------------------|------------|---------|")

        for metric_name, (vals1, vals2) in comp_metrics.items():
            mean1 = np.mean(vals1)
            mean2 = np.mean(vals2)
            diff = mean1 - mean2

            # Two-sample t-test
            if len(vals1) >= 2 and len(vals2) >= 2:
                t_stat, p_val = stats.ttest_ind(vals1, vals2)
                p_str = f"{p_val:.4f}" if p_val >= 0.001 else "<0.001"
            else:
                p_str = "N/A"

            sig = "*" if len(vals1) >= 2 and len(vals2) >= 2 and p_val < 0.05 else ""
            lines.append(
                f"| {metric_name} | {mean1:.4f} | {mean2:.4f} | "
                f"{diff:+.4f}{sig} | {p_str} |"
            )

        lines.append("")
        lines.append("*Note: * indicates p < 0.05*")
        lines.append("")

        # Interpretation
        lines.append("### Interpretation")
        lines.append("")

        spec_vals1 = [r.specialization_score for r in seed_results]
        spec_vals2 = [r.specialization_score for r in comparison_results]
        spec_diff = np.mean(spec_vals1) - np.mean(spec_vals2)

        if spec_diff > 0.01:
            lines.append(
                f"> **{condition_name}** shows higher specialization scores, "
                f"suggesting the condition promotes behavioral differentiation."
            )
        elif spec_diff < -0.01:
            lines.append(
                f"> **{comparison_name}** shows higher specialization scores, "
                f"suggesting the comparison condition better promotes specialization."
            )
        else:
            lines.append(
                "> Both conditions show similar specialization levels. "
                "The manipulation may not significantly affect emergence."
            )
        lines.append("")

    # Statistical validity section
    lines.append("## Statistical Validity")
    lines.append("")
    lines.append(f"- **Sample size**: N = {n_seeds} independent seeds")
    lines.append("- **Confidence intervals**: 95% CI using t-distribution")
    if comparison_results:
        lines.append("- **Comparison test**: Two-sample independent t-test")
    lines.append("")

    if n_seeds < 10:
        lines.append(
            "> **Warning**: Small sample size (N < 10). "
            "Results may have high variance. Consider running more seeds."
        )
    elif n_seeds < 30:
        lines.append(
            "> **Note**: Moderate sample size. Results are indicative but "
            "may benefit from additional seeds for stronger statistical power."
        )
    else:
        lines.append(
            "> **Good**: Large sample size (N >= 30) provides strong "
            "statistical power for reliable conclusions."
        )
    lines.append("")

    # Methodology section
    lines.append("## Methodology")
    lines.append("")
    lines.append("Each seed was analyzed as follows:")
    lines.append("")
    lines.append("1. **Weight divergence**: Pairwise cosine distance between agent weight vectors")
    lines.append("2. **Behavioral features**: Extracted from recorded episodes (movement entropy, "
                 "food collection rate, reproduction rate, etc.)")
    lines.append("3. **Specialization score**: Composite of silhouette score (50%), "
                 "weight divergence (25%), and behavioral variance (25%)")
    lines.append("4. **Optimal k**: Number of behavioral clusters determined by silhouette analysis")
    lines.append("")

    report_text = "\n".join(lines)

    # Write report
    report_path = output_dir / "multi_seed_report.md"
    report_path.write_text(report_text)

    return report_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate multi-seed analysis report"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory containing seed checkpoints",
    )
    parser.add_argument(
        "--compare-dir",
        type=str,
        default=None,
        help="Optional: Directory with comparison condition checkpoints (e.g., field_off)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/multi_seed",
        help="Output directory for report",
    )
    parser.add_argument(
        "--condition-name",
        type=str,
        default="field_on",
        help="Name of the primary condition",
    )
    parser.add_argument(
        "--compare-name",
        type=str,
        default="field_off",
        help="Name of the comparison condition",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=3,
        help="Number of episodes per seed for feature extraction",
    )
    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    if args.config is not None:
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    config.log.wandb = False

    print("=" * 70)
    print("Emergence Lab -- Multi-Seed Report Generator")
    print("=" * 70)
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Output directory: {output_dir}")
    if args.compare_dir:
        print(f"Comparison directory: {args.compare_dir}")
    print()

    # Find all seed checkpoints
    print("Finding seed checkpoints...")
    checkpoints = find_all_seed_checkpoints(args.checkpoint_dir)
    print(f"Found {len(checkpoints)} seeds: {[s for s, _ in checkpoints]}")

    if not checkpoints:
        print("ERROR: No checkpoints found!")
        return

    # Analyze each seed
    print("\nAnalyzing seeds...")
    seed_results = []
    for seed_id, ckpt_path in checkpoints:
        try:
            result = analyze_seed(seed_id, ckpt_path, config, args.num_episodes)
            seed_results.append(result)
        except Exception as e:
            print(f"  ERROR analyzing seed {seed_id}: {e}")

    print(f"\nSuccessfully analyzed {len(seed_results)} seeds")

    # Analyze comparison condition if provided
    comparison_results = None
    if args.compare_dir:
        print(f"\nAnalyzing comparison condition: {args.compare_dir}")
        comp_checkpoints = find_all_seed_checkpoints(args.compare_dir)
        print(f"Found {len(comp_checkpoints)} comparison seeds")

        comparison_results = []
        for seed_id, ckpt_path in comp_checkpoints:
            try:
                result = analyze_seed(seed_id, ckpt_path, config, args.num_episodes)
                comparison_results.append(result)
            except Exception as e:
                print(f"  ERROR analyzing comparison seed {seed_id}: {e}")

        print(f"Successfully analyzed {len(comparison_results)} comparison seeds")

    # Generate report
    print("\nGenerating report...")
    report_text = generate_report(
        seed_results=seed_results,
        output_dir=output_dir,
        condition_name=args.condition_name,
        comparison_results=comparison_results,
        comparison_name=args.compare_name,
    )

    print(f"\nReport written to: {output_dir / 'multi_seed_report.md'}")

    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if seed_results:
        spec_scores = [r.specialization_score for r in seed_results]
        div_scores = [r.weight_divergence_mean for r in seed_results]

        print(f"Condition: {args.condition_name}")
        print(f"  Seeds analyzed: {len(seed_results)}")
        print(f"  Specialization score: {np.mean(spec_scores):.4f} +/- {np.std(spec_scores):.4f}")
        print(f"  Weight divergence: {np.mean(div_scores):.6f} +/- {np.std(div_scores):.6f}")

    if comparison_results:
        comp_spec = [r.specialization_score for r in comparison_results]
        comp_div = [r.weight_divergence_mean for r in comparison_results]

        print()
        print(f"Condition: {args.compare_name}")
        print(f"  Seeds analyzed: {len(comparison_results)}")
        print(f"  Specialization score: {np.mean(comp_spec):.4f} +/- {np.std(comp_spec):.4f}")
        print(f"  Weight divergence: {np.mean(comp_div):.6f} +/- {np.std(comp_div):.6f}")

        # Quick t-test
        if len(seed_results) >= 2 and len(comparison_results) >= 2:
            t_stat, p_val = stats.ttest_ind(spec_scores, comp_spec)
            sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print()
            print(f"Specialization difference: {np.mean(spec_scores) - np.mean(comp_spec):+.4f} (p={p_val:.4f}){sig}")

    print()
    print(f"Full report: {output_dir / 'multi_seed_report.md'}")


if __name__ == "__main__":
    main()
