#!/usr/bin/env python
"""Generate a comprehensive specialization analysis report.

Runs a trained model through the full analysis pipeline and produces:
- A markdown report with metrics, species info, and interpretive text
- PNG visualizations (behavior clusters, weight divergence, field usage,
  specialization score)

Usage:
    python scripts/generate_specialization_report.py --checkpoint checkpoints/params.pkl
    python scripts/generate_specialization_report.py --checkpoint checkpoints/params.pkl --output-dir reports/
    python scripts/generate_specialization_report.py  # trains briefly then analyzes
"""

import argparse
import os
import pickle
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from src.agents.network import ActorCritic
from src.analysis.ablation import (
    specialization_ablation_test,
)
from src.analysis.lineage import LineageTracker
from src.analysis.specialization import (
    analyze_field_usage,
    compute_weight_divergence,
    correlate_lineage_strategy,
    detect_species,
    extract_behavior_features,
    find_optimal_clusters,
    specialization_score,
    SpecializationTracker,
)
from src.analysis.trajectory import record_episode
from src.analysis.visualization import (
    BEHAVIOR_FEATURE_NAMES,
    plot_behavior_clusters,
    plot_field_usage_by_cluster,
    plot_specialization_score_over_time,
    plot_weight_divergence_over_time,
)
from src.configs import Config
from src.environment.obs import obs_dim
from src.training.train import create_train_state, train_step


def _train_model(
    config: Config, iterations: int, seed: int
) -> tuple[Any, Any, Any, "SpecializationTracker"]:
    """Train a model and return params, agent_params, alive_mask, tracker."""
    key = jax.random.PRNGKey(seed)
    print("  Initializing training state...")
    runner_state = create_train_state(config, key)
    print("  Training state initialized.")

    # Specialization tracker for collecting history
    tracker = SpecializationTracker(config)

    @jax.jit
    def jit_train_step(rs: Any) -> Any:
        return train_step(rs, config)

    print("  JIT compiling train_step...")
    t0 = time.time()
    runner_state, metrics = jit_train_step(runner_state)
    jax.block_until_ready(metrics)
    print(f"  JIT compilation done ({time.time() - t0:.1f}s)")

    steps_per_iter = (
        config.train.num_envs * config.train.num_steps * config.evolution.max_agents
    )

    # Record specialization at step 0
    if config.evolution.enabled and runner_state.env_state.agent_params is not None:
        agent_p = jax.tree_util.tree_map(
            lambda x: x[0], runner_state.env_state.agent_params
        )
        alive_m = np.asarray(runner_state.env_state.agent_alive[0])
        tracker.update(agent_p, alive_m, step=0)

    t_start = time.time()
    for i in range(1, iterations):
        runner_state, metrics = jit_train_step(runner_state)

        current_step = (i + 1) * steps_per_iter

        # Track specialization periodically
        if (
            config.evolution.enabled
            and runner_state.env_state.agent_params is not None
            and i % max(1, iterations // 20) == 0
        ):
            agent_p = jax.tree_util.tree_map(
                lambda x: x[0], runner_state.env_state.agent_params
            )
            alive_m = np.asarray(runner_state.env_state.agent_alive[0])
            tracker.update(agent_p, alive_m, step=current_step)

        if i % 10 == 0 or i == iterations - 1:
            reward = float(metrics["mean_reward"])
            loss = float(metrics["total_loss"])
            print(
                f"    iter {i:>4d}/{iterations} | "
                f"reward={reward:.4f} | loss={loss:.4f}"
            )

    jax.block_until_ready(metrics)
    print(f"  Training complete ({time.time() - t_start:.1f}s)")

    params = runner_state.params
    trained_agent_params = None
    trained_alive_mask = None
    if config.evolution.enabled and runner_state.env_state.agent_params is not None:
        trained_agent_params = jax.tree_util.tree_map(
            lambda x: x[0], runner_state.env_state.agent_params
        )
        trained_alive_mask = runner_state.env_state.agent_alive[0]

    return params, trained_agent_params, trained_alive_mask, tracker


def _generate_markdown_report(
    output_dir: Path,
    config: Config,
    div_result: dict[str, Any],
    spec_result: dict[str, Any],
    species_result: dict[str, Any],
    field_usage: dict[str, Any],
    lineage_result: dict[str, Any] | None,
    ablation_results: dict[str, Any] | None,
    tracker_summary: dict[str, Any] | None,
    num_alive: int,
    num_trajectories: int,
) -> str:
    """Generate a markdown report and write to disk. Returns report text."""
    lines: list[str] = []

    lines.append("# Specialization Analysis Report")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- **Grid size**: {config.env.grid_size}x{config.env.grid_size}")
    lines.append(f"- **Max agents**: {config.evolution.max_agents}")
    lines.append(f"- **Alive agents**: {num_alive}")
    lines.append(f"- **Trajectory episodes recorded**: {num_trajectories}")
    lines.append(
        f"- **Mutation std**: {config.evolution.mutation_std}"
    )
    lines.append(f"- **Evolution enabled**: {config.evolution.enabled}")
    lines.append("")

    # --- Specialization Score ---
    lines.append("## Specialization Score")
    lines.append("")
    lines.append(f"**Composite Score: {spec_result['score']:.4f}** (0 = identical, 1 = fully specialized)")
    lines.append("")
    lines.append("| Component | Score |")
    lines.append("|-----------|-------|")
    lines.append(f"| Silhouette | {spec_result['silhouette_component']:.4f} |")
    lines.append(f"| Weight Divergence | {spec_result['divergence_component']:.4f} |")
    lines.append(f"| Behavioral Variance | {spec_result['variance_component']:.4f} |")
    lines.append(f"| Optimal Clusters (k) | {spec_result['optimal_k']} |")
    lines.append("")

    # --- Weight Divergence ---
    lines.append("## Weight Divergence")
    lines.append("")
    lines.append(f"- **Mean divergence**: {div_result['mean_divergence']:.6f}")
    lines.append(f"- **Max divergence**: {div_result['max_divergence']:.6f}")
    lines.append(f"- **Agents compared**: {len(div_result['agent_indices'])}")
    lines.append("")

    if tracker_summary:
        lines.append("### Training History")
        lines.append("")
        if "weight_divergence_mean" in tracker_summary:
            lines.append(
                f"- Mean divergence over training: "
                f"{tracker_summary['weight_divergence_mean']:.6f} "
                f"(std: {tracker_summary.get('weight_divergence_std', 0.0):.6f})"
            )
        if "weight_divergence_final" in tracker_summary:
            lines.append(
                f"- Final divergence: {tracker_summary['weight_divergence_final']:.6f}"
            )
        if tracker_summary.get("total_events", 0) > 0:
            lines.append(f"- Specialization events detected: {tracker_summary['total_events']}")
            for event_str in tracker_summary.get("events", []):
                lines.append(f"  - {event_str}")
        lines.append("")

    lines.append("![Weight Divergence Over Time](figures/weight_divergence.png)")
    lines.append("")

    # --- Species Detection ---
    lines.append("## Species Detection")
    lines.append("")
    lines.append(
        f"- **Species detected**: {species_result['num_species']}"
    )
    lines.append(f"- **Silhouette**: {species_result['silhouette']:.4f}")
    lines.append(f"- **Optimal k**: {species_result['optimal_k']}")
    lines.append(f"- **Heredity score**: {species_result['heredity_score']:.4f}")
    lines.append(
        f"- **Speciation observed**: {'Yes' if species_result['is_speciated'] else 'No'}"
    )
    lines.append("")

    if species_result["species"]:
        lines.append("### Detected Species")
        lines.append("")
        lines.append("| Species | Members | Heredity | Role | Key Features |")
        lines.append("|---------|---------|----------|------|--------------|")
        for sp in species_result["species"]:
            # Summarize top features
            feat_strs = []
            for i, fname in enumerate(BEHAVIOR_FEATURE_NAMES):
                if i < len(sp.mean_features):
                    feat_strs.append(f"{fname}={sp.mean_features[i]:.3f}")
            top_feats = ", ".join(feat_strs[:3])
            lines.append(
                f"| Cluster {sp.cluster_id} | {sp.num_members} | "
                f"{sp.heredity_score:.2f} | {sp.role} | {top_feats} |"
            )
        lines.append("")

    lines.append("![Behavior Clusters](figures/behavior_clusters_pca.png)")
    lines.append("")

    # --- Field Usage ---
    lines.append("## Field Usage by Cluster")
    lines.append("")
    lines.append(f"- **Clusters analyzed**: {field_usage['num_clusters']}")
    lines.append("")

    lines.append("| Cluster | Role | Write Freq | Mean Field | Movement | Spread | Field-Action Corr |")
    lines.append("|---------|------|------------|------------|----------|--------|-------------------|")
    for cid in sorted(field_usage["per_cluster"].keys()):
        stats = field_usage["per_cluster"][cid]
        role = field_usage["cluster_roles"].get(cid, "unknown")
        lines.append(
            f"| {cid} | {role} | {stats['write_frequency']:.3f} | "
            f"{stats['mean_field_value']:.3f} | {stats['movement_rate']:.3f} | "
            f"{stats['spatial_spread']:.3f} | {stats['field_action_correlation']:.3f} |"
        )
    lines.append("")

    lines.append("![Field Usage by Cluster](figures/field_usage.png)")
    lines.append("")

    # --- Lineage-Strategy Correlation ---
    if lineage_result is not None:
        lines.append("## Lineage-Strategy Correlation")
        lines.append("")
        lines.append(f"- **Lineages analyzed**: {lineage_result['num_lineages']}")
        lines.append(
            f"- **Specialist lineages**: {lineage_result['num_specialist_lineages']}"
        )
        lines.append(
            f"- **Mean homogeneity**: {lineage_result['mean_homogeneity']:.4f}"
        )
        lines.append("")

        if lineage_result["specialist_lineages"]:
            lines.append("### Specialist Lineages")
            lines.append("")
            lines.append("| Root Ancestor | Dominant Cluster | Homogeneity |")
            lines.append("|---------------|------------------|-------------|")
            for root_id, dom_cluster, homog in lineage_result["specialist_lineages"][:10]:
                lines.append(f"| {root_id} | {dom_cluster} | {homog:.2f} |")
            lines.append("")

    # --- Ablation Results ---
    if ablation_results is not None:
        lines.append("## Diversity vs Performance Ablation")
        lines.append("")
        lines.append("| Condition | Mean Reward | Food Collected | Population Stability | Survival Rate |")
        lines.append("|-----------|-------------|----------------|----------------------|---------------|")
        for condition_name in ["divergent", "uniform", "random_weights"]:
            if condition_name in ablation_results:
                r = ablation_results[condition_name]
                lines.append(
                    f"| {r.condition} | {r.mean_reward:.4f} | "
                    f"{r.food_collected:.1f} | {r.population_stability:.4f} | "
                    f"{r.survival_rate:.2f} |"
                )
        lines.append("")

        # Interpretation
        if "divergent" in ablation_results and "uniform" in ablation_results:
            div_food = ablation_results["divergent"].food_collected
            uni_food = ablation_results["uniform"].food_collected
            if div_food > uni_food:
                lines.append(
                    "> **Conclusion**: Specialized (divergent) weights collect MORE food "
                    "than uniform weights, suggesting specialization improves collective performance."
                )
            elif uni_food > div_food:
                lines.append(
                    "> **Conclusion**: Uniform weights outperform divergent weights. "
                    "Specialization has not yet emerged as beneficial."
                )
            else:
                lines.append(
                    "> **Conclusion**: Performance is similar between conditions. "
                    "More training may reveal clearer patterns."
                )
            lines.append("")

    # --- Specialization Score Over Time ---
    if tracker_summary and tracker_summary.get("total_updates", 0) > 1:
        lines.append("## Specialization Over Training")
        lines.append("")
        lines.append("![Specialization Score Over Time](figures/specialization_score.png)")
        lines.append("")

    # --- Visualizations ---
    lines.append("## Visualizations")
    lines.append("")
    lines.append("All figures saved to the `figures/` subdirectory:")
    lines.append("")
    lines.append("- `behavior_clusters_pca.png` — PCA scatter of agent behaviors")
    lines.append("- `behavior_clusters_tsne.png` — t-SNE scatter of agent behaviors")
    lines.append("- `weight_divergence.png` — Weight divergence over training")
    lines.append("- `field_usage.png` — Field usage metrics by cluster")
    lines.append("- `specialization_score.png` — Specialization score over training")
    lines.append("")

    report_text = "\n".join(lines)
    report_path = output_dir / "specialization_report.md"
    report_path.write_text(report_text)
    return report_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate comprehensive specialization analysis report"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (params.pkl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output directory for report and figures (default: reports/)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Training iterations if no checkpoint (default: 50)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="Number of episodes for trajectory recording (default: 5)",
    )
    parser.add_argument(
        "--ablation-episodes",
        type=int,
        default=10,
        help="Number of episodes per ablation condition (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Skip the ablation comparison (faster)",
    )
    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load or create config
    if args.config is not None:
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    config.log.wandb = False
    config.train.seed = args.seed

    print("=" * 70)
    print("Emergence Lab -- Specialization Report Generator")
    print("=" * 70)

    # --- Phase 1: Load or train model ---
    tracker_summary: dict[str, Any] | None = None
    tracker: SpecializationTracker | None = None
    divergence_history: dict[str, list[Any]] | None = None

    if args.checkpoint is not None:
        print(f"\nLoading checkpoint from {args.checkpoint}...")
        with open(args.checkpoint, "rb") as f:
            checkpoint_data = pickle.load(f)

        if isinstance(checkpoint_data, dict) and "agent_params" in checkpoint_data:
            # Structured checkpoint (from run_specialization_ablation.py)
            params = checkpoint_data["params"]
            trained_agent_params = checkpoint_data.get("agent_params")
            trained_alive_mask = checkpoint_data.get("alive_mask")
            if trained_alive_mask is not None:
                trained_alive_mask = jnp.array(trained_alive_mask)
        else:
            # Raw Flax variables dict (from train.py)
            params = checkpoint_data
            trained_agent_params = None
            trained_alive_mask = None

        print("Checkpoint loaded.")
    else:
        print(f"\nPhase 1: Training ({args.iterations} iterations)")
        print("-" * 40)
        steps_per_iter = (
            config.train.num_envs
            * config.train.num_steps
            * config.evolution.max_agents
        )
        config.train.total_steps = args.iterations * steps_per_iter

        params, trained_agent_params, trained_alive_mask, tracker = _train_model(
            config, args.iterations, args.seed
        )
        tracker_summary = tracker.get_summary()

        # Build divergence history from tracker
        if tracker.step_count > 0:
            divergence_history = {
                "steps": list(tracker.steps),
                "weight_divergence": list(tracker.history["weight_divergence"]),
                "max_divergence": list(tracker.history["max_divergence"]),
            }

    # Create network
    network = ActorCritic(
        hidden_dims=tuple(config.agent.hidden_dims),
        num_actions=config.agent.num_actions,
    )
    # Verify params
    observation_dim = obs_dim(config)
    dummy_obs = jnp.zeros((observation_dim,))
    network.apply(params, dummy_obs)

    # --- Phase 2: Analysis ---
    print(f"\nPhase 2: Specialization Analysis")
    print("-" * 40)

    if trained_agent_params is None or trained_alive_mask is None:
        print("ERROR: No per-agent params available. Enable evolution and train first.")
        print("Run without --checkpoint to train, or use a checkpoint with evolution enabled.")
        return

    alive_mask_np = np.asarray(trained_alive_mask)
    num_alive = int(np.sum(alive_mask_np))
    print(f"  Alive agents: {num_alive}")

    # Weight divergence
    print("  Computing weight divergence...")
    div_result = compute_weight_divergence(trained_agent_params, alive_mask_np)
    print(f"    Mean: {div_result['mean_divergence']:.6f}, Max: {div_result['max_divergence']:.6f}")

    # Record trajectories (multiple episodes for robustness)
    print(f"  Recording {args.num_episodes} evaluation trajectories...")
    all_features: list[np.ndarray] = []
    last_trajectory = None
    for ep in range(args.num_episodes):
        traj_key = jax.random.PRNGKey(args.seed + 1000 + ep)
        trajectory = record_episode(network, params, config, traj_key)
        features = extract_behavior_features(trajectory)
        all_features.append(features)
        last_trajectory = trajectory

    # Average features across episodes for more stable clustering
    avg_features = np.mean(all_features, axis=0)
    assert last_trajectory is not None
    print(f"    Feature shape: {avg_features.shape}")

    # Specialization score
    print("  Computing specialization score...")
    spec_result = specialization_score(
        avg_features,
        agent_params=trained_agent_params,
        alive_mask=alive_mask_np,
    )
    print(f"    Score: {spec_result['score']:.4f} (k={spec_result['optimal_k']})")

    # Optimal clustering
    print("  Finding optimal clusters...")
    clustering = find_optimal_clusters(avg_features)
    labels = clustering["labels"]

    # Species detection (without lineage tracker from checkpoint)
    print("  Detecting species...")
    agent_ids = np.arange(avg_features.shape[0])
    species_result = detect_species(
        avg_features,
        threshold=0.5,  # slightly lower for report (may not have long training)
    )
    print(f"    Species found: {species_result['num_species']}")

    # Field usage analysis
    print("  Analyzing field usage...")
    field_usage = analyze_field_usage(last_trajectory, species_result["all_labels"])
    print(f"    Cluster roles: {field_usage['cluster_roles']}")

    # Lineage-strategy correlation (only if we have a lineage tracker)
    lineage_result: dict[str, Any] | None = None

    # --- Phase 3: Ablation (optional) ---
    ablation_results = None
    if not args.skip_ablation:
        print(f"\nPhase 3: Specialization Ablation ({args.ablation_episodes} episodes/condition)")
        print("-" * 40)
        ablation_results = specialization_ablation_test(
            network=network,
            params=params,
            config=config,
            trained_agent_params=trained_agent_params,
            trained_alive_mask=trained_alive_mask,
            num_episodes=args.ablation_episodes,
            seed=args.seed,
        )
        for cond, res in ablation_results.items():
            print(f"    {cond}: reward={res.mean_reward:.4f}, food={res.food_collected:.1f}")
    else:
        print("\nPhase 3: Ablation skipped (--skip-ablation)")

    # --- Phase 4: Visualizations ---
    print(f"\nPhase 4: Generating Visualizations")
    print("-" * 40)

    # Behavior clusters (PCA)
    print("  Plotting behavior clusters (PCA)...")
    plot_behavior_clusters(
        avg_features,
        species_result["all_labels"],
        method="pca",
        output_path=figures_dir / "behavior_clusters_pca.png",
    )

    # Behavior clusters (t-SNE) — only if enough samples
    if avg_features.shape[0] >= 3:
        print("  Plotting behavior clusters (t-SNE)...")
        plot_behavior_clusters(
            avg_features,
            species_result["all_labels"],
            method="tsne",
            output_path=figures_dir / "behavior_clusters_tsne.png",
        )

    # Weight divergence over time
    if divergence_history and len(divergence_history["steps"]) > 1:
        print("  Plotting weight divergence over time...")
        plot_weight_divergence_over_time(
            divergence_history,
            output_path=figures_dir / "weight_divergence.png",
        )

    # Field usage by cluster
    print("  Plotting field usage by cluster...")
    plot_field_usage_by_cluster(
        field_usage,
        output_path=figures_dir / "field_usage.png",
    )

    # Specialization score over time (only if we have training history)
    if tracker and tracker.step_count > 1:
        print("  Plotting specialization score over time...")
        # Compute specialization scores at each tracked step
        score_history: dict[str, list[Any]] = {
            "steps": list(tracker.steps),
            "scores": [],
            "silhouette_component": [],
            "divergence_component": [],
            "variance_component": [],
        }
        # Use the final features for a single score timeline based on divergence
        # We approximate the score timeline using the tracked divergence history
        for i in range(tracker.step_count):
            div_val = tracker.history["weight_divergence"][i]
            # Approximate components from divergence alone
            div_comp = float(np.clip(div_val / 2.0, 0.0, 1.0))
            # Use final silhouette and variance as constants (we only have snapshots)
            sil_comp = spec_result["silhouette_component"]
            var_comp = spec_result["variance_component"]
            approx_score = 0.5 * sil_comp + 0.25 * div_comp + 0.25 * var_comp
            score_history["scores"].append(float(np.clip(approx_score, 0.0, 1.0)))
            score_history["silhouette_component"].append(sil_comp)
            score_history["divergence_component"].append(div_comp)
            score_history["variance_component"].append(var_comp)

        plot_specialization_score_over_time(
            score_history,
            output_path=figures_dir / "specialization_score.png",
        )

    # If no divergence history (loaded checkpoint), create a single-point plot
    if divergence_history is None:
        # Create a minimal divergence plot with just the current state
        single_point_hist = {
            "steps": [0],
            "weight_divergence": [div_result["mean_divergence"]],
            "max_divergence": [div_result["max_divergence"]],
        }
        print("  Plotting current weight divergence snapshot...")
        plot_weight_divergence_over_time(
            single_point_hist,
            output_path=figures_dir / "weight_divergence.png",
        )

    # --- Phase 5: Generate Report ---
    print(f"\nPhase 5: Generating Report")
    print("-" * 40)

    report_text = _generate_markdown_report(
        output_dir=output_dir,
        config=config,
        div_result=div_result,
        spec_result=spec_result,
        species_result=species_result,
        field_usage=field_usage,
        lineage_result=lineage_result,
        ablation_results=ablation_results,
        tracker_summary=tracker_summary,
        num_alive=num_alive,
        num_trajectories=args.num_episodes,
    )

    print(f"  Report written to: {output_dir / 'specialization_report.md'}")
    print(f"  Figures written to: {figures_dir}/")

    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Specialization score:  {spec_result['score']:.4f}")
    print(f"  Weight divergence:     {div_result['mean_divergence']:.6f}")
    print(f"  Species detected:      {species_result['num_species']}")
    print(f"  Optimal clusters:      {spec_result['optimal_k']}")
    print(f"  Speciation observed:   {'Yes' if species_result['is_speciated'] else 'No'}")
    if ablation_results and "divergent" in ablation_results:
        print(f"  Ablation (divergent):  food={ablation_results['divergent'].food_collected:.1f}")
        if "uniform" in ablation_results:
            print(f"  Ablation (uniform):    food={ablation_results['uniform'].food_collected:.1f}")
    print(f"\n  Full report: {output_dir / 'specialization_report.md'}")


if __name__ == "__main__":
    main()
