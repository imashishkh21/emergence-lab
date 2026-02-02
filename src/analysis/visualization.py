"""Visualization functions for specialization analysis.

Produces PNG files from existing analysis data structures.
"""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

BEHAVIOR_FEATURE_NAMES = [
    "movement_entropy",
    "food_collection_rate",
    "distance_per_step",
    "reproduction_rate",
    "mean_energy",
    "exploration_ratio",
    "action_stay_fraction",
]

CLUSTER_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _save_and_close(fig: Figure, output_path: str | Path | None) -> Figure:
    """Save figure to disk if path given, then close to free memory."""
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_behavior_clusters(
    features: np.ndarray,
    labels: np.ndarray,
    method: str = "pca",
    output_path: str | Path | None = None,
) -> Figure:
    """Scatter plot of agent behavior features reduced to 2D, colored by cluster.

    Args:
        features: Array of shape (n_agents, 7) with behavioral features.
        labels: Array of shape (n_agents,) with integer cluster labels.
        method: Dimensionality reduction method, "pca" or "tsne".
        output_path: If given, save PNG to this path.

    Returns:
        The matplotlib Figure (closed but still inspectable).
    """
    n_samples = features.shape[0]
    fig, ax = plt.subplots(figsize=(8, 6))

    if n_samples < 2:
        # Not enough points for DR — plot raw first 2 features
        coords = features[:, :2] if features.shape[1] >= 2 else features
        ax.set_xlabel(BEHAVIOR_FEATURE_NAMES[0])
        ax.set_ylabel(
            BEHAVIOR_FEATURE_NAMES[1] if features.shape[1] >= 2 else "feature"
        )
    elif method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = min(30, n_samples - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        coords = tsne.fit_transform(features)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
    else:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(features)
        var = pca.explained_variance_ratio_ * 100
        ax.set_xlabel(f"PC1 ({var[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({var[1]:.1f}%)")

    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        color = CLUSTER_COLORS[int(label) % len(CLUSTER_COLORS)]
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=color,
            label=f"Cluster {label}",
            alpha=0.7,
            edgecolors="white",
            linewidths=0.5,
        )

    ax.legend()
    ax.set_title("Agent Behavior Clusters")
    fig.tight_layout()
    return _save_and_close(fig, output_path)


def plot_weight_divergence_over_time(
    divergence_history: dict,
    output_path: str | Path | None = None,
) -> Figure:
    """Line chart of weight divergence over training steps.

    Args:
        divergence_history: Dict with keys "steps" (list[int]),
            "weight_divergence" (list[float]), optionally "max_divergence"
            (list[float]).
        output_path: If given, save PNG to this path.

    Returns:
        The matplotlib Figure.
    """
    steps = divergence_history["steps"]
    mean_div = divergence_history["weight_divergence"]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(steps, mean_div, label="Mean Divergence", color="#1f77b4", linewidth=2)
    ax.fill_between(steps, 0, mean_div, alpha=0.15, color="#1f77b4")

    if "max_divergence" in divergence_history:
        max_div = divergence_history["max_divergence"]
        ax.plot(
            steps,
            max_div,
            label="Max Divergence",
            color="#d62728",
            linewidth=1.5,
            linestyle="--",
        )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Weight Divergence (Cosine Distance)")
    ax.set_title("Weight Divergence Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_and_close(fig, output_path)


def plot_field_usage_by_cluster(
    usage_data: dict,
    output_path: str | Path | None = None,
) -> Figure:
    """Grouped bar chart of field usage metrics per cluster.

    Args:
        usage_data: Dict returned by analyze_field_usage() with keys
            "per_cluster", "cluster_roles", "num_clusters".
        output_path: If given, save PNG to this path.

    Returns:
        The matplotlib Figure.
    """
    metric_names = [
        "write_frequency",
        "mean_field_value",
        "movement_rate",
        "spatial_spread",
        "field_action_correlation",
    ]
    cluster_ids = sorted(usage_data["per_cluster"].keys())
    n_clusters = len(cluster_ids)
    n_metrics = len(metric_names)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(n_metrics)
    width = 0.8 / max(n_clusters, 1)

    for i, cid in enumerate(cluster_ids):
        role = usage_data["cluster_roles"].get(cid, "unknown")
        values = [usage_data["per_cluster"][cid].get(m, 0.0) for m in metric_names]
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        offset = (i - (n_clusters - 1) / 2) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=f"Cluster {cid} ({role})",
            color=color,
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace("_", " ").title() for m in metric_names], rotation=30, ha="right"
    )
    ax.set_ylabel("Value")
    ax.set_title("Field Usage by Cluster")
    ax.legend()
    ax.yaxis.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_and_close(fig, output_path)


def plot_specialization_score_over_time(
    scores: dict,
    output_path: str | Path | None = None,
) -> Figure:
    """Line chart of specialization score over training steps.

    Args:
        scores: Dict with keys "steps" (list[int]), "scores" (list[float]),
            optionally "silhouette_component", "divergence_component",
            "variance_component" (each list[float]).
        output_path: If given, save PNG to this path.

    Returns:
        The matplotlib Figure.
    """
    steps = scores["steps"]
    composite = scores["scores"]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        steps,
        composite,
        label="Composite Score",
        color="#d62728",
        linewidth=2.5,
    )

    component_keys = [
        ("silhouette_component", "Silhouette", "#1f77b4"),
        ("divergence_component", "Divergence", "#ff7f0e"),
        ("variance_component", "Variance", "#2ca02c"),
    ]
    for key, label, color in component_keys:
        if key in scores:
            ax.plot(
                steps,
                scores[key],
                label=label,
                color=color,
                linewidth=1.5,
                linestyle="--",
            )

    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Score")
    ax.set_title("Specialization Score Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_and_close(fig, output_path)
