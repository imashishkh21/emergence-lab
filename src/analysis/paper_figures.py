"""Publication-quality figures for Phase 5 emergence experiments.

All figures follow publication standards:
    - 300 DPI resolution
    - Colorblind-safe palette
    - PDF + PNG output formats
    - Consistent font sizes and styling

Figures included:
    US-016 (Scaling + Ablation):
        - Scaling curves with power law fits
        - Ablation bar charts with IQM and CIs
        - Performance profiles (CDFs)
        - Probability of improvement heatmaps

    US-017 (Emergence Metrics):
        - PID synergy heatmaps
        - O-information trajectories
        - Causal emergence gap plots
        - Phase transition diagrams

Reference: rliable best practices (Agarwal et al., 2021).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

# Use non-interactive backend for server/CI compatibility
matplotlib.use("Agg")


# =============================================================================
# Style Constants
# =============================================================================

FONT_SIZE = 12
TITLE_SIZE = 14
DPI = 300
FIGURE_SIZE = (8, 6)

# Colorblind-safe palette from Paul Tol
# https://personal.sron.nl/~pault/
COLORBLIND_PALETTE = [
    "#0077BB",  # blue
    "#33BBEE",  # cyan
    "#009988",  # teal
    "#EE7733",  # orange
    "#CC3311",  # red
    "#EE3377",  # magenta
    "#BBBBBB",  # gray
]

# Method-specific colors for consistent visualization
METHOD_COLORS = {
    "ours": "#009988",  # teal (our method stands out)
    "normal": "#009988",  # teal (same as ours)
    "ippo": "#0077BB",  # blue
    "aco_fixed": "#EE7733",  # orange
    "aco_hybrid": "#CC3311",  # red
    "mappo": "#33BBEE",  # cyan
    "zeroed": "#BBBBBB",  # gray
    "random": "#EE3377",  # magenta
    "frozen": "#0077BB",  # blue
    "no_field": "#CC3311",  # red
    "write_only": "#EE7733",  # orange
}


def setup_publication_style() -> None:
    """Configure matplotlib for publication-quality figures.

    Sets font sizes, line widths, and other parameters for
    consistent, professional-looking figures.
    """
    plt.rcParams.update({
        # Font settings
        "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 1,
        # Line widths
        "lines.linewidth": 2.0,
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.5,
        # Figure settings
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "figure.figsize": FIGURE_SIZE,
        # Grid
        "axes.grid": True,
        "grid.alpha": 0.3,
        # Legend
        "legend.framealpha": 0.8,
        "legend.edgecolor": "gray",
        # Remove top/right spines
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def get_color(name: str, index: int = 0) -> str:
    """Get color for a method/condition name.

    Args:
        name: Method or condition name.
        index: Fallback index if name not in palette.

    Returns:
        Hex color string.
    """
    if name.lower() in METHOD_COLORS:
        return METHOD_COLORS[name.lower()]
    return COLORBLIND_PALETTE[index % len(COLORBLIND_PALETTE)]


# =============================================================================
# US-016: Scaling and Ablation Figures
# =============================================================================


def plot_scaling_curve(
    scaling_results: list[Any],  # list[ScalingAnalysis]
    output_path: str,
    show_reference: bool = True,
    figsize: tuple[float, float] = (8, 6),
) -> Figure:
    """Plot scaling curves with power law fits.

    Creates a log-log plot of total food vs number of agents for
    each field condition. Includes power law fit lines and reference
    lines for linear/superlinear scaling.

    Args:
        scaling_results: List of ScalingAnalysis objects, one per condition.
        output_path: Base path for output (will add .pdf and .png).
        show_reference: Whether to show alpha=1.0 reference line.
        figsize: Figure size in inches.

    Returns:
        Matplotlib Figure object.
    """
    setup_publication_style()
    fig, ax = plt.subplots(figsize=figsize)

    for i, analysis in enumerate(scaling_results):
        # Extract data
        n_agents = np.array(analysis.n_agents_list)
        mean_foods = np.array(analysis.mean_foods)
        std_foods = np.array(analysis.std_foods) if hasattr(analysis, "std_foods") else None
        alpha = getattr(analysis, "alpha", 1.0)
        condition = analysis.field_condition

        color = get_color(condition, i)

        # Plot data points with error bars
        if std_foods is not None and len(std_foods) == len(mean_foods):
            ax.errorbar(
                n_agents,
                mean_foods,
                yerr=std_foods,
                fmt="o",
                color=color,
                capsize=3,
                capthick=1,
                label=f"{condition} (data)",
                markersize=8,
                zorder=3,
            )
        else:
            ax.plot(
                n_agents,
                mean_foods,
                "o",
                color=color,
                label=f"{condition} (data)",
                markersize=8,
                zorder=3,
            )

        # Plot power law fit line
        if len(n_agents) >= 2 and alpha != 0:
            n_fit = np.linspace(n_agents[0], n_agents[-1], 100)
            # log(F) = alpha * log(N) + c
            # F = exp(c) * N^alpha
            c = getattr(analysis, "c", 0.0)
            f_fit = np.exp(c) * n_fit ** alpha
            ax.plot(
                n_fit,
                f_fit,
                "--",
                color=color,
                alpha=0.7,
                label=f"{condition} fit (alpha={alpha:.2f})",
                zorder=2,
            )

    # Reference line for linear scaling (alpha = 1.0)
    if show_reference and len(scaling_results) > 0:
        # Use first result's data range
        n_agents = np.array(scaling_results[0].n_agents_list)
        if len(n_agents) >= 2:
            n_ref = np.linspace(n_agents[0], n_agents[-1], 100)
            # Linear scaling: F = constant * N
            f_ref = (scaling_results[0].mean_foods[0] / n_agents[0]) * n_ref
            ax.plot(
                n_ref,
                f_ref,
                ":",
                color="black",
                alpha=0.5,
                label="Linear (alpha=1.0)",
                zorder=1,
            )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Number of Agents (N)")
    ax.set_ylabel("Total Food Collected")
    ax.set_title("Scaling Analysis: Food Collection vs Agent Count")
    ax.legend(loc="upper left", fontsize=FONT_SIZE - 2)

    plt.tight_layout()
    save_figure(fig, output_path)
    return fig


def plot_ablation_bars(
    ablation_results: dict[str, Any],  # dict[str, ExtendedAblationResult]
    output_path: str,
    metric: str = "iqm",
    figsize: tuple[float, float] = (10, 6),
) -> Figure:
    """Plot ablation results as bar chart with error bars.

    Creates a horizontal bar chart comparing methods/conditions
    with IQM values and bootstrap confidence intervals.

    Args:
        ablation_results: Dict mapping condition name to results.
            Each result should have mean_reward, std_reward, and optionally
            ci_lower, ci_upper, iqm fields.
        output_path: Base path for output.
        metric: Which metric to display ("iqm", "mean", "median").
        figsize: Figure size in inches.

    Returns:
        Matplotlib Figure object.
    """
    setup_publication_style()
    fig, ax = plt.subplots(figsize=figsize)

    # Sort by metric value (descending)
    conditions = list(ablation_results.keys())

    def get_metric_value(result: Any) -> float:
        if metric == "iqm" and hasattr(result, "iqm"):
            return float(result.iqm)
        elif metric == "median" and hasattr(result, "median"):
            return float(result.median)
        return float(result.mean_reward)

    conditions = sorted(conditions, key=lambda c: get_metric_value(ablation_results[c]), reverse=True)

    values = []
    errors_lower = []
    errors_upper = []
    colors = []

    for i, cond in enumerate(conditions):
        result = ablation_results[cond]
        val = get_metric_value(result)
        values.append(val)

        # Get CI if available and valid (non-zero)
        ci_lower = getattr(result, "ci_lower", None)
        ci_upper = getattr(result, "ci_upper", None)
        if ci_lower is not None and ci_upper is not None and ci_upper > 0:
            # Use CI bounds - ensure non-negative errors
            errors_lower.append(max(0.0, val - ci_lower))
            errors_upper.append(max(0.0, ci_upper - val))
        else:
            # Fall back to std
            std = getattr(result, "std_reward", 0.0)
            errors_lower.append(std)
            errors_upper.append(std)

        colors.append(get_color(cond, i))

    y_pos = np.arange(len(conditions))

    # Horizontal bar chart
    bars = ax.barh(
        y_pos,
        values,
        xerr=[errors_lower, errors_upper],
        color=colors,
        capsize=3,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(conditions)
    ax.invert_yaxis()  # Top to bottom
    ax.set_xlabel(f"{metric.upper()} Reward")
    ax.set_title("Ablation Study: Field Condition Comparison")

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}",
            va="center",
            fontsize=FONT_SIZE - 2,
        )

    plt.tight_layout()
    save_figure(fig, output_path)
    return fig


def plot_performance_profiles(
    score_dict: dict[str, np.ndarray],
    output_path: str,
    tau_range: tuple[float, float] = (0, 2),
    n_points: int = 100,
    figsize: tuple[float, float] = (8, 6),
) -> Figure:
    """Plot performance profiles (CDFs) for multiple methods.

    Performance profiles show the fraction of runs where a method
    achieves at least a given fraction of the best score.

    Args:
        score_dict: Dict mapping method name to array of scores.
        output_path: Base path for output.
        tau_range: Range of tau values (normalized score threshold).
        n_points: Number of points in the CDF.
        figsize: Figure size in inches.

    Returns:
        Matplotlib Figure object.
    """
    setup_publication_style()
    fig, ax = plt.subplots(figsize=figsize)

    # Find maximum score across all methods for normalization
    all_scores = np.concatenate(list(score_dict.values()))
    max_score = np.max(all_scores)
    if max_score == 0:
        max_score = 1.0

    tau_values = np.linspace(tau_range[0], tau_range[1], n_points)

    for i, (method, scores) in enumerate(score_dict.items()):
        # Normalize scores
        normalized = scores / max_score

        # Compute CDF
        fractions = []
        for tau in tau_values:
            fraction = np.mean(normalized >= tau)
            fractions.append(fraction)

        color = get_color(method, i)
        ax.plot(
            tau_values,
            fractions,
            "-",
            color=color,
            label=method,
            linewidth=2,
        )

    ax.set_xlabel("Normalized Score Threshold (tau)")
    ax.set_ylabel("Fraction of Runs >= tau")
    ax.set_title("Performance Profiles")
    ax.legend(loc="lower left")
    ax.set_xlim(tau_range)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    save_figure(fig, output_path)
    return fig


def plot_probability_of_improvement(
    scores_dict: dict[str, np.ndarray],
    output_path: str,
    figsize: tuple[float, float] = (8, 7),
) -> Figure:
    """Plot probability of improvement heatmap.

    Creates a matrix showing P(method_i > method_j) for all pairs.

    Args:
        scores_dict: Dict mapping method name to array of scores.
        output_path: Base path for output.
        figsize: Figure size in inches.

    Returns:
        Matplotlib Figure object.
    """
    setup_publication_style()

    methods = list(scores_dict.keys())
    n_methods = len(methods)

    # Compute pairwise probability of improvement
    poi_matrix = np.zeros((n_methods, n_methods))

    for i, method_i in enumerate(methods):
        for j, method_j in enumerate(methods):
            if i == j:
                poi_matrix[i, j] = 0.5  # Diagonal
            else:
                scores_i = scores_dict[method_i]
                scores_j = scores_dict[method_j]
                # P(method_i > method_j) via pairwise comparison
                n_comparisons: int = 0
                n_wins: float = 0.0
                for si in scores_i:
                    for sj in scores_j:
                        n_comparisons += 1
                        if si > sj:
                            n_wins += 1.0
                        elif si == sj:
                            n_wins += 0.5
                poi_matrix[i, j] = n_wins / n_comparisons if n_comparisons > 0 else 0.5

    fig, ax = plt.subplots(figsize=figsize)

    # Diverging colormap centered at 0.5
    im = ax.imshow(
        poi_matrix,
        cmap="RdBu",
        vmin=0,
        vmax=1,
        aspect="auto",
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("P(row > column)")

    # Add annotations
    for i in range(n_methods):
        for j in range(n_methods):
            text_color = "white" if abs(poi_matrix[i, j] - 0.5) > 0.3 else "black"
            ax.text(
                j,
                i,
                f"{poi_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=FONT_SIZE - 1,
            )

    ax.set_xticks(range(n_methods))
    ax.set_yticks(range(n_methods))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_yticklabels(methods)
    ax.set_xlabel("Method (column)")
    ax.set_ylabel("Method (row)")
    ax.set_title("Probability of Improvement: P(row > column)")

    plt.tight_layout()
    save_figure(fig, output_path)
    return fig


# =============================================================================
# US-017: Emergence Metrics Figures
# =============================================================================


def plot_pid_synergy_heatmap(
    synergy_matrix: np.ndarray,
    output_path: str,
    agent_labels: list[str] | None = None,
    figsize: tuple[float, float] = (8, 7),
) -> Figure:
    """Plot PID synergy heatmap for agent pairs.

    Creates an agent-pair matrix showing synergy values.
    Diagonal is marked as N/A (no self-synergy).

    Args:
        synergy_matrix: (n_agents, n_agents) array of synergy values.
        output_path: Base path for output.
        agent_labels: Optional labels for agents.
        figsize: Figure size in inches.

    Returns:
        Matplotlib Figure object.
    """
    setup_publication_style()

    n_agents = synergy_matrix.shape[0]
    if agent_labels is None:
        agent_labels = [f"A{i}" for i in range(n_agents)]

    fig, ax = plt.subplots(figsize=figsize)

    # Mask diagonal
    masked_matrix = np.ma.array(synergy_matrix, mask=np.eye(n_agents, dtype=bool))

    # Diverging colormap (blue=negative/synergy, red=positive)
    im = ax.imshow(
        masked_matrix,
        cmap="RdBu_r",  # Reversed: blue = positive (synergy)
        aspect="auto",
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("PID Synergy")

    # Annotate cells
    for i in range(n_agents):
        for j in range(n_agents):
            if i == j:
                ax.text(j, i, "N/A", ha="center", va="center", color="gray", fontsize=FONT_SIZE - 2)
            else:
                val = synergy_matrix[i, j]
                text_color = "white" if abs(val) > np.std(masked_matrix.compressed()) else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=FONT_SIZE - 2)

    ax.set_xticks(range(n_agents))
    ax.set_yticks(range(n_agents))
    ax.set_xticklabels(agent_labels, rotation=45, ha="right")
    ax.set_yticklabels(agent_labels)
    ax.set_xlabel("Agent j (field source)")
    ax.set_ylabel("Agent i (action source)")
    ax.set_title("PID Synergy: I(action_i, field_j; outcome_i)")

    plt.tight_layout()
    save_figure(fig, output_path)
    return fig


def plot_o_information_trajectory(
    o_info_values: list[float],
    steps: list[int],
    output_path: str,
    surrogate_ci: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (10, 5),
) -> Figure:
    """Plot O-information over training steps.

    Shows how O-information (synergy indicator) evolves during training.
    Includes optional surrogate CI band.

    Args:
        o_info_values: O-information values at each step.
        steps: Training step numbers.
        output_path: Base path for output.
        surrogate_ci: Optional (lower, upper) bounds from surrogate test.
        figsize: Figure size in inches.

    Returns:
        Matplotlib Figure object.
    """
    setup_publication_style()
    fig, ax = plt.subplots(figsize=figsize)

    steps_arr = np.array(steps)
    o_info_arr = np.array(o_info_values)

    # Plot O-information trajectory
    ax.plot(steps_arr, o_info_arr, "-", color=COLORBLIND_PALETTE[0], linewidth=2, label="O-information")

    # Surrogate CI band
    if surrogate_ci is not None:
        ax.fill_between(
            steps_arr,
            surrogate_ci[0],
            surrogate_ci[1],
            alpha=0.2,
            color="gray",
            label=f"Surrogate 95% CI [{surrogate_ci[0]:.3f}, {surrogate_ci[1]:.3f}]",
        )

    # Zero line (synergy/redundancy boundary)
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5, label="Synergy/Redundancy boundary")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("O-Information")
    ax.set_title("O-Information Over Training (Omega < 0 = Synergy)")
    ax.legend(loc="best")

    # Shade synergy region
    y_min, y_max = ax.get_ylim()
    if y_min < 0:
        ax.axhspan(y_min, 0, alpha=0.1, color=COLORBLIND_PALETTE[0], label=None)
        ax.text(
            steps_arr[0],
            y_min + 0.1 * abs(y_min),
            "Synergy dominates",
            fontsize=FONT_SIZE - 2,
            color=COLORBLIND_PALETTE[0],
        )

    plt.tight_layout()
    save_figure(fig, output_path)
    return fig


def plot_causal_emergence_gap(
    macro_ei: list[float],
    micro_ei: list[float],
    steps: list[int],
    output_path: str,
    figsize: tuple[float, float] = (10, 5),
) -> Figure:
    """Plot causal emergence: macro EI vs micro EI over time.

    Shows the EI gap (macro - micro) growing over training,
    which indicates increasing causal emergence.

    Args:
        macro_ei: Macro-scale Effective Information values.
        micro_ei: Micro-scale Effective Information values.
        steps: Training step numbers.
        output_path: Base path for output.
        figsize: Figure size in inches.

    Returns:
        Matplotlib Figure object.
    """
    setup_publication_style()
    fig, ax = plt.subplots(figsize=figsize)

    steps_arr = np.array(steps)
    macro_arr = np.array(macro_ei)
    micro_arr = np.array(micro_ei)
    gap = macro_arr - micro_arr

    # Plot both lines
    ax.plot(steps_arr, macro_arr, "-", color=COLORBLIND_PALETTE[0], linewidth=2, label="Macro EI")
    ax.plot(steps_arr, micro_arr, "--", color=COLORBLIND_PALETTE[3], linewidth=2, label="Micro EI")

    # Shade the gap region
    where_mask = list(macro_arr > micro_arr)  # Convert to list for type compat
    ax.fill_between(
        steps_arr,
        micro_arr,
        macro_arr,
        where=where_mask,
        alpha=0.3,
        color=COLORBLIND_PALETTE[2],
        label="Emergence gap (macro > micro)",
    )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Effective Information (bits)")
    ax.set_title("Causal Emergence: Macro vs Micro Effective Information")
    ax.legend(loc="best")

    # Annotate final gap
    final_gap = gap[-1] if len(gap) > 0 else 0
    ax.text(
        steps_arr[-1] * 0.7,
        (macro_arr[-1] + micro_arr[-1]) / 2,
        f"Final gap: {final_gap:.2f} bits",
        fontsize=FONT_SIZE,
        color=COLORBLIND_PALETTE[2],
    )

    plt.tight_layout()
    save_figure(fig, output_path)
    return fig


def plot_phase_transitions(
    metrics_dict: dict[str, list[float]],
    steps: list[int],
    output_path: str,
    transition_events: list[dict[str, Any]] | None = None,
    figsize: tuple[float, float] = (12, 6),
) -> Figure:
    """Plot multiple metrics with phase transition annotations.

    Baker et al. (2019) style visualization showing how multiple
    metrics evolve together with marked transition points.

    Args:
        metrics_dict: Dict mapping metric name to list of values.
        steps: Training step numbers.
        output_path: Base path for output.
        transition_events: Optional list of transition events, each with
            'step' (int) and 'label' (str) keys.
        figsize: Figure size in inches.

    Returns:
        Matplotlib Figure object.
    """
    setup_publication_style()
    fig, axes = plt.subplots(
        len(metrics_dict),
        1,
        figsize=(figsize[0], figsize[1] * len(metrics_dict) / 3),
        sharex=True,
    )

    if len(metrics_dict) == 1:
        axes = [axes]

    steps_arr = np.array(steps)

    for idx, (metric_name, values) in enumerate(metrics_dict.items()):
        ax = axes[idx]
        values_arr = np.array(values)

        color = COLORBLIND_PALETTE[idx % len(COLORBLIND_PALETTE)]
        ax.plot(steps_arr, values_arr, "-", color=color, linewidth=2)
        ax.set_ylabel(metric_name.replace("_", "\n"), fontsize=FONT_SIZE - 1)

        # Mark transition events
        if transition_events:
            for event in transition_events:
                event_step = event.get("step", 0)
                event_label = event.get("label", "")
                if steps_arr[0] <= event_step <= steps_arr[-1]:
                    ax.axvline(
                        x=event_step,
                        color="red",
                        linestyle="--",
                        alpha=0.7,
                    )
                    if idx == 0:  # Only label on top subplot
                        ax.text(
                            event_step,
                            ax.get_ylim()[1],
                            event_label,
                            fontsize=FONT_SIZE - 2,
                            rotation=90,
                            va="top",
                            ha="right",
                        )

    axes[-1].set_xlabel("Training Step")
    axes[0].set_title("Phase Transitions: Multiple Metrics Over Training")

    plt.tight_layout()
    save_figure(fig, output_path)
    return fig


# =============================================================================
# Utility Functions
# =============================================================================


def save_figure(
    fig: Figure,
    output_path: str,
    formats: list[str] | None = None,
    dpi: int = DPI,
) -> None:
    """Save figure in multiple formats.

    Args:
        fig: Matplotlib Figure object.
        output_path: Base path (without extension).
        formats: List of formats (default: ["pdf", "png"]).
        dpi: Resolution for raster formats.
    """
    if formats is None:
        formats = ["pdf", "png"]

    # Ensure parent directory exists
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Remove extension if present
    base_path = str(path.with_suffix(""))

    for fmt in formats:
        save_path = f"{base_path}.{fmt}"
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format=fmt)


def create_figure_grid(
    n_figures: int,
    figsize_per_panel: tuple[float, float] = (4, 3),
    max_cols: int = 3,
) -> tuple[Figure, list[plt.Axes]]:
    """Create a grid of subplots for multiple figures.

    Args:
        n_figures: Number of panels needed.
        figsize_per_panel: Size of each panel in inches.
        max_cols: Maximum number of columns.

    Returns:
        Tuple of (Figure, list of Axes).
    """
    setup_publication_style()

    n_cols = min(n_figures, max_cols)
    n_rows = (n_figures + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
    )

    # Flatten axes array
    if n_rows == 1 and n_cols == 1:
        axes_list = [axes]
    else:
        axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    # Hide unused axes
    for i in range(n_figures, len(axes_list)):
        axes_list[i].set_visible(False)

    return fig, axes_list[:n_figures]
