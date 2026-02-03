#!/usr/bin/env python3
"""CLI for generating publication-quality figures.

Usage:
    python scripts/generate_paper_figures.py \\
        --results-dir results/ \\
        --output-dir figures/

    # Dry run (show what figures would be generated)
    python scripts/generate_paper_figures.py --dry-run \\
        --results-dir results/ --output-dir figures/

    # Generate specific figures only
    python scripts/generate_paper_figures.py \\
        --results-dir results/ --output-dir figures/ \\
        --figures scaling,ablation

Reference: US-016 in Phase 5 PRD.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

# Available figure types
AVAILABLE_FIGURES = [
    "scaling",
    "ablation",
    "performance_profile",
    "poi",
    "synergy_heatmap",
    "o_information",
    "causal_emergence",
    "phase_transitions",
]


def load_results(results_dir: str, filename: str) -> dict | None:
    """Load results from pickle file.

    Args:
        results_dir: Directory containing results.
        filename: Filename to load.

    Returns:
        Loaded dict or None if not found.
    """
    path = Path(results_dir) / filename
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  Warning: Could not load {path}: {e}")
        return None


def generate_scaling_figure(results_dir: str, output_dir: str, verbose: bool) -> bool:
    """Generate scaling curve figure.

    Returns:
        True if successful, False otherwise.
    """
    from src.analysis.paper_figures import plot_scaling_curve

    # Look for scaling results
    results = load_results(results_dir, "scaling_results.pkl")
    if results is None:
        print("  Skipping: No scaling_results.pkl found")
        return False

    if verbose:
        print(f"  Loaded scaling results with {len(results)} conditions")

    output_path = str(Path(output_dir) / "scaling_curve")
    plot_scaling_curve(results, output_path)
    print(f"  Saved: {output_path}.pdf, {output_path}.png")
    return True


def generate_ablation_figure(results_dir: str, output_dir: str, verbose: bool) -> bool:
    """Generate ablation bar chart.

    Returns:
        True if successful, False otherwise.
    """
    from src.analysis.paper_figures import plot_ablation_bars

    results = load_results(results_dir, "ablation_results.pkl")
    if results is None:
        print("  Skipping: No ablation_results.pkl found")
        return False

    if verbose:
        print(f"  Loaded ablation results with {len(results)} conditions")

    output_path = str(Path(output_dir) / "ablation_bars")
    plot_ablation_bars(results, output_path)
    print(f"  Saved: {output_path}.pdf, {output_path}.png")
    return True


def generate_performance_profile_figure(results_dir: str, output_dir: str, verbose: bool) -> bool:
    """Generate performance profile figure.

    Returns:
        True if successful, False otherwise.
    """
    from src.analysis.paper_figures import plot_performance_profiles

    results = load_results(results_dir, "baselines_comparison.pkl")
    if results is None:
        print("  Skipping: No baselines_comparison.pkl found")
        return False

    # Extract scores per method
    score_dict = {}
    if hasattr(results, "method_results"):
        for method, method_result in results.method_results.items():
            if hasattr(method_result, "scores"):
                score_dict[method] = np.array(method_result.scores)
    elif isinstance(results, dict):
        for method, method_result in results.items():
            if isinstance(method_result, dict) and "scores" in method_result:
                score_dict[method] = np.array(method_result["scores"])
            elif hasattr(method_result, "scores"):
                score_dict[method] = np.array(method_result.scores)

    if not score_dict:
        print("  Skipping: No score data found in results")
        return False

    if verbose:
        print(f"  Loaded scores for {len(score_dict)} methods")

    output_path = str(Path(output_dir) / "performance_profiles")
    plot_performance_profiles(score_dict, output_path)
    print(f"  Saved: {output_path}.pdf, {output_path}.png")
    return True


def generate_poi_figure(results_dir: str, output_dir: str, verbose: bool) -> bool:
    """Generate probability of improvement heatmap.

    Returns:
        True if successful, False otherwise.
    """
    from src.analysis.paper_figures import plot_probability_of_improvement

    results = load_results(results_dir, "baselines_comparison.pkl")
    if results is None:
        print("  Skipping: No baselines_comparison.pkl found")
        return False

    # Extract scores per method
    score_dict = {}
    if hasattr(results, "method_results"):
        for method, method_result in results.method_results.items():
            if hasattr(method_result, "scores"):
                score_dict[method] = np.array(method_result.scores)
    elif isinstance(results, dict):
        for method, method_result in results.items():
            if isinstance(method_result, dict) and "scores" in method_result:
                score_dict[method] = np.array(method_result["scores"])
            elif hasattr(method_result, "scores"):
                score_dict[method] = np.array(method_result.scores)

    if not score_dict:
        print("  Skipping: No score data found in results")
        return False

    if verbose:
        print(f"  Loaded scores for {len(score_dict)} methods")

    output_path = str(Path(output_dir) / "probability_of_improvement")
    plot_probability_of_improvement(score_dict, output_path)
    print(f"  Saved: {output_path}.pdf, {output_path}.png")
    return True


def generate_synergy_heatmap_figure(results_dir: str, output_dir: str, verbose: bool) -> bool:
    """Generate PID synergy heatmap.

    Returns:
        True if successful, False otherwise.
    """
    from src.analysis.paper_figures import plot_pid_synergy_heatmap

    results = load_results(results_dir, "emergence_metrics.pkl")
    if results is None:
        results = load_results(results_dir, "synergy_matrix.pkl")
    if results is None:
        print("  Skipping: No emergence_metrics.pkl or synergy_matrix.pkl found")
        return False

    # Extract synergy matrix
    synergy_matrix = None
    if isinstance(results, np.ndarray):
        synergy_matrix = results
    elif hasattr(results, "synergy_matrix"):
        synergy_matrix = results.synergy_matrix
    elif isinstance(results, dict) and "synergy_matrix" in results:
        synergy_matrix = results["synergy_matrix"]

    if synergy_matrix is None:
        print("  Skipping: No synergy_matrix found in results")
        return False

    if verbose:
        print(f"  Loaded synergy matrix of shape {synergy_matrix.shape}")

    output_path = str(Path(output_dir) / "synergy_heatmap")
    plot_pid_synergy_heatmap(synergy_matrix, output_path)
    print(f"  Saved: {output_path}.pdf, {output_path}.png")
    return True


def generate_o_information_figure(results_dir: str, output_dir: str, verbose: bool) -> bool:
    """Generate O-information trajectory figure.

    Returns:
        True if successful, False otherwise.
    """
    from src.analysis.paper_figures import plot_o_information_trajectory

    results = load_results(results_dir, "emergence_metrics.pkl")
    if results is None:
        results = load_results(results_dir, "o_information.pkl")
    if results is None:
        print("  Skipping: No emergence_metrics.pkl or o_information.pkl found")
        return False

    # Extract O-information trajectory
    o_info_values = None
    steps = None

    if hasattr(results, "windowed_results"):
        # From EmergenceReport
        windowed = results.windowed_results
        if windowed:
            o_info_values = [w["metrics"].get("o_information", 0) for w in windowed]
            steps = [w["window_start"] for w in windowed]
    elif isinstance(results, dict):
        if "o_info_values" in results:
            o_info_values = results["o_info_values"]
            steps = results.get("steps", list(range(len(o_info_values))))
        elif "windowed_results" in results:
            windowed = results["windowed_results"]
            o_info_values = [w["metrics"].get("o_information", 0) for w in windowed]
            steps = [w["window_start"] for w in windowed]

    if o_info_values is None or steps is None:
        print("  Skipping: No O-information trajectory data found")
        return False

    if verbose:
        print(f"  Loaded O-information trajectory with {len(o_info_values)} points")

    output_path = str(Path(output_dir) / "o_information_trajectory")
    plot_o_information_trajectory(o_info_values, steps, output_path)
    print(f"  Saved: {output_path}.pdf, {output_path}.png")
    return True


def generate_causal_emergence_figure(results_dir: str, output_dir: str, verbose: bool) -> bool:
    """Generate causal emergence gap figure.

    Returns:
        True if successful, False otherwise.
    """
    from src.analysis.paper_figures import plot_causal_emergence_gap

    results = load_results(results_dir, "emergence_metrics.pkl")
    if results is None:
        results = load_results(results_dir, "causal_emergence.pkl")
    if results is None:
        print("  Skipping: No emergence_metrics.pkl or causal_emergence.pkl found")
        return False

    # Extract EI data
    macro_ei = None
    micro_ei = None
    steps = None

    if isinstance(results, dict):
        if "macro_ei" in results:
            macro_ei = results["macro_ei"]
            micro_ei = results.get("micro_ei", [0] * len(macro_ei))
            steps = results.get("steps", list(range(len(macro_ei))))
        elif "windowed_results" in results:
            windowed = results["windowed_results"]
            macro_ei = [w["metrics"].get("macro_ei", 0) for w in windowed]
            micro_ei = [w["metrics"].get("micro_ei", 0) for w in windowed]
            steps = [w["window_start"] for w in windowed]

    if macro_ei is None or steps is None:
        print("  Skipping: No causal emergence data found")
        return False

    if verbose:
        print(f"  Loaded causal emergence data with {len(macro_ei)} points")

    output_path = str(Path(output_dir) / "causal_emergence_gap")
    plot_causal_emergence_gap(macro_ei, micro_ei, steps, output_path)
    print(f"  Saved: {output_path}.pdf, {output_path}.png")
    return True


def generate_phase_transitions_figure(results_dir: str, output_dir: str, verbose: bool) -> bool:
    """Generate phase transitions figure.

    Returns:
        True if successful, False otherwise.
    """
    from src.analysis.paper_figures import plot_phase_transitions

    results = load_results(results_dir, "emergence_metrics.pkl")
    if results is None:
        results = load_results(results_dir, "phase_transitions.pkl")
    if results is None:
        print("  Skipping: No emergence_metrics.pkl or phase_transitions.pkl found")
        return False

    # Extract metrics dict and events
    metrics_dict = {}
    steps = None
    events = None

    if isinstance(results, dict):
        if "metrics_dict" in results:
            metrics_dict = results["metrics_dict"]
            steps = results.get("steps", [])
            events = results.get("transition_events", [])
        elif "windowed_results" in results:
            windowed = results["windowed_results"]
            if windowed:
                steps = [w["window_start"] for w in windowed]
                # Collect all metrics
                for key in windowed[0].get("metrics", {}).keys():
                    metrics_dict[key] = [w["metrics"].get(key, 0) for w in windowed]

    if not metrics_dict or not steps:
        print("  Skipping: No phase transition data found")
        return False

    if verbose:
        print(f"  Loaded {len(metrics_dict)} metrics over {len(steps)} steps")

    output_path = str(Path(output_dir) / "phase_transitions")
    plot_phase_transitions(metrics_dict, steps, output_path, events)
    print(f"  Saved: {output_path}.pdf, {output_path}.png")
    return True


# Figure generators mapping
FIGURE_GENERATORS = {
    "scaling": generate_scaling_figure,
    "ablation": generate_ablation_figure,
    "performance_profile": generate_performance_profile_figure,
    "poi": generate_poi_figure,
    "synergy_heatmap": generate_synergy_heatmap_figure,
    "o_information": generate_o_information_figure,
    "causal_emergence": generate_causal_emergence_figure,
    "phase_transitions": generate_phase_transitions_figure,
}


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for Phase 5 experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing experiment results (pickles)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for output figures",
    )
    parser.add_argument(
        "--figures",
        type=str,
        default="all",
        help=f"Comma-separated list of figures to generate (default: all). Available: {', '.join(AVAILABLE_FIGURES)}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what figures would be generated without running",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output",
    )

    args = parser.parse_args()

    # Parse figure list
    if args.figures == "all":
        figures_to_generate = AVAILABLE_FIGURES
    else:
        figures_to_generate = [f.strip() for f in args.figures.split(",")]
        for fig in figures_to_generate:
            if fig not in AVAILABLE_FIGURES:
                print(f"ERROR: Unknown figure type: {fig}")
                print(f"Available: {', '.join(AVAILABLE_FIGURES)}")
                return 1

    # Dry run mode
    if args.dry_run:
        print("DRY RUN - Would generate figures:")
        print(f"  Results dir: {args.results_dir}")
        print(f"  Output dir: {args.output_dir}")
        print("  Figures to generate:")
        for fig in figures_to_generate:
            print(f"    - {fig}")
        print()
        print("Required result files:")
        print("  - scaling_results.pkl (for scaling figure)")
        print("  - ablation_results.pkl (for ablation figure)")
        print("  - baselines_comparison.pkl (for performance_profile, poi)")
        print("  - emergence_metrics.pkl (for synergy, o_information, causal_emergence, phase_transitions)")
        return 0

    # Check results directory exists
    if not Path(args.results_dir).exists():
        print(f"ERROR: Results directory not found: {args.results_dir}")
        return 1

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Import here to avoid slow import on --help or --dry-run
    try:
        from src.analysis.paper_figures import setup_publication_style
        setup_publication_style()
    except ImportError as e:
        print(f"ERROR: Failed to import paper_figures module: {e}")
        return 1

    print(f"Generating figures from: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Generate each figure
    n_success = 0
    n_skipped = 0

    for fig_name in figures_to_generate:
        print(f"[{fig_name}]")
        generator = FIGURE_GENERATORS.get(fig_name)
        if generator is None:
            print(f"  ERROR: No generator for {fig_name}")
            continue

        try:
            success = generator(args.results_dir, args.output_dir, args.verbose)
            if success:
                n_success += 1
            else:
                n_skipped += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            n_skipped += 1

    print()
    print(f"Done: {n_success} figures generated, {n_skipped} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
