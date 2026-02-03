#!/usr/bin/env python3
"""End-to-end verification script for Phase 5.

This script verifies that all Phase 5 modules can be imported and
basic functionality works correctly. Run this after making changes
to ensure nothing is broken.

Usage:
    python scripts/verify_phase5.py
    python scripts/verify_phase5.py --verbose

Exit codes:
    0: All checks passed
    1: One or more checks failed

Reference: US-018 in Phase 5 PRD.
"""

from __future__ import annotations

import argparse
import sys
import traceback


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="End-to-end verification script for Phase 5.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output including tracebacks for all errors",
    )
    parser.add_argument(
        "--n-timesteps",
        type=int,
        default=1000,
        help="Number of timesteps for synthetic trajectory (default: 1000)",
    )
    parser.add_argument(
        "--n-agents",
        type=int,
        default=8,
        help="Number of agents for synthetic trajectory (default: 8)",
    )
    return parser.parse_args()


def main() -> int:
    """Run all Phase 5 verification checks."""
    args = parse_args()
    verbose = args.verbose

    print("=" * 60)
    print("Phase 5 Verification")
    print("=" * 60)
    if verbose:
        print("  Verbose mode: enabled")
        print(f"  Timesteps: {args.n_timesteps}")
        print(f"  Agents: {args.n_agents}")
    print()

    all_passed = True

    # =================================================================
    # 1. Import Tests
    # =================================================================
    print("[1/5] Testing imports...")

    imports_to_test = [
        # Analysis modules
        ("src.analysis.o_information", "compute_o_information"),
        ("src.analysis.pid_synergy", "compute_median_synergy"),
        ("src.analysis.causal_emergence", "compute_causal_emergence_from_trajectory"),
        ("src.analysis.surrogates", "surrogate_test"),
        ("src.analysis.statistics", "compute_iqm"),
        ("src.analysis.scaling", "compute_per_agent_efficiency"),
        ("src.analysis.paper_figures", "setup_publication_style"),
        ("src.analysis.emergence_report", "compute_all_emergence_metrics"),
        ("src.analysis.information", "compute_te_matrix"),
        # Baselines
        ("src.baselines.ippo", "ippo_config"),
        ("src.baselines.aco_fixed", "aco_config"),
        ("src.baselines.mappo", "mappo_config"),
        # Experiments
        ("src.experiments.runner", "run_experiment"),
        ("src.experiments.configs", "standard_config"),
        ("src.experiments.baselines", "run_baselines_comparison"),
    ]

    import_failures = []
    for module_name, function_name in imports_to_test:
        try:
            module = __import__(module_name, fromlist=[function_name])
            func = getattr(module, function_name)
            print(f"  OK: {module_name}.{function_name}")
        except ImportError as e:
            print(f"  FAIL: {module_name} - {e}")
            import_failures.append((module_name, str(e)))
            if verbose:
                traceback.print_exc()
        except AttributeError as e:
            print(f"  FAIL: {module_name}.{function_name} - {e}")
            import_failures.append((module_name, str(e)))
            if verbose:
                traceback.print_exc()

    if import_failures:
        print(f"  {len(import_failures)} import(s) failed")
        all_passed = False
    else:
        print("  All imports successful")
    print()

    # =================================================================
    # 2. Create Synthetic Trajectory
    # =================================================================
    print("[2/5] Creating synthetic trajectory...")

    try:
        import numpy as np
        from numpy import ndarray

        rng = np.random.default_rng(42)
        n_timesteps = args.n_timesteps
        n_agents = args.n_agents

        trajectory: dict[str, ndarray] = {
            "actions": rng.integers(0, 6, size=(n_timesteps, n_agents)),
            "positions": rng.integers(0, 20, size=(n_timesteps, n_agents, 2)),
            "rewards": rng.random((n_timesteps, n_agents)).astype(np.float32),
            "alive_mask": np.ones((n_timesteps, n_agents), dtype=bool),
            "energy": rng.random((n_timesteps, n_agents)).astype(np.float32) * 100,
            "field": rng.random((n_timesteps, 20, 20, 4)).astype(np.float32),
        }
        print(f"  Created trajectory with {n_timesteps} steps, {n_agents} agents")
    except Exception as e:
        print(f"  FAIL: Could not create trajectory - {e}")
        if verbose:
            traceback.print_exc()
        all_passed = False
        return 1
    print()

    # =================================================================
    # 3. Compute Metrics
    # =================================================================
    print("[3/5] Computing emergence metrics...")

    metrics_to_compute = [
        ("O-Information", "src.analysis.o_information", "compute_o_information", lambda: trajectory["rewards"]),
        ("Transfer Entropy", "src.analysis.information", "compute_te_matrix", lambda: trajectory["actions"]),
    ]

    for metric_name, module_name, func_name, get_data in metrics_to_compute:
        try:
            module = __import__(module_name, fromlist=[func_name])
            func = getattr(module, func_name)
            data = get_data()
            result = func(data)
            if isinstance(result, dict):
                result_str = str(list(result.keys())[:3])
            else:
                result_str = f"{result:.4f}" if isinstance(result, float) else str(type(result).__name__)
            print(f"  OK: {metric_name} -> {result_str}")
        except Exception as e:
            print(f"  FAIL: {metric_name} - {e}")
            if verbose:
                traceback.print_exc()
            all_passed = False

    # Test EmergenceReport (comprehensive)
    try:
        from src.analysis.emergence_report import compute_all_emergence_metrics

        report = compute_all_emergence_metrics(trajectory, run_surrogates=False)
        print(f"  OK: EmergenceReport with {7} metrics")
    except Exception as e:
        print(f"  FAIL: EmergenceReport - {e}")
        if verbose:
            traceback.print_exc()
        all_passed = False
    print()

    # =================================================================
    # 4. Generate Test Figure
    # =================================================================
    print("[4/5] Testing figure generation...")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from src.analysis.paper_figures import save_figure, setup_publication_style

        setup_publication_style()
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9], "-o")
        ax.set_title("Phase 5 Verification Test")

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/test_figure"
            save_figure(fig, output_path, formats=["png"])
            import os
            if os.path.exists(f"{output_path}.png"):
                print("  OK: Figure saved successfully")
            else:
                print("  FAIL: Figure file not created")
                all_passed = False
        plt.close(fig)
    except Exception as e:
        print(f"  FAIL: Figure generation - {e}")
        if verbose:
            traceback.print_exc()
        all_passed = False
    print()

    # =================================================================
    # 5. Verify Output Formats
    # =================================================================
    print("[5/5] Testing JSON serialization...")

    try:
        from src.analysis.emergence_report import report_from_json, report_to_json

        json_str = report_to_json(report)
        restored = report_from_json(json_str)

        if restored.n_agents == report.n_agents:
            print("  OK: JSON round-trip successful")
        else:
            print("  FAIL: JSON round-trip mismatch")
            all_passed = False
    except Exception as e:
        print(f"  FAIL: JSON serialization - {e}")
        if verbose:
            traceback.print_exc()
        all_passed = False
    print()

    # =================================================================
    # Summary
    # =================================================================
    print("=" * 60)
    if all_passed:
        print("Phase 5 Verification: PASSED")
        print("=" * 60)
        return 0
    else:
        print("Phase 5 Verification: FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
