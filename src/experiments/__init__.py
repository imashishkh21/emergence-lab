"""Experiment harness and utilities for multi-seed experiments.

This module provides infrastructure for running experiments with:
- Multiple seeds for statistical significance
- Paired seed support (same seed across methods for reduced variance)
- Standardized result formats compatible with rliable analysis
- Save/load functionality for experiment results

Key components:
    - ExperimentConfig: Configuration for an experiment run
    - ExperimentResult: Aggregated results with statistics
    - run_experiment: Main function to run experiments
    - Environment configs: standard, hidden_resources, food_scarcity
"""
