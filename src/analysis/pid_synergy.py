"""Pairwise PID Synergy via dit library.

Implements Partial Information Decomposition (PID) over agent pairs using
the dit library. PID decomposes mutual information I(S1, S2; T) into:
  - Synergy: information that S1 and S2 provide about T only when combined
  - Redundancy: information that both S1 and S2 individually provide about T
  - Unique_S1: information only S1 provides about T
  - Unique_S2: information only S2 provides about T

Variables:
  S1 = agent_i action (discrete, 6 values)
  S2 = field summary at agent_i position (K=2 quantile bins)
  T  = future food collected by agent_i (K=2 quantile bins)

Key insight: High synergy means the field + action together predict
outcomes better than either alone — a signature of emergence.

Reference: Williams & Beer (2010), Riedl et al. (2025).
"""

from __future__ import annotations

import importlib.util
from itertools import combinations
from typing import Any

import numpy as np

# Check if dit is available
_dit_available = importlib.util.find_spec("dit") is not None


def _patch_numpy_for_dit() -> None:
    """Patch np.alltrue removed in NumPy 2.0 (dit compat)."""
    if not hasattr(np, "alltrue"):
        np.alltrue = np.all  # type: ignore[attr-defined]


def discretize_continuous(data: np.ndarray, num_bins: int = 2) -> np.ndarray:
    """Discretize continuous data into K quantile bins.

    Uses quantile-based binning to ensure roughly equal counts per bin.

    Args:
        data: 1D array of continuous values.
        num_bins: Number of bins (default K=2 per PRD).

    Returns:
        1D array of integer bin indices (0 to num_bins-1).
    """
    data = np.asarray(data)

    if len(data) == 0:
        return np.array([], dtype=np.int64)

    # Handle constant data
    if np.std(data) < 1e-10:
        return np.zeros(len(data), dtype=np.int64)

    # Get unique values for the special case of discrete data
    unique_vals = np.unique(data)

    # If data is already discrete with <= num_bins values, map directly
    if len(unique_vals) <= num_bins:
        # Map unique values to bin indices
        val_to_bin = {v: i for i, v in enumerate(sorted(unique_vals))}
        binned = np.array([val_to_bin[v] for v in data], dtype=np.int64)
        return binned

    # For continuous data with many unique values, use quantile binning
    # Compute quantile edges
    quantiles = np.linspace(0, 100, num_bins + 1)
    edges = np.percentile(data, quantiles)

    # Use the interior edges for digitize
    # edges[0] is min, edges[-1] is max, we want edges[1:-1] as thresholds
    interior_edges = edges[1:-1]

    # Handle case where quantiles collapse (many ties)
    if len(np.unique(interior_edges)) == 0:
        # Fall back to median split
        median = float(np.median(data))
        binned_arr = (data > median).astype(np.int64)
        return np.asarray(binned_arr, dtype=np.int64)

    binned = np.digitize(data, interior_edges, right=False)

    return binned.astype(np.int64)


def apply_jeffreys_smoothing(counts: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Apply Jeffreys smoothing (alpha=0.5 pseudocounts) to count array.

    Jeffreys prior is the standard for avoiding zero probabilities in
    information-theoretic computations.

    Args:
        counts: Array of counts (histogram).
        alpha: Pseudocount to add to each bin (default 0.5 for Jeffreys).

    Returns:
        Smoothed count array (floats).
    """
    counts = np.asarray(counts, dtype=np.float64)
    return counts + alpha


def _build_joint_distribution(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    alpha: float = 0.5,
) -> tuple[list[str], list[float]]:
    """Build a joint distribution from three discrete variables with smoothing.

    Args:
        x, y, z: 1D arrays of discrete values (integers).
        alpha: Jeffreys smoothing pseudocount.

    Returns:
        Tuple of (outcomes, probabilities) for dit.Distribution.
    """
    x = np.asarray(x, dtype=int)
    y = np.asarray(y, dtype=int)
    z = np.asarray(z, dtype=int)

    if len(x) == 0:
        return [], []

    # Get unique values for each variable
    x_vals = sorted(set(x))
    y_vals = sorted(set(y))
    z_vals = sorted(set(z))

    # Build count dictionary for all possible combinations
    counts: dict[tuple[int, int, int], float] = {}
    for xv in x_vals:
        for yv in y_vals:
            for zv in z_vals:
                counts[(xv, yv, zv)] = alpha  # Start with pseudocount

    # Add observed counts
    for xi, yi, zi in zip(x, y, z):
        counts[(xi, yi, zi)] += 1

    # Convert to outcomes and probabilities
    total = sum(counts.values())
    outcomes = []
    probs = []

    for (xv, yv, zv), count in sorted(counts.items()):
        # Convert to string representation for dit
        outcomes.append(f"{xv}{yv}{zv}")
        probs.append(count / total)

    return outcomes, probs


def compute_interaction_information(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> float:
    """Compute Interaction Information II(X; Y; Z).

    Interaction Information is a quick screening function for synergy:
      II(X; Y; Z) = I(X; Y | Z) - I(X; Y)

    Our convention (matching PRD): II < 0 means synergy dominates.
    Note: dit library uses opposite convention, so we negate.

    Args:
        x, y, z: 1D arrays of discrete values (same length).

    Returns:
        Interaction Information value.
        - Negative = synergy dominates
        - Positive = redundancy dominates
        - Returns 0.0 for degenerate cases
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # Edge cases
    if len(x) == 0 or len(x) < 2:
        return 0.0

    # Check for constant variables
    if np.std(x) < 1e-10 or np.std(y) < 1e-10 or np.std(z) < 1e-10:
        return 0.0

    if not _dit_available:
        raise ImportError(
            "dit library is required for interaction information computation. "
            "Install with: pip install -e '.[phase5]'"
        )

    _patch_numpy_for_dit()

    from dit import Distribution
    from dit.multivariate import coinformation

    try:
        # Build joint distribution with Jeffreys smoothing
        outcomes, probs = _build_joint_distribution(x, y, z, alpha=0.5)

        if not outcomes:
            return 0.0

        d = Distribution(outcomes, probs)

        # Coinformation has the sign convention we want:
        # CI < 0 means synergy (same as "II < 0 means synergy" in PRD)
        ci = coinformation(d, [[0], [1], [2]])

        if not np.isfinite(ci):
            return 0.0

        return float(ci)

    except Exception:
        return 0.0


def compute_pairwise_pid(
    actions: np.ndarray,
    field_summary: np.ndarray,
    future_food: np.ndarray,
    num_bins: int = 2,
    alpha: float = 0.5,
) -> dict[str, float]:
    """Compute pairwise PID decomposition.

    Decomposes I(actions, field; future_food) into synergy, redundancy,
    and unique information components using Williams-Beer PID.

    Args:
        actions: 1D array of discrete action values (0-5 for 6 actions).
        field_summary: 1D array of continuous field readings (will be binned).
        future_food: 1D array of continuous future food values (will be binned).
        num_bins: Number of bins for discretization (default K=2).
        alpha: Jeffreys smoothing pseudocount (default 0.5).

    Returns:
        Dict with keys: synergy, redundancy, unique_s1, unique_s2
        All values are non-negative floats.
    """
    actions = np.asarray(actions)
    field_summary = np.asarray(field_summary)
    future_food = np.asarray(future_food)

    zero_result = {
        "synergy": 0.0,
        "redundancy": 0.0,
        "unique_s1": 0.0,
        "unique_s2": 0.0,
    }

    # Edge cases
    if len(actions) == 0 or len(actions) < 2:
        return zero_result

    # Discretize continuous variables
    field_binned = discretize_continuous(field_summary, num_bins)
    food_binned = discretize_continuous(future_food, num_bins)

    # Check for constant variables after binning
    if (
        np.std(actions) < 1e-10
        or np.std(field_binned) < 1e-10
        or np.std(food_binned) < 1e-10
    ):
        return zero_result

    if not _dit_available:
        raise ImportError(
            "dit library is required for PID computation. "
            "Install with: pip install -e '.[phase5]'"
        )

    _patch_numpy_for_dit()

    from dit import Distribution
    from dit.pid import PID_WB

    try:
        # Build joint distribution
        # S1 = actions, S2 = field, T = food
        outcomes, probs = _build_joint_distribution(
            actions.astype(int),
            field_binned,
            food_binned,
            alpha=alpha,
        )

        if not outcomes:
            return zero_result

        d = Distribution(outcomes, probs)

        # Compute PID with sources [S1], [S2] and target [T]
        # Variable indices: 0=actions, 1=field, 2=food
        pid = PID_WB(d, [[0], [1]], [2])

        # Extract PID components
        # pid lattice nodes:
        # ((0, 1),) = synergy (both sources together)
        # ((0,),) = unique to source 0 (actions)
        # ((1,),) = unique to source 1 (field)
        # ((0,), (1,)) = redundancy (shared by both)
        synergy = float(pid[((0, 1),)])
        unique_s1 = float(pid[((0,),)])
        unique_s2 = float(pid[((1,),)])
        redundancy = float(pid[((0,), (1,))])

        # Ensure non-negative (numerical precision)
        return {
            "synergy": max(0.0, synergy),
            "redundancy": max(0.0, redundancy),
            "unique_s1": max(0.0, unique_s1),
            "unique_s2": max(0.0, unique_s2),
        }

    except Exception:
        return zero_result


def compute_median_synergy(
    agent_actions: np.ndarray,
    field_summaries: np.ndarray,
    future_foods: np.ndarray,
    num_bins: int = 2,
) -> dict[str, Any]:
    """Compute median synergy across all agent pairs.

    For each pair of agents (i, j), computes PID synergy using:
    - Agent i's actions + Agent j's field summary -> Agent i's future food

    Args:
        agent_actions: (n_samples, n_agents) array of actions.
        field_summaries: (n_samples, n_agents) array of field readings.
        future_foods: (n_samples, n_agents) array of future food values.
        num_bins: Number of bins for discretization.

    Returns:
        Dict with keys:
        - median_synergy: median across all pairs
        - mean_synergy: mean across all pairs
        - std_synergy: std across all pairs
        - num_pairs: number of agent pairs
        - synergy_per_pair: list of (i, j, synergy) tuples
    """
    agent_actions = np.asarray(agent_actions)
    field_summaries = np.asarray(field_summaries)
    future_foods = np.asarray(future_foods)

    zero_result: dict[str, Any] = {
        "median_synergy": 0.0,
        "mean_synergy": 0.0,
        "std_synergy": 0.0,
        "num_pairs": 0,
        "synergy_per_pair": [],
    }

    # Check dimensions
    if agent_actions.ndim != 2:
        return zero_result

    n_samples, n_agents = agent_actions.shape

    if n_samples == 0 or n_agents < 2:
        return zero_result

    # Compute synergy for each pair
    synergies = []
    synergy_per_pair: list[tuple[int, int, float]] = []

    for i, j in combinations(range(n_agents), 2):
        # Use agent i's actions + agent j's field -> agent i's future food
        pid_result = compute_pairwise_pid(
            agent_actions[:, i],
            field_summaries[:, j],
            future_foods[:, i],
            num_bins=num_bins,
        )
        syn = pid_result["synergy"]
        synergies.append(syn)
        synergy_per_pair.append((i, j, syn))

    if not synergies:
        return zero_result

    synergies_arr = np.array(synergies)

    return {
        "median_synergy": float(np.median(synergies_arr)),
        "mean_synergy": float(np.mean(synergies_arr)),
        "std_synergy": float(np.std(synergies_arr)),
        "num_pairs": len(synergies),
        "synergy_per_pair": synergy_per_pair,
    }


def surrogate_significance_test(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    metric: str = "interaction_information",
    n_surrogates: int = 100,
    shuffle_type: str = "row",
    seed: int | None = None,
) -> dict[str, Any]:
    """Test statistical significance via surrogate testing.

    Generates surrogate data by shuffling and compares observed metric
    to the surrogate distribution.

    Args:
        x, y, z: 1D arrays of values.
        metric: Which metric to compute ("interaction_information" or "synergy").
        n_surrogates: Number of surrogate samples.
        shuffle_type: "row" (break cross-variable coordination) or
                      "column" (break temporal dependencies).
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys:
        - observed: observed metric value
        - surrogate_mean: mean of surrogate distribution
        - surrogate_std: std of surrogate distribution
        - p_value: fraction of surrogates more extreme than observed
        - significant: bool, True if p < 0.05
        - n_surrogates: number of surrogates used
        - shuffle_type: type of shuffle used
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    rng = np.random.default_rng(seed)

    # Compute observed metric
    if metric == "interaction_information":
        observed = compute_interaction_information(x, y, z)
    elif metric == "synergy":
        # Discretize first for PID
        x_disc = discretize_continuous(x.astype(float), num_bins=2) if x.dtype == float else x
        pid_result = compute_pairwise_pid(
            x_disc.astype(int),
            y.astype(float),
            z.astype(float),
        )
        observed = pid_result["synergy"]
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Generate surrogates
    surrogate_values = []

    for _ in range(n_surrogates):
        if shuffle_type == "row":
            # Shuffle one variable to break cross-variable coordination
            # Shuffle y relative to x and z
            y_shuffled = rng.permutation(y)
            if metric == "interaction_information":
                surr = compute_interaction_information(x, y_shuffled, z)
            else:
                x_disc = discretize_continuous(x.astype(float), num_bins=2) if x.dtype == float else x
                pid_result = compute_pairwise_pid(
                    x_disc.astype(int),
                    y_shuffled.astype(float),
                    z.astype(float),
                )
                surr = pid_result["synergy"]
        elif shuffle_type == "column":
            # Shuffle all variables together (break temporal structure)
            # This is a circular shift
            shift = rng.integers(1, len(x)) if len(x) > 1 else 0
            x_shifted = np.roll(x, shift)
            y_shifted = np.roll(y, shift)
            z_shifted = np.roll(z, shift)
            if metric == "interaction_information":
                surr = compute_interaction_information(x_shifted, y_shifted, z_shifted)
            else:
                x_disc = discretize_continuous(x_shifted.astype(float), num_bins=2) if x_shifted.dtype == float else x_shifted
                pid_result = compute_pairwise_pid(
                    x_disc.astype(int),
                    y_shifted.astype(float),
                    z_shifted.astype(float),
                )
                surr = pid_result["synergy"]
        else:
            raise ValueError(f"Unknown shuffle_type: {shuffle_type}")

        surrogate_values.append(surr)

    surrogate_arr = np.array(surrogate_values)
    surrogate_mean = float(np.mean(surrogate_arr))
    surrogate_std = float(np.std(surrogate_arr))

    # Compute p-value
    # For interaction information: synergy means observed < surrogates
    # For synergy: we want observed > surrogates (more synergy than null)
    if metric == "interaction_information":
        # How many surrogates are more negative (more synergistic)?
        n_more_extreme = np.sum(surrogate_arr <= observed)
    else:
        # How many surrogates have higher synergy?
        n_more_extreme = np.sum(surrogate_arr >= observed)

    p_value = float(n_more_extreme) / n_surrogates

    return {
        "observed": observed,
        "surrogate_mean": surrogate_mean,
        "surrogate_std": surrogate_std,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "n_surrogates": n_surrogates,
        "shuffle_type": shuffle_type,
    }
