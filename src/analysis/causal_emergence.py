"""Causal Emergence metrics for measuring macro-scale causal power.

Implements two complementary causal emergence measures:
1. Hoel's Effective Information (EI): compares determinism and degeneracy of
   micro vs macro transition probability matrices (TPMs).
2. Rosas' Psi: measures whether a macro variable has causal power beyond the
   sum of its micro-level parts.

Key interpretation:
  - EI gap (macro_EI - micro_EI) > 0: causal emergence at macro scale
  - Psi > 0: macro variable has emergent causal power

References:
  - Hoel et al. (2013), "Quantifying Causal Emergence Shows That Macro Can Beat Micro"
  - Rosas et al. (2020), "Reconciling Causation and Abstraction Through Emergence"
  - Comolatti & Hoel (2022), "Causal Emergence in Biological Networks"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


def discretize_to_bins(
    values: np.ndarray,
    num_bins: int = 4,
    method: str = "quantile",
) -> np.ndarray:
    """Discretize continuous values into bins.

    Args:
        values: (N,) array of continuous values.
        num_bins: Number of bins for discretization.
        method: Binning method:
            - "quantile": equal-frequency bins (default)
            - "uniform": equal-width bins

    Returns:
        (N,) array of bin indices (0 to num_bins-1).
    """
    values = np.asarray(values, dtype=np.float64)

    # Handle constant values
    if np.std(values) < 1e-10:
        return np.zeros(len(values), dtype=np.int32)

    # Handle NaN by replacing with median
    if np.any(np.isnan(values)):
        median_val = np.nanmedian(values)
        values = np.where(np.isnan(values), median_val, values)

    if method == "quantile":
        # Equal-frequency bins
        try:
            bins = np.percentile(
                values, np.linspace(0, 100, num_bins + 1)[1:-1]
            )
            bins = np.unique(bins)  # Remove duplicates for constant-ish data
            if len(bins) == 0:
                return np.zeros(len(values), dtype=np.int32)
            return np.digitize(values, bins).astype(np.int32)
        except Exception:
            return np.zeros(len(values), dtype=np.int32)
    elif method == "uniform":
        # Equal-width bins
        min_val, max_val = np.min(values), np.max(values)
        if max_val - min_val < 1e-10:
            return np.zeros(len(values), dtype=np.int32)
        bins = np.linspace(min_val, max_val, num_bins + 1)[1:-1]
        return np.digitize(values, bins).astype(np.int32)
    else:
        raise ValueError(f"Unknown binning method: {method}")


def build_tpm(
    states_t: np.ndarray,
    states_t1: np.ndarray,
    num_states: int | None = None,
    smoothing: float = 1e-10,
) -> np.ndarray:
    """Build transition probability matrix from state sequences.

    Args:
        states_t: (N,) array of discrete states at time t.
        states_t1: (N,) array of discrete states at time t+1.
        num_states: Number of possible states. If None, inferred from data.
        smoothing: Laplace smoothing constant to avoid zero probabilities.

    Returns:
        (num_states, num_states) TPM where TPM[i,j] = P(state_t1=j | state_t=i).
    """
    states_t = np.asarray(states_t, dtype=np.int32)
    states_t1 = np.asarray(states_t1, dtype=np.int32)

    if num_states is None:
        num_states = max(np.max(states_t), np.max(states_t1)) + 1

    # Count transitions
    tpm = np.zeros((num_states, num_states), dtype=np.float64) + smoothing

    for i in range(len(states_t)):
        if 0 <= states_t[i] < num_states and 0 <= states_t1[i] < num_states:
            tpm[states_t[i], states_t1[i]] += 1

    # Normalize rows to get probabilities
    row_sums = tpm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-10, 1.0, row_sums)  # Avoid division by zero
    result: np.ndarray = tpm / row_sums

    return result


def compute_tpm_entropy(tpm: np.ndarray) -> float:
    """Compute average row entropy of a TPM (conditional entropy).

    This measures the average uncertainty in the next state given the current state.
    Lower entropy = more deterministic transitions.

    Args:
        tpm: (num_states, num_states) transition probability matrix.

    Returns:
        Average row entropy in bits.
    """
    tpm = np.asarray(tpm, dtype=np.float64)

    # Compute entropy for each row, then average
    # H(row) = -sum(p * log2(p)) for p > 0
    row_entropies = []
    for row in tpm:
        # Avoid log(0)
        row = np.clip(row, 1e-10, 1.0)
        row = row / row.sum()  # Renormalize after clipping
        h = -np.sum(row * np.log2(row))
        row_entropies.append(h)

    return float(np.mean(row_entropies))


def compute_effective_information(
    micro_tpm: np.ndarray,
    macro_tpm: np.ndarray,
) -> dict[str, float]:
    """Compute Hoel's Effective Information and the EI gap.

    Effective Information measures how much information is generated by
    the causal structure of a system. It balances determinism (low output
    entropy given input) against degeneracy (variety of outputs reached).

    EI = H_max(output) - <H(output | input)>
       = log(N) - average_row_entropy(TPM)

    A positive EI gap (macro_EI > micro_EI) indicates causal emergence:
    the macro-level description captures more causal information than
    the micro-level description.

    Args:
        micro_tpm: (n_micro, n_micro) TPM at micro-scale (e.g., joint agent states).
        macro_tpm: (n_macro, n_macro) TPM at macro-scale (e.g., population count).

    Returns:
        Dict with keys:
            - "micro_ei": Effective Information at micro scale.
            - "macro_ei": Effective Information at macro scale.
            - "ei_gap": macro_EI - micro_EI (positive = causal emergence).
            - "micro_entropy": Average conditional entropy at micro scale.
            - "macro_entropy": Average conditional entropy at macro scale.
    """
    micro_tpm = np.asarray(micro_tpm, dtype=np.float64)
    macro_tpm = np.asarray(macro_tpm, dtype=np.float64)

    n_micro = micro_tpm.shape[0]
    n_macro = macro_tpm.shape[0]

    # Compute conditional entropies (average row entropy)
    micro_entropy = compute_tpm_entropy(micro_tpm)
    macro_entropy = compute_tpm_entropy(macro_tpm)

    # Effective Information: H_max - H_conditional
    # H_max = log2(N) for uniform input distribution (interventional perspective)
    micro_ei = np.log2(n_micro) - micro_entropy if n_micro > 1 else 0.0
    macro_ei = np.log2(n_macro) - macro_entropy if n_macro > 1 else 0.0

    # EI gap: positive means causal emergence at macro scale
    ei_gap = macro_ei - micro_ei

    return {
        "micro_ei": float(micro_ei),
        "macro_ei": float(macro_ei),
        "ei_gap": float(ei_gap),
        "micro_entropy": float(micro_entropy),
        "macro_entropy": float(macro_entropy),
    }


def compute_mutual_information_discrete(
    x: np.ndarray,
    y: np.ndarray,
    num_states_x: int | None = None,
    num_states_y: int | None = None,
) -> float:
    """Compute mutual information I(X; Y) for discrete variables.

    Args:
        x: (N,) array of discrete states for variable X.
        y: (N,) array of discrete states for variable Y.
        num_states_x: Number of possible states for X. If None, inferred.
        num_states_y: Number of possible states for Y. If None, inferred.

    Returns:
        Mutual information in bits.
    """
    x = np.asarray(x, dtype=np.int32)
    y = np.asarray(y, dtype=np.int32)

    if len(x) != len(y):
        raise ValueError(f"x and y must have same length: {len(x)} vs {len(y)}")

    if num_states_x is None:
        num_states_x = np.max(x) + 1
    if num_states_y is None:
        num_states_y = np.max(y) + 1

    # Build joint distribution with Laplace smoothing
    smoothing = 1e-10
    joint = np.zeros((num_states_x, num_states_y), dtype=np.float64) + smoothing

    for i in range(len(x)):
        if 0 <= x[i] < num_states_x and 0 <= y[i] < num_states_y:
            joint[x[i], y[i]] += 1

    # Normalize
    joint = joint / joint.sum()

    # Marginals
    p_x = joint.sum(axis=1)
    p_y = joint.sum(axis=0)

    # MI = sum_{x,y} p(x,y) * log(p(x,y) / (p(x) * p(y)))
    mi = 0.0
    for i in range(num_states_x):
        for j in range(num_states_y):
            if joint[i, j] > 1e-10:
                mi += joint[i, j] * np.log2(joint[i, j] / (p_x[i] * p_y[j] + 1e-10))

    return max(0.0, float(mi))


def compute_rosas_psi(
    macro_var_t: np.ndarray,
    macro_var_t1: np.ndarray,
    micro_vars_t: np.ndarray,
    num_bins: int = 4,
) -> dict[str, float]:
    """Compute Rosas' Psi metric for causal emergence.

    Psi measures whether a macro variable V has causal power beyond the sum
    of its micro-level components. Positive Psi indicates emergent causation.

    Psi(V) = I(V_t; V_{t+1}) - sum_i I(X_{i,t}; V_{t+1})

    Where:
      - V is the macro variable (e.g., population count)
      - X_i are micro variables (e.g., individual agent states)
      - t and t+1 are consecutive time steps

    Args:
        macro_var_t: (N,) array of macro variable at time t.
        macro_var_t1: (N,) array of macro variable at time t+1.
        micro_vars_t: (N, n_micro) array of micro variables at time t.
        num_bins: Number of bins for discretizing continuous variables.

    Returns:
        Dict with keys:
            - "psi": The Psi metric (positive = emergent causation).
            - "macro_mi": I(V_t; V_{t+1}) - macro self-prediction.
            - "sum_micro_mi": sum_i I(X_i,t; V_{t+1}) - micro contributions.
            - "n_micro": Number of micro variables.
    """
    macro_t = np.asarray(macro_var_t, dtype=np.float64)
    macro_t1 = np.asarray(macro_var_t1, dtype=np.float64)
    micro_t = np.asarray(micro_vars_t, dtype=np.float64)

    if len(macro_t) != len(macro_t1):
        raise ValueError(
            f"macro_var_t and macro_var_t1 must have same length: "
            f"{len(macro_t)} vs {len(macro_t1)}"
        )

    if micro_t.ndim == 1:
        micro_t = micro_t[:, np.newaxis]

    n_samples, n_micro = micro_t.shape

    if n_samples != len(macro_t):
        raise ValueError(
            f"micro_vars_t samples ({n_samples}) must match macro ({len(macro_t)})"
        )

    # Handle degenerate cases
    if n_samples < 10:
        return {
            "psi": 0.0,
            "macro_mi": 0.0,
            "sum_micro_mi": 0.0,
            "n_micro": n_micro,
        }

    # Discretize all variables
    macro_t_discrete = discretize_to_bins(macro_t, num_bins=num_bins)
    macro_t1_discrete = discretize_to_bins(macro_t1, num_bins=num_bins)

    # Compute macro self-prediction: I(V_t; V_{t+1})
    macro_mi = compute_mutual_information_discrete(macro_t_discrete, macro_t1_discrete)

    # Compute sum of micro contributions: sum_i I(X_i,t; V_{t+1})
    sum_micro_mi = 0.0
    for i in range(n_micro):
        micro_i_discrete = discretize_to_bins(micro_t[:, i], num_bins=num_bins)
        mi_i = compute_mutual_information_discrete(micro_i_discrete, macro_t1_discrete)
        sum_micro_mi += mi_i

    # Psi = macro_mi - sum_micro_mi
    psi = macro_mi - sum_micro_mi

    return {
        "psi": float(psi),
        "macro_mi": float(macro_mi),
        "sum_micro_mi": float(sum_micro_mi),
        "n_micro": n_micro,
    }


def extract_macro_variables(
    trajectory: dict[str, np.ndarray],
    num_bins: int = 4,
) -> dict[str, np.ndarray]:
    """Extract macro-level variables from a trajectory for causal emergence analysis.

    Macro variable candidates from PRD:
      - population_count: sum(alive)
      - mean_field_intensity: mean(field_values)
      - total_food_collected: cumulative food
      - spatial_dispersion: std(agent_positions)
      - field_entropy: shannon_entropy(field)

    Args:
        trajectory: Dict with keys like 'alive_mask' (T, num_agents),
            'field' (T, H, W, C), 'positions' (T, num_agents, 2),
            'rewards' (T, num_agents), etc.
        num_bins: Number of bins for discretization (used in TPM building).

    Returns:
        Dict mapping macro variable names to (T,) arrays of values.
    """
    macros: dict[str, np.ndarray] = {}

    # Population count: sum of alive agents per timestep
    if "alive_mask" in trajectory:
        alive = np.asarray(trajectory["alive_mask"], dtype=bool)
        macros["population_count"] = alive.sum(axis=1).astype(np.float64)

    # Mean field intensity
    if "field" in trajectory:
        field = np.asarray(trajectory["field"], dtype=np.float64)
        # field shape: (T, H, W, C) or (T, H, W)
        axes_to_mean = tuple(range(1, field.ndim))
        macros["mean_field_intensity"] = field.mean(axis=axes_to_mean)

    # Spatial dispersion: std of agent positions (only alive agents)
    if "positions" in trajectory:
        positions = np.asarray(trajectory["positions"], dtype=np.float64)
        # positions shape: (n_steps, num_agents, 2)
        n_steps = positions.shape[0]
        dispersion = np.zeros(n_steps, dtype=np.float64)

        if "alive_mask" in trajectory:
            alive = np.asarray(trajectory["alive_mask"], dtype=bool)
            for t in range(n_steps):
                alive_pos = positions[t, alive[t], :]
                if len(alive_pos) > 1:
                    # Dispersion = mean of std across x and y coordinates
                    dispersion[t] = np.mean(np.std(alive_pos, axis=0))
        else:
            for t in range(n_steps):
                dispersion[t] = np.mean(np.std(positions[t], axis=0))

        macros["spatial_dispersion"] = dispersion

    # Total food collected (cumulative)
    if "rewards" in trajectory:
        rewards = np.asarray(trajectory["rewards"], dtype=np.float64)
        # rewards shape: (T, num_agents)
        total_per_step = rewards.sum(axis=1)
        macros["total_food_collected"] = np.cumsum(total_per_step)

    # Field entropy (Shannon entropy of discretized field values)
    if "field" in trajectory:
        field = np.asarray(trajectory["field"], dtype=np.float64)
        n_field_steps = field.shape[0]
        entropies = np.zeros(n_field_steps, dtype=np.float64)

        for t in range(n_field_steps):
            flat_field = field[t].flatten()
            # Discretize field values
            binned = discretize_to_bins(flat_field, num_bins=num_bins)
            # Compute entropy
            counts = np.bincount(binned, minlength=num_bins)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            if len(probs) > 0:
                entropies[t] = float(stats.entropy(probs, base=2))

        macros["field_entropy"] = entropies

    return macros


def compute_causal_emergence_from_trajectory(
    trajectory: dict[str, np.ndarray],
    macro_name: str = "population_count",
    num_bins: int = 4,
    window_size: int | None = None,
) -> dict[str, Any]:
    """Compute causal emergence metrics from a trajectory.

    This is the main entry point for trajectory-based analysis.
    It extracts macro/micro variables and computes both EI and Psi.

    Args:
        trajectory: Trajectory dict with 'alive_mask', 'field', 'positions', etc.
        macro_name: Which macro variable to analyze. Options:
            "population_count", "mean_field_intensity", "total_food_collected",
            "spatial_dispersion", "field_entropy".
        num_bins: Number of bins for discretization.
        window_size: If provided, only use the last window_size timesteps.

    Returns:
        Dict with keys:
            - "macro_name": Name of the macro variable analyzed.
            - "ei_gap": Effective Information gap (positive = emergence).
            - "micro_ei": EI at micro scale.
            - "macro_ei": EI at macro scale.
            - "psi": Rosas' Psi metric (positive = emergent causation).
            - "macro_mi": Macro self-prediction MI.
            - "sum_micro_mi": Sum of micro contributions.
            - "causal_emergence": bool, True if both EI gap > 0 and Psi > 0.
    """
    # Extract macro variables
    macros = extract_macro_variables(trajectory, num_bins=num_bins)

    if macro_name not in macros:
        available = list(macros.keys())
        raise ValueError(
            f"Macro variable '{macro_name}' not available. "
            f"Available: {available}. Check trajectory contents."
        )

    macro_var = macros[macro_name]

    # Apply window if specified
    if window_size is not None and len(macro_var) > window_size:
        start_idx = len(macro_var) - window_size
        macro_var = macro_var[start_idx:]

        # Also window other variables
        for key in macros:
            if len(macros[key]) > window_size:
                macros[key] = macros[key][start_idx:]

    n_timesteps = len(macro_var)
    if n_timesteps < 3:
        return {
            "macro_name": macro_name,
            "ei_gap": 0.0,
            "micro_ei": 0.0,
            "macro_ei": 0.0,
            "psi": 0.0,
            "macro_mi": 0.0,
            "sum_micro_mi": 0.0,
            "causal_emergence": False,
        }

    # Build macro TPM
    macro_discrete = discretize_to_bins(macro_var, num_bins=num_bins)
    macro_tpm = build_tpm(macro_discrete[:-1], macro_discrete[1:], num_states=num_bins)

    # Build micro TPM from agent-level data
    # Use alive_mask as micro variables (binary per agent)
    if "alive_mask" in trajectory:
        alive = np.asarray(trajectory["alive_mask"], dtype=np.float64)
        if window_size is not None and len(alive) > window_size:
            alive = alive[-window_size:]

        # Create micro state from alive pattern
        # Each unique alive pattern is a micro state
        n_agents = alive.shape[1]

        # For large agent counts, use hash of alive pattern
        if n_agents <= 8:
            # Binary encoding: each alive pattern is a unique state
            micro_states = (alive * (2 ** np.arange(n_agents))).sum(axis=1).astype(int)
            n_micro_states = min(2 ** n_agents, num_bins * 4)  # Cap for tractability
        else:
            # Use population count as micro state proxy for large populations
            micro_states = alive.sum(axis=1).astype(int)
            n_micro_states = n_agents + 1

        micro_tpm = build_tpm(
            micro_states[:-1], micro_states[1:], num_states=n_micro_states
        )

        # Compute EI
        ei_result = compute_effective_information(micro_tpm, macro_tpm)

        # Compute Psi using individual agent alive states as micro variables
        psi_result = compute_rosas_psi(
            macro_var[:-1],  # V_t
            macro_var[1:],   # V_{t+1}
            alive[:-1, :],   # X_i,t (individual agent states)
            num_bins=num_bins,
        )
    else:
        # No micro data available, can only compute macro EI
        ei_result = {
            "micro_ei": 0.0,
            "macro_ei": float(np.log2(num_bins) - compute_tpm_entropy(macro_tpm)),
            "ei_gap": 0.0,
            "micro_entropy": 0.0,
            "macro_entropy": float(compute_tpm_entropy(macro_tpm)),
        }
        psi_result = {
            "psi": 0.0,
            "macro_mi": 0.0,
            "sum_micro_mi": 0.0,
            "n_micro": 0,
        }

    # Determine if causal emergence is present
    causal_emergence = ei_result["ei_gap"] > 0 and psi_result["psi"] > 0

    return {
        "macro_name": macro_name,
        "ei_gap": ei_result["ei_gap"],
        "micro_ei": ei_result["micro_ei"],
        "macro_ei": ei_result["macro_ei"],
        "psi": psi_result["psi"],
        "macro_mi": psi_result["macro_mi"],
        "sum_micro_mi": psi_result["sum_micro_mi"],
        "causal_emergence": causal_emergence,
    }


def compute_windowed_causal_emergence(
    trajectory: dict[str, np.ndarray],
    macro_name: str = "population_count",
    num_bins: int = 4,
    window_size: int = 1000,
    overlap: float = 0.5,
) -> list[dict[str, Any]]:
    """Compute causal emergence metrics over sliding windows.

    This enables tracking how causal emergence evolves over training.
    Per PRD: 1M-step windows with 50% overlap for non-stationarity analysis.

    Args:
        trajectory: Trajectory dict.
        macro_name: Which macro variable to analyze.
        num_bins: Discretization bins.
        window_size: Window size in timesteps.
        overlap: Fraction of overlap between windows (0.0 to 1.0).

    Returns:
        List of result dicts, one per window, each with:
            - "window_start": Start index of window.
            - "window_end": End index of window.
            - All keys from compute_causal_emergence_from_trajectory.
    """
    # Determine trajectory length from any available key
    traj_len = 0
    for key in ["alive_mask", "field", "positions", "rewards"]:
        if key in trajectory:
            traj_len = len(trajectory[key])
            break

    if traj_len == 0 or traj_len < window_size:
        # Not enough data for windowed analysis
        return []

    step = int(window_size * (1.0 - overlap))
    step = max(1, step)  # At least 1 step forward

    results = []
    start_idx = 0

    while start_idx + window_size <= traj_len:
        # Extract window from trajectory
        windowed_traj = {}
        for key, val in trajectory.items():
            arr = np.asarray(val)
            if len(arr) >= start_idx + window_size:
                windowed_traj[key] = arr[start_idx : start_idx + window_size]

        # Compute metrics for this window
        result = compute_causal_emergence_from_trajectory(
            windowed_traj,
            macro_name=macro_name,
            num_bins=num_bins,
        )
        result["window_start"] = start_idx
        result["window_end"] = start_idx + window_size
        results.append(result)

        start_idx += step

    return results


@dataclass
class CausalEmergenceEvent:
    """A detected change in causal emergence metrics."""

    step: int
    metric_name: str
    old_value: float
    new_value: float
    z_score: float

    def __str__(self) -> str:
        direction = "increase" if self.new_value > self.old_value else "decrease"
        return (
            f"[step {self.step}] Causal emergence shift in {self.metric_name}: "
            f"{self.old_value:.4f} -> {self.new_value:.4f} "
            f"({direction}, z={self.z_score:.2f})"
        )


class CausalEmergenceTracker:
    """Tracks causal emergence metrics over training.

    Follows the same pattern as TransferEntropyTracker and OInformationTracker:
    rolling window z-score detection for phase transitions in causal emergence.

    Tracks both Hoel's EI gap and Rosas' Psi to provide complementary
    measures of macro-level causal power.

    Attributes:
        window_size: Rolling window size for z-score detection.
        z_threshold: Number of standard deviations for event detection.
        num_bins: Number of bins for discretization.
        history: Dict mapping metric name to list of values.
        steps: List of training steps corresponding to history entries.
        events: List of detected CausalEmergenceEvent instances.
        step_count: Number of update calls made.
    """

    def __init__(
        self,
        window_size: int = 20,
        z_threshold: float = 3.0,
        num_bins: int = 4,
    ) -> None:
        """Initialize the causal emergence tracker.

        Args:
            window_size: Number of recent values for z-score baseline.
            z_threshold: Z-score threshold for detecting significant changes.
            num_bins: Number of bins for discretization.
        """
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.num_bins = num_bins

        self.history: dict[str, list[float]] = {
            "ei_gap": [],
            "psi": [],
            "macro_ei": [],
            "micro_ei": [],
            "emergence_ratio": [],  # fraction with positive EI gap AND positive Psi
        }
        self.steps: list[int] = []
        self.events: list[CausalEmergenceEvent] = []
        self.step_count: int = 0

    def update(
        self,
        trajectory: dict[str, np.ndarray],
        step: int,
        macro_name: str = "population_count",
    ) -> list[CausalEmergenceEvent]:
        """Compute causal emergence metrics and detect phase transitions.

        Args:
            trajectory: Trajectory dict with 'alive_mask', 'field', etc.
            step: Current training step.
            macro_name: Which macro variable to analyze.

        Returns:
            List of CausalEmergenceEvent instances detected at this step.
        """
        result = compute_causal_emergence_from_trajectory(
            trajectory,
            macro_name=macro_name,
            num_bins=self.num_bins,
        )

        new_values = {
            "ei_gap": result["ei_gap"],
            "psi": result["psi"],
            "macro_ei": result["macro_ei"],
            "micro_ei": result["micro_ei"],
        }

        new_events: list[CausalEmergenceEvent] = []

        for name, value in new_values.items():
            hist = self.history[name]

            # Check for phase transition
            if len(hist) >= self.window_size:
                recent = hist[-self.window_size :]
                mean = float(np.mean(recent))
                std = float(np.std(recent))
                if std < 1e-8:
                    std = 1e-8
                z_score = abs(value - mean) / std

                if z_score > self.z_threshold:
                    event = CausalEmergenceEvent(
                        step=step,
                        metric_name=name,
                        old_value=mean,
                        new_value=value,
                        z_score=z_score,
                    )
                    self.events.append(event)
                    new_events.append(event)

            hist.append(value)

        # Track emergence ratio (cumulative)
        n_emergence = sum(
            1
            for i in range(len(self.history["ei_gap"]))
            if self.history["ei_gap"][i] > 0 and self.history["psi"][i] > 0
        )
        emergence_ratio = n_emergence / len(self.history["ei_gap"])
        self.history["emergence_ratio"].append(emergence_ratio)

        self.steps.append(step)
        self.step_count += 1
        return new_events

    def get_metrics(self) -> dict[str, float]:
        """Return current metric values for logging.

        Returns:
            Dict with keys like 'causal_emergence/ei_gap', 'causal_emergence/psi', etc.
        """
        metrics: dict[str, float] = {}
        for name, hist in self.history.items():
            if len(hist) > 0:
                metrics[f"causal_emergence/{name}"] = hist[-1]
        metrics["causal_emergence/num_events"] = float(len(self.events))
        return metrics

    def get_summary(self) -> dict[str, Any]:
        """Return a summary of causal emergence tracking.

        Returns:
            Dict with overall statistics and event list.
        """
        summary: dict[str, Any] = {
            "total_updates": self.step_count,
            "total_events": len(self.events),
            "events": [str(e) for e in self.events],
        }

        for name, hist in self.history.items():
            if len(hist) > 0:
                summary[f"{name}_final"] = hist[-1]
                summary[f"{name}_mean"] = float(np.mean(hist))
                summary[f"{name}_std"] = float(np.std(hist))
                summary[f"{name}_min"] = float(np.min(hist))
                summary[f"{name}_max"] = float(np.max(hist))

        # Key emergence indicators
        if len(self.history["ei_gap"]) > 0:
            summary["positive_ei_gap_fraction"] = sum(
                1 for v in self.history["ei_gap"] if v > 0
            ) / len(self.history["ei_gap"])
            summary["positive_psi_fraction"] = sum(
                1 for v in self.history["psi"] if v > 0
            ) / len(self.history["psi"])
            summary["emergence_fraction"] = self.history["emergence_ratio"][-1]
            summary["max_ei_gap"] = float(np.max(self.history["ei_gap"]))
            summary["max_psi"] = float(np.max(self.history["psi"]))

        return summary
