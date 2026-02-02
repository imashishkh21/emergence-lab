"""Information-theoretic metrics for measuring agent coordination.

Implements Transfer Entropy (TE) to quantify directed information flow between
agent pairs. High TE between agents indicates that one agent's behavior is
predictive of another's — a hallmark of coordination or communication via
the shared field.

Reference: Schreiber (2000), "Measuring Information Transfer", PRL 85(2).
k-NN estimator: Kraskov et al. (2004), "Estimating Mutual Information", PRE 69.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

import numpy as np
from scipy.spatial import KDTree


def compute_transfer_entropy(
    source_history: np.ndarray,
    target_history: np.ndarray,
    lag: int = 1,
    k: int = 4,
) -> float:
    """Compute transfer entropy from source to target using k-NN estimator.

    Transfer entropy TE(X→Y) measures how much the past of X reduces
    uncertainty about Y's future, beyond Y's own past. Uses the
    Kraskov-Stögbauer-Grassberger (KSG) estimator with digamma functions
    for continuous variables.

    For discrete/low-dimensional action histories, we add small jitter
    to avoid zero distances in k-NN lookups.

    Args:
        source_history: (T,) or (T, D) array of source agent's state history.
            For actions, pass (T,) int or float array.
        target_history: (T,) or (T, D) array of target agent's state history.
        lag: Time lag for transfer (default 1 step).
        k: Number of nearest neighbors for density estimation.

    Returns:
        Transfer entropy in nats (≥ 0). Returns 0.0 for degenerate cases.
    """
    source = np.asarray(source_history, dtype=np.float64)
    target = np.asarray(target_history, dtype=np.float64)

    # Ensure 2D
    if source.ndim == 1:
        source = source[:, np.newaxis]
    if target.ndim == 1:
        target = target[:, np.newaxis]

    T = min(len(source), len(target))
    if T < lag + 2 or T < k + 2:
        return 0.0

    # Construct the embedding vectors:
    # target_future: Y(t+lag)
    # target_past:   Y(t)
    # source_past:   X(t)
    target_future = target[lag:T]  # (N, D)
    target_past = target[: T - lag]  # (N, D)
    source_past = source[: T - lag]  # (N, D)
    N = len(target_future)

    if N < k + 1:
        return 0.0

    # Add small jitter to avoid zero distances (important for discrete data)
    rng = np.random.RandomState(42)
    jitter_scale = 1e-8
    target_future = target_future + rng.normal(0, jitter_scale, target_future.shape)
    target_past = target_past + rng.normal(0, jitter_scale, target_past.shape)
    source_past = source_past + rng.normal(0, jitter_scale, source_past.shape)

    # Joint space: (Y_future, Y_past, X_past)
    joint = np.hstack([target_future, target_past, source_past])  # (N, 3D)

    # Marginal spaces:
    # Z1 = (Y_past, X_past) — full conditioning
    # Z2 = (Y_future, Y_past) — no source
    # Z3 = (Y_past,) — just target past
    z1 = np.hstack([target_past, source_past])  # (N, 2D)
    z2 = np.hstack([target_future, target_past])  # (N, 2D)
    z3 = target_past  # (N, D)

    # Use Chebyshev (L-infinity) distance as in KSG estimator
    try:
        tree_joint = KDTree(joint)
    except Exception:
        return 0.0

    # Find k-th neighbor distance in joint space for each point
    # query k+1 because the point itself is included
    dists_joint, _ = tree_joint.query(joint, k=k + 1, p=np.inf)
    # k-th neighbor distance (0-indexed: index k is the k-th neighbor)
    eps = dists_joint[:, k]  # (N,)

    # Count neighbors within eps in each marginal space
    # Using Chebyshev balls
    from scipy.special import digamma

    tree_z1 = KDTree(z1)
    tree_z2 = KDTree(z2)
    tree_z3 = KDTree(z3)

    # Count points within distance eps[i] for each marginal
    # query_ball_point with p=inf gives Chebyshev ball
    n_z1 = np.zeros(N, dtype=np.float64)
    n_z2 = np.zeros(N, dtype=np.float64)
    n_z3 = np.zeros(N, dtype=np.float64)

    for i in range(N):
        # Subtract 1 to exclude the point itself
        n_z1[i] = max(len(tree_z1.query_ball_point(z1[i], eps[i], p=np.inf)) - 1, 0)
        n_z2[i] = max(len(tree_z2.query_ball_point(z2[i], eps[i], p=np.inf)) - 1, 0)
        n_z3[i] = max(len(tree_z3.query_ball_point(z3[i], eps[i], p=np.inf)) - 1, 0)

    # Avoid log(0) by setting minimum count to 1
    n_z1 = np.maximum(n_z1, 1)
    n_z2 = np.maximum(n_z2, 1)
    n_z3 = np.maximum(n_z3, 1)

    # KSG estimator for conditional mutual information:
    # TE(X→Y) = I(Y_future; X_past | Y_past)
    #          = digamma(k) - <digamma(n_z1) + digamma(n_z2) - digamma(n_z3)>
    te = float(
        digamma(k)
        - np.mean(digamma(n_z1) + digamma(n_z2) - digamma(n_z3))
    )

    # TE should be non-negative; small negative values are estimation noise
    return max(te, 0.0)


def compute_te_matrix(
    agent_histories: np.ndarray,
    alive_mask: np.ndarray | None = None,
    lag: int = 1,
    k: int = 4,
) -> dict[str, Any]:
    """Compute pairwise transfer entropy matrix for all agent pairs.

    Args:
        agent_histories: (T, num_agents) or (T, num_agents, D) array of
            per-agent state histories (typically action sequences).
        alive_mask: (num_agents,) bool array. If provided, only compute
            TE between alive agents. Dead agents get zero TE.
        lag: Time lag for TE computation.
        k: Number of nearest neighbors for KSG estimator.

    Returns:
        Dict with keys:
            - 'te_matrix': (num_agents, num_agents) float64 array of pairwise TE.
                te_matrix[i, j] = TE(agent_i → agent_j).
            - 'mean_te': float, mean TE across all alive pairs.
            - 'max_te': float, maximum TE value.
            - 'te_density': float, fraction of alive pairs with TE > threshold.
            - 'agent_indices': list of alive agent indices.
    """
    histories = np.asarray(agent_histories, dtype=np.float64)
    if histories.ndim == 2:
        # (T, num_agents) → add feature dim
        T, num_agents = histories.shape
    elif histories.ndim == 3:
        T, num_agents, D = histories.shape
    else:
        raise ValueError(
            f"agent_histories must be 2D or 3D, got shape {histories.shape}"
        )

    if alive_mask is not None:
        alive = np.asarray(alive_mask, dtype=bool)
    else:
        alive = np.ones(num_agents, dtype=bool)

    alive_idx = np.where(alive)[0]
    n_alive = len(alive_idx)

    te_matrix = np.zeros((num_agents, num_agents), dtype=np.float64)

    if n_alive < 2:
        return {
            "te_matrix": te_matrix,
            "mean_te": 0.0,
            "max_te": 0.0,
            "te_density": 0.0,
            "agent_indices": alive_idx.tolist(),
        }

    # Compute pairwise TE for alive agents
    te_values: list[float] = []
    for i in alive_idx:
        for j in alive_idx:
            if i == j:
                continue
            if histories.ndim == 2:
                source_hist = histories[:, i]
                target_hist = histories[:, j]
            else:
                source_hist = histories[:, i, :]
                target_hist = histories[:, j, :]

            te_val = compute_transfer_entropy(source_hist, target_hist, lag=lag, k=k)
            te_matrix[i, j] = te_val
            te_values.append(te_val)

    # Aggregate statistics
    te_arr = np.array(te_values) if te_values else np.array([0.0])
    mean_te = float(np.mean(te_arr))
    max_te = float(np.max(te_arr))

    # TE density: fraction of pairs with TE > small threshold
    # (indicates meaningful information flow, not just noise)
    te_threshold = 0.01
    density = float(np.mean(te_arr > te_threshold)) if len(te_arr) > 0 else 0.0

    return {
        "te_matrix": te_matrix,
        "mean_te": mean_te,
        "max_te": max_te,
        "te_density": density,
        "agent_indices": alive_idx.tolist(),
    }


def transfer_entropy_from_trajectory(
    trajectory: dict[str, np.ndarray],
    feature: str = "actions",
    lag: int = 1,
    k: int = 4,
) -> dict[str, Any]:
    """Compute transfer entropy from a trajectory dict.

    Convenience wrapper that extracts agent histories from a trajectory
    dict (as produced by TrajectoryRecorder) and computes the TE matrix.

    Args:
        trajectory: Dict with 'actions' (T, num_agents), 'positions' (T, num_agents, 2),
            'alive_mask' (T, num_agents), etc.
        feature: Which feature to use for TE computation. Options:
            - 'actions': raw action indices (default)
            - 'positions': 2D position vectors
        lag: Time lag for TE computation.
        k: Number of nearest neighbors.

    Returns:
        Same dict as compute_te_matrix, plus:
            - 'feature': which feature was used.
    """
    if feature == "actions":
        histories = np.asarray(trajectory["actions"], dtype=np.float64)
    elif feature == "positions":
        histories = np.asarray(trajectory["positions"], dtype=np.float64)
    else:
        raise ValueError(f"Unknown feature: {feature}. Use 'actions' or 'positions'.")

    # Use the final alive mask (last timestep) for filtering
    alive_input = trajectory.get("alive_mask")
    alive_mask: np.ndarray | None = None
    if alive_input is not None:
        alive_arr = np.asarray(alive_input)
        # Use any-alive across time (agent was alive at any point)
        alive_mask = np.asarray(alive_arr.any(axis=0))

    result = compute_te_matrix(histories, alive_mask=alive_mask, lag=lag, k=k)
    result["feature"] = feature
    return result


@dataclass
class TEEvent:
    """A detected change in transfer entropy patterns."""

    step: int
    metric_name: str
    old_value: float
    new_value: float
    z_score: float

    def __str__(self) -> str:
        direction = "increase" if self.new_value > self.old_value else "decrease"
        return (
            f"[step {self.step}] TE shift in {self.metric_name}: "
            f"{self.old_value:.4f} -> {self.new_value:.4f} "
            f"({direction}, z={self.z_score:.2f})"
        )


class TransferEntropyTracker:
    """Tracks transfer entropy metrics over training.

    Follows the same pattern as EmergenceTracker and SpecializationTracker:
    rolling window z-score detection for phase transitions in information
    flow between agents.

    Attributes:
        window_size: Rolling window size for z-score detection.
        z_threshold: Number of standard deviations for event detection.
        history: Dict mapping metric name to list of values.
        steps: List of training steps corresponding to history entries.
        events: List of detected TEEvent instances.
        step_count: Number of update calls made.
    """

    def __init__(
        self,
        window_size: int = 20,
        z_threshold: float = 3.0,
    ) -> None:
        self.window_size = window_size
        self.z_threshold = z_threshold

        self.history: dict[str, list[float]] = {
            "mean_te": [],
            "max_te": [],
            "te_density": [],
        }
        self.steps: list[int] = []
        self.events: list[TEEvent] = []
        self.step_count: int = 0

    def update(
        self,
        agent_histories: np.ndarray,
        alive_mask: np.ndarray | None,
        step: int,
        lag: int = 1,
        k: int = 4,
    ) -> list[TEEvent]:
        """Compute TE metrics and detect phase transitions.

        Args:
            agent_histories: (T, num_agents) or (T, num_agents, D) state history.
            alive_mask: (num_agents,) bool. If None, all agents assumed alive.
            step: Current training step.
            lag: Time lag for TE.
            k: k-NN parameter.

        Returns:
            List of TEEvent instances detected at this step (empty if none).
        """
        result = compute_te_matrix(
            agent_histories, alive_mask=alive_mask, lag=lag, k=k
        )

        new_values = {
            "mean_te": result["mean_te"],
            "max_te": result["max_te"],
            "te_density": result["te_density"],
        }

        new_events: list[TEEvent] = []

        for name, value in new_values.items():
            hist = self.history[name]

            # Check for phase transition (need enough history)
            if len(hist) >= self.window_size:
                recent = hist[-self.window_size :]
                mean = float(np.mean(recent))
                std = float(np.std(recent))
                if std < 1e-8:
                    std = 1e-8
                z_score = abs(value - mean) / std

                if z_score > self.z_threshold:
                    event = TEEvent(
                        step=step,
                        metric_name=name,
                        old_value=mean,
                        new_value=value,
                        z_score=z_score,
                    )
                    self.events.append(event)
                    new_events.append(event)

            hist.append(value)

        self.steps.append(step)
        self.step_count += 1
        return new_events

    def get_metrics(self) -> dict[str, float]:
        """Return current metric values for logging.

        Returns:
            Dict with keys like 'information/mean_te', 'information/max_te',
            'information/te_density', 'information/num_events'.
        """
        metrics: dict[str, float] = {}
        for name, hist in self.history.items():
            if len(hist) > 0:
                metrics[f"information/{name}"] = hist[-1]
        metrics["information/num_events"] = float(len(self.events))
        return metrics

    def get_summary(self) -> dict[str, Any]:
        """Return a summary of transfer entropy tracking.

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
        return summary
