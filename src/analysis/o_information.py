"""O-information metric for measuring higher-order interactions between agents.

O-information (Omega) measures the balance between redundancy and synergy in
multi-agent systems. Omega = TC - DTC where:
  - TC (Total Correlation): shared information among all variables
  - DTC (Dual Total Correlation): information about the whole not in parts

Key interpretation:
  - Omega < 0: synergy dominates (emergence signal!)
  - Omega > 0: redundancy dominates (no emergence)
  - Omega = 0: balanced

Reference: Rosas et al. (2019), "Quantifying High-order Interdependencies via
Multivariate Extensions of the Mutual Information", PRE 100.

Library: hoi (JAX-native, O(n) scaling via Gaussian Copula).
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

import numpy as np

# Check if hoi is available
_hoi_available = importlib.util.find_spec("hoi") is not None


def compute_o_information(
    behavioral_features: np.ndarray,
    method: str = "gc",
) -> float:
    """Compute O-information (Omega) over agent behavioral features.

    Uses the hoi library's Oinfo class with Gaussian Copula (gc) method
    for efficient computation on continuous data.

    O-information formula:
        Omega(X) = TC(X) - DTC(X)
                 = (n-2) * H(X) + sum_i H(X_i) - sum_i H(X_{-i})

    Args:
        behavioral_features: (n_samples, n_agents) or (n_samples, n_agents, n_features)
            array of behavioral features per agent. Each column is one agent's
            feature trajectory over time.
        method: Estimation method for hoi. Options:
            - "gc": Gaussian Copula (default, fast, handles continuous data)
            - "binning": Discretization-based (slower, more accurate for discrete)

    Returns:
        O-information value (scalar float).
        - Negative = synergy dominates (emergence!)
        - Positive = redundancy dominates
        - Returns 0.0 for degenerate cases (too few agents, constant features, etc.)

    Raises:
        ImportError: If hoi is not installed.
    """
    if not _hoi_available:
        raise ImportError(
            "hoi library is required for O-information computation. "
            "Install with: pip install -e '.[phase5]'"
        )

    from hoi.metrics import Oinfo

    features = np.asarray(behavioral_features, dtype=np.float64)

    # Handle NaN inputs
    if np.any(np.isnan(features)):
        return 0.0

    # Ensure 3D: (n_samples, n_variables, n_features)
    if features.ndim == 1:
        # Single agent, can't compute O-info
        return 0.0
    elif features.ndim == 2:
        # (n_samples, n_agents) -> (n_samples, n_agents, 1)
        n_samples, n_agents = features.shape
        features = features[:, :, np.newaxis]
    elif features.ndim == 3:
        n_samples, n_agents, _ = features.shape
    else:
        raise ValueError(
            f"behavioral_features must be 2D or 3D, got shape {features.shape}"
        )

    # Need at least 3 agents for meaningful O-information
    # (with 2 agents, O-info = 0 by definition)
    if n_agents < 3:
        return 0.0

    # Need sufficient samples
    if n_samples < 10:
        return 0.0

    # Check for constant features (would cause numerical issues)
    feature_std = np.std(features, axis=0)
    if np.all(feature_std < 1e-10):
        return 0.0

    # Add small noise to constant dimensions to avoid numerical issues
    # but preserve the signal in varying dimensions
    noise_scale = 1e-8
    rng = np.random.default_rng(42)
    features = features + rng.normal(0, noise_scale, features.shape)

    try:
        # hoi's Oinfo computes O-information for all possible multiplets
        # We want the O-info for the full system (all agents together)
        model = Oinfo(features)

        # method="gc" uses Gaussian Copula, fast and handles continuous data
        # minsize=n_agents ensures we get the full-system multiplet
        # maxsize=n_agents ensures we only get the full-system multiplet
        result = model.fit(method=method, minsize=n_agents, maxsize=n_agents)

        # Result shape: (n_multiplets, n_features)
        # We want the scalar O-info for the full system
        if result.size == 0:
            return 0.0

        # Average across features if multi-feature
        o_info = float(np.mean(result))

        # Handle any remaining NaN/inf
        if not np.isfinite(o_info):
            return 0.0

        return o_info

    except Exception:
        # Catch any numerical issues from hoi
        return 0.0


def compute_o_information_by_condition(
    behavioral_features: np.ndarray,
    field_condition: str,
    method: str = "gc",
) -> dict[str, Any]:
    """Compute O-information and tag with field condition.

    Convenience function for ablation studies comparing O-info
    across field conditions (normal, zeroed, random).

    Args:
        behavioral_features: Agent behavioral features array.
        field_condition: One of "normal", "zeroed", "random".
        method: hoi estimation method.

    Returns:
        Dict with keys:
            - "o_information": the computed O-info value
            - "field_condition": the condition string
            - "synergy_dominant": bool, True if O-info < 0
    """
    o_info = compute_o_information(behavioral_features, method=method)
    return {
        "o_information": o_info,
        "field_condition": field_condition,
        "synergy_dominant": o_info < 0,
    }


@dataclass
class OInfoEvent:
    """A detected change in O-information patterns."""

    step: int
    old_value: float
    new_value: float
    z_score: float

    def __str__(self) -> str:
        direction = "increase" if self.new_value > self.old_value else "decrease"
        synergy = "synergy+" if self.new_value < self.old_value else "redundancy+"
        return (
            f"[step {self.step}] O-information shift: "
            f"{self.old_value:.4f} -> {self.new_value:.4f} "
            f"({direction}, {synergy}, z={self.z_score:.2f})"
        )


class OInformationTracker:
    """Tracks O-information metrics over training.

    Follows the same pattern as TransferEntropyTracker:
    rolling window z-score detection for phase transitions in
    higher-order interactions between agents.

    O-information becoming more negative over training indicates
    increasing synergy — a signature of emergent collective behavior.

    Attributes:
        window_size: Rolling window size for z-score detection.
        z_threshold: Number of standard deviations for event detection.
        history: Dict mapping metric name to list of values.
        steps: List of training steps corresponding to history entries.
        events: List of detected OInfoEvent instances.
        step_count: Number of update calls made.
    """

    def __init__(
        self,
        window_size: int = 20,
        z_threshold: float = 3.0,
    ) -> None:
        """Initialize the O-information tracker.

        Args:
            window_size: Number of recent values for z-score baseline.
            z_threshold: Z-score threshold for detecting significant changes.
        """
        self.window_size = window_size
        self.z_threshold = z_threshold

        self.history: dict[str, list[float]] = {
            "o_information": [],
            "synergy_ratio": [],  # fraction of negative O-info values
        }
        self.steps: list[int] = []
        self.events: list[OInfoEvent] = []
        self.step_count: int = 0

    def update(
        self,
        behavioral_features: np.ndarray,
        step: int,
        method: str = "gc",
    ) -> list[OInfoEvent]:
        """Compute O-information and detect phase transitions.

        Args:
            behavioral_features: (n_samples, n_agents) or (n_samples, n_agents, n_features)
                array of agent behavioral features.
            step: Current training step.
            method: hoi estimation method ("gc" or "binning").

        Returns:
            List of OInfoEvent instances detected at this step (empty if none).
        """
        o_info = compute_o_information(behavioral_features, method=method)

        # Track synergy ratio: is O-info negative? (cumulative fraction)
        self.history["o_information"].append(o_info)
        n_negative = sum(1 for v in self.history["o_information"] if v < 0)
        synergy_ratio = n_negative / len(self.history["o_information"])
        self.history["synergy_ratio"].append(synergy_ratio)

        new_events: list[OInfoEvent] = []

        # Check for phase transition in O-information
        o_info_hist = self.history["o_information"]
        if len(o_info_hist) >= self.window_size:
            recent = o_info_hist[-self.window_size :]
            mean = float(np.mean(recent))
            std = float(np.std(recent))
            if std < 1e-8:
                std = 1e-8
            z_score = abs(o_info - mean) / std

            if z_score > self.z_threshold:
                event = OInfoEvent(
                    step=step,
                    old_value=mean,
                    new_value=o_info,
                    z_score=z_score,
                )
                self.events.append(event)
                new_events.append(event)

        self.steps.append(step)
        self.step_count += 1
        return new_events

    def get_metrics(self) -> dict[str, float]:
        """Return current metric values for logging.

        Returns:
            Dict with keys like 'o_information/value', 'o_information/synergy_ratio',
            'o_information/num_events'.
        """
        metrics: dict[str, float] = {}
        if len(self.history["o_information"]) > 0:
            metrics["o_information/value"] = self.history["o_information"][-1]
        if len(self.history["synergy_ratio"]) > 0:
            metrics["o_information/synergy_ratio"] = self.history["synergy_ratio"][-1]
        metrics["o_information/num_events"] = float(len(self.events))
        return metrics

    def get_summary(self) -> dict[str, Any]:
        """Return a summary of O-information tracking.

        Returns:
            Dict with overall statistics and event list.
        """
        summary: dict[str, Any] = {
            "total_updates": self.step_count,
            "total_events": len(self.events),
            "events": [str(e) for e in self.events],
        }

        o_info_hist = self.history["o_information"]
        if len(o_info_hist) > 0:
            summary["o_information_final"] = o_info_hist[-1]
            summary["o_information_mean"] = float(np.mean(o_info_hist))
            summary["o_information_std"] = float(np.std(o_info_hist))
            summary["o_information_min"] = float(np.min(o_info_hist))
            summary["o_information_max"] = float(np.max(o_info_hist))

            # Key emergence indicators
            summary["synergy_dominant_fraction"] = sum(
                1 for v in o_info_hist if v < 0
            ) / len(o_info_hist)
            summary["most_negative_o_info"] = float(np.min(o_info_hist))

        synergy_hist = self.history["synergy_ratio"]
        if len(synergy_hist) > 0:
            summary["synergy_ratio_final"] = synergy_hist[-1]

        return summary
