"""Emergence detection: tracks field metrics over training and detects phase transitions.

Includes:
- EmergenceTracker: rolling z-score detection on field metrics
- PhaseTransitionDetector: susceptibility + autocorrelation-based phase detection
  on any order parameter (e.g., specialization score)
"""

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

import numpy as np

from src.analysis.field_metrics import field_entropy, field_structure
from src.configs import Config
from src.field.field import FieldState


@dataclass
class EmergenceEvent:
    """A detected phase transition in field metrics."""

    step: int
    metric_name: str
    old_value: float
    new_value: float
    z_score: float

    def __str__(self) -> str:
        direction = "increase" if self.new_value > self.old_value else "decrease"
        return (
            f"[step {self.step}] Phase transition in {self.metric_name}: "
            f"{self.old_value:.4f} -> {self.new_value:.4f} "
            f"({direction}, z={self.z_score:.2f})"
        )


@dataclass
class PhaseTransitionEvent:
    """A phase transition detected via susceptibility spike + autocorrelation increase."""

    step: int
    order_parameter: float
    susceptibility: float
    susceptibility_z: float
    autocorrelation: float
    prev_autocorrelation: float

    def __str__(self) -> str:
        return (
            f"[step {self.step}] ⚡ Phase transition detected: "
            f"order={self.order_parameter:.4f}, "
            f"susceptibility={self.susceptibility:.4f} (z={self.susceptibility_z:.2f}), "
            f"autocorr={self.prev_autocorrelation:.4f}->{self.autocorrelation:.4f}"
        )


@dataclass
class MetricHistory:
    """Rolling history for a single metric."""

    values: list[float] = dataclass_field(default_factory=list)
    steps: list[int] = dataclass_field(default_factory=list)

    def append(self, value: float, step: int) -> None:
        self.values.append(value)
        self.steps.append(step)

    def __len__(self) -> int:
        return len(self.values)

    def recent_mean(self, window: int) -> float:
        """Mean of the last `window` values."""
        if len(self.values) == 0:
            return 0.0
        recent = self.values[-window:]
        return float(np.mean(recent))

    def recent_std(self, window: int) -> float:
        """Std of the last `window` values."""
        if len(self.values) < 2:
            return 1.0
        recent = self.values[-window:]
        std = float(np.std(recent))
        return std if std > 1e-8 else 1e-8


class EmergenceTracker:
    """Tracks field metrics over training and detects phase transitions.

    Phase transitions are detected when a new metric value deviates
    significantly (beyond `z_threshold` standard deviations) from
    the rolling window mean.

    Attributes:
        config: Master configuration.
        history: Dict mapping metric name to MetricHistory.
        events: List of detected EmergenceEvent instances.
        step_count: Number of update calls made.
    """

    def __init__(
        self,
        config: Config,
        window_size: int = 20,
        z_threshold: float = 3.0,
    ) -> None:
        self.config = config
        self.window_size = window_size
        self.z_threshold = z_threshold

        self.history: dict[str, MetricHistory] = {
            "entropy": MetricHistory(),
            "structure": MetricHistory(),
        }
        self.events: list[EmergenceEvent] = []
        self.step_count: int = 0

    def update(self, field: FieldState, step: int) -> list[EmergenceEvent]:
        """Compute field metrics and check for phase transitions.

        Args:
            field: Current field state with values of shape (H, W, C).
            step: Current training step.

        Returns:
            List of EmergenceEvent instances detected at this step
            (empty if no transitions detected).
        """
        # Compute metrics
        entropy_val = float(field_entropy(field))
        structure_val = float(field_structure(field))

        new_values = {
            "entropy": entropy_val,
            "structure": structure_val,
        }

        new_events: list[EmergenceEvent] = []

        for name, value in new_values.items():
            hist = self.history[name]

            # Check for phase transition (need enough history)
            if len(hist) >= self.window_size:
                mean = hist.recent_mean(self.window_size)
                std = hist.recent_std(self.window_size)
                z_score = abs(value - mean) / std

                if z_score > self.z_threshold:
                    event = EmergenceEvent(
                        step=step,
                        metric_name=name,
                        old_value=mean,
                        new_value=value,
                        z_score=z_score,
                    )
                    self.events.append(event)
                    new_events.append(event)

            hist.append(value, step)

        self.step_count += 1
        return new_events

    def get_metrics(self) -> dict[str, float]:
        """Return current metric values for logging.

        Returns:
            Dict with keys like 'emergence/entropy', 'emergence/structure',
            'emergence/num_events'.
        """
        metrics: dict[str, float] = {}
        for name, hist in self.history.items():
            if len(hist) > 0:
                metrics[f"emergence/{name}"] = hist.values[-1]
        metrics["emergence/num_events"] = float(len(self.events))
        return metrics

    def get_summary(self) -> dict[str, Any]:
        """Return a summary of emergence tracking.

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
                summary[f"{name}_final"] = hist.values[-1]
                summary[f"{name}_mean"] = float(np.mean(hist.values))
                summary[f"{name}_std"] = float(np.std(hist.values))
        return summary


def _autocorrelation_time(values: list[float], window: int) -> float:
    """Estimate autocorrelation time from a time series.

    Computes the lag-1 autocorrelation of the last ``window`` values,
    then converts to an autocorrelation time via tau = -1/ln(|r|).
    Higher tau means the system is "remembering" longer — a sign of
    critical slowing down near phase transitions.

    Args:
        values: Full history of the order parameter.
        window: Number of recent values to use.

    Returns:
        Estimated autocorrelation time (>= 0). Returns 0.0 when
        insufficient data or zero variance.
    """
    if len(values) < 3:
        return 0.0
    recent = np.array(values[-window:], dtype=np.float64)
    if len(recent) < 3:
        return 0.0
    mean = np.mean(recent)
    var = np.var(recent)
    if var < 1e-12:
        return 0.0
    # Lag-1 autocorrelation
    shifted = recent[1:] - mean
    original = recent[:-1] - mean
    r = float(np.sum(shifted * original)) / (float(var) * len(original))
    r = np.clip(r, -0.999, 0.999)
    abs_r = abs(r)
    if abs_r < 1e-8:
        return 0.0
    return float(-1.0 / np.log(abs_r))


class PhaseTransitionDetector:
    """Detects phase transitions via susceptibility spikes and critical slowing down.

    Tracks an "order parameter" (e.g., specialization score) over time.
    A phase transition is flagged when BOTH conditions hold:
      1. Susceptibility (variance of the order parameter over a rolling window)
         spikes beyond ``z_threshold`` standard deviations of its own history.
      2. Autocorrelation time is increasing (critical slowing down).

    This implements the classical statistical mechanics approach to
    detecting phase transitions — near a transition, fluctuations diverge
    (high susceptibility) and the system takes longer to decorrelate
    (increasing autocorrelation time).

    Attributes:
        window_size: Rolling window for susceptibility/autocorrelation.
        z_threshold: Z-score threshold for susceptibility spike.
        order_values: History of the order parameter.
        susceptibility_values: History of susceptibility (variance) values.
        autocorrelation_values: History of autocorrelation time values.
        steps: Training steps corresponding to each update.
        events: Detected PhaseTransitionEvent list.
        step_count: Number of update calls.
    """

    def __init__(
        self,
        window_size: int = 20,
        z_threshold: float = 3.0,
    ) -> None:
        self.window_size = window_size
        self.z_threshold = z_threshold

        self.order_values: list[float] = []
        self.susceptibility_values: list[float] = []
        self.autocorrelation_values: list[float] = []
        self.steps: list[int] = []
        self.events: list[PhaseTransitionEvent] = []
        self.step_count: int = 0

    def update(
        self,
        order_parameter: float,
        step: int,
    ) -> list[PhaseTransitionEvent]:
        """Record a new order parameter value and check for phase transitions.

        Args:
            order_parameter: Current value of the order parameter
                (e.g., specialization score in [0, 1]).
            step: Current training step.

        Returns:
            List of PhaseTransitionEvent instances detected at this step
            (empty if no transition detected).
        """
        self.order_values.append(float(order_parameter))
        self.steps.append(step)

        new_events: list[PhaseTransitionEvent] = []

        # Need at least window_size values for susceptibility
        if len(self.order_values) >= self.window_size:
            recent = np.array(
                self.order_values[-self.window_size :], dtype=np.float64
            )
            susceptibility = float(np.var(recent))
        else:
            susceptibility = 0.0

        self.susceptibility_values.append(susceptibility)

        # Compute autocorrelation time
        autocorr = _autocorrelation_time(self.order_values, self.window_size)
        self.autocorrelation_values.append(autocorr)

        # Check for phase transition:
        # 1. Susceptibility spike (z > threshold compared to susceptibility history)
        # 2. Autocorrelation increasing (current > previous)
        n_susc = len(self.susceptibility_values)
        if n_susc >= self.window_size:
            susc_recent = np.array(
                self.susceptibility_values[-self.window_size :], dtype=np.float64
            )
            susc_mean = float(np.mean(susc_recent))
            susc_std = float(np.std(susc_recent))
            if susc_std < 1e-12:
                susc_std = 1e-12
            susc_z = (susceptibility - susc_mean) / susc_std

            # Autocorrelation increasing?
            prev_autocorr = (
                self.autocorrelation_values[-2]
                if len(self.autocorrelation_values) >= 2
                else 0.0
            )
            autocorr_increasing = autocorr > prev_autocorr

            if susc_z > self.z_threshold and autocorr_increasing:
                event = PhaseTransitionEvent(
                    step=step,
                    order_parameter=float(order_parameter),
                    susceptibility=susceptibility,
                    susceptibility_z=susc_z,
                    autocorrelation=autocorr,
                    prev_autocorrelation=prev_autocorr,
                )
                self.events.append(event)
                new_events.append(event)

        self.step_count += 1
        return new_events

    def get_metrics(self) -> dict[str, float]:
        """Return current metric values for logging.

        Returns:
            Dict with keys prefixed by ``'phase_transition/'``.
        """
        metrics: dict[str, float] = {}
        if self.order_values:
            metrics["phase_transition/order_parameter"] = self.order_values[-1]
        if self.susceptibility_values:
            metrics["phase_transition/susceptibility"] = self.susceptibility_values[-1]
        if self.autocorrelation_values:
            metrics["phase_transition/autocorrelation"] = (
                self.autocorrelation_values[-1]
            )
        metrics["phase_transition/num_events"] = float(len(self.events))
        return metrics

    def get_summary(self) -> dict[str, Any]:
        """Return a summary of phase transition detection.

        Returns:
            Dict with overall statistics and event list.
        """
        summary: dict[str, Any] = {
            "total_updates": self.step_count,
            "total_events": len(self.events),
            "events": [str(e) for e in self.events],
        }
        if self.order_values:
            summary["order_parameter_final"] = self.order_values[-1]
            summary["order_parameter_mean"] = float(np.mean(self.order_values))
            summary["order_parameter_std"] = float(np.std(self.order_values))
        if self.susceptibility_values:
            summary["susceptibility_final"] = self.susceptibility_values[-1]
            summary["susceptibility_mean"] = float(np.mean(self.susceptibility_values))
            summary["susceptibility_max"] = float(np.max(self.susceptibility_values))
        if self.autocorrelation_values:
            summary["autocorrelation_final"] = self.autocorrelation_values[-1]
            summary["autocorrelation_mean"] = float(
                np.mean(self.autocorrelation_values)
            )
        return summary
