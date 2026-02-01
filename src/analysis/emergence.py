"""Emergence detection: tracks field metrics over training and detects phase transitions."""

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
