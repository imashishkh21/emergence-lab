"""Weights & Biases logging utilities for experiment tracking."""

import dataclasses
from typing import Any

import numpy as np

from src.configs import Config


def _config_to_dict(config: Config) -> dict[str, Any]:
    """Convert Config dataclass to flat dict for W&B."""
    result: dict[str, Any] = {}
    top = dataclasses.asdict(config)
    for section_name, section_dict in top.items():
        if isinstance(section_dict, dict):
            for k, v in section_dict.items():
                result[f"{section_name}.{k}"] = v
        else:
            result[section_name] = section_dict
    return result


def init_wandb(config: Config) -> None:
    """Initialize a Weights & Biases run.

    Args:
        config: Master configuration. The run is created under
                config.log.project with all hyperparameters logged.
    """
    import wandb

    wandb.init(
        project=config.log.project,
        config=_config_to_dict(config),
    )


def log_metrics(metrics: dict[str, Any], step: int) -> None:
    """Log scalar metrics to W&B.

    Args:
        metrics: Dictionary of metric name to scalar value.
                 JAX arrays are converted to Python floats.
        step: Global training step for the x-axis.
    """
    import wandb

    logged: dict[str, float] = {}
    for k, v in metrics.items():
        logged[k] = float(v)
    wandb.log(logged, step=step)


def log_video(frames: list[np.ndarray], name: str, step: int, fps: int = 30) -> None:
    """Log a video to W&B.

    Args:
        frames: List of RGB images as numpy arrays (H, W, 3), uint8.
        name: Name for the video in the W&B dashboard.
        step: Global training step.
        fps: Frames per second for playback.
    """
    import wandb

    # Stack frames into (T, H, W, C) then transpose to (T, C, H, W) for wandb
    video_array = np.stack(frames, axis=0)  # (T, H, W, C)
    video_array = np.transpose(video_array, (0, 3, 1, 2))  # (T, C, H, W)
    wandb.log({name: wandb.Video(video_array, fps=fps)}, step=step)


def finish_wandb() -> None:
    """Close the current W&B run."""
    import wandb

    wandb.finish()
