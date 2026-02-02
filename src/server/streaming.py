"""Training bridge for streaming simulation state to WebSocket clients.

The TrainingBridge connects the JAX training loop to the FastAPI WebSocket
server. It receives state snapshots from the training loop (in the training
thread) and makes them available for the server to stream to connected
dashboard clients.

Thread-safety: The training loop runs in its own thread (or the main thread),
while the server's async event loop processes WebSocket connections. We use a
threading.Lock to protect the shared latest_frame buffer so both sides can
safely read/write.
"""

from __future__ import annotations

import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import msgpack
import numpy as np


@dataclass
class Frame:
    """A single frame of training state ready for streaming.

    All numpy arrays are pre-converted from JAX arrays for zero-copy
    serialization. The frame is designed to be small enough for 30Hz
    streaming over WebSocket.
    """

    step: int
    positions: np.ndarray  # (max_agents, 2) float32
    alive: np.ndarray  # (max_agents,) bool
    energy: np.ndarray  # (max_agents,) float32
    food_positions: np.ndarray  # (num_food, 2) float32
    food_collected: np.ndarray  # (num_food,) bool
    field_values: np.ndarray  # (H, W, C) float32
    cluster_labels: np.ndarray | None  # (max_agents,) int8 or None
    metrics: dict[str, float]
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()


def pack_frame(frame: Frame) -> bytes:
    """Serialize a Frame to MessagePack binary format.

    Numpy arrays are serialized as raw bytes with dtype/shape metadata
    so the client can reconstruct typed arrays.

    Returns:
        MessagePack-encoded bytes ready for WebSocket send.
    """
    data: dict[str, Any] = {
        "step": frame.step,
        "timestamp": frame.timestamp,
        "positions": _pack_array(frame.positions.astype(np.float32)),
        "alive": _pack_array(frame.alive.astype(np.uint8)),
        "energy": _pack_array(frame.energy.astype(np.float32)),
        "food_positions": _pack_array(frame.food_positions.astype(np.float32)),
        "food_collected": _pack_array(frame.food_collected.astype(np.uint8)),
        "field": _pack_array(frame.field_values.astype(np.float32)),
        "metrics": frame.metrics,
    }
    if frame.cluster_labels is not None:
        data["clusters"] = _pack_array(frame.cluster_labels.astype(np.int8))
    result: bytes = msgpack.packb(data, use_bin_type=True)
    return result


def _pack_array(arr: np.ndarray) -> dict[str, Any]:
    """Pack a numpy array into a dict with shape, dtype, and raw bytes."""
    return {
        "shape": list(arr.shape),
        "dtype": arr.dtype.str,
        "data": arr.tobytes(),
    }


class TrainingBridge:
    """Bridge between the training loop and the WebSocket server.

    The training loop calls `publish_frame()` to push new state snapshots.
    The server calls `get_latest_frame()` to read the most recent snapshot
    for streaming to clients.

    This class is thread-safe: publish happens in the training thread,
    while get_latest_frame is called from the async server event loop.
    """

    def __init__(self, target_fps: float = 30.0) -> None:
        self._lock = threading.Lock()
        self._latest_frame: Frame | None = None
        self._frame_count: int = 0
        self._target_fps = target_fps
        self._min_interval = 1.0 / target_fps if target_fps > 0 else 0.0
        self._last_publish_time: float = 0.0
        self._paused: bool = False
        self._commands: list[dict[str, Any]] = []

    @property
    def frame_count(self) -> int:
        """Total number of frames published."""
        return self._frame_count

    @property
    def target_fps(self) -> float:
        """Target frames per second for streaming."""
        return self._target_fps

    @property
    def paused(self) -> bool:
        """Whether training is paused."""
        return self._paused

    @paused.setter
    def paused(self, value: bool) -> None:
        self._paused = value

    def publish_frame(self, frame: Frame) -> bool:
        """Publish a new frame from the training loop.

        Rate-limited to target_fps. Returns True if the frame was
        accepted, False if skipped due to rate limiting.

        Args:
            frame: The training state snapshot to publish.

        Returns:
            True if accepted, False if rate-limited.
        """
        now = time.time()
        if now - self._last_publish_time < self._min_interval:
            return False

        with self._lock:
            self._latest_frame = frame
            self._frame_count += 1
            self._last_publish_time = now
        return True

    def get_latest_frame(self) -> Frame | None:
        """Get the most recent frame (called from server async loop).

        Returns:
            The latest Frame, or None if no frame has been published yet.
        """
        with self._lock:
            return self._latest_frame

    def get_latest_packed(self) -> bytes | None:
        """Get the most recent frame as packed MessagePack bytes.

        Returns:
            MessagePack bytes, or None if no frame available.
        """
        frame = self.get_latest_frame()
        if frame is None:
            return None
        return pack_frame(frame)

    def push_command(self, command: dict[str, Any]) -> None:
        """Push a command from the server to the training loop.

        Commands are JSON dicts like {"type": "pause"} or
        {"type": "set_param", "key": "mutation_std", "value": 0.05}.

        Args:
            command: A command dict from the dashboard client.
        """
        with self._lock:
            self._commands.append(command)

    def pop_commands(self) -> list[dict[str, Any]]:
        """Pop all pending commands (called from training loop).

        Returns:
            List of command dicts, possibly empty.
        """
        with self._lock:
            commands = self._commands
            self._commands = []
            return commands

    def create_frame_from_state(
        self,
        env_state: Any,
        metrics: dict[str, Any],
        step: int,
        cluster_labels: np.ndarray | None = None,
    ) -> Frame:
        """Create a Frame from a JAX EnvState and metrics dict.

        Converts JAX arrays to numpy for serialization. Uses the first
        environment when the state is batched across num_envs.

        Args:
            env_state: The EnvState (possibly batched).
            metrics: Dict of scalar metrics from train_step.
            step: Current training step count.
            cluster_labels: Optional cluster labels for agents.

        Returns:
            A Frame ready for publishing.
        """
        # Extract from first env if batched
        positions = np.asarray(env_state.agent_positions[0]).astype(np.float32)
        alive = np.asarray(env_state.agent_alive[0]).astype(bool)
        energy = np.asarray(env_state.agent_energy[0]).astype(np.float32)
        food_pos = np.asarray(env_state.food_positions[0]).astype(np.float32)
        food_col = np.asarray(env_state.food_collected[0]).astype(bool)
        field_vals = np.asarray(env_state.field_state.values[0]).astype(np.float32)

        # Convert metrics to plain floats
        float_metrics: dict[str, float] = {}
        for k, v in metrics.items():
            try:
                float_metrics[k] = float(v)
            except (TypeError, ValueError):
                continue

        return Frame(
            step=step,
            positions=positions,
            alive=alive,
            energy=energy,
            food_positions=food_pos,
            food_collected=food_col,
            field_values=field_vals,
            cluster_labels=cluster_labels,
            metrics=float_metrics,
        )
