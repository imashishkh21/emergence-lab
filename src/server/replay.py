"""Recording and replay of training sessions.

Records frames to disk as compressed MessagePack files for later playback.
Each session is stored as a directory containing:
  - metadata.json: session info (start/end step, duration, config)
  - frames.msgpack: compressed array of serialized frames
  - bookmarks.json: user-bookmarked moments

The replay system supports:
  - Variable-speed playback (0.5x, 1x, 2x, 4x)
  - Scrubbing to arbitrary positions
  - Bookmarking interesting moments
"""

from __future__ import annotations

import gzip
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import msgpack
import numpy as np

from src.server.streaming import Frame, _pack_array


@dataclass
class Bookmark:
    """A bookmarked moment in a recorded session."""

    step: int
    frame_index: int
    label: str
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "frame_index": self.frame_index,
            "label": self.label,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Bookmark:
        return cls(
            step=d["step"],
            frame_index=d["frame_index"],
            label=d["label"],
            timestamp=d.get("timestamp", 0.0),
        )


def _serialize_frame(frame: Frame) -> dict[str, Any]:
    """Serialize a Frame to a dict suitable for MessagePack storage.

    Similar to pack_frame() but returns a dict instead of bytes,
    so multiple frames can be batched into one MessagePack file.
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
        "training_mode": frame.training_mode,
    }
    if frame.cluster_labels is not None:
        data["clusters"] = _pack_array(frame.cluster_labels.astype(np.int8))
    if frame.agent_ids is not None:
        data["agent_ids"] = _pack_array(frame.agent_ids.astype(np.int32))
    if frame.parent_ids is not None:
        data["parent_ids"] = _pack_array(frame.parent_ids.astype(np.int32))
    if frame.birth_steps is not None:
        data["birth_steps"] = _pack_array(frame.birth_steps.astype(np.int32))
    if frame.lineage_data is not None:
        data["lineage_data"] = frame.lineage_data
    return data


def _deserialize_frame(data: dict[str, Any]) -> dict[str, Any]:
    """Return the raw dict as-is for streaming to clients.

    The stored format is already in the same shape as pack_frame() output,
    so it can be sent directly over WebSocket after msgpack.packb().
    """
    return data


class SessionRecorder:
    """Records training frames to disk for later replay.

    Frames are accumulated in memory and flushed to disk when save() is called
    or when the recorder is used as a context manager.

    Usage:
        recorder = SessionRecorder("recordings/my_session")
        recorder.start()
        for frame in training_frames:
            recorder.record(frame)
        recorder.save()
    """

    def __init__(
        self,
        output_dir: str | Path,
        max_frames: int = 0,
        sample_rate: int = 1,
    ) -> None:
        """Initialize the recorder.

        Args:
            output_dir: Directory to store the recording.
            max_frames: Maximum frames to record (0 = unlimited).
            sample_rate: Record every Nth frame (1 = all, 2 = every other, etc.).
        """
        self._output_dir = Path(output_dir)
        self._max_frames = max_frames
        self._sample_rate = max(1, sample_rate)
        self._frames: list[dict[str, Any]] = []
        self._bookmarks: list[Bookmark] = []
        self._frame_counter: int = 0
        self._recording: bool = False
        self._start_time: float = 0.0
        self._start_step: int = 0
        self._end_step: int = 0

    @property
    def frame_count(self) -> int:
        """Number of frames recorded so far."""
        return len(self._frames)

    @property
    def recording(self) -> bool:
        """Whether the recorder is currently active."""
        return self._recording

    @property
    def bookmarks(self) -> list[Bookmark]:
        """List of bookmarked moments."""
        return list(self._bookmarks)

    def start(self) -> None:
        """Start recording."""
        self._recording = True
        self._start_time = time.time()

    def stop(self) -> None:
        """Stop recording."""
        self._recording = False

    def record(self, frame: Frame) -> bool:
        """Record a frame.

        Respects sample_rate and max_frames limits.

        Args:
            frame: The training frame to record.

        Returns:
            True if the frame was recorded, False if skipped.
        """
        if not self._recording:
            return False

        self._frame_counter += 1
        if self._frame_counter % self._sample_rate != 0:
            return False

        if self._max_frames > 0 and len(self._frames) >= self._max_frames:
            return False

        serialized = _serialize_frame(frame)
        self._frames.append(serialized)

        if len(self._frames) == 1:
            self._start_step = frame.step
        self._end_step = frame.step

        return True

    def add_bookmark(self, label: str, step: int | None = None) -> Bookmark:
        """Bookmark the current or specified moment.

        Args:
            label: Human-readable label for this moment.
            step: Step to bookmark. If None, uses the latest frame's step.

        Returns:
            The created Bookmark.
        """
        if step is None and self._frames:
            step = self._frames[-1]["step"]
        elif step is None:
            step = 0

        # Find the frame index closest to the given step
        frame_index = 0
        for i, f in enumerate(self._frames):
            if f["step"] <= step:
                frame_index = i
            else:
                break

        bookmark = Bookmark(step=step, frame_index=frame_index, label=label)
        self._bookmarks.append(bookmark)
        return bookmark

    def save(self) -> Path:
        """Save the recording to disk.

        Creates the output directory and writes:
          - metadata.json
          - frames.msgpack.gz (gzip-compressed MessagePack)
          - bookmarks.json

        Returns:
            Path to the output directory.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Write metadata
        metadata = {
            "start_step": self._start_step,
            "end_step": self._end_step,
            "frame_count": len(self._frames),
            "sample_rate": self._sample_rate,
            "duration_seconds": time.time() - self._start_time
            if self._start_time > 0
            else 0.0,
            "recorded_at": time.time(),
        }
        metadata_path = self._output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Write frames (gzip-compressed MessagePack)
        frames_path = self._output_dir / "frames.msgpack.gz"
        packed: bytes = msgpack.packb(self._frames, use_bin_type=True)
        with gzip.open(frames_path, "wb") as f:
            f.write(packed)

        # Write bookmarks
        bookmarks_path = self._output_dir / "bookmarks.json"
        with open(bookmarks_path, "w") as f:
            json.dump(
                [b.to_dict() for b in self._bookmarks],
                f,
                indent=2,
            )

        return self._output_dir

    def __enter__(self) -> SessionRecorder:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()
        self.save()


class SessionPlayer:
    """Plays back a recorded training session.

    Supports variable-speed playback and scrubbing to arbitrary positions.

    Usage:
        player = SessionPlayer("recordings/my_session")
        player.load()
        player.play()
        while player.playing:
            frame_data = player.current_frame()
            player.advance()
    """

    def __init__(self, session_dir: str | Path) -> None:
        """Initialize the player.

        Args:
            session_dir: Directory containing the recorded session.
        """
        self._session_dir = Path(session_dir)
        self._frames: list[dict[str, Any]] = []
        self._bookmarks: list[Bookmark] = []
        self._metadata: dict[str, Any] = {}
        self._position: int = 0
        self._playing: bool = False
        self._speed: float = 1.0
        self._loaded: bool = False

    @property
    def loaded(self) -> bool:
        """Whether the session has been loaded."""
        return self._loaded

    @property
    def frame_count(self) -> int:
        """Total number of frames in the session."""
        return len(self._frames)

    @property
    def position(self) -> int:
        """Current playback position (frame index)."""
        return self._position

    @property
    def playing(self) -> bool:
        """Whether playback is active."""
        return self._playing

    @property
    def speed(self) -> float:
        """Playback speed multiplier."""
        return self._speed

    @speed.setter
    def speed(self, value: float) -> None:
        self._speed = max(0.25, min(16.0, value))

    @property
    def metadata(self) -> dict[str, Any]:
        """Session metadata."""
        return dict(self._metadata)

    @property
    def bookmarks(self) -> list[Bookmark]:
        """Session bookmarks."""
        return list(self._bookmarks)

    @property
    def progress(self) -> float:
        """Playback progress as a fraction 0.0–1.0."""
        if self.frame_count == 0:
            return 0.0
        return self._position / max(1, self.frame_count - 1)

    def load(self) -> None:
        """Load the session from disk.

        Raises:
            FileNotFoundError: If the session directory doesn't exist.
            ValueError: If required files are missing.
        """
        if not self._session_dir.exists():
            raise FileNotFoundError(
                f"Session directory not found: {self._session_dir}"
            )

        # Load metadata
        metadata_path = self._session_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self._metadata = json.load(f)
        else:
            self._metadata = {}

        # Load frames
        frames_path = self._session_dir / "frames.msgpack.gz"
        if not frames_path.exists():
            raise ValueError(f"Frames file not found: {frames_path}")

        with gzip.open(frames_path, "rb") as f:
            raw = f.read()
        self._frames = msgpack.unpackb(raw, raw=False)

        # Load bookmarks
        bookmarks_path = self._session_dir / "bookmarks.json"
        if bookmarks_path.exists():
            with open(bookmarks_path) as f:
                bookmark_data = json.load(f)
            self._bookmarks = [Bookmark.from_dict(b) for b in bookmark_data]
        else:
            self._bookmarks = []

        self._position = 0
        self._loaded = True

    def play(self) -> None:
        """Start or resume playback."""
        self._playing = True

    def pause(self) -> None:
        """Pause playback."""
        self._playing = False

    def seek(self, position: int) -> None:
        """Seek to a specific frame position.

        Args:
            position: Frame index to seek to (clamped to valid range).
        """
        self._position = max(0, min(position, max(0, self.frame_count - 1)))

    def seek_to_step(self, target_step: int) -> None:
        """Seek to the frame closest to a given training step.

        Args:
            target_step: The training step to seek to.
        """
        best_idx = 0
        best_diff = float("inf")
        for i, frame in enumerate(self._frames):
            diff = abs(frame.get("step", 0) - target_step)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        self._position = best_idx

    def seek_to_bookmark(self, bookmark_index: int) -> None:
        """Seek to a bookmarked position.

        Args:
            bookmark_index: Index into the bookmarks list.
        """
        if 0 <= bookmark_index < len(self._bookmarks):
            self.seek(self._bookmarks[bookmark_index].frame_index)

    def current_frame(self) -> dict[str, Any] | None:
        """Get the current frame data for streaming.

        Returns:
            Frame dict ready for MessagePack packing, or None if no frames.
        """
        if not self._frames or self._position >= len(self._frames):
            return None
        return self._frames[self._position]

    def advance(self) -> bool:
        """Advance to the next frame.

        Returns:
            True if advanced, False if at end of recording.
        """
        if not self._playing:
            return False
        next_pos = self._position + 1
        if next_pos >= self.frame_count:
            self._playing = False
            return False
        self._position = next_pos
        return True

    def current_step(self) -> int:
        """Get the training step of the current frame."""
        frame = self.current_frame()
        if frame is None:
            return 0
        return int(frame.get("step", 0))


def list_sessions(recordings_dir: str | Path) -> list[dict[str, Any]]:
    """List all recorded sessions in a directory.

    Args:
        recordings_dir: Directory containing session subdirectories.

    Returns:
        List of session info dicts with name, path, and metadata.
    """
    recordings_dir = Path(recordings_dir)
    sessions: list[dict[str, Any]] = []

    if not recordings_dir.exists():
        return sessions

    for entry in sorted(recordings_dir.iterdir()):
        if not entry.is_dir():
            continue
        frames_path = entry / "frames.msgpack.gz"
        if not frames_path.exists():
            continue

        metadata: dict[str, Any] = {}
        metadata_path = entry / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        sessions.append({
            "name": entry.name,
            "path": str(entry),
            "metadata": metadata,
        })

    return sessions
