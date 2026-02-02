"""Checkpointing module for saving and loading full training state.

Saves and restores: params, opt_state, agent_params, prng_key, step,
config, and tracker state. Uses enhanced pickle with JAX→numpy conversion
for cross-platform compatibility.

Checkpoint rotation: keeps the last N checkpoints and deletes the oldest.
"""

import dataclasses
import glob
import os
import pickle
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def _jax_to_numpy(tree: Any) -> Any:
    """Convert all JAX arrays in a pytree to numpy arrays."""
    return jax.tree_util.tree_map(
        lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x,
        tree,
    )


def _numpy_to_jax(tree: Any) -> Any:
    """Convert all numpy arrays in a pytree to JAX arrays."""
    return jax.tree_util.tree_map(
        lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x,
        tree,
    )


def _config_to_dict(config: Any) -> dict[str, Any]:
    """Serialize a Config dataclass to a plain dict."""

    def _convert(obj: object) -> object:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {k: _convert(v) for k, v in dataclasses.asdict(obj).items()}
        return obj

    return _convert(config)  # type: ignore[return-value]


def save_checkpoint(
    path: str,
    state_dict: dict[str, Any],
    max_checkpoints: int = 5,
) -> str:
    """Save a checkpoint to disk.

    All JAX arrays are converted to numpy before pickling for
    cross-platform compatibility (GPU→CPU safe).

    Args:
        path: File path for the checkpoint (e.g. "checkpoints/step_100000.pkl").
        state_dict: Dict containing training state. Expected keys:
            - params: Shared policy parameters (Flax pytree).
            - opt_state: Optimizer state (optax pytree).
            - agent_params: Per-agent parameters (pytree with leading max_agents dim),
              or None if evolution is disabled.
            - prng_key: JAX PRNG key.
            - step: Current training step (int).
            - config: Config dataclass or dict.
            - tracker_state: Dict of tracker serializations (optional).
        max_checkpoints: Maximum number of checkpoints to keep in the
            same directory. Oldest are deleted. Set to 0 to disable rotation.

    Returns:
        The absolute path to the saved checkpoint.
    """
    # Convert JAX arrays to numpy for cross-platform pickling
    serializable: dict[str, Any] = {}

    for key, value in state_dict.items():
        if key == "config":
            # Serialize Config to plain dict if it's a dataclass
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                serializable[key] = _config_to_dict(value)
            else:
                serializable[key] = value
        elif key in ("params", "opt_state", "agent_params", "prng_key"):
            if value is not None:
                serializable[key] = _jax_to_numpy(value)
            else:
                serializable[key] = None
        else:
            # step, tracker_state, etc. — pass through
            serializable[key] = value

    # Ensure parent directory exists
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # Write atomically: write to temp file then rename
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(serializable, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)

    # Create/update a "latest" symlink or copy for convenience
    if parent_dir:
        latest_path = os.path.join(parent_dir, "latest.pkl")
        # Use a relative path for the symlink
        rel_target = os.path.basename(path)
        # Remove existing symlink/file
        if os.path.exists(latest_path) or os.path.islink(latest_path):
            os.remove(latest_path)
        os.symlink(rel_target, latest_path)

    # Checkpoint rotation: keep only the last max_checkpoints
    if max_checkpoints > 0 and parent_dir:
        _rotate_checkpoints(parent_dir, max_checkpoints)

    return os.path.abspath(path)


def _rotate_checkpoints(checkpoint_dir: str, max_checkpoints: int) -> None:
    """Delete oldest checkpoints, keeping only the most recent ones.

    Only considers files matching step_*.pkl pattern. The latest.pkl
    symlink is never deleted.

    Args:
        checkpoint_dir: Directory containing checkpoint files.
        max_checkpoints: Maximum number of checkpoint files to keep.
    """
    pattern = os.path.join(checkpoint_dir, "step_*.pkl")
    checkpoint_files = sorted(glob.glob(pattern), key=os.path.getmtime)

    # Delete oldest files beyond the limit
    while len(checkpoint_files) > max_checkpoints:
        oldest = checkpoint_files.pop(0)
        os.remove(oldest)


def load_checkpoint(path: str) -> dict[str, Any]:
    """Load a checkpoint from disk.

    Numpy arrays are converted back to JAX arrays.

    Args:
        path: Path to the checkpoint file.

    Returns:
        Dict with all saved state. Keys typically include:
            - params: Shared policy parameters (JAX pytree).
            - opt_state: Optimizer state (JAX pytree).
            - agent_params: Per-agent parameters (JAX pytree) or None.
            - prng_key: JAX PRNG key.
            - step: Training step (int).
            - config: Config dict.
            - tracker_state: Dict of tracker serializations (if saved).

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        pickle.UnpicklingError: If the file is corrupted.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    # Convert numpy arrays back to JAX arrays
    result: dict[str, Any] = {}
    for key, value in data.items():
        if key in ("params", "opt_state", "agent_params", "prng_key"):
            if value is not None:
                result[key] = _numpy_to_jax(value)
            else:
                result[key] = None
        else:
            # config (dict), step (int), tracker_state (dict) — pass through
            result[key] = value

    return result
