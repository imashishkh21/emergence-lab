"""Tests for the checkpointing module (save/load full training state).

Covers: JAX→numpy conversion, numpy→JAX round-trip, checkpoint rotation,
latest symlink, atomic writes, config serialization, tracker state
preservation, and edge cases.
"""

import os
import pickle
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from src.configs import Config
from src.training.checkpointing import (
    _config_to_dict,
    _jax_to_numpy,
    _numpy_to_jax,
    _rotate_checkpoints,
    load_checkpoint,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config() -> Config:
    """Return a minimal Config for testing."""
    return Config()


def _make_params() -> dict:
    """Create a simple Flax-like params pytree with JAX arrays."""
    key = jax.random.PRNGKey(0)
    return {
        "params": {
            "Dense_0": {
                "kernel": jax.random.normal(key, (10, 64)),
                "bias": jnp.zeros(64),
            },
            "Dense_1": {
                "kernel": jax.random.normal(key, (64, 6)),
                "bias": jnp.zeros(6),
            },
        }
    }


def _make_agent_params(max_agents: int = 8) -> dict:
    """Create per-agent params pytree with leading (max_agents,) dim."""
    key = jax.random.PRNGKey(1)
    return {
        "params": {
            "Dense_0": {
                "kernel": jax.random.normal(key, (max_agents, 10, 64)),
                "bias": jnp.zeros((max_agents, 64)),
            },
        }
    }


def _make_opt_state(params: dict) -> Any:
    """Create an optax optimizer state from params."""
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(3e-4, eps=1e-5),
    )
    return optimizer.init(params)


def _make_full_state_dict() -> dict:
    """Create a full state dict with all expected checkpoint keys."""
    params = _make_params()
    return {
        "params": params,
        "opt_state": _make_opt_state(params),
        "agent_params": _make_agent_params(),
        "prng_key": jax.random.PRNGKey(42),
        "step": 100000,
        "config": _make_config(),
        "tracker_state": {
            "emergence": {"window_size": 20, "z_threshold": 3.0, "step_count": 5},
            "specialization": {"window_size": 20, "step_count": 3},
        },
    }



# ===========================================================================
# JAX ↔ numpy conversion
# ===========================================================================


class TestJaxNumpyConversion:
    """Tests for _jax_to_numpy and _numpy_to_jax helpers."""

    def test_jax_to_numpy_converts_arrays(self):
        tree = {"a": jnp.array([1.0, 2.0]), "b": jnp.zeros(3)}
        result = _jax_to_numpy(tree)
        assert isinstance(result["a"], np.ndarray)
        assert isinstance(result["b"], np.ndarray)

    def test_numpy_to_jax_converts_arrays(self):
        tree = {"a": np.array([1.0, 2.0]), "b": np.zeros(3)}
        result = _numpy_to_jax(tree)
        assert isinstance(result["a"], jnp.ndarray)
        assert isinstance(result["b"], jnp.ndarray)

    def test_round_trip_preserves_values(self):
        original = {"x": jnp.array([1.0, 2.0, 3.0])}
        as_numpy = _jax_to_numpy(original)
        back_to_jax = _numpy_to_jax(as_numpy)
        np.testing.assert_allclose(
            np.array(back_to_jax["x"]), np.array(original["x"])
        )

    def test_round_trip_preserves_nested_pytree(self):
        params = _make_params()
        as_numpy = _jax_to_numpy(params)
        back_to_jax = _numpy_to_jax(as_numpy)
        # Check a specific leaf
        np.testing.assert_allclose(
            np.array(back_to_jax["params"]["Dense_0"]["kernel"]),
            np.array(params["params"]["Dense_0"]["kernel"]),
        )

    def test_jax_to_numpy_passes_non_arrays(self):
        tree = {"a": jnp.array(1.0), "b": 42, "c": "hello"}
        result = _jax_to_numpy(tree)
        assert result["b"] == 42
        assert result["c"] == "hello"

    def test_numpy_to_jax_passes_non_arrays(self):
        tree = {"a": np.array(1.0), "b": 42, "c": "hello"}
        result = _numpy_to_jax(tree)
        assert result["b"] == 42
        assert result["c"] == "hello"

    def test_jax_to_numpy_preserves_dtype(self):
        tree = {"f32": jnp.zeros(3, dtype=jnp.float32), "i32": jnp.zeros(3, dtype=jnp.int32)}
        result = _jax_to_numpy(tree)
        assert result["f32"].dtype == np.float32
        assert result["i32"].dtype == np.int32


# ===========================================================================
# Config serialization
# ===========================================================================


class TestConfigSerialization:
    """Tests for _config_to_dict."""

    def test_returns_dict(self):
        config = _make_config()
        result = _config_to_dict(config)
        assert isinstance(result, dict)

    def test_contains_subconfigs(self):
        config = _make_config()
        result = _config_to_dict(config)
        assert "env" in result
        assert "field" in result
        assert "agent" in result
        assert "train" in result
        assert "log" in result

    def test_preserves_values(self):
        config = _make_config()
        result = _config_to_dict(config)
        assert result["env"]["grid_size"] == config.env.grid_size
        assert result["train"]["seed"] == config.train.seed
        assert result["log"]["save_interval"] == config.log.save_interval

    def test_is_picklable(self):
        config = _make_config()
        result = _config_to_dict(config)
        # Should be pickle-safe (no dataclass instances)
        pickled = pickle.dumps(result)
        restored = pickle.loads(pickled)
        assert restored["env"]["grid_size"] == config.env.grid_size


# ===========================================================================
# save_checkpoint
# ===========================================================================


class TestSaveCheckpoint:
    """Tests for save_checkpoint."""

    def test_creates_file(self, tmp_path):
        state_dict = _make_full_state_dict()
        path = str(tmp_path / "ckpt" / "step_100000.pkl")
        save_checkpoint(path, state_dict)
        assert os.path.exists(path)

    def test_creates_parent_directory(self, tmp_path):
        state_dict = _make_full_state_dict()
        path = str(tmp_path / "deep" / "nested" / "step_100.pkl")
        save_checkpoint(path, state_dict)
        assert os.path.exists(path)

    def test_creates_latest_symlink(self, tmp_path):
        state_dict = _make_full_state_dict()
        ckpt_dir = tmp_path / "checkpoints"
        path = str(ckpt_dir / "step_100000.pkl")
        save_checkpoint(path, state_dict)
        latest = ckpt_dir / "latest.pkl"
        assert latest.exists() or latest.is_symlink()
        # Symlink should point to the checkpoint file
        assert os.path.realpath(str(latest)) == os.path.realpath(path)

    def test_latest_symlink_updates(self, tmp_path):
        state_dict = _make_full_state_dict()
        ckpt_dir = tmp_path / "checkpoints"
        path1 = str(ckpt_dir / "step_100000.pkl")
        path2 = str(ckpt_dir / "step_200000.pkl")
        save_checkpoint(path1, state_dict)
        save_checkpoint(path2, state_dict)
        latest = ckpt_dir / "latest.pkl"
        assert os.path.realpath(str(latest)) == os.path.realpath(path2)

    def test_returns_absolute_path(self, tmp_path):
        state_dict = _make_full_state_dict()
        path = str(tmp_path / "step_1.pkl")
        result = save_checkpoint(path, state_dict)
        assert os.path.isabs(result)

    def test_jax_arrays_converted_to_numpy(self, tmp_path):
        state_dict = _make_full_state_dict()
        path = str(tmp_path / "step_1.pkl")
        save_checkpoint(path, state_dict)
        # Load raw pickle and check types
        with open(path, "rb") as f:
            raw = pickle.load(f)
        kernel = raw["params"]["params"]["Dense_0"]["kernel"]
        assert isinstance(kernel, np.ndarray), f"Expected np.ndarray, got {type(kernel)}"

    def test_config_serialized_as_dict(self, tmp_path):
        state_dict = _make_full_state_dict()
        path = str(tmp_path / "step_1.pkl")
        save_checkpoint(path, state_dict)
        with open(path, "rb") as f:
            raw = pickle.load(f)
        assert isinstance(raw["config"], dict)
        assert "env" in raw["config"]

    def test_none_agent_params_handled(self, tmp_path):
        state_dict = _make_full_state_dict()
        state_dict["agent_params"] = None
        path = str(tmp_path / "step_1.pkl")
        save_checkpoint(path, state_dict)
        with open(path, "rb") as f:
            raw = pickle.load(f)
        assert raw["agent_params"] is None

    def test_step_preserved_as_int(self, tmp_path):
        state_dict = _make_full_state_dict()
        state_dict["step"] = 42
        path = str(tmp_path / "step_42.pkl")
        save_checkpoint(path, state_dict)
        with open(path, "rb") as f:
            raw = pickle.load(f)
        assert raw["step"] == 42
        assert isinstance(raw["step"], int)

    def test_tracker_state_preserved(self, tmp_path):
        state_dict = _make_full_state_dict()
        path = str(tmp_path / "step_1.pkl")
        save_checkpoint(path, state_dict)
        with open(path, "rb") as f:
            raw = pickle.load(f)
        assert "tracker_state" in raw
        assert raw["tracker_state"]["emergence"]["window_size"] == 20


# ===========================================================================
# Checkpoint rotation
# ===========================================================================


class TestCheckpointRotation:
    """Tests for checkpoint rotation (keep last N)."""

    def test_keeps_max_checkpoints(self, tmp_path):
        ckpt_dir = str(tmp_path / "checkpoints")
        os.makedirs(ckpt_dir)
        state_dict = _make_full_state_dict()
        # Save 7 checkpoints with max_checkpoints=5
        for i in range(7):
            path = os.path.join(ckpt_dir, f"step_{i * 100000}.pkl")
            save_checkpoint(path, state_dict, max_checkpoints=5)
        # Count step_*.pkl files (exclude latest.pkl)
        import glob
        step_files = glob.glob(os.path.join(ckpt_dir, "step_*.pkl"))
        assert len(step_files) == 5

    def test_oldest_deleted_first(self, tmp_path):
        ckpt_dir = str(tmp_path / "checkpoints")
        os.makedirs(ckpt_dir)
        state_dict = _make_full_state_dict()
        for i in range(4):
            path = os.path.join(ckpt_dir, f"step_{i * 100000}.pkl")
            save_checkpoint(path, state_dict, max_checkpoints=3)
        # The first checkpoint (step_0) should have been deleted
        assert not os.path.exists(os.path.join(ckpt_dir, "step_0.pkl"))
        # The last three should remain
        assert os.path.exists(os.path.join(ckpt_dir, "step_100000.pkl"))
        assert os.path.exists(os.path.join(ckpt_dir, "step_200000.pkl"))
        assert os.path.exists(os.path.join(ckpt_dir, "step_300000.pkl"))

    def test_rotation_disabled_with_zero(self, tmp_path):
        ckpt_dir = str(tmp_path / "checkpoints")
        os.makedirs(ckpt_dir)
        state_dict = _make_full_state_dict()
        for i in range(5):
            path = os.path.join(ckpt_dir, f"step_{i * 100000}.pkl")
            save_checkpoint(path, state_dict, max_checkpoints=0)
        import glob
        step_files = glob.glob(os.path.join(ckpt_dir, "step_*.pkl"))
        assert len(step_files) == 5  # All kept

    def test_latest_symlink_not_counted(self, tmp_path):
        ckpt_dir = str(tmp_path / "checkpoints")
        os.makedirs(ckpt_dir)
        state_dict = _make_full_state_dict()
        for i in range(3):
            path = os.path.join(ckpt_dir, f"step_{i * 100000}.pkl")
            save_checkpoint(path, state_dict, max_checkpoints=3)
        # latest.pkl should exist but not be counted as a checkpoint
        latest = os.path.join(ckpt_dir, "latest.pkl")
        assert os.path.exists(latest) or os.path.islink(latest)
        import glob
        step_files = glob.glob(os.path.join(ckpt_dir, "step_*.pkl"))
        assert len(step_files) == 3

    def test_rotate_checkpoints_direct(self, tmp_path):
        ckpt_dir = str(tmp_path)
        # Create 5 step files manually
        for i in range(5):
            path = os.path.join(ckpt_dir, f"step_{i}.pkl")
            with open(path, "wb") as f:
                pickle.dump({}, f)
        _rotate_checkpoints(ckpt_dir, max_checkpoints=3)
        import glob
        remaining = glob.glob(os.path.join(ckpt_dir, "step_*.pkl"))
        assert len(remaining) == 3


# ===========================================================================
# load_checkpoint
# ===========================================================================


class TestLoadCheckpoint:
    """Tests for load_checkpoint."""

    def test_loads_saved_checkpoint(self, tmp_path):
        state_dict = _make_full_state_dict()
        path = str(tmp_path / "step_100000.pkl")
        save_checkpoint(path, state_dict)
        loaded = load_checkpoint(path)
        assert "params" in loaded
        assert "opt_state" in loaded
        assert "step" in loaded

    def test_numpy_converted_back_to_jax(self, tmp_path):
        state_dict = _make_full_state_dict()
        path = str(tmp_path / "step_1.pkl")
        save_checkpoint(path, state_dict)
        loaded = load_checkpoint(path)
        kernel = loaded["params"]["params"]["Dense_0"]["kernel"]
        assert isinstance(kernel, jnp.ndarray), f"Expected jnp.ndarray, got {type(kernel)}"

    def test_params_values_preserved(self, tmp_path):
        state_dict = _make_full_state_dict()
        path = str(tmp_path / "step_1.pkl")
        save_checkpoint(path, state_dict)
        loaded = load_checkpoint(path)
        original_kernel = np.array(state_dict["params"]["params"]["Dense_0"]["kernel"])
        loaded_kernel = np.array(loaded["params"]["params"]["Dense_0"]["kernel"])
        np.testing.assert_allclose(loaded_kernel, original_kernel)

    def test_agent_params_preserved(self, tmp_path):
        state_dict = _make_full_state_dict()
        path = str(tmp_path / "step_1.pkl")
        save_checkpoint(path, state_dict)
        loaded = load_checkpoint(path)
        original = np.array(
            state_dict["agent_params"]["params"]["Dense_0"]["kernel"]
        )
        restored = np.array(
            loaded["agent_params"]["params"]["Dense_0"]["kernel"]
        )
        np.testing.assert_allclose(restored, original)

    def test_agent_params_none_preserved(self, tmp_path):
        state_dict = _make_full_state_dict()
        state_dict["agent_params"] = None
        path = str(tmp_path / "step_1.pkl")
        save_checkpoint(path, state_dict)
        loaded = load_checkpoint(path)
        assert loaded["agent_params"] is None

    def test_prng_key_preserved(self, tmp_path):
        state_dict = _make_full_state_dict()
        path = str(tmp_path / "step_1.pkl")
        save_checkpoint(path, state_dict)
        loaded = load_checkpoint(path)
        np.testing.assert_array_equal(
            np.array(loaded["prng_key"]),
            np.array(state_dict["prng_key"]),
        )

    def test_step_preserved(self, tmp_path):
        state_dict = _make_full_state_dict()
        state_dict["step"] = 999999
        path = str(tmp_path / "step_1.pkl")
        save_checkpoint(path, state_dict)
        loaded = load_checkpoint(path)
        assert loaded["step"] == 999999

    def test_config_dict_preserved(self, tmp_path):
        state_dict = _make_full_state_dict()
        path = str(tmp_path / "step_1.pkl")
        save_checkpoint(path, state_dict)
        loaded = load_checkpoint(path)
        assert isinstance(loaded["config"], dict)
        assert loaded["config"]["env"]["grid_size"] == 20

    def test_tracker_state_preserved(self, tmp_path):
        state_dict = _make_full_state_dict()
        path = str(tmp_path / "step_1.pkl")
        save_checkpoint(path, state_dict)
        loaded = load_checkpoint(path)
        assert loaded["tracker_state"]["emergence"]["window_size"] == 20

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_checkpoint("/nonexistent/path/checkpoint.pkl")

    def test_load_via_latest_symlink(self, tmp_path):
        state_dict = _make_full_state_dict()
        state_dict["step"] = 777
        ckpt_dir = tmp_path / "checkpoints"
        path = str(ckpt_dir / "step_777.pkl")
        save_checkpoint(path, state_dict)
        # Load via the latest symlink
        latest_path = str(ckpt_dir / "latest.pkl")
        loaded = load_checkpoint(latest_path)
        assert loaded["step"] == 777


# ===========================================================================
# Full round-trip
# ===========================================================================


class TestFullRoundTrip:
    """Tests for complete save→load round-trip with all state."""

    def test_full_state_round_trip(self, tmp_path):
        state_dict = _make_full_state_dict()
        path = str(tmp_path / "step_100000.pkl")
        save_checkpoint(path, state_dict)
        loaded = load_checkpoint(path)

        # params preserved
        np.testing.assert_allclose(
            np.array(loaded["params"]["params"]["Dense_0"]["kernel"]),
            np.array(state_dict["params"]["params"]["Dense_0"]["kernel"]),
        )
        # agent_params preserved
        np.testing.assert_allclose(
            np.array(loaded["agent_params"]["params"]["Dense_0"]["kernel"]),
            np.array(state_dict["agent_params"]["params"]["Dense_0"]["kernel"]),
        )
        # step preserved
        assert loaded["step"] == state_dict["step"]
        # config preserved
        assert loaded["config"]["env"]["grid_size"] == 20
        # tracker state preserved
        assert loaded["tracker_state"] == state_dict["tracker_state"]

    def test_opt_state_round_trip(self, tmp_path):
        """Optimizer state round-trip — structure preserved after save/load."""
        state_dict = _make_full_state_dict()
        path = str(tmp_path / "step_1.pkl")
        save_checkpoint(path, state_dict)
        loaded = load_checkpoint(path)

        # The optax state is a nested tuple/namedtuple structure.
        # After round-trip, leaves should have the same values.
        original_leaves = jax.tree_util.tree_leaves(state_dict["opt_state"])
        loaded_leaves = jax.tree_util.tree_leaves(loaded["opt_state"])
        assert len(original_leaves) == len(loaded_leaves)
        for orig, rest in zip(original_leaves, loaded_leaves):
            np.testing.assert_allclose(
                np.array(rest), np.array(orig), atol=1e-7
            )

    def test_multiple_save_load_cycles(self, tmp_path):
        """Save, load, modify, save again, load again — no corruption."""
        state_dict = _make_full_state_dict()
        path1 = str(tmp_path / "step_1.pkl")
        save_checkpoint(path1, state_dict)

        loaded1 = load_checkpoint(path1)
        loaded1["step"] = 200000

        path2 = str(tmp_path / "step_2.pkl")
        save_checkpoint(path2, loaded1)

        loaded2 = load_checkpoint(path2)
        assert loaded2["step"] == 200000
        # Params should still match the original
        np.testing.assert_allclose(
            np.array(loaded2["params"]["params"]["Dense_0"]["kernel"]),
            np.array(state_dict["params"]["params"]["Dense_0"]["kernel"]),
        )

    def test_cross_platform_pickle(self, tmp_path):
        """Checkpoint contains only numpy arrays (no JAX-device-specific data)."""
        state_dict = _make_full_state_dict()
        path = str(tmp_path / "step_1.pkl")
        save_checkpoint(path, state_dict)

        # Load raw and verify no JAX arrays in the serialized form
        with open(path, "rb") as f:
            raw = pickle.load(f)

        def _check_no_jax(tree: Any, prefix: str = "") -> None:
            if isinstance(tree, dict):
                for k, v in tree.items():
                    _check_no_jax(v, f"{prefix}.{k}")
            elif isinstance(tree, (list, tuple)):
                for i, v in enumerate(tree):
                    _check_no_jax(v, f"{prefix}[{i}]")
            elif isinstance(tree, jnp.ndarray):
                raise AssertionError(
                    f"Found JAX array at {prefix}: {type(tree)}"
                )

        _check_no_jax(raw["params"], "params")
        if raw["agent_params"] is not None:
            _check_no_jax(raw["agent_params"], "agent_params")
