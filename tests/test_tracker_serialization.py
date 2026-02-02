"""Tests for tracker serialization (to_dict/from_dict round-trips).

Covers EmergenceTracker and SpecializationTracker serialization,
ensuring full state round-trip preservation.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.analysis.emergence import EmergenceTracker
from src.analysis.specialization import SpecializationTracker
from src.configs import Config
from src.field.field import FieldState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config() -> Config:
    """Return a minimal default Config for tracker construction."""
    return Config()


def _make_field_state(grid_size: int = 20, num_channels: int = 4) -> FieldState:
    """Create a FieldState with random values."""
    values = np.random.default_rng(42).random(
        (grid_size, grid_size, num_channels)
    ).astype(np.float32)
    return FieldState(values=values)


def _make_agent_params(max_agents: int = 8, param_dim: int = 16) -> dict:
    """Create a simple agent_params pytree with one leaf."""
    rng = np.random.default_rng(0)
    return {"Dense_0": {"kernel": rng.normal(size=(max_agents, param_dim)).astype(np.float32)}}


# ===========================================================================
# EmergenceTracker serialization
# ===========================================================================


class TestEmergenceTrackerSerialization:
    """Tests for EmergenceTracker.to_dict() and from_dict()."""

    def test_to_dict_returns_dict(self):
        tracker = EmergenceTracker(_make_config())
        d = tracker.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_keys(self):
        tracker = EmergenceTracker(_make_config())
        d = tracker.to_dict()
        assert "window_size" in d
        assert "z_threshold" in d
        assert "history" in d
        assert "events" in d
        assert "step_count" in d

    def test_empty_tracker_round_trip(self):
        config = _make_config()
        tracker = EmergenceTracker(config, window_size=15, z_threshold=2.5)
        d = tracker.to_dict()
        restored = EmergenceTracker.from_dict(d, config)
        assert restored.window_size == 15
        assert restored.z_threshold == 2.5
        assert restored.step_count == 0
        assert len(restored.events) == 0
        for name in ("entropy", "structure"):
            assert len(restored.history[name]) == 0

    def test_round_trip_preserves_history(self):
        config = _make_config()
        tracker = EmergenceTracker(config)
        rng = np.random.default_rng(1)
        for i in range(10):
            field = FieldState(
                values=rng.random((20, 20, 4)).astype(np.float32)
            )
            tracker.update(field, step=i * 100)

        d = tracker.to_dict()
        restored = EmergenceTracker.from_dict(d, config)

        assert restored.step_count == tracker.step_count
        for name in ("entropy", "structure"):
            assert restored.history[name].values == tracker.history[name].values
            assert restored.history[name].steps == tracker.history[name].steps

    def test_round_trip_preserves_events(self):
        """Create a tracker that detects an event, serialize, and verify."""
        config = _make_config()
        tracker = EmergenceTracker(config, window_size=5, z_threshold=2.0)
        rng = np.random.default_rng(2)

        # Feed steady values first to build baseline
        for i in range(10):
            field = FieldState(values=np.full((20, 20, 4), 0.5, dtype=np.float32))
            tracker.update(field, step=i * 100)

        # Now feed a spike to trigger an event
        field_spike = FieldState(
            values=rng.random((20, 20, 4)).astype(np.float32) * 10.0
        )
        tracker.update(field_spike, step=1100)

        d = tracker.to_dict()
        restored = EmergenceTracker.from_dict(d, config)

        assert len(restored.events) == len(tracker.events)
        for orig, rest in zip(tracker.events, restored.events):
            assert rest.step == orig.step
            assert rest.metric_name == orig.metric_name
            assert rest.old_value == pytest.approx(orig.old_value)
            assert rest.new_value == pytest.approx(orig.new_value)
            assert rest.z_score == pytest.approx(orig.z_score)

    def test_round_trip_get_metrics_match(self):
        config = _make_config()
        tracker = EmergenceTracker(config)
        for i in range(5):
            field = FieldState(
                values=np.random.default_rng(i).random((20, 20, 4)).astype(np.float32)
            )
            tracker.update(field, step=i * 100)

        d = tracker.to_dict()
        restored = EmergenceTracker.from_dict(d, config)

        orig_metrics = tracker.get_metrics()
        rest_metrics = restored.get_metrics()
        for key in orig_metrics:
            assert rest_metrics[key] == pytest.approx(orig_metrics[key])

    def test_round_trip_get_summary_match(self):
        config = _make_config()
        tracker = EmergenceTracker(config)
        for i in range(5):
            field = FieldState(
                values=np.random.default_rng(i).random((20, 20, 4)).astype(np.float32)
            )
            tracker.update(field, step=i * 100)

        d = tracker.to_dict()
        restored = EmergenceTracker.from_dict(d, config)

        orig_summary = tracker.get_summary()
        rest_summary = restored.get_summary()
        for key in orig_summary:
            if isinstance(orig_summary[key], float):
                assert rest_summary[key] == pytest.approx(orig_summary[key])
            else:
                assert rest_summary[key] == orig_summary[key]

    def test_to_dict_is_json_serializable(self):
        config = _make_config()
        tracker = EmergenceTracker(config)
        for i in range(3):
            field = FieldState(
                values=np.random.default_rng(i).random((20, 20, 4)).astype(np.float32)
            )
            tracker.update(field, step=i * 100)

        d = tracker.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    def test_restored_tracker_can_continue_updating(self):
        config = _make_config()
        tracker = EmergenceTracker(config)
        for i in range(5):
            field = FieldState(
                values=np.random.default_rng(i).random((20, 20, 4)).astype(np.float32)
            )
            tracker.update(field, step=i * 100)

        d = tracker.to_dict()
        restored = EmergenceTracker.from_dict(d, config)

        # Continue updating the restored tracker
        for i in range(5, 10):
            field = FieldState(
                values=np.random.default_rng(i).random((20, 20, 4)).astype(np.float32)
            )
            restored.update(field, step=i * 100)

        assert restored.step_count == 10
        for name in ("entropy", "structure"):
            assert len(restored.history[name]) == 10

    def test_custom_window_and_threshold(self):
        config = _make_config()
        tracker = EmergenceTracker(config, window_size=30, z_threshold=4.5)
        d = tracker.to_dict()
        restored = EmergenceTracker.from_dict(d, config)
        assert restored.window_size == 30
        assert restored.z_threshold == 4.5


# ===========================================================================
# SpecializationTracker serialization
# ===========================================================================


class TestSpecializationTrackerSerialization:
    """Tests for SpecializationTracker.to_dict() and from_dict()."""

    def test_to_dict_returns_dict(self):
        tracker = SpecializationTracker(_make_config())
        d = tracker.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_keys(self):
        tracker = SpecializationTracker(_make_config())
        d = tracker.to_dict()
        assert "window_size" in d
        assert "z_threshold" in d
        assert "history" in d
        assert "steps" in d
        assert "events" in d
        assert "step_count" in d

    def test_empty_tracker_round_trip(self):
        config = _make_config()
        tracker = SpecializationTracker(config, window_size=10, z_threshold=2.0)
        d = tracker.to_dict()
        restored = SpecializationTracker.from_dict(d, config)
        assert restored.window_size == 10
        assert restored.z_threshold == 2.0
        assert restored.step_count == 0
        assert len(restored.events) == 0
        assert len(restored.steps) == 0
        for name in ("weight_divergence", "max_divergence", "num_alive"):
            assert len(restored.history[name]) == 0

    def test_round_trip_preserves_history(self):
        config = _make_config()
        tracker = SpecializationTracker(config)
        params = _make_agent_params()
        alive = np.ones(8, dtype=bool)

        for i in range(10):
            tracker.update(params, alive, step=i * 1000)

        d = tracker.to_dict()
        restored = SpecializationTracker.from_dict(d, config)

        assert restored.step_count == tracker.step_count
        assert restored.steps == tracker.steps
        for name in ("weight_divergence", "max_divergence", "num_alive"):
            assert restored.history[name] == pytest.approx(tracker.history[name])

    def test_round_trip_preserves_events(self):
        config = _make_config()
        tracker = SpecializationTracker(config, window_size=5, z_threshold=2.0)

        # Feed identical params to build baseline
        params_same = {
            "Dense_0": {"kernel": np.zeros((8, 16), dtype=np.float32)}
        }
        alive = np.ones(8, dtype=bool)
        for i in range(10):
            tracker.update(params_same, alive, step=i * 1000)

        # Feed very different params to trigger event
        rng = np.random.default_rng(99)
        params_diff = {
            "Dense_0": {"kernel": rng.normal(size=(8, 16)).astype(np.float32) * 10.0}
        }
        tracker.update(params_diff, alive, step=10000)

        d = tracker.to_dict()
        restored = SpecializationTracker.from_dict(d, config)

        assert len(restored.events) == len(tracker.events)
        for orig, rest in zip(tracker.events, restored.events):
            assert rest.step == orig.step
            assert rest.metric_name == orig.metric_name
            assert rest.old_value == pytest.approx(orig.old_value)
            assert rest.new_value == pytest.approx(orig.new_value)
            assert rest.z_score == pytest.approx(orig.z_score)

    def test_round_trip_get_metrics_match(self):
        config = _make_config()
        tracker = SpecializationTracker(config)
        params = _make_agent_params()
        alive = np.ones(8, dtype=bool)

        for i in range(5):
            tracker.update(params, alive, step=i * 1000)

        d = tracker.to_dict()
        restored = SpecializationTracker.from_dict(d, config)

        orig_metrics = tracker.get_metrics()
        rest_metrics = restored.get_metrics()
        for key in orig_metrics:
            assert rest_metrics[key] == pytest.approx(orig_metrics[key])

    def test_round_trip_get_summary_match(self):
        config = _make_config()
        tracker = SpecializationTracker(config)
        params = _make_agent_params()
        alive = np.ones(8, dtype=bool)

        for i in range(5):
            tracker.update(params, alive, step=i * 1000)

        d = tracker.to_dict()
        restored = SpecializationTracker.from_dict(d, config)

        orig_summary = tracker.get_summary()
        rest_summary = restored.get_summary()
        for key in orig_summary:
            if isinstance(orig_summary[key], float):
                assert rest_summary[key] == pytest.approx(orig_summary[key])
            else:
                assert rest_summary[key] == orig_summary[key]

    def test_to_dict_is_json_serializable(self):
        config = _make_config()
        tracker = SpecializationTracker(config)
        params = _make_agent_params()
        alive = np.ones(8, dtype=bool)

        for i in range(3):
            tracker.update(params, alive, step=i * 1000)

        d = tracker.to_dict()
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    def test_restored_tracker_can_continue_updating(self):
        config = _make_config()
        tracker = SpecializationTracker(config)
        params = _make_agent_params()
        alive = np.ones(8, dtype=bool)

        for i in range(5):
            tracker.update(params, alive, step=i * 1000)

        d = tracker.to_dict()
        restored = SpecializationTracker.from_dict(d, config)

        # Continue updating
        for i in range(5, 10):
            restored.update(params, alive, step=i * 1000)

        assert restored.step_count == 10
        assert len(restored.steps) == 10
        for name in ("weight_divergence", "max_divergence", "num_alive"):
            assert len(restored.history[name]) == 10

    def test_custom_window_and_threshold(self):
        config = _make_config()
        tracker = SpecializationTracker(config, window_size=50, z_threshold=5.0)
        d = tracker.to_dict()
        restored = SpecializationTracker.from_dict(d, config)
        assert restored.window_size == 50
        assert restored.z_threshold == 5.0

    def test_jax_alive_mask_round_trip(self):
        """Ensure JAX arrays in update don't break serialization."""
        config = _make_config()
        tracker = SpecializationTracker(config)
        params = _make_agent_params()
        alive_jax = jnp.ones(8, dtype=jnp.bool_)

        tracker.update(params, alive_jax, step=0)

        d = tracker.to_dict()
        restored = SpecializationTracker.from_dict(d, config)
        assert restored.step_count == 1
        assert restored.history["num_alive"] == [8.0]


# ===========================================================================
# Cross-tracker tests
# ===========================================================================


class TestCrossTrackerSerialization:
    """Tests for combined tracker serialization (as used in checkpoints)."""

    def test_both_trackers_serialize_together(self):
        """Simulate how checkpointing will store both tracker states."""
        config = _make_config()
        em_tracker = EmergenceTracker(config)
        sp_tracker = SpecializationTracker(config)

        # Update emergence tracker
        for i in range(5):
            field = FieldState(
                values=np.random.default_rng(i).random((20, 20, 4)).astype(np.float32)
            )
            em_tracker.update(field, step=i * 100)

        # Update specialization tracker
        params = _make_agent_params()
        alive = np.ones(8, dtype=bool)
        for i in range(5):
            sp_tracker.update(params, alive, step=i * 1000)

        # Combined state dict
        checkpoint = {
            "emergence_tracker": em_tracker.to_dict(),
            "specialization_tracker": sp_tracker.to_dict(),
        }

        # Verify JSON serializable
        json_str = json.dumps(checkpoint)
        checkpoint_restored = json.loads(json_str)

        # Restore both
        em_restored = EmergenceTracker.from_dict(
            checkpoint_restored["emergence_tracker"], config
        )
        sp_restored = SpecializationTracker.from_dict(
            checkpoint_restored["specialization_tracker"], config
        )

        assert em_restored.step_count == em_tracker.step_count
        assert sp_restored.step_count == sp_tracker.step_count

    def test_pickle_round_trip(self):
        """Verify trackers survive pickle serialization (checkpoint format)."""
        import pickle

        config = _make_config()
        em_tracker = EmergenceTracker(config)
        sp_tracker = SpecializationTracker(config)

        for i in range(3):
            field = FieldState(
                values=np.random.default_rng(i).random((20, 20, 4)).astype(np.float32)
            )
            em_tracker.update(field, step=i * 100)

        params = _make_agent_params()
        alive = np.ones(8, dtype=bool)
        for i in range(3):
            sp_tracker.update(params, alive, step=i * 1000)

        state = {
            "emergence_tracker": em_tracker.to_dict(),
            "specialization_tracker": sp_tracker.to_dict(),
        }

        pickled = pickle.dumps(state)
        unpickled = pickle.loads(pickled)

        em_restored = EmergenceTracker.from_dict(
            unpickled["emergence_tracker"], config
        )
        sp_restored = SpecializationTracker.from_dict(
            unpickled["specialization_tracker"], config
        )

        assert em_restored.step_count == 3
        assert sp_restored.step_count == 3
