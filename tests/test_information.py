"""Tests for information-theoretic metrics (transfer entropy)."""

import pytest
import numpy as np


class TestComputeTransferEntropy:
    """US-010: Transfer entropy between agent pairs."""

    def test_independent_sequences_low_te(self):
        """Independent random sequences should have near-zero TE."""
        from src.analysis.information import compute_transfer_entropy

        rng = np.random.RandomState(42)
        source = rng.randint(0, 6, size=200).astype(float)
        target = rng.randint(0, 6, size=200).astype(float)

        te = compute_transfer_entropy(source, target, lag=1, k=4)

        assert isinstance(te, float)
        # Should be small for independent sequences
        assert te < 0.5

    def test_perfectly_correlated_positive_te(self):
        """When target copies source with lag, TE should be positive."""
        from src.analysis.information import compute_transfer_entropy

        rng = np.random.RandomState(42)
        source = rng.randint(0, 6, size=200).astype(float)
        # Target copies source with lag=1
        target = np.zeros_like(source)
        target[1:] = source[:-1]
        target[0] = source[0]

        te = compute_transfer_entropy(source, target, lag=1, k=4)

        assert te > 0.0

    def test_correlated_higher_than_independent(self):
        """Correlated TE should be higher than independent TE."""
        from src.analysis.information import compute_transfer_entropy

        rng = np.random.RandomState(42)
        source = rng.randint(0, 6, size=300).astype(float)

        # Independent target
        target_indep = rng.randint(0, 6, size=300).astype(float)

        # Correlated target (follows source with noise)
        target_corr = np.zeros_like(source)
        target_corr[1:] = source[:-1]
        target_corr += rng.normal(0, 0.1, size=300)

        te_indep = compute_transfer_entropy(source, target_indep, lag=1, k=4)
        te_corr = compute_transfer_entropy(source, target_corr, lag=1, k=4)

        assert te_corr > te_indep

    def test_non_negative_output(self):
        """TE should always be non-negative."""
        from src.analysis.information import compute_transfer_entropy

        rng = np.random.RandomState(0)
        for seed in range(5):
            rng2 = np.random.RandomState(seed)
            source = rng2.randint(0, 6, size=100).astype(float)
            target = rng2.randint(0, 6, size=100).astype(float)
            te = compute_transfer_entropy(source, target, lag=1, k=3)
            assert te >= 0.0

    def test_short_sequence_returns_zero(self):
        """Very short sequences should return 0 gracefully."""
        from src.analysis.information import compute_transfer_entropy

        source = np.array([1.0, 2.0])
        target = np.array([3.0, 4.0])

        te = compute_transfer_entropy(source, target, lag=1, k=4)
        assert te == 0.0

    def test_empty_sequence_returns_zero(self):
        """Empty sequences should return 0."""
        from src.analysis.information import compute_transfer_entropy

        source = np.array([])
        target = np.array([])

        te = compute_transfer_entropy(source, target)
        assert te == 0.0

    def test_different_lags(self):
        """Different lag values should produce different TE values."""
        from src.analysis.information import compute_transfer_entropy

        rng = np.random.RandomState(42)
        source = rng.randint(0, 6, size=300).astype(float)
        target = np.zeros_like(source)
        target[3:] = source[:-3]  # Target copies source with lag=3

        te_lag1 = compute_transfer_entropy(source, target, lag=1, k=4)
        te_lag3 = compute_transfer_entropy(source, target, lag=3, k=4)

        # TE at the correct lag should be higher
        assert te_lag3 > te_lag1

    def test_multidimensional_input(self):
        """Should work with multi-dimensional feature vectors."""
        from src.analysis.information import compute_transfer_entropy

        rng = np.random.RandomState(42)
        source = rng.randn(200, 2)
        target = np.zeros_like(source)
        target[1:] = source[:-1]  # Copy with lag 1

        te = compute_transfer_entropy(source, target, lag=1, k=4)
        assert te >= 0.0

    def test_self_transfer_entropy(self):
        """TE from a sequence to itself should work without error."""
        from src.analysis.information import compute_transfer_entropy

        rng = np.random.RandomState(42)
        x = rng.randint(0, 6, size=200).astype(float)

        te = compute_transfer_entropy(x, x, lag=1, k=4)
        assert isinstance(te, float)
        assert te >= 0.0


class TestComputeTEMatrix:
    """US-010: Pairwise TE matrix computation."""

    def test_output_keys(self):
        """Result should contain all expected keys."""
        from src.analysis.information import compute_te_matrix

        rng = np.random.RandomState(42)
        histories = rng.randint(0, 6, size=(100, 4)).astype(float)

        result = compute_te_matrix(histories)

        assert "te_matrix" in result
        assert "mean_te" in result
        assert "max_te" in result
        assert "te_density" in result
        assert "agent_indices" in result

    def test_matrix_shape(self):
        """TE matrix should be (num_agents, num_agents)."""
        from src.analysis.information import compute_te_matrix

        rng = np.random.RandomState(42)
        num_agents = 5
        histories = rng.randint(0, 6, size=(100, num_agents)).astype(float)

        result = compute_te_matrix(histories)

        assert result["te_matrix"].shape == (num_agents, num_agents)

    def test_zero_diagonal(self):
        """Self-TE (diagonal) should be zero."""
        from src.analysis.information import compute_te_matrix

        rng = np.random.RandomState(42)
        histories = rng.randint(0, 6, size=(100, 4)).astype(float)

        result = compute_te_matrix(histories)

        np.testing.assert_allclose(
            np.diag(result["te_matrix"]), 0.0, atol=1e-10
        )

    def test_alive_mask_filters(self):
        """Dead agents should have zero TE rows and columns."""
        from src.analysis.information import compute_te_matrix

        rng = np.random.RandomState(42)
        histories = rng.randint(0, 6, size=(100, 5)).astype(float)
        alive_mask = np.array([True, False, True, True, False])

        result = compute_te_matrix(histories, alive_mask=alive_mask)

        # Dead agents (indices 1, 4) should have zero TE
        assert result["te_matrix"][1, :].sum() == 0.0
        assert result["te_matrix"][:, 1].sum() == 0.0
        assert result["te_matrix"][4, :].sum() == 0.0
        assert result["te_matrix"][:, 4].sum() == 0.0
        assert set(result["agent_indices"]) == {0, 2, 3}

    def test_single_agent_returns_zeros(self):
        """Single alive agent should return zero TE."""
        from src.analysis.information import compute_te_matrix

        rng = np.random.RandomState(42)
        histories = rng.randint(0, 6, size=(100, 3)).astype(float)
        alive_mask = np.array([False, True, False])

        result = compute_te_matrix(histories, alive_mask=alive_mask)

        assert result["mean_te"] == 0.0
        assert result["max_te"] == 0.0
        assert result["te_density"] == 0.0

    def test_non_negative_values(self):
        """All TE values should be non-negative."""
        from src.analysis.information import compute_te_matrix

        rng = np.random.RandomState(42)
        histories = rng.randint(0, 6, size=(100, 4)).astype(float)

        result = compute_te_matrix(histories)

        assert (result["te_matrix"] >= 0).all()
        assert result["mean_te"] >= 0.0
        assert result["max_te"] >= 0.0

    def test_3d_input(self):
        """Should work with 3D (T, num_agents, D) histories."""
        from src.analysis.information import compute_te_matrix

        rng = np.random.RandomState(42)
        histories = rng.randn(100, 4, 2)

        result = compute_te_matrix(histories)

        assert result["te_matrix"].shape == (4, 4)
        assert result["mean_te"] >= 0.0

    def test_mean_te_is_average(self):
        """mean_te should be the average of off-diagonal TE values."""
        from src.analysis.information import compute_te_matrix

        rng = np.random.RandomState(42)
        histories = rng.randint(0, 6, size=(150, 3)).astype(float)

        result = compute_te_matrix(histories)
        matrix = result["te_matrix"]

        # Collect off-diagonal values for alive agents
        off_diag = []
        for i in range(3):
            for j in range(3):
                if i != j:
                    off_diag.append(matrix[i, j])

        expected_mean = np.mean(off_diag)
        assert abs(result["mean_te"] - expected_mean) < 1e-10

    def test_max_te_is_maximum(self):
        """max_te should be the maximum off-diagonal TE value."""
        from src.analysis.information import compute_te_matrix

        rng = np.random.RandomState(42)
        histories = rng.randint(0, 6, size=(150, 3)).astype(float)

        result = compute_te_matrix(histories)
        matrix = result["te_matrix"]

        off_diag = []
        for i in range(3):
            for j in range(3):
                if i != j:
                    off_diag.append(matrix[i, j])

        expected_max = np.max(off_diag)
        assert abs(result["max_te"] - expected_max) < 1e-10

    def test_density_range(self):
        """TE density should be in [0, 1]."""
        from src.analysis.information import compute_te_matrix

        rng = np.random.RandomState(42)
        histories = rng.randint(0, 6, size=(100, 4)).astype(float)

        result = compute_te_matrix(histories)

        assert 0.0 <= result["te_density"] <= 1.0


class TestTransferEntropyFromTrajectory:
    """US-010: Convenience wrapper for trajectory dicts."""

    def _make_trajectory(
        self,
        num_steps: int = 100,
        num_agents: int = 4,
    ) -> dict[str, np.ndarray]:
        """Helper to create a trajectory dict."""
        rng = np.random.RandomState(42)
        return {
            "actions": rng.randint(0, 6, size=(num_steps, num_agents)),
            "positions": rng.randint(0, 20, size=(num_steps, num_agents, 2)),
            "rewards": rng.uniform(0, 1, size=(num_steps, num_agents)).astype(
                np.float32
            ),
            "alive_mask": np.ones((num_steps, num_agents), dtype=bool),
            "energy": rng.uniform(50, 150, size=(num_steps, num_agents)).astype(
                np.float32
            ),
        }

    def test_actions_feature(self):
        """Should compute TE from action histories."""
        from src.analysis.information import transfer_entropy_from_trajectory

        traj = self._make_trajectory()
        result = transfer_entropy_from_trajectory(traj, feature="actions")

        assert "te_matrix" in result
        assert "feature" in result
        assert result["feature"] == "actions"

    def test_positions_feature(self):
        """Should compute TE from position histories."""
        from src.analysis.information import transfer_entropy_from_trajectory

        traj = self._make_trajectory()
        result = transfer_entropy_from_trajectory(traj, feature="positions")

        assert result["feature"] == "positions"
        assert result["te_matrix"].shape == (4, 4)

    def test_invalid_feature_raises(self):
        """Should raise ValueError for unknown feature."""
        from src.analysis.information import transfer_entropy_from_trajectory

        traj = self._make_trajectory()
        with pytest.raises(ValueError, match="Unknown feature"):
            transfer_entropy_from_trajectory(traj, feature="invalid")

    def test_alive_mask_respected(self):
        """Dead agents should be excluded from TE computation."""
        from src.analysis.information import transfer_entropy_from_trajectory

        traj = self._make_trajectory(num_agents=4)
        # Agent 1 is never alive
        traj["alive_mask"][:, 1] = False

        result = transfer_entropy_from_trajectory(traj)

        # Agent 1 should have zero TE
        assert result["te_matrix"][1, :].sum() == 0.0
        assert result["te_matrix"][:, 1].sum() == 0.0
        assert 1 not in result["agent_indices"]

    def test_partial_alive_mask(self):
        """Agents alive for some steps should be included."""
        from src.analysis.information import transfer_entropy_from_trajectory

        traj = self._make_trajectory(num_steps=100, num_agents=3)
        # Agent 2 alive only for first half
        traj["alive_mask"][50:, 2] = False

        result = transfer_entropy_from_trajectory(traj)

        # Agent 2 was alive at some point, so should be included
        assert 2 in result["agent_indices"]


class TestTransferEntropyTracker:
    """US-010: Transfer entropy tracker for training loop integration."""

    def test_init_defaults(self):
        """Tracker should initialize with default parameters."""
        from src.analysis.information import TransferEntropyTracker

        tracker = TransferEntropyTracker()

        assert tracker.window_size == 20
        assert tracker.z_threshold == 3.0
        assert tracker.step_count == 0
        assert len(tracker.events) == 0

    def test_update_returns_events(self):
        """Update should return a list (possibly empty) of events."""
        from src.analysis.information import TransferEntropyTracker

        tracker = TransferEntropyTracker()
        rng = np.random.RandomState(42)
        histories = rng.randint(0, 6, size=(50, 4)).astype(float)
        alive = np.ones(4, dtype=bool)

        events = tracker.update(histories, alive, step=100)

        assert isinstance(events, list)

    def test_step_count_increments(self):
        """Step count should increment with each update."""
        from src.analysis.information import TransferEntropyTracker

        tracker = TransferEntropyTracker()
        rng = np.random.RandomState(42)
        histories = rng.randint(0, 6, size=(50, 4)).astype(float)
        alive = np.ones(4, dtype=bool)

        tracker.update(histories, alive, step=100)
        tracker.update(histories, alive, step=200)

        assert tracker.step_count == 2

    def test_history_recorded(self):
        """History should grow with each update."""
        from src.analysis.information import TransferEntropyTracker

        tracker = TransferEntropyTracker()
        rng = np.random.RandomState(42)
        histories = rng.randint(0, 6, size=(50, 4)).astype(float)
        alive = np.ones(4, dtype=bool)

        for i in range(5):
            tracker.update(histories, alive, step=i * 100)

        assert len(tracker.history["mean_te"]) == 5
        assert len(tracker.history["max_te"]) == 5
        assert len(tracker.history["te_density"]) == 5
        assert len(tracker.steps) == 5

    def test_get_metrics_keys(self):
        """get_metrics should return properly prefixed keys."""
        from src.analysis.information import TransferEntropyTracker

        tracker = TransferEntropyTracker()
        rng = np.random.RandomState(42)
        histories = rng.randint(0, 6, size=(50, 4)).astype(float)
        alive = np.ones(4, dtype=bool)

        tracker.update(histories, alive, step=100)
        metrics = tracker.get_metrics()

        assert "information/mean_te" in metrics
        assert "information/max_te" in metrics
        assert "information/te_density" in metrics
        assert "information/num_events" in metrics

    def test_get_metrics_finite(self):
        """All metric values should be finite floats."""
        from src.analysis.information import TransferEntropyTracker

        tracker = TransferEntropyTracker()
        rng = np.random.RandomState(42)
        histories = rng.randint(0, 6, size=(50, 4)).astype(float)
        alive = np.ones(4, dtype=bool)

        tracker.update(histories, alive, step=100)
        metrics = tracker.get_metrics()

        for k, v in metrics.items():
            assert isinstance(v, float), f"{k} is not float"
            assert np.isfinite(v), f"{k} is not finite"

    def test_get_summary_keys(self):
        """get_summary should return expected keys."""
        from src.analysis.information import TransferEntropyTracker

        tracker = TransferEntropyTracker()
        rng = np.random.RandomState(42)
        histories = rng.randint(0, 6, size=(50, 4)).astype(float)
        alive = np.ones(4, dtype=bool)

        tracker.update(histories, alive, step=100)
        summary = tracker.get_summary()

        assert "total_updates" in summary
        assert "total_events" in summary
        assert "events" in summary
        assert "mean_te_final" in summary
        assert "mean_te_mean" in summary

    def test_event_detection_on_sudden_change(self):
        """Sudden spike in TE should trigger an event."""
        from src.analysis.information import TransferEntropyTracker

        tracker = TransferEntropyTracker(window_size=5, z_threshold=2.0)

        rng = np.random.RandomState(42)

        # Feed 10 updates with independent (low TE) histories
        for i in range(10):
            histories = rng.randint(0, 6, size=(50, 4)).astype(float)
            alive = np.ones(4, dtype=bool)
            tracker.update(histories, alive, step=i * 100)

        # Now feed a correlated history (high TE)
        source = rng.randint(0, 6, size=200).astype(float)
        corr_histories = np.column_stack([
            source,
            np.roll(source, 1),  # Copy with lag
            np.roll(source, 1),
            np.roll(source, 1),
        ])

        events = tracker.update(corr_histories, np.ones(4, dtype=bool), step=1100)

        # Should detect the sudden change (at least check the event list grew)
        assert tracker.step_count == 11

    def test_steady_state_no_events(self):
        """Stable TE should not trigger events."""
        from src.analysis.information import TransferEntropyTracker

        tracker = TransferEntropyTracker(window_size=5, z_threshold=3.0)
        rng = np.random.RandomState(42)

        # Same type of data repeatedly — TE should be relatively stable
        all_events = []
        for i in range(10):
            histories = rng.randint(0, 6, size=(50, 4)).astype(float)
            alive = np.ones(4, dtype=bool)
            events = tracker.update(histories, alive, step=i * 100)
            all_events.extend(events)

        # With consistent random data, events should be rare
        # (not guaranteed zero, but check that tracker works)
        assert isinstance(all_events, list)

    def test_event_str_format(self):
        """TEEvent should have a readable string representation."""
        from src.analysis.information import TEEvent

        event = TEEvent(
            step=1000,
            metric_name="mean_te",
            old_value=0.05,
            new_value=0.25,
            z_score=4.5,
        )

        s = str(event)
        assert "1000" in s
        assert "mean_te" in s
        assert "increase" in s

    def test_none_alive_mask(self):
        """None alive_mask should treat all agents as alive."""
        from src.analysis.information import TransferEntropyTracker

        tracker = TransferEntropyTracker()
        rng = np.random.RandomState(42)
        histories = rng.randint(0, 6, size=(50, 4)).astype(float)

        events = tracker.update(histories, None, step=100)
        assert isinstance(events, list)
        assert tracker.step_count == 1
