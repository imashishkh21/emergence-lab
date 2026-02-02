"""Smoke tests for Phase 5 optional dependencies.

Each test is @pytest.mark.skipif guarded so CI doesn't break
if the libraries aren't installed.
"""

import importlib

import numpy as np
import pytest

# ── Import availability flags ──────────────────────────────────────────

_dit_available = importlib.util.find_spec("dit") is not None
_hoi_available = importlib.util.find_spec("hoi") is not None
_rliable_available = importlib.util.find_spec("rliable") is not None
_jaxmarl_available = importlib.util.find_spec("jaxmarl") is not None


# ── dit ────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _dit_available, reason="dit not installed")
class TestDit:
    """Smoke tests for the dit library (discrete information theory)."""

    @staticmethod
    def _patch_numpy() -> None:
        """Patch np.alltrue removed in NumPy 2.0 (dit compat)."""
        if not hasattr(np, "alltrue"):
            np.alltrue = np.all  # type: ignore[attr-defined]

    def test_import(self) -> None:
        self._patch_numpy()
        import dit  # noqa: F401

    def test_distribution_creation(self) -> None:
        self._patch_numpy()
        import dit

        d = dit.Distribution(["00", "01", "10", "11"], [0.25] * 4)
        assert len(d.outcomes) == 4

    def test_shannon_entropy(self) -> None:
        self._patch_numpy()
        import dit

        d = dit.Distribution(["00", "01", "10", "11"], [0.25] * 4)
        h = dit.shannon.entropy(d)
        assert abs(h - 2.0) < 1e-10, f"Expected entropy=2.0, got {h}"

    def test_mutual_information(self) -> None:
        self._patch_numpy()
        import dit

        # Perfectly correlated: X=Y
        d = dit.Distribution(["00", "11"], [0.5, 0.5])
        mi = dit.shannon.mutual_information(d, [0], [1])
        assert mi > 0.99, f"Expected MI~1.0, got {mi}"

    def test_pid_import(self) -> None:
        """Verify PID modules are accessible (used in US-003)."""
        self._patch_numpy()
        from dit.pid import PID_WB  # noqa: F401


# ── hoi ────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _hoi_available, reason="hoi not installed")
class TestHoi:
    """Smoke tests for the hoi library (higher-order interactions)."""

    def test_import(self) -> None:
        import hoi  # noqa: F401

    def test_oinfo_import(self) -> None:
        from hoi.metrics import Oinfo  # noqa: F401

    def test_oinfo_computation(self) -> None:
        """Compute O-information on synthetic data."""
        from hoi.metrics import Oinfo

        rng = np.random.default_rng(42)
        # Shape: (n_samples, n_variables, n_features)
        x = rng.standard_normal((100, 4, 1)).astype(np.float64)
        model = Oinfo(x)
        result = model.fit(method="gc")
        # Result shape: (n_multiplets, n_features)
        assert result.shape[1] == 1
        assert result.shape[0] > 0


# ── rliable ────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _rliable_available, reason="rliable not installed")
class TestRliable:
    """Smoke tests for the rliable library (statistical reporting)."""

    def test_import(self) -> None:
        from rliable import metrics  # noqa: F401

    def test_aggregate_iqm(self) -> None:
        """Compute IQM (Interquartile Mean) on synthetic scores."""
        from rliable import metrics

        rng = np.random.default_rng(42)
        scores = rng.standard_normal((20, 3))
        iqm = metrics.aggregate_iqm(scores)
        assert np.isfinite(iqm), f"IQM should be finite, got {iqm}"

    def test_aggregate_mean(self) -> None:
        from rliable import metrics

        scores = np.ones((10, 3))
        mean = metrics.aggregate_mean(scores)
        assert abs(mean - 1.0) < 1e-10

    def test_aggregate_median(self) -> None:
        from rliable import metrics

        scores = np.ones((10, 3)) * 5.0
        median = metrics.aggregate_median(scores)
        assert abs(median - 5.0) < 1e-10


# ── jaxmarl ────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _jaxmarl_available, reason="jaxmarl not installed")
class TestJaxmarl:
    """Smoke tests for jaxmarl (reference only — we write custom MAPPO).

    Note: jaxmarl pins jax<=0.4.38 and scipy<=1.12, which conflict with
    our jax>=0.9 and scipy>=1.17. It's excluded from the phase5 group
    in pyproject.toml. These tests will be skipped in normal CI.
    """

    def test_import(self) -> None:
        import jaxmarl  # noqa: F401
