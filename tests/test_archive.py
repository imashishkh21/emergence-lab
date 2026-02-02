"""Tests for MAP-Elites Behavioral Archive (US-004)."""

import numpy as np
import pytest

from src.analysis.archive import ArchiveEntry, BehavioralArchive, extract_descriptors
from src.configs import ArchiveConfig, Config


class TestBehavioralArchiveInit:
    """Tests for BehavioralArchive initialization and properties."""

    def test_empty_archive_size(self):
        """New archive should have zero occupied cells."""
        archive = BehavioralArchive(grid_size=10)
        assert archive.size == 0

    def test_capacity(self):
        """Capacity should be grid_size ** 2."""
        archive = BehavioralArchive(grid_size=50)
        assert archive.capacity == 2500

    def test_default_grid_size(self):
        """Default grid size should be 100."""
        archive = BehavioralArchive()
        assert archive.grid_size == 100
        assert archive.capacity == 10000

    def test_invalid_grid_size(self):
        """grid_size < 1 should raise ValueError."""
        with pytest.raises(ValueError):
            BehavioralArchive(grid_size=0)
        with pytest.raises(ValueError):
            BehavioralArchive(grid_size=-1)

    def test_repr(self):
        """repr should contain key info."""
        archive = BehavioralArchive(grid_size=10)
        r = repr(archive)
        assert "grid_size=10" in r
        assert "occupied=0" in r

    def test_coverage_empty(self):
        """Coverage of empty archive should be 0."""
        archive = BehavioralArchive(grid_size=10)
        assert archive.coverage() == 0.0


class TestBehavioralArchiveAdd:
    """Tests for the add() method."""

    def test_add_to_empty_cell(self):
        """Adding to an empty cell should always succeed."""
        archive = BehavioralArchive(grid_size=10)
        params = {"w": np.array([1.0, 2.0])}
        result = archive.add(params, fitness=5.0, descriptor=np.array([0.5, 0.5]))
        assert result is True
        assert archive.size == 1

    def test_add_higher_fitness_replaces(self):
        """Higher fitness agent should replace lower fitness in same cell."""
        archive = BehavioralArchive(grid_size=10)
        d = np.array([0.5, 0.5])
        archive.add({"w": np.array([1.0])}, fitness=3.0, descriptor=d)
        result = archive.add({"w": np.array([2.0])}, fitness=7.0, descriptor=d)
        assert result is True
        entry = archive.get_cell(d)
        assert entry is not None
        assert entry.fitness == 7.0
        assert archive.size == 1  # still one cell

    def test_add_lower_fitness_rejected(self):
        """Lower fitness agent should not replace higher fitness occupant."""
        archive = BehavioralArchive(grid_size=10)
        d = np.array([0.5, 0.5])
        archive.add({"w": np.array([1.0])}, fitness=10.0, descriptor=d)
        result = archive.add({"w": np.array([2.0])}, fitness=5.0, descriptor=d)
        assert result is False
        entry = archive.get_cell(d)
        assert entry is not None
        assert entry.fitness == 10.0

    def test_add_equal_fitness_rejected(self):
        """Equal fitness should not replace (must be strictly higher)."""
        archive = BehavioralArchive(grid_size=10)
        d = np.array([0.3, 0.7])
        archive.add({"w": np.array([1.0])}, fitness=5.0, descriptor=d)
        result = archive.add({"w": np.array([2.0])}, fitness=5.0, descriptor=d)
        assert result is False

    def test_add_different_cells(self):
        """Adding to different cells should increase size."""
        archive = BehavioralArchive(grid_size=10)
        archive.add({"w": 1}, fitness=1.0, descriptor=np.array([0.1, 0.1]))
        archive.add({"w": 2}, fitness=2.0, descriptor=np.array([0.9, 0.9]))
        assert archive.size == 2

    def test_add_descriptor_clipping(self):
        """Descriptors outside [0,1] should be clipped."""
        archive = BehavioralArchive(grid_size=10)
        # This should not raise — values get clipped
        result = archive.add({"w": 1}, fitness=1.0, descriptor=np.array([-0.5, 1.5]))
        assert result is True
        assert archive.size == 1

    def test_add_boundary_descriptors(self):
        """Descriptors at 0.0 and 1.0 should map to valid cells."""
        archive = BehavioralArchive(grid_size=10)
        archive.add({"w": 1}, fitness=1.0, descriptor=np.array([0.0, 0.0]))
        archive.add({"w": 2}, fitness=1.0, descriptor=np.array([1.0, 1.0]))
        # 0.0 maps to cell (0,0), 1.0 maps to cell (9,9)
        assert archive.size == 2

    def test_add_invalid_descriptor_shape(self):
        """Descriptor with wrong number of elements should raise."""
        archive = BehavioralArchive(grid_size=10)
        with pytest.raises(ValueError):
            archive.add({"w": 1}, fitness=1.0, descriptor=np.array([0.5]))
        with pytest.raises(ValueError):
            archive.add({"w": 1}, fitness=1.0, descriptor=np.array([0.1, 0.2, 0.3]))


class TestBehavioralArchiveSample:
    """Tests for the sample() method."""

    def test_sample_empty_archive(self):
        """Sampling from empty archive should return empty list."""
        archive = BehavioralArchive(grid_size=10)
        result = archive.sample(5)
        assert result == []

    def test_sample_zero(self):
        """Sampling 0 entries should return empty list."""
        archive = BehavioralArchive(grid_size=10)
        archive.add({"w": 1}, fitness=1.0, descriptor=np.array([0.5, 0.5]))
        result = archive.sample(0)
        assert result == []

    def test_sample_negative_raises(self):
        """Negative sample count should raise ValueError."""
        archive = BehavioralArchive(grid_size=10)
        with pytest.raises(ValueError):
            archive.sample(-1)

    def test_sample_returns_entries(self):
        """Sampled entries should be valid ArchiveEntry instances."""
        archive = BehavioralArchive(grid_size=10)
        for i in range(5):
            archive.add(
                {"w": i}, fitness=float(i), descriptor=np.array([i / 10, i / 10])
            )
        samples = archive.sample(3)
        assert len(samples) == 3
        for s in samples:
            assert isinstance(s, ArchiveEntry)
            assert hasattr(s, "params")
            assert hasattr(s, "fitness")
            assert hasattr(s, "descriptor")

    def test_sample_without_replacement(self):
        """When n <= size, samples should be unique (no replacement)."""
        archive = BehavioralArchive(grid_size=100, seed=0)
        for i in range(10):
            archive.add(
                {"w": i}, fitness=float(i), descriptor=np.array([i / 100, i / 100])
            )
        samples = archive.sample(10)
        fitnesses = [s.fitness for s in samples]
        assert len(set(fitnesses)) == 10  # all unique

    def test_sample_with_replacement(self):
        """When n > size, samples use replacement."""
        archive = BehavioralArchive(grid_size=10, seed=42)
        archive.add({"w": 1}, fitness=1.0, descriptor=np.array([0.5, 0.5]))
        archive.add({"w": 2}, fitness=2.0, descriptor=np.array([0.1, 0.1]))
        samples = archive.sample(10)
        assert len(samples) == 10
        # All samples must come from the 2 entries
        for s in samples:
            assert s.fitness in [1.0, 2.0]

    def test_sample_single_entry(self):
        """Sampling from archive with one entry."""
        archive = BehavioralArchive(grid_size=10)
        archive.add({"w": 1}, fitness=5.0, descriptor=np.array([0.5, 0.5]))
        samples = archive.sample(1)
        assert len(samples) == 1
        assert samples[0].fitness == 5.0


class TestBehavioralArchiveQuery:
    """Tests for get_cell, as_feature_array, and statistics methods."""

    def test_get_cell_empty(self):
        """get_cell on empty archive should return None."""
        archive = BehavioralArchive(grid_size=10)
        assert archive.get_cell(np.array([0.5, 0.5])) is None

    def test_get_cell_occupied(self):
        """get_cell should return the stored entry."""
        archive = BehavioralArchive(grid_size=10)
        d = np.array([0.35, 0.65])
        archive.add({"w": 42}, fitness=3.14, descriptor=d)
        entry = archive.get_cell(d)
        assert entry is not None
        assert entry.fitness == 3.14

    def test_as_feature_array_empty(self):
        """as_feature_array on empty archive should be (0, 2)."""
        archive = BehavioralArchive(grid_size=10)
        arr = archive.as_feature_array()
        assert arr.shape == (0, 2)

    def test_as_feature_array_shape(self):
        """as_feature_array shape should match archive size."""
        archive = BehavioralArchive(grid_size=10)
        for i in range(5):
            archive.add(
                {"w": i}, fitness=float(i), descriptor=np.array([i / 10, (9 - i) / 10])
            )
        arr = archive.as_feature_array()
        assert arr.shape == (5, 2)

    def test_as_feature_array_values(self):
        """as_feature_array should contain actual descriptor values."""
        archive = BehavioralArchive(grid_size=10)
        d = np.array([0.33, 0.77])
        archive.add({"w": 1}, fitness=1.0, descriptor=d)
        arr = archive.as_feature_array()
        assert arr.shape == (1, 2)
        np.testing.assert_allclose(arr[0], d, atol=1e-10)

    def test_fitness_stats_empty(self):
        """fitness_stats on empty archive should return all zeros."""
        archive = BehavioralArchive(grid_size=10)
        stats = archive.fitness_stats()
        assert stats["mean"] == 0.0
        assert stats["std"] == 0.0
        assert stats["min"] == 0.0
        assert stats["max"] == 0.0

    def test_fitness_stats_values(self):
        """fitness_stats should compute correct summary statistics."""
        archive = BehavioralArchive(grid_size=100)
        fitnesses = [1.0, 2.0, 3.0, 4.0, 5.0]
        for i, f in enumerate(fitnesses):
            archive.add({"w": i}, fitness=f, descriptor=np.array([i / 10, i / 10]))
        stats = archive.fitness_stats()
        assert stats["mean"] == pytest.approx(3.0)
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(5.0)
        assert stats["std"] == pytest.approx(np.std(fitnesses))

    def test_clear(self):
        """clear() should empty the archive."""
        archive = BehavioralArchive(grid_size=10)
        for i in range(5):
            archive.add({"w": i}, fitness=float(i), descriptor=np.array([i / 10, i / 10]))
        assert archive.size == 5
        archive.clear()
        assert archive.size == 0
        assert archive.coverage() == 0.0

    def test_coverage_partial(self):
        """Coverage should reflect fraction of cells occupied."""
        archive = BehavioralArchive(grid_size=10)  # 100 cells
        for i in range(25):
            archive.add(
                {"w": i},
                fitness=float(i),
                descriptor=np.array([(i % 10) / 10, (i // 10) / 10]),
            )
        assert archive.coverage() == pytest.approx(25 / 100)


class TestDescriptorMapping:
    """Tests for the internal descriptor-to-cell mapping."""

    def test_zero_maps_to_origin(self):
        """Descriptor (0, 0) should map to cell (0, 0)."""
        archive = BehavioralArchive(grid_size=10)
        cell = archive._descriptor_to_cell(np.array([0.0, 0.0]))
        assert cell == (0, 0)

    def test_one_maps_to_max(self):
        """Descriptor (1.0, 1.0) should map to cell (grid_size-1, grid_size-1)."""
        archive = BehavioralArchive(grid_size=10)
        cell = archive._descriptor_to_cell(np.array([1.0, 1.0]))
        assert cell == (9, 9)

    def test_midpoint(self):
        """Descriptor (0.5, 0.5) should map to cell (5, 5)."""
        archive = BehavioralArchive(grid_size=10)
        cell = archive._descriptor_to_cell(np.array([0.5, 0.5]))
        assert cell == (5, 5)

    def test_clipping_negative(self):
        """Negative descriptors should be clipped to 0."""
        archive = BehavioralArchive(grid_size=10)
        cell = archive._descriptor_to_cell(np.array([-1.0, -0.5]))
        assert cell == (0, 0)

    def test_clipping_above_one(self):
        """Descriptors > 1 should be clipped to 1."""
        archive = BehavioralArchive(grid_size=10)
        cell = archive._descriptor_to_cell(np.array([2.0, 1.5]))
        assert cell == (9, 9)

    def test_fine_grained_cells(self):
        """Nearby but different descriptors should map to different cells."""
        archive = BehavioralArchive(grid_size=100)
        cell_a = archive._descriptor_to_cell(np.array([0.01, 0.01]))
        cell_b = archive._descriptor_to_cell(np.array([0.02, 0.02]))
        assert cell_a != cell_b


class TestExtractDescriptors:
    """Tests for extract_descriptors() function."""

    def test_output_shape(self):
        """Should return (num_agents, 2)."""
        num_agents = 5
        num_steps = 20
        trajectory = {
            "actions": np.random.randint(0, 6, (num_steps, num_agents)),
            "alive_mask": np.ones((num_steps, num_agents), dtype=bool),
        }
        result = extract_descriptors(trajectory)
        assert result.shape == (num_agents, 2)

    def test_values_in_range(self):
        """All descriptor values should be in [0, 1]."""
        num_agents = 8
        num_steps = 50
        trajectory = {
            "actions": np.random.randint(0, 6, (num_steps, num_agents)),
            "alive_mask": np.ones((num_steps, num_agents), dtype=bool),
        }
        result = extract_descriptors(trajectory)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_dead_agent_zeros(self):
        """Agents that are never alive should get [0, 0] descriptors."""
        num_agents = 4
        num_steps = 10
        alive = np.ones((num_steps, num_agents), dtype=bool)
        alive[:, 2] = False  # Agent 2 always dead
        trajectory = {
            "actions": np.random.randint(0, 6, (num_steps, num_agents)),
            "alive_mask": alive,
        }
        result = extract_descriptors(trajectory)
        np.testing.assert_array_equal(result[2], [0.0, 0.0])

    def test_all_alive_field_write_one(self):
        """Agent alive every step should have field_write_frequency = 1.0."""
        num_agents = 3
        num_steps = 20
        trajectory = {
            "actions": np.random.randint(0, 6, (num_steps, num_agents)),
            "alive_mask": np.ones((num_steps, num_agents), dtype=bool),
        }
        result = extract_descriptors(trajectory)
        np.testing.assert_allclose(result[:, 1], 1.0, atol=1e-10)

    def test_half_alive_field_write(self):
        """Agent alive for half the steps should have field_write_frequency ~ 0.5."""
        num_agents = 2
        num_steps = 20
        alive = np.zeros((num_steps, num_agents), dtype=bool)
        alive[:10, 0] = True  # Agent 0 alive for half
        alive[:, 1] = True  # Agent 1 alive always
        trajectory = {
            "actions": np.zeros((num_steps, num_agents), dtype=int),
            "alive_mask": alive,
        }
        result = extract_descriptors(trajectory)
        assert result[0, 1] == pytest.approx(0.5)
        assert result[1, 1] == pytest.approx(1.0)

    def test_deterministic_action_low_entropy(self):
        """Agent that always takes action 0 should have low movement entropy."""
        num_agents = 2
        num_steps = 100
        actions = np.zeros((num_steps, num_agents), dtype=int)
        actions[:, 1] = np.random.randint(0, 6, num_steps)  # Agent 1 random
        trajectory = {
            "actions": actions,
            "alive_mask": np.ones((num_steps, num_agents), dtype=bool),
        }
        result = extract_descriptors(trajectory)
        # Agent 0: always same action → entropy ~ 0
        assert result[0, 0] < 0.1
        # Agent 1: random → entropy close to 1
        assert result[1, 0] > 0.7

    def test_uniform_action_high_entropy(self):
        """Agent with uniform action distribution should have high entropy."""
        num_agents = 1
        num_steps = 6000  # lots of steps for uniform dist
        # Perfectly uniform: 1000 of each action
        actions = np.tile(np.arange(6), num_steps // 6).reshape(-1, 1)
        trajectory = {
            "actions": actions,
            "alive_mask": np.ones((num_steps, num_agents), dtype=bool),
        }
        result = extract_descriptors(trajectory)
        assert result[0, 0] == pytest.approx(1.0, abs=0.01)


class TestArchiveWithDescriptors:
    """Integration tests: extract descriptors → add to archive."""

    def test_full_pipeline(self):
        """Extract descriptors from trajectory and add to archive."""
        num_agents = 8
        num_steps = 50
        trajectory = {
            "actions": np.random.randint(0, 6, (num_steps, num_agents)),
            "alive_mask": np.ones((num_steps, num_agents), dtype=bool),
        }

        descriptors = extract_descriptors(trajectory)
        archive = BehavioralArchive(grid_size=50)

        # Add all agents with their food-collected count as fitness
        fitnesses = np.random.rand(num_agents) * 10
        for i in range(num_agents):
            archive.add(
                params={"agent_idx": i},
                fitness=float(fitnesses[i]),
                descriptor=descriptors[i],
            )

        assert archive.size > 0
        assert archive.size <= num_agents

        # Feature array should have descriptors of occupied cells
        features = archive.as_feature_array()
        assert features.shape[0] == archive.size
        assert features.shape[1] == 2

        # Sample should work
        samples = archive.sample(3)
        assert len(samples) == 3

    def test_archive_preserves_best_fitness(self):
        """When multiple agents map to same cell, best fitness wins."""
        archive = BehavioralArchive(grid_size=10)
        d = np.array([0.55, 0.55])

        # Add several agents with same descriptor but different fitnesses
        archive.add({"id": 1}, fitness=3.0, descriptor=d)
        archive.add({"id": 2}, fitness=7.0, descriptor=d)
        archive.add({"id": 3}, fitness=5.0, descriptor=d)

        entry = archive.get_cell(d)
        assert entry is not None
        assert entry.fitness == 7.0
        assert entry.params["id"] == 2  # agent 2 had highest fitness

    def test_novelty_score_compatibility(self):
        """Archive feature array should work with novelty_score()."""
        from src.analysis.specialization import novelty_score

        archive = BehavioralArchive(grid_size=50)
        for i in range(10):
            d = np.array([i / 10, (9 - i) / 10])
            archive.add({"w": i}, fitness=float(i), descriptor=d)

        archive_features = archive.as_feature_array()
        agent_features = np.array([[0.5, 0.5], [0.1, 0.9]])

        scores = novelty_score(agent_features, archive_features, k=3)
        assert scores.shape == (2,)
        assert np.all(scores >= 0)


class TestArchiveConfig:
    """Tests for ArchiveConfig in configs.py."""

    def test_default_values(self):
        """Default config should match PRD spec."""
        cfg = ArchiveConfig()
        assert cfg.grid_size == 100
        assert cfg.enabled is False

    def test_config_integration(self):
        """ArchiveConfig should be accessible from master Config."""
        config = Config()
        assert hasattr(config, "archive")
        assert config.archive.grid_size == 100
        assert config.archive.enabled is False

    def test_config_custom_values(self):
        """Custom values should be accepted."""
        cfg = ArchiveConfig(grid_size=50, enabled=True)
        assert cfg.grid_size == 50
        assert cfg.enabled is True
