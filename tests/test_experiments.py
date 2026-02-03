"""Tests for the experiment harness and environment configurations.

Tests the experiment runner infrastructure including:
- ExperimentConfig and ExperimentResult dataclasses
- run_experiment() function with various configurations
- Paired seed support for reduced variance
- Save/load functionality for experiment results
- Environment configuration presets (standard, hidden_resources, food_scarcity)
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.configs import Config


class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""

    def test_experiment_config_defaults(self):
        """ExperimentConfig should have sensible defaults."""
        from src.experiments.runner import ExperimentConfig

        config = ExperimentConfig(method_name="test_method")

        assert config.method_name == "test_method"
        assert config.n_seeds == 20  # DR-4 gold standard
        assert config.env_config_name == "standard"
        assert config.paired_seeds is True
        assert config.seed_offset == 0
        assert config.n_episodes == 10
        assert config.save_per_seed_results is True

    def test_experiment_config_custom_values(self):
        """ExperimentConfig should accept custom values."""
        from src.experiments.runner import ExperimentConfig

        config = ExperimentConfig(
            method_name="custom",
            n_seeds=5,
            env_config_name="food_scarcity",
            paired_seeds=False,
            seed_offset=100,
            n_episodes=20,
            save_per_seed_results=False,
        )

        assert config.method_name == "custom"
        assert config.n_seeds == 5
        assert config.env_config_name == "food_scarcity"
        assert config.paired_seeds is False
        assert config.seed_offset == 100
        assert config.n_episodes == 20
        assert config.save_per_seed_results is False


class TestExperimentResult:
    """Tests for ExperimentResult dataclass."""

    def test_experiment_result_defaults(self):
        """ExperimentResult should have sensible defaults."""
        from src.experiments.runner import ExperimentResult

        result = ExperimentResult(
            method_name="test",
            env_config_name="standard",
            n_seeds=5,
        )

        assert result.method_name == "test"
        assert result.env_config_name == "standard"
        assert result.n_seeds == 5
        assert result.seed_list == []
        assert result.per_seed_rewards == []
        assert result.mean_reward == 0.0
        assert result.iqm_reward == 0.0
        assert result.config is None

    def test_experiment_result_str(self):
        """ExperimentResult should have a readable __str__."""
        from src.experiments.runner import ExperimentResult

        result = ExperimentResult(
            method_name="test",
            env_config_name="standard",
            n_seeds=5,
            mean_reward=100.5,
            std_reward=10.2,
            iqm_reward=98.3,
            ci_lower=90.1,
            ci_upper=110.9,
            mean_food=25.5,
            std_food=5.2,
            mean_population=8.0,
        )

        result_str = str(result)

        assert "test" in result_str
        assert "standard" in result_str
        assert "100.50" in result_str
        assert "98.30" in result_str
        assert "25.50" in result_str


class TestComputeIQM:
    """Tests for the IQM computation function."""

    def test_iqm_basic(self):
        """compute_iqm should correctly compute interquartile mean."""
        from src.experiments.runner import compute_iqm

        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        iqm = compute_iqm(values)

        # 25th percentile = 2.75, 75th percentile = 6.25
        # Values in range: [3.0, 4.0, 5.0, 6.0]
        # Mean = 4.5
        assert 3.5 <= iqm <= 5.5  # Allow some tolerance

    def test_iqm_empty(self):
        """compute_iqm should handle empty array."""
        from src.experiments.runner import compute_iqm

        values = np.array([])
        iqm = compute_iqm(values)

        assert iqm == 0.0

    def test_iqm_single_value(self):
        """compute_iqm should handle single value."""
        from src.experiments.runner import compute_iqm

        values = np.array([5.0])
        iqm = compute_iqm(values)

        assert iqm == 5.0

    def test_iqm_outlier_robustness(self):
        """IQM should be robust to outliers."""
        from src.experiments.runner import compute_iqm

        # Normal values + extreme outliers
        values = np.array([10.0, 11.0, 12.0, 13.0, 1000.0, 0.0])
        iqm = compute_iqm(values)

        # IQM should be near the normal values, not influenced by outliers
        assert 5.0 <= iqm <= 50.0


class TestBootstrapCI:
    """Tests for bootstrap confidence interval computation."""

    def test_bootstrap_ci_basic(self):
        """bootstrap_ci_simple should return valid confidence interval."""
        from src.experiments.runner import bootstrap_ci_simple

        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ci_lower, ci_upper = bootstrap_ci_simple(values, n_bootstrap=1000, seed=42)

        assert ci_lower <= np.mean(values) <= ci_upper
        assert ci_lower < ci_upper

    def test_bootstrap_ci_empty(self):
        """bootstrap_ci_simple should handle empty array."""
        from src.experiments.runner import bootstrap_ci_simple

        values = np.array([])
        ci_lower, ci_upper = bootstrap_ci_simple(values)

        assert ci_lower == 0.0
        assert ci_upper == 0.0

    def test_bootstrap_ci_single_value(self):
        """bootstrap_ci_simple should handle single value."""
        from src.experiments.runner import bootstrap_ci_simple

        values = np.array([5.0])
        ci_lower, ci_upper = bootstrap_ci_simple(values)

        assert ci_lower == 5.0
        assert ci_upper == 5.0

    def test_bootstrap_ci_reproducible(self):
        """bootstrap_ci_simple should be reproducible with same seed."""
        from src.experiments.runner import bootstrap_ci_simple

        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        ci1 = bootstrap_ci_simple(values, n_bootstrap=1000, seed=42)
        ci2 = bootstrap_ci_simple(values, n_bootstrap=1000, seed=42)

        assert ci1 == ci2


class TestRunExperiment:
    """Tests for the run_experiment function."""

    def test_run_experiment_basic(self):
        """run_experiment should run method for each seed and aggregate results."""
        from src.experiments.runner import ExperimentConfig, run_experiment

        config = ExperimentConfig(
            method_name="mock",
            n_seeds=3,
            env_config_name="standard",
        )

        # Simple mock method that returns deterministic results based on seed
        def mock_method(seed: int) -> dict:
            return {
                "total_reward": float(seed * 10),
                "food_collected": float(seed * 5),
                "final_population": seed + 5,
            }

        result = run_experiment(config, mock_method)

        assert result.method_name == "mock"
        assert result.env_config_name == "standard"
        assert result.n_seeds == 3
        assert len(result.seed_list) == 3
        assert result.seed_list == [0, 1, 2]
        assert len(result.per_seed_rewards) == 3
        assert result.per_seed_rewards == [0.0, 10.0, 20.0]
        assert len(result.per_seed_food) == 3
        assert result.per_seed_food == [0.0, 5.0, 10.0]
        assert result.mean_reward == 10.0

    def test_run_experiment_with_seed_offset(self):
        """run_experiment should respect seed_offset."""
        from src.experiments.runner import ExperimentConfig, run_experiment

        config = ExperimentConfig(
            method_name="test",
            n_seeds=3,
            seed_offset=100,
        )

        seeds_used = []

        def method_fn(seed: int) -> dict:
            seeds_used.append(seed)
            return {"total_reward": 1.0, "food_collected": 1.0, "final_population": 1}

        result = run_experiment(config, method_fn)

        assert seeds_used == [100, 101, 102]
        assert result.seed_list == [100, 101, 102]

    def test_run_experiment_override_seed_offset(self):
        """run_experiment should allow overriding seed_offset."""
        from src.experiments.runner import ExperimentConfig, run_experiment

        config = ExperimentConfig(
            method_name="test",
            n_seeds=3,
            seed_offset=100,  # This will be overridden
        )

        seeds_used = []

        def method_fn(seed: int) -> dict:
            seeds_used.append(seed)
            return {"total_reward": 1.0, "food_collected": 1.0, "final_population": 1}

        result = run_experiment(config, method_fn, seed_offset=200)

        assert seeds_used == [200, 201, 202]
        assert result.seed_list == [200, 201, 202]

    def test_run_experiment_saves_per_seed_results(self):
        """run_experiment should save per-seed results when configured."""
        from src.experiments.runner import ExperimentConfig, run_experiment

        config = ExperimentConfig(
            method_name="test",
            n_seeds=2,
            save_per_seed_results=True,
        )

        def method_fn(seed: int) -> dict:
            return {
                "total_reward": float(seed),
                "food_collected": float(seed * 2),
                "final_population": seed + 1,
                "extra_data": f"seed_{seed}",
            }

        result = run_experiment(config, method_fn)

        assert len(result.per_seed_results) == 2
        assert result.per_seed_results[0]["extra_data"] == "seed_0"
        assert result.per_seed_results[1]["extra_data"] == "seed_1"

    def test_run_experiment_without_per_seed_results(self):
        """run_experiment should not save per-seed results when disabled."""
        from src.experiments.runner import ExperimentConfig, run_experiment

        config = ExperimentConfig(
            method_name="test",
            n_seeds=2,
            save_per_seed_results=False,
        )

        def method_fn(seed: int) -> dict:
            return {
                "total_reward": float(seed),
                "food_collected": float(seed),
                "final_population": 1,
            }

        result = run_experiment(config, method_fn)

        assert result.per_seed_results == []

    def test_run_experiment_computes_statistics(self):
        """run_experiment should compute aggregate statistics correctly."""
        from src.experiments.runner import ExperimentConfig, run_experiment

        config = ExperimentConfig(
            method_name="test",
            n_seeds=5,
        )

        # Predictable values for statistics validation
        values = [10.0, 20.0, 30.0, 40.0, 50.0]

        def method_fn(seed: int) -> dict:
            return {
                "total_reward": values[seed],
                "food_collected": values[seed] / 2,
                "final_population": 8,
            }

        result = run_experiment(config, method_fn)

        assert result.mean_reward == 30.0
        assert np.isclose(result.std_reward, np.std(values), atol=0.1)
        assert result.median_reward == 30.0
        assert result.mean_food == 15.0
        assert result.mean_population == 8.0

    def test_run_experiment_stores_config(self):
        """run_experiment should store the config in the result."""
        from src.experiments.runner import ExperimentConfig, run_experiment

        config = ExperimentConfig(
            method_name="test",
            n_seeds=1,
        )

        def method_fn(seed: int) -> dict:
            return {"total_reward": 1.0, "food_collected": 1.0, "final_population": 1}

        result = run_experiment(config, method_fn)

        assert result.config is config


class TestSaveLoadExperimentResult:
    """Tests for save/load functionality."""

    def test_save_and_load_result(self):
        """Should be able to save and load ExperimentResult."""
        from src.experiments.runner import (
            ExperimentResult,
            load_experiment_result,
            save_experiment_result,
        )

        result = ExperimentResult(
            method_name="test_save",
            env_config_name="standard",
            n_seeds=3,
            seed_list=[0, 1, 2],
            per_seed_rewards=[10.0, 20.0, 30.0],
            per_seed_food=[5.0, 10.0, 15.0],
            per_seed_population=[8, 8, 8],
            mean_reward=20.0,
            std_reward=8.16,
            iqm_reward=20.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.pkl"
            save_experiment_result(result, path)

            assert path.exists()

            loaded = load_experiment_result(path)

            assert loaded.method_name == result.method_name
            assert loaded.env_config_name == result.env_config_name
            assert loaded.n_seeds == result.n_seeds
            assert loaded.seed_list == result.seed_list
            assert loaded.per_seed_rewards == result.per_seed_rewards
            assert loaded.mean_reward == result.mean_reward

    def test_save_creates_parent_dirs(self):
        """save_experiment_result should create parent directories."""
        from src.experiments.runner import (
            ExperimentResult,
            save_experiment_result,
        )

        result = ExperimentResult(
            method_name="test",
            env_config_name="standard",
            n_seeds=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "deeply" / "result.pkl"
            save_experiment_result(result, path)

            assert path.exists()

    def test_load_nonexistent_file_raises(self):
        """load_experiment_result should raise FileNotFoundError for missing file."""
        from src.experiments.runner import load_experiment_result

        with pytest.raises(FileNotFoundError):
            load_experiment_result("/nonexistent/path/result.pkl")


class TestRunPairedExperiment:
    """Tests for paired experiment running."""

    def test_run_paired_experiment_uses_same_seeds(self):
        """run_paired_experiment should use same seeds for all methods."""
        from src.experiments.runner import ExperimentConfig, run_paired_experiment

        configs = [
            ExperimentConfig(method_name="method_a", n_seeds=3),
            ExperimentConfig(method_name="method_b", n_seeds=3),
        ]

        seeds_a = []
        seeds_b = []

        def method_a(seed: int) -> dict:
            seeds_a.append(seed)
            return {"total_reward": 1.0, "food_collected": 1.0, "final_population": 1}

        def method_b(seed: int) -> dict:
            seeds_b.append(seed)
            return {"total_reward": 2.0, "food_collected": 2.0, "final_population": 2}

        results = run_paired_experiment(configs, [method_a, method_b], seed_offset=10)

        assert seeds_a == seeds_b == [10, 11, 12]
        assert len(results) == 2
        assert results[0].method_name == "method_a"
        assert results[1].method_name == "method_b"

    def test_run_paired_experiment_validates_n_seeds(self):
        """run_paired_experiment should raise if n_seeds don't match."""
        from src.experiments.runner import ExperimentConfig, run_paired_experiment

        configs = [
            ExperimentConfig(method_name="a", n_seeds=3),
            ExperimentConfig(method_name="b", n_seeds=5),  # Different!
        ]

        def method_fn(seed: int) -> dict:
            return {"total_reward": 1.0, "food_collected": 1.0, "final_population": 1}

        with pytest.raises(ValueError, match="must have same n_seeds"):
            run_paired_experiment(configs, [method_fn, method_fn])

    def test_run_paired_experiment_validates_length(self):
        """run_paired_experiment should raise if configs and fns don't match."""
        from src.experiments.runner import ExperimentConfig, run_paired_experiment

        configs = [
            ExperimentConfig(method_name="a", n_seeds=3),
        ]

        def method_fn(seed: int) -> dict:
            return {"total_reward": 1.0, "food_collected": 1.0, "final_population": 1}

        with pytest.raises(ValueError, match="Number of configs"):
            run_paired_experiment(configs, [method_fn, method_fn])

    def test_run_paired_experiment_empty(self):
        """run_paired_experiment should handle empty input."""
        from src.experiments.runner import run_paired_experiment

        results = run_paired_experiment([], [])

        assert results == []


class TestCompareExperimentResults:
    """Tests for comparing experiment results."""

    def test_compare_results_basic(self):
        """compare_experiment_results should rank and compare methods."""
        from src.experiments.runner import ExperimentResult, compare_experiment_results

        results = [
            ExperimentResult(
                method_name="method_a",
                env_config_name="standard",
                n_seeds=3,
                iqm_reward=100.0,
                mean_reward=100.0,
                ci_lower=90.0,
                ci_upper=110.0,
            ),
            ExperimentResult(
                method_name="method_b",
                env_config_name="standard",
                n_seeds=3,
                iqm_reward=150.0,  # Higher - should be ranked first
                mean_reward=150.0,
                ci_lower=140.0,
                ci_upper=160.0,
            ),
        ]

        comparison = compare_experiment_results(results)

        assert comparison["best_method"] == "method_b"
        assert len(comparison["rankings"]) == 2
        assert comparison["rankings"][0]["method"] == "method_b"
        assert comparison["rankings"][0]["rank"] == 1
        assert comparison["rankings"][1]["method"] == "method_a"
        assert comparison["rankings"][1]["rank"] == 2

    def test_compare_results_pairwise_diffs(self):
        """compare_experiment_results should compute pairwise differences."""
        from src.experiments.runner import ExperimentResult, compare_experiment_results

        results = [
            ExperimentResult(
                method_name="a",
                env_config_name="standard",
                n_seeds=1,
                iqm_reward=100.0,
                mean_reward=100.0,
            ),
            ExperimentResult(
                method_name="b",
                env_config_name="standard",
                n_seeds=1,
                iqm_reward=120.0,
                mean_reward=120.0,
            ),
        ]

        comparison = compare_experiment_results(results)

        assert "a" in comparison["pairwise_diffs"]
        assert comparison["pairwise_diffs"]["a"]["b"] == -20.0  # 100 - 120
        assert comparison["pairwise_diffs"]["b"]["a"] == 20.0   # 120 - 100

    def test_compare_results_empty(self):
        """compare_experiment_results should handle empty list."""
        from src.experiments.runner import compare_experiment_results

        comparison = compare_experiment_results([])

        assert comparison["rankings"] == []
        assert comparison["pairwise_diffs"] == {}
        assert comparison["best_method"] is None


class TestStandardConfig:
    """Tests for the standard environment configuration."""

    def test_standard_config_creates_valid_config(self):
        """standard_config() should return a valid Config object."""
        from src.experiments.configs import standard_config

        config = standard_config()

        assert isinstance(config, Config)
        assert config.env.grid_size == 20
        assert config.env.num_agents == 8
        assert config.env.num_food == 10
        assert config.env.max_steps == 500
        assert config.field.num_channels == 4
        assert config.evolution.enabled is True
        assert config.evolution.max_agents == 32

    def test_standard_config_with_base_config(self):
        """standard_config() should preserve unrelated settings from base."""
        from src.experiments.configs import standard_config

        base = Config()
        base.train.learning_rate = 0.001  # Should be preserved

        config = standard_config(base)

        assert config.train.learning_rate == 0.001


class TestHiddenResourcesConfig:
    """Tests for the hidden resources environment configuration."""

    def test_hidden_resources_config_creates_valid_config(self):
        """hidden_resources_config() should return a valid Config object."""
        from src.experiments.configs import hidden_resources_config

        config = hidden_resources_config()

        assert isinstance(config, Config)
        assert config.env.num_agents == 12  # More agents for coordination
        assert config.env.num_food == 15
        assert config.env.max_steps == 1000  # Longer episodes
        assert config.evolution.max_agents == 48

    def test_hidden_resources_config_with_base_config(self):
        """hidden_resources_config() should preserve unrelated settings."""
        from src.experiments.configs import hidden_resources_config

        base = Config()
        base.train.seed = 12345

        config = hidden_resources_config(base)

        assert config.train.seed == 12345


class TestFoodScarcityConfig:
    """Tests for the food scarcity environment configuration."""

    def test_food_scarcity_config_creates_valid_config(self):
        """food_scarcity_config() should return a valid Config object."""
        from src.experiments.configs import food_scarcity_config

        config = food_scarcity_config()

        assert isinstance(config, Config)
        assert config.env.num_food == 5  # Scarce resources
        assert config.env.grid_size == 16  # Smaller grid
        assert config.env.food_respawn_prob == 0.15  # Faster respawn
        assert config.evolution.food_energy == 75  # Higher reward per food
        assert config.evolution.reproduce_threshold == 180  # Harder to reproduce
        assert config.evolution.max_agents == 24

    def test_food_scarcity_config_with_base_config(self):
        """food_scarcity_config() should preserve unrelated settings."""
        from src.experiments.configs import food_scarcity_config

        base = Config()
        base.agent.hidden_dims = (128, 128)

        config = food_scarcity_config(base)

        assert config.agent.hidden_dims == (128, 128)


class TestGetEnvConfig:
    """Tests for the get_env_config utility function."""

    def test_get_env_config_standard(self):
        """get_env_config('standard') should return standard config."""
        from src.experiments.configs import get_env_config

        config = get_env_config("standard")

        assert config.env.num_food == 10

    def test_get_env_config_hidden_resources(self):
        """get_env_config('hidden_resources') should return hidden resources config."""
        from src.experiments.configs import get_env_config

        config = get_env_config("hidden_resources")

        assert config.env.num_agents == 12

    def test_get_env_config_food_scarcity(self):
        """get_env_config('food_scarcity') should return food scarcity config."""
        from src.experiments.configs import get_env_config

        config = get_env_config("food_scarcity")

        assert config.env.num_food == 5

    def test_get_env_config_invalid_raises(self):
        """get_env_config() should raise ValueError for unknown config name."""
        from src.experiments.configs import get_env_config

        with pytest.raises(ValueError, match="Unknown config name"):
            get_env_config("nonexistent")

    def test_get_env_config_with_base_config(self):
        """get_env_config() should accept base_config parameter."""
        from src.experiments.configs import get_env_config

        base = Config()
        base.train.seed = 999

        config = get_env_config("standard", base)

        assert config.train.seed == 999


class TestListEnvConfigs:
    """Tests for list_env_configs utility."""

    def test_list_env_configs(self):
        """list_env_configs() should return all config names."""
        from src.experiments.configs import list_env_configs

        names = list_env_configs()

        assert "standard" in names
        assert "hidden_resources" in names
        assert "food_scarcity" in names
        assert len(names) == 3


class TestExperimentsImportability:
    """Tests for module importability."""

    def test_import_experiments_module(self):
        """experiments module should be importable."""
        from src import experiments

        assert experiments is not None

    def test_import_runner_module(self):
        """experiments.runner module should be importable."""
        from src.experiments import runner

        assert runner is not None

    def test_import_configs_module(self):
        """experiments.configs module should be importable."""
        from src.experiments import configs

        assert configs is not None

    def test_import_experiment_config(self):
        """ExperimentConfig should be importable."""
        from src.experiments.runner import ExperimentConfig

        assert ExperimentConfig is not None

    def test_import_experiment_result(self):
        """ExperimentResult should be importable."""
        from src.experiments.runner import ExperimentResult

        assert ExperimentResult is not None

    def test_import_run_experiment(self):
        """run_experiment should be importable."""
        from src.experiments.runner import run_experiment

        assert callable(run_experiment)

    def test_import_all_configs(self):
        """All config functions should be importable."""
        from src.experiments.configs import (
            food_scarcity_config,
            get_env_config,
            hidden_resources_config,
            list_env_configs,
            standard_config,
        )

        assert callable(standard_config)
        assert callable(hidden_resources_config)
        assert callable(food_scarcity_config)
        assert callable(get_env_config)
        assert callable(list_env_configs)

    def test_import_save_load_functions(self):
        """Save/load functions should be importable."""
        from src.experiments.runner import (
            load_experiment_result,
            save_experiment_result,
        )

        assert callable(save_experiment_result)
        assert callable(load_experiment_result)


class TestCLIVerification:
    """Tests verifying CLI compatibility."""

    def test_experiment_config_printable(self):
        """ExperimentConfig should be printable for CLI verification."""
        from src.experiments.runner import ExperimentConfig

        config = ExperimentConfig(method_name="test", n_seeds=5)
        config_str = str(config)

        assert "test" in config_str
        assert "5" in config_str

    def test_print_run_experiment_function(self):
        """run_experiment import statement should work as documented."""
        # This mimics the CLI verification command from PRD:
        # python -c "from src.experiments.runner import ExperimentConfig, run_experiment; print('OK')"
        from src.experiments.runner import ExperimentConfig, run_experiment

        assert ExperimentConfig is not None
        assert run_experiment is not None
        print("OK")  # Should not raise
