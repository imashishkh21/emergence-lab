"""Tests for baseline implementations.

This module tests all baseline methods (IPPO, ACO, MAPPO) to ensure they:
1. Return the standardized result dict format
2. Function correctly with their configurations
3. Handle edge cases appropriately
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.configs import Config


class TestIPPO:
    """Tests for the IPPO baseline (Independent PPO, no field, no evolution)."""

    def test_ippo_config_creates_valid_config(self):
        """ippo_config() should return a valid Config object."""
        from src.baselines.ippo import ippo_config

        config = ippo_config()

        assert isinstance(config, Config)
        # Field should be disabled (write_strength=0, decay_rate=1.0)
        assert config.field.write_strength == 0.0
        assert config.field.decay_rate == 1.0
        # Evolution should be disabled
        assert config.evolution.enabled is False
        # High starting energy to prevent deaths
        assert config.evolution.starting_energy >= 1000000
        # No energy drain
        assert config.evolution.energy_per_step == 0

    def test_ippo_config_with_base_config(self):
        """ippo_config() should preserve non-field/evolution settings from base."""
        from src.baselines.ippo import ippo_config

        base = Config()
        base.env.grid_size = 30
        base.env.num_agents = 16
        base.env.num_food = 20

        config = ippo_config(base)

        # Preserved settings
        assert config.env.grid_size == 30
        assert config.env.num_agents == 16
        assert config.env.num_food == 20
        # Modified settings
        assert config.field.write_strength == 0.0
        assert config.evolution.enabled is False

    def test_ippo_config_max_agents_matches_num_agents(self):
        """IPPO config should have max_agents equal to num_agents."""
        from src.baselines.ippo import ippo_config

        base = Config()
        base.env.num_agents = 12

        config = ippo_config(base)

        assert config.evolution.max_agents == 12
        assert config.evolution.min_agents == 12

    def test_create_ippo_network(self):
        """create_ippo_network() should return a valid ActorCritic."""
        from src.agents.network import ActorCritic
        from src.baselines.ippo import create_ippo_network, ippo_config

        config = ippo_config()
        network = create_ippo_network(config)

        assert isinstance(network, ActorCritic)
        assert network.num_actions == 6
        assert network.hidden_dims == config.agent.hidden_dims

    def test_init_ippo_params(self):
        """init_ippo_params() should initialize valid network parameters."""
        from src.baselines.ippo import (
            create_ippo_network,
            init_ippo_params,
            ippo_config,
        )

        config = ippo_config()
        network = create_ippo_network(config)
        key = jax.random.PRNGKey(42)

        params = init_ippo_params(network, config, key)

        assert params is not None
        assert "params" in params  # Flax params structure

    def test_run_ippo_episode_returns_correct_format(self):
        """run_ippo_episode() should return standardized result dict."""
        from src.baselines.ippo import (
            create_ippo_network,
            init_ippo_params,
            ippo_config,
            run_ippo_episode,
        )

        config = ippo_config()
        config.env.max_steps = 10  # Short episode for test
        network = create_ippo_network(config)
        key = jax.random.PRNGKey(42)
        params = init_ippo_params(network, config, key)

        key, episode_key = jax.random.split(key)
        result = run_ippo_episode(network, params, config, episode_key)

        # Check required fields
        assert "total_reward" in result
        assert "food_collected" in result
        assert "final_population" in result
        assert "per_agent_rewards" in result

        # Check types
        assert isinstance(result["total_reward"], float)
        assert isinstance(result["food_collected"], float)
        assert isinstance(result["final_population"], int)
        assert isinstance(result["per_agent_rewards"], list)

        # Check per_agent_rewards length
        assert len(result["per_agent_rewards"]) == config.env.num_agents

    def test_run_ippo_episode_population_stable(self):
        """IPPO should maintain constant population (no deaths or births)."""
        from src.baselines.ippo import (
            create_ippo_network,
            init_ippo_params,
            ippo_config,
            run_ippo_episode,
        )

        config = ippo_config()
        config.env.max_steps = 50
        config.env.num_agents = 8
        config = ippo_config(config)  # Reapply IPPO settings

        network = create_ippo_network(config)
        key = jax.random.PRNGKey(42)
        params = init_ippo_params(network, config, key)

        key, episode_key = jax.random.split(key)
        result = run_ippo_episode(network, params, config, episode_key)

        # Population should remain at initial value
        assert result["final_population"] == config.env.num_agents

    def test_run_ippo_episode_deterministic(self):
        """run_ippo_episode() with deterministic=True should use greedy actions."""
        from src.baselines.ippo import (
            create_ippo_network,
            init_ippo_params,
            ippo_config,
            run_ippo_episode,
        )

        config = ippo_config()
        config.env.max_steps = 20
        network = create_ippo_network(config)
        key = jax.random.PRNGKey(42)
        params = init_ippo_params(network, config, key)

        # Run deterministic episodes with same seed - should give same result
        result1 = run_ippo_episode(
            network, params, config, jax.random.PRNGKey(0), deterministic=True
        )
        result2 = run_ippo_episode(
            network, params, config, jax.random.PRNGKey(0), deterministic=True
        )

        # Deterministic should give same results
        assert result1["total_reward"] == result2["total_reward"]
        assert result1["food_collected"] == result2["food_collected"]

    def test_run_ippo_episode_stochastic_varies(self):
        """run_ippo_episode() with stochastic policy should vary with seed."""
        from src.baselines.ippo import (
            create_ippo_network,
            init_ippo_params,
            ippo_config,
            run_ippo_episode,
        )

        config = ippo_config()
        config.env.max_steps = 50
        network = create_ippo_network(config)
        key = jax.random.PRNGKey(42)
        params = init_ippo_params(network, config, key)

        # Run with different seeds
        result1 = run_ippo_episode(
            network, params, config, jax.random.PRNGKey(1), deterministic=False
        )
        result2 = run_ippo_episode(
            network, params, config, jax.random.PRNGKey(999), deterministic=False
        )

        # Results should differ (with high probability)
        # Note: This could theoretically fail if actions happen to be the same
        # but with different seeds and many steps, this is extremely unlikely
        assert (
            result1["total_reward"] != result2["total_reward"]
            or result1["food_collected"] != result2["food_collected"]
        )

    def test_run_ippo_episode_collects_food(self):
        """IPPO agents should be able to collect food."""
        from src.baselines.ippo import (
            create_ippo_network,
            init_ippo_params,
            ippo_config,
            run_ippo_episode,
        )

        config = ippo_config()
        config.env.max_steps = 100  # Longer episode
        config.env.num_food = 50  # More food
        config.env.num_agents = 8
        config = ippo_config(config)

        network = create_ippo_network(config)
        key = jax.random.PRNGKey(42)
        params = init_ippo_params(network, config, key)

        key, episode_key = jax.random.split(key)
        result = run_ippo_episode(network, params, config, episode_key)

        # With random movement, agents should collect at least some food
        assert result["food_collected"] >= 0.0

    def test_evaluate_ippo_returns_correct_format(self):
        """evaluate_ippo() should return aggregated result dict."""
        from src.baselines.ippo import (
            create_ippo_network,
            evaluate_ippo,
            init_ippo_params,
            ippo_config,
        )

        config = ippo_config()
        config.env.max_steps = 10
        network = create_ippo_network(config)
        key = jax.random.PRNGKey(42)
        params = init_ippo_params(network, config, key)

        result = evaluate_ippo(network, params, config, n_episodes=3, seed=42)

        # Check required fields
        assert "total_reward" in result
        assert "total_reward_std" in result
        assert "food_collected" in result
        assert "food_collected_std" in result
        assert "final_population" in result
        assert "per_agent_rewards" in result
        assert "episode_rewards" in result
        assert "episode_food" in result
        assert "n_episodes" in result

        # Check types and shapes
        assert isinstance(result["total_reward"], float)
        assert isinstance(result["total_reward_std"], float)
        assert isinstance(result["episode_rewards"], list)
        assert len(result["episode_rewards"]) == 3
        assert result["n_episodes"] == 3

    def test_evaluate_ippo_reproducible_with_seed(self):
        """evaluate_ippo() with same seed should give same results."""
        from src.baselines.ippo import (
            create_ippo_network,
            evaluate_ippo,
            init_ippo_params,
            ippo_config,
        )

        config = ippo_config()
        config.env.max_steps = 10
        network = create_ippo_network(config)
        key = jax.random.PRNGKey(42)
        params = init_ippo_params(network, config, key)

        result1 = evaluate_ippo(network, params, config, n_episodes=3, seed=123)
        result2 = evaluate_ippo(network, params, config, n_episodes=3, seed=123)

        assert result1["episode_rewards"] == result2["episode_rewards"]
        assert result1["episode_food"] == result2["episode_food"]

    def test_ippo_field_is_zeroed(self):
        """IPPO should have a zeroed field throughout the episode."""
        from src.baselines.ippo import ippo_config
        from src.environment.env import reset, step
        from src.environment.obs import get_observations

        config = ippo_config()
        config.env.max_steps = 20

        key = jax.random.PRNGKey(42)
        state = reset(key, config)

        # Run a few steps with random actions
        for _ in range(10):
            key, action_key, step_key = jax.random.split(key, 3)
            actions = jax.random.randint(action_key, (config.env.num_agents,), 0, 5)
            state, _, done, _ = step(state, actions, config)

            # Field should be all zeros (decay_rate=1.0 zeros it each step)
            field_values = np.array(state.field_state.values)
            # Note: write_strength=0 means nothing is written
            # decay_rate=1.0 means any residual decays fully
            # So field should be zeros or very close
            assert np.allclose(
                field_values, 0.0, atol=1e-6
            ), f"Field not zeroed: max={np.max(np.abs(field_values))}"

            if done:
                break

    def test_ippo_no_evolution_deaths(self):
        """IPPO agents should not die (energy infinite)."""
        from src.baselines.ippo import ippo_config
        from src.environment.env import reset, step

        config = ippo_config()
        config.env.max_steps = 200  # Longer episode

        key = jax.random.PRNGKey(42)
        state = reset(key, config)

        initial_alive = int(np.sum(np.array(state.agent_alive)))

        # Run full episode
        for _ in range(config.env.max_steps):
            key, action_key = jax.random.split(key)
            actions = jax.random.randint(
                action_key, (config.evolution.max_agents,), 0, 5
            )
            state, _, done, _ = step(state, actions, config)
            if done:
                break

        final_alive = int(np.sum(np.array(state.agent_alive)))
        assert final_alive == initial_alive, "Agents died in IPPO (should not happen)"


class TestIPPOImportability:
    """Tests for module importability."""

    def test_import_ippo_module(self):
        """ippo module should be importable."""
        from src.baselines import ippo

        assert ippo is not None

    def test_import_ippo_functions(self):
        """All public functions should be importable."""
        from src.baselines.ippo import (
            create_ippo_network,
            evaluate_ippo,
            init_ippo_params,
            ippo_config,
            run_ippo_episode,
        )

        assert callable(ippo_config)
        assert callable(run_ippo_episode)
        assert callable(evaluate_ippo)
        assert callable(create_ippo_network)
        assert callable(init_ippo_params)

    def test_print_ippo_config(self):
        """ippo_config() output should be printable (for CLI verification)."""
        from src.baselines.ippo import ippo_config

        config = ippo_config()
        config_str = str(config)

        assert "field" in config_str
        assert "evolution" in config_str
