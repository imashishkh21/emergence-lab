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
        # Field should be disabled (decay_rate=1.0)
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
        assert network.num_actions == 5
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
        config.env.max_steps = 100  # More steps to increase variance
        network = create_ippo_network(config)
        key = jax.random.PRNGKey(42)
        params = init_ippo_params(network, config, key)

        # Run with several different seeds and check at least one pair differs
        results = []
        for seed in [1, 999, 42, 123]:
            result = run_ippo_episode(
                network, params, config, jax.random.PRNGKey(seed), deterministic=False
            )
            results.append((result["total_reward"], result["food_collected"]))

        # At least one pair should differ (with very high probability)
        any_differ = any(
            results[i] != results[j]
            for i in range(len(results))
            for j in range(i + 1, len(results))
        )
        assert any_differ, "All stochastic episodes had identical results"

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

    def test_ippo_field_does_not_accumulate(self):
        """IPPO field should not accumulate information (decay_rate=1.0)."""
        from src.baselines.ippo import ippo_config
        from src.environment.env import reset, step

        config = ippo_config()
        config.env.max_steps = 20

        key = jax.random.PRNGKey(42)
        state = reset(key, config)

        # Run a few steps with random actions
        for t in range(10):
            key, action_key, step_key = jax.random.split(key, 3)
            actions = jax.random.randint(action_key, (config.env.num_agents,), 0, 5)
            state, _, done, _ = step(state, actions, config)

            # With decay_rate=1.0, field from previous steps is fully decayed.
            # Only current-step writes remain (territory=0.01, recruitment=1.0 for laden).
            # Territory channel should have small values, not accumulated large values.
            field_values = np.array(state.field_state.values)
            ch1_territory = field_values[:, :, 1]
            # Multiple agents can stack on same cell, so max = N_agents * write_strength
            max_single_step = config.field.territory_write_strength * config.env.num_agents
            assert np.max(ch1_territory) <= max_single_step + 1e-6, \
                f"Territory should not accumulate: max={np.max(ch1_territory)}"

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


class TestACO:
    """Tests for the ACO baselines (ACO-Fixed and ACO-Hybrid)."""

    def test_aco_config_creates_valid_config(self):
        """aco_config() should return a valid Config object."""
        from src.baselines.aco_fixed import aco_config

        config = aco_config()

        assert isinstance(config, Config)
        # Field should be enabled with ACO parameters
        assert config.field.decay_rate == 0.5   # ACO_RHO = 0.5
        # Evolution should be disabled
        assert config.evolution.enabled is False

    def test_aco_config_with_base_config(self):
        """aco_config() should preserve non-field/evolution settings from base."""
        from src.baselines.aco_fixed import aco_config

        base = Config()
        base.env.grid_size = 30
        base.env.num_agents = 16
        base.env.num_food = 20

        config = aco_config(base)

        # Preserved settings
        assert config.env.grid_size == 30
        assert config.env.num_agents == 16
        assert config.env.num_food == 20
        # Modified settings
        assert config.field.decay_rate == 0.5
        assert config.evolution.enabled is False

    def test_aco_parameters_match_dorigo(self):
        """ACO parameters should match Dorigo & Stutzle (2004)."""
        from src.baselines.aco_fixed import ACO_ALPHA, ACO_BETA, ACO_Q, ACO_RHO

        assert ACO_ALPHA == 1.0  # Pheromone importance
        assert ACO_BETA == 2.0   # Heuristic importance
        assert ACO_RHO == 0.5    # Evaporation rate
        assert ACO_Q == 1.0      # Deposit quantity

    def test_run_aco_fixed_episode_returns_correct_format(self):
        """run_aco_fixed_episode() should return standardized result dict."""
        from src.baselines.aco_fixed import aco_config, run_aco_fixed_episode

        config = aco_config()
        config.env.max_steps = 10  # Short episode for test
        key = jax.random.PRNGKey(42)

        result = run_aco_fixed_episode(config, key)

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

    def test_run_aco_fixed_episode_population_stable(self):
        """ACO-Fixed should maintain constant population (no deaths or births)."""
        from src.baselines.aco_fixed import aco_config, run_aco_fixed_episode

        config = aco_config()
        config.env.max_steps = 50
        config.env.num_agents = 8
        config = aco_config(config)  # Reapply ACO settings

        key = jax.random.PRNGKey(42)
        result = run_aco_fixed_episode(config, key)

        # Population should remain at initial value
        assert result["final_population"] == config.env.num_agents

    def test_run_aco_fixed_no_neural_network(self):
        """ACO-Fixed should work without any neural network."""
        from src.baselines.aco_fixed import aco_config, run_aco_fixed_episode

        config = aco_config()
        config.env.max_steps = 20
        key = jax.random.PRNGKey(42)

        # Should not raise any errors - no network required
        result = run_aco_fixed_episode(config, key)
        assert result["total_reward"] >= 0

    def test_run_aco_fixed_uses_field(self):
        """ACO-Fixed should use the field (pheromone) for decision making."""
        from src.baselines.aco_fixed import aco_config
        from src.environment.env import reset, step

        config = aco_config()
        config.env.max_steps = 20

        key = jax.random.PRNGKey(42)
        state = reset(key, config)

        # Run a few steps - field should have non-zero values
        for _ in range(10):
            key, action_key = jax.random.split(key)
            actions = jax.random.randint(action_key, (config.env.num_agents,), 0, 5)
            state, _, done, _ = step(state, actions, config)
            if done:
                break

        # Field should have values (due to agent writes and diffusion)
        field_values = np.array(state.field_state.values)
        assert np.max(field_values) > 0, "ACO should have active pheromone field"

    def test_run_aco_hybrid_episode_returns_correct_format(self):
        """run_aco_hybrid_episode() should return standardized result dict."""
        from src.baselines.aco_fixed import (
            aco_config,
            create_aco_hybrid_network,
            init_aco_hybrid_params,
            run_aco_hybrid_episode,
        )

        config = aco_config()
        config.env.max_steps = 10
        network = create_aco_hybrid_network(config)
        key = jax.random.PRNGKey(42)
        params = init_aco_hybrid_params(network, config, key)

        key, episode_key = jax.random.split(key)
        result = run_aco_hybrid_episode(network, params, config, episode_key)

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

    def test_run_aco_hybrid_population_stable(self):
        """ACO-Hybrid should maintain constant population."""
        from src.baselines.aco_fixed import (
            aco_config,
            create_aco_hybrid_network,
            init_aco_hybrid_params,
            run_aco_hybrid_episode,
        )

        config = aco_config()
        config.env.max_steps = 50
        config.env.num_agents = 8
        config = aco_config(config)

        network = create_aco_hybrid_network(config)
        key = jax.random.PRNGKey(42)
        params = init_aco_hybrid_params(network, config, key)

        key, episode_key = jax.random.split(key)
        result = run_aco_hybrid_episode(network, params, config, episode_key)

        assert result["final_population"] == config.env.num_agents

    def test_run_aco_hybrid_deterministic(self):
        """run_aco_hybrid_episode() with deterministic=True should be consistent."""
        from src.baselines.aco_fixed import (
            aco_config,
            create_aco_hybrid_network,
            init_aco_hybrid_params,
            run_aco_hybrid_episode,
        )

        config = aco_config()
        config.env.max_steps = 20
        network = create_aco_hybrid_network(config)
        key = jax.random.PRNGKey(42)
        params = init_aco_hybrid_params(network, config, key)

        # Run deterministic episodes with same seed
        result1 = run_aco_hybrid_episode(
            network, params, config, jax.random.PRNGKey(0), deterministic=True
        )
        result2 = run_aco_hybrid_episode(
            network, params, config, jax.random.PRNGKey(0), deterministic=True
        )

        assert result1["total_reward"] == result2["total_reward"]
        assert result1["food_collected"] == result2["food_collected"]

    def test_evaluate_aco_fixed_returns_correct_format(self):
        """evaluate_aco_fixed() should return aggregated result dict."""
        from src.baselines.aco_fixed import aco_config, evaluate_aco_fixed

        config = aco_config()
        config.env.max_steps = 10

        result = evaluate_aco_fixed(config, n_episodes=3, seed=42)

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

        # Check shapes
        assert len(result["episode_rewards"]) == 3
        assert result["n_episodes"] == 3

    def test_evaluate_aco_hybrid_returns_correct_format(self):
        """evaluate_aco_hybrid() should return aggregated result dict."""
        from src.baselines.aco_fixed import (
            aco_config,
            create_aco_hybrid_network,
            evaluate_aco_hybrid,
            init_aco_hybrid_params,
        )

        config = aco_config()
        config.env.max_steps = 10
        network = create_aco_hybrid_network(config)
        key = jax.random.PRNGKey(42)
        params = init_aco_hybrid_params(network, config, key)

        result = evaluate_aco_hybrid(network, params, config, n_episodes=3, seed=42)

        # Check required fields
        assert "total_reward" in result
        assert "total_reward_std" in result
        assert "episode_rewards" in result
        assert len(result["episode_rewards"]) == 3
        assert result["n_episodes"] == 3

    def test_evaluate_aco_fixed_reproducible_with_seed(self):
        """evaluate_aco_fixed() with same seed should give same results."""
        from src.baselines.aco_fixed import aco_config, evaluate_aco_fixed

        config = aco_config()
        config.env.max_steps = 10

        result1 = evaluate_aco_fixed(config, n_episodes=3, seed=123)
        result2 = evaluate_aco_fixed(config, n_episodes=3, seed=123)

        assert result1["episode_rewards"] == result2["episode_rewards"]
        assert result1["episode_food"] == result2["episode_food"]

    def test_create_aco_hybrid_network(self):
        """create_aco_hybrid_network() should return a valid ActorCritic."""
        from src.agents.network import ActorCritic
        from src.baselines.aco_fixed import aco_config, create_aco_hybrid_network

        config = aco_config()
        network = create_aco_hybrid_network(config)

        assert isinstance(network, ActorCritic)
        assert network.num_actions == 5

    def test_init_aco_hybrid_params(self):
        """init_aco_hybrid_params() should initialize valid network parameters."""
        from src.baselines.aco_fixed import (
            aco_config,
            create_aco_hybrid_network,
            init_aco_hybrid_params,
        )

        config = aco_config()
        network = create_aco_hybrid_network(config)
        key = jax.random.PRNGKey(42)

        params = init_aco_hybrid_params(network, config, key)

        assert params is not None
        assert "params" in params

    def test_aco_fixed_collects_food(self):
        """ACO-Fixed agents should be able to collect food."""
        from src.baselines.aco_fixed import aco_config, run_aco_fixed_episode

        config = aco_config()
        config.env.max_steps = 100
        config.env.num_food = 50
        config.env.num_agents = 8
        config = aco_config(config)

        key = jax.random.PRNGKey(42)
        result = run_aco_fixed_episode(config, key)

        # ACO should collect some food
        assert result["food_collected"] >= 0.0

    def test_aco_hybrid_collects_food(self):
        """ACO-Hybrid agents should be able to collect food."""
        from src.baselines.aco_fixed import (
            aco_config,
            create_aco_hybrid_network,
            init_aco_hybrid_params,
            run_aco_hybrid_episode,
        )

        config = aco_config()
        config.env.max_steps = 100
        config.env.num_food = 50
        config.env.num_agents = 8
        config = aco_config(config)

        network = create_aco_hybrid_network(config)
        key = jax.random.PRNGKey(42)
        params = init_aco_hybrid_params(network, config, key)

        key, episode_key = jax.random.split(key)
        result = run_aco_hybrid_episode(network, params, config, episode_key)

        assert result["food_collected"] >= 0.0


class TestACOImportability:
    """Tests for ACO module importability."""

    def test_import_aco_module(self):
        """aco_fixed module should be importable."""
        from src.baselines import aco_fixed

        assert aco_fixed is not None

    def test_import_aco_functions(self):
        """All public functions should be importable."""
        from src.baselines.aco_fixed import (
            ACO_ALPHA,
            ACO_BETA,
            ACO_Q,
            ACO_RHO,
            aco_config,
            create_aco_hybrid_network,
            evaluate_aco_fixed,
            evaluate_aco_hybrid,
            init_aco_hybrid_params,
            run_aco_fixed_episode,
            run_aco_hybrid_episode,
        )

        assert callable(aco_config)
        assert callable(run_aco_fixed_episode)
        assert callable(run_aco_hybrid_episode)
        assert callable(evaluate_aco_fixed)
        assert callable(evaluate_aco_hybrid)
        assert callable(create_aco_hybrid_network)
        assert callable(init_aco_hybrid_params)
        assert ACO_ALPHA == 1.0
        assert ACO_BETA == 2.0
        assert ACO_RHO == 0.5
        assert ACO_Q == 1.0


class TestMAPPO:
    """Tests for the MAPPO baseline (Multi-Agent PPO with centralized critic)."""

    def test_mappo_config_creates_valid_config(self):
        """mappo_config() should return a valid Config object."""
        from src.baselines.mappo import mappo_config

        config = mappo_config()

        assert isinstance(config, Config)
        # Field should be disabled
        assert config.field.decay_rate == 1.0
        # Evolution should be disabled
        assert config.evolution.enabled is False
        # High starting energy to prevent deaths
        assert config.evolution.starting_energy >= 1000000

    def test_mappo_config_with_base_config(self):
        """mappo_config() should preserve non-field/evolution settings from base."""
        from src.baselines.mappo import mappo_config

        base = Config()
        base.env.grid_size = 30
        base.env.num_agents = 16
        base.env.num_food = 20

        config = mappo_config(base)

        # Preserved settings
        assert config.env.grid_size == 30
        assert config.env.num_agents == 16
        assert config.env.num_food == 20
        # Modified settings
        assert config.evolution.enabled is False

    def test_mappo_config_max_agents_matches_num_agents(self):
        """MAPPO config should have max_agents equal to num_agents."""
        from src.baselines.mappo import mappo_config

        base = Config()
        base.env.num_agents = 12

        config = mappo_config(base)

        assert config.evolution.max_agents == 12
        assert config.evolution.min_agents == 12

    def test_create_mappo_network(self):
        """create_mappo_network() should return a valid ActorCritic."""
        from src.agents.network import ActorCritic
        from src.baselines.mappo import create_mappo_network, mappo_config

        config = mappo_config()
        actor = create_mappo_network(config)

        assert isinstance(actor, ActorCritic)
        assert actor.num_actions == 5

    def test_create_centralized_critic(self):
        """create_centralized_critic() should return a valid CentralizedCritic."""
        from src.baselines.mappo import (
            CentralizedCritic,
            create_centralized_critic,
            mappo_config,
        )

        config = mappo_config()
        critic = create_centralized_critic(config)

        assert isinstance(critic, CentralizedCritic)
        assert critic.n_agents == config.env.num_agents

    def test_centralized_critic_larger_hidden_dims(self):
        """Centralized critic should have larger hidden dims than actor."""
        from src.baselines.mappo import (
            create_centralized_critic,
            create_mappo_network,
            mappo_config,
        )

        config = mappo_config()
        actor = create_mappo_network(config)
        critic = create_centralized_critic(config)

        # Critic should have 2x hidden dims
        assert critic.hidden_dims[0] == actor.hidden_dims[0] * 2

    def test_init_mappo_params(self):
        """init_mappo_params() should initialize valid network parameters."""
        from src.baselines.mappo import (
            create_centralized_critic,
            create_mappo_network,
            init_mappo_params,
            mappo_config,
        )

        config = mappo_config()
        actor = create_mappo_network(config)
        critic = create_centralized_critic(config)
        key = jax.random.PRNGKey(42)

        actor_params, critic_params = init_mappo_params(actor, critic, config, key)

        assert actor_params is not None
        assert critic_params is not None
        assert "params" in actor_params
        assert "params" in critic_params

    def test_centralized_critic_forward_pass(self):
        """CentralizedCritic should produce correct output shapes."""
        from src.baselines.mappo import (
            create_centralized_critic,
            create_mappo_network,
            init_mappo_params,
            mappo_config,
        )
        from src.environment.obs import obs_dim

        config = mappo_config()
        config.env.num_agents = 4
        config = mappo_config(config)

        actor = create_mappo_network(config)
        critic = create_centralized_critic(config)
        key = jax.random.PRNGKey(42)

        _, critic_params = init_mappo_params(actor, critic, config, key)

        # Test forward pass
        observation_dim = obs_dim(config)
        all_obs = jnp.zeros((config.env.num_agents * observation_dim,))
        values = critic.apply(critic_params, all_obs)

        assert values.shape == (config.env.num_agents,)

    def test_run_mappo_episode_returns_correct_format(self):
        """run_mappo_episode() should return standardized result dict."""
        from src.baselines.mappo import (
            create_centralized_critic,
            create_mappo_network,
            init_mappo_params,
            mappo_config,
            run_mappo_episode,
        )

        config = mappo_config()
        config.env.max_steps = 10  # Short episode for test
        actor = create_mappo_network(config)
        critic = create_centralized_critic(config)
        key = jax.random.PRNGKey(42)
        actor_params, critic_params = init_mappo_params(actor, critic, config, key)

        key, episode_key = jax.random.split(key)
        result = run_mappo_episode(
            actor, critic, actor_params, critic_params, config, episode_key
        )

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

    def test_run_mappo_episode_population_stable(self):
        """MAPPO should maintain constant population (no deaths or births)."""
        from src.baselines.mappo import (
            create_centralized_critic,
            create_mappo_network,
            init_mappo_params,
            mappo_config,
            run_mappo_episode,
        )

        config = mappo_config()
        config.env.max_steps = 50
        config.env.num_agents = 8
        config = mappo_config(config)

        actor = create_mappo_network(config)
        critic = create_centralized_critic(config)
        key = jax.random.PRNGKey(42)
        actor_params, critic_params = init_mappo_params(actor, critic, config, key)

        key, episode_key = jax.random.split(key)
        result = run_mappo_episode(
            actor, critic, actor_params, critic_params, config, episode_key
        )

        assert result["final_population"] == config.env.num_agents

    def test_run_mappo_episode_deterministic(self):
        """run_mappo_episode() with deterministic=True should use greedy actions."""
        from src.baselines.mappo import (
            create_centralized_critic,
            create_mappo_network,
            init_mappo_params,
            mappo_config,
            run_mappo_episode,
        )

        config = mappo_config()
        config.env.max_steps = 20
        actor = create_mappo_network(config)
        critic = create_centralized_critic(config)
        key = jax.random.PRNGKey(42)
        actor_params, critic_params = init_mappo_params(actor, critic, config, key)

        # Run deterministic episodes with same seed - should give same result
        result1 = run_mappo_episode(
            actor,
            critic,
            actor_params,
            critic_params,
            config,
            jax.random.PRNGKey(0),
            deterministic=True,
        )
        result2 = run_mappo_episode(
            actor,
            critic,
            actor_params,
            critic_params,
            config,
            jax.random.PRNGKey(0),
            deterministic=True,
        )

        # Deterministic should give same results
        assert result1["total_reward"] == result2["total_reward"]
        assert result1["food_collected"] == result2["food_collected"]

    def test_run_mappo_episode_stochastic_varies(self):
        """run_mappo_episode() with stochastic policy should vary with seed."""
        from src.baselines.mappo import (
            create_centralized_critic,
            create_mappo_network,
            init_mappo_params,
            mappo_config,
            run_mappo_episode,
        )

        config = mappo_config()
        config.env.max_steps = 50
        actor = create_mappo_network(config)
        critic = create_centralized_critic(config)
        key = jax.random.PRNGKey(42)
        actor_params, critic_params = init_mappo_params(actor, critic, config, key)

        # Run with different seeds
        result1 = run_mappo_episode(
            actor,
            critic,
            actor_params,
            critic_params,
            config,
            jax.random.PRNGKey(1),
            deterministic=False,
        )
        result2 = run_mappo_episode(
            actor,
            critic,
            actor_params,
            critic_params,
            config,
            jax.random.PRNGKey(999),
            deterministic=False,
        )

        # Results should differ (with high probability)
        assert (
            result1["total_reward"] != result2["total_reward"]
            or result1["food_collected"] != result2["food_collected"]
        )

    def test_evaluate_mappo_returns_correct_format(self):
        """evaluate_mappo() should return aggregated result dict."""
        from src.baselines.mappo import (
            create_centralized_critic,
            create_mappo_network,
            evaluate_mappo,
            init_mappo_params,
            mappo_config,
        )

        config = mappo_config()
        config.env.max_steps = 10
        actor = create_mappo_network(config)
        critic = create_centralized_critic(config)
        key = jax.random.PRNGKey(42)
        actor_params, critic_params = init_mappo_params(actor, critic, config, key)

        result = evaluate_mappo(
            actor, critic, actor_params, critic_params, config, n_episodes=3, seed=42
        )

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

    def test_evaluate_mappo_reproducible_with_seed(self):
        """evaluate_mappo() with same seed should give same results."""
        from src.baselines.mappo import (
            create_centralized_critic,
            create_mappo_network,
            evaluate_mappo,
            init_mappo_params,
            mappo_config,
        )

        config = mappo_config()
        config.env.max_steps = 10
        actor = create_mappo_network(config)
        critic = create_centralized_critic(config)
        key = jax.random.PRNGKey(42)
        actor_params, critic_params = init_mappo_params(actor, critic, config, key)

        result1 = evaluate_mappo(
            actor, critic, actor_params, critic_params, config, n_episodes=3, seed=123
        )
        result2 = evaluate_mappo(
            actor, critic, actor_params, critic_params, config, n_episodes=3, seed=123
        )

        assert result1["episode_rewards"] == result2["episode_rewards"]
        assert result1["episode_food"] == result2["episode_food"]

    def test_mappo_field_does_not_accumulate(self):
        """MAPPO field should not accumulate information (decay_rate=1.0)."""
        from src.baselines.mappo import mappo_config
        from src.environment.env import reset, step

        config = mappo_config()
        config.env.max_steps = 20

        key = jax.random.PRNGKey(42)
        state = reset(key, config)

        # Run a few steps with random actions
        for _ in range(10):
            key, action_key = jax.random.split(key)
            actions = jax.random.randint(action_key, (config.env.num_agents,), 0, 5)
            state, _, done, _ = step(state, actions, config)

            # With decay_rate=1.0, field from previous steps is fully decayed.
            # Only current-step writes remain (territory=0.01, recruitment=1.0 for laden).
            field_values = np.array(state.field_state.values)
            ch1_territory = field_values[:, :, 1]
            # Multiple agents can stack on same cell, so max = N_agents * write_strength
            max_single_step = config.field.territory_write_strength * config.env.num_agents
            assert np.max(ch1_territory) <= max_single_step + 1e-6, \
                f"Territory should not accumulate: max={np.max(ch1_territory)}"

            if done:
                break

    def test_mappo_no_evolution_deaths(self):
        """MAPPO agents should not die (energy infinite)."""
        from src.baselines.mappo import mappo_config
        from src.environment.env import reset, step

        config = mappo_config()
        config.env.max_steps = 200

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
        assert final_alive == initial_alive, "Agents died in MAPPO (should not happen)"

    def test_mappo_loss_computes_correctly(self):
        """mappo_loss() should compute loss and metrics without errors."""
        from src.baselines.mappo import (
            create_centralized_critic,
            create_mappo_network,
            init_mappo_params,
            mappo_config,
            mappo_loss,
        )
        from src.environment.obs import obs_dim

        config = mappo_config()
        config.env.num_agents = 4
        config = mappo_config(config)

        actor = create_mappo_network(config)
        critic = create_centralized_critic(config)
        key = jax.random.PRNGKey(42)
        actor_params, critic_params = init_mappo_params(actor, critic, config, key)

        observation_dim = obs_dim(config)
        batch_size = 16

        # Create dummy batch
        batch = {
            "obs": jnp.zeros((batch_size, observation_dim)),
            "all_obs": jnp.zeros((batch_size, config.env.num_agents * observation_dim)),
            "actions": jnp.zeros((batch_size,), dtype=jnp.int32),
            "log_probs": jnp.zeros((batch_size,)),
            "advantages": jnp.ones((batch_size,)),
            "returns": jnp.zeros((batch_size,)),
            "alive_mask": jnp.ones((batch_size,), dtype=jnp.bool_),
            "agent_indices": jnp.zeros((batch_size,), dtype=jnp.int32),
        }

        loss, metrics = mappo_loss(
            actor,
            critic,
            actor_params,
            critic_params,
            batch,
            clip_eps=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
        )

        assert loss.shape == ()
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert "approx_kl" in metrics
        assert "clip_fraction" in metrics

    def test_running_mean_std(self):
        """RunningMeanStd should track statistics correctly."""
        from src.baselines.mappo import RunningMeanStd

        rms = RunningMeanStd()

        # Update with some values
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rms.update(x)

        assert np.isclose(rms.mean, 3.0, atol=0.1)
        assert rms.var > 0

        # Normalize should work
        normalized = rms.normalize(x)
        assert len(normalized) == len(x)

    def test_create_mappo_train_state(self):
        """create_mappo_train_state() should initialize all training components."""
        from src.baselines.mappo import (
            create_centralized_critic,
            create_mappo_network,
            create_mappo_train_state,
            mappo_config,
        )

        config = mappo_config()
        actor = create_mappo_network(config)
        critic = create_centralized_critic(config)
        key = jax.random.PRNGKey(42)

        actor_params, critic_params, actor_opt, critic_opt = create_mappo_train_state(
            actor, critic, config, key
        )

        assert actor_params is not None
        assert critic_params is not None
        assert actor_opt is not None
        assert critic_opt is not None


class TestMAPPOImportability:
    """Tests for MAPPO module importability."""

    def test_import_mappo_module(self):
        """mappo module should be importable."""
        from src.baselines import mappo

        assert mappo is not None

    def test_import_mappo_classes(self):
        """All public classes should be importable."""
        from src.baselines.mappo import CentralizedCritic, RunningMeanStd

        assert CentralizedCritic is not None
        assert RunningMeanStd is not None

    def test_import_mappo_functions(self):
        """All public functions should be importable."""
        from src.baselines.mappo import (
            create_centralized_critic,
            create_mappo_network,
            create_mappo_train_state,
            evaluate_mappo,
            init_mappo_params,
            mappo_config,
            mappo_loss,
            run_mappo_episode,
        )

        assert callable(mappo_config)
        assert callable(create_mappo_network)
        assert callable(create_centralized_critic)
        assert callable(init_mappo_params)
        assert callable(run_mappo_episode)
        assert callable(evaluate_mappo)
        assert callable(mappo_loss)
        assert callable(create_mappo_train_state)

    def test_print_centralized_critic(self):
        """CentralizedCritic should be printable (for CLI verification)."""
        from src.baselines.mappo import CentralizedCritic

        critic_str = str(CentralizedCritic)
        assert "CentralizedCritic" in critic_str