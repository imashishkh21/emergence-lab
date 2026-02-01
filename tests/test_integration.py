"""Integration tests for Phase 2: Full pipeline test with evolution.

Tests the complete pipeline: init → train → evaluate → render → analyze.
Phase 2 additions: population dynamics, lineage tracking, weight divergence.
"""

import os
import tempfile

import pytest
import jax
import jax.numpy as jnp
import numpy as np


def _small_config():
    """Create a small config suitable for fast integration testing."""
    from src.configs import Config

    config = Config()
    config.train.num_envs = 4
    config.train.num_steps = 32
    config.train.total_steps = 1000
    config.train.num_epochs = 2
    config.train.minibatch_size = 64
    config.env.num_agents = 4
    config.env.grid_size = 10
    config.env.num_food = 5
    config.env.max_steps = 20
    config.field.num_channels = 2
    config.agent.hidden_dims = (32, 32)
    config.log.wandb = False
    return config


@pytest.mark.timeout(120)  # 2 minute timeout
class TestIntegration:
    """Full pipeline integration tests."""

    def test_full_pipeline(self):
        """Test: init → train 1000 steps → evaluate → render → analyze.

        This is the primary integration test covering the full pipeline:
        1. Initialize training state
        2. Train for ~1000 env steps
        3. Evaluate with deterministic policy
        4. Render frames
        5. Analyze field metrics (entropy, structure, mutual information)
        6. Run emergence tracker
        7. Run ablation test
        8. Verify Phase 2 population metrics
        """
        from src.configs import Config
        from src.training.train import create_train_state, train_step
        from src.environment.env import reset, step
        from src.environment.obs import get_observations
        from src.environment.render import render_frame
        from src.agents.network import ActorCritic
        from src.agents.policy import get_deterministic_actions
        from src.analysis.field_metrics import field_entropy, field_structure, field_food_mi
        from src.analysis.emergence import EmergenceTracker
        from src.analysis.ablation import ablation_test
        from src.utils.video import record_episode

        config = _small_config()
        key = jax.random.PRNGKey(42)

        # ---- 1. Initialize training state ----
        runner_state = create_train_state(config, key)
        assert runner_state is not None
        assert runner_state.params is not None

        # ---- 2. Train for some steps ----
        steps_per_iter = config.train.num_envs * config.train.num_steps * config.env.num_agents
        num_updates = max(config.train.total_steps // steps_per_iter, 3)
        # Run at least a few updates but cap to keep test fast
        num_updates = min(num_updates, 10)

        all_metrics = []
        for i in range(num_updates):
            runner_state, metrics = train_step(runner_state, config)
            all_metrics.append(metrics)

        # Training should complete without NaN/Inf
        for i, m in enumerate(all_metrics):
            for name, value in m.items():
                if isinstance(value, (float, jnp.ndarray)):
                    assert not jnp.isnan(value).any(), f"NaN in {name} at update {i}"
                    assert not jnp.isinf(value).any(), f"Inf in {name} at update {i}"

        # Verify expected metric keys exist (including Phase 2 population metrics)
        expected_keys = {"total_loss", "policy_loss", "value_loss", "entropy",
                         "mean_reward", "mean_value",
                         "population_size", "births_this_step", "deaths_this_step",
                         "mean_energy", "max_energy", "min_energy",
                         "oldest_agent_age"}
        assert expected_keys.issubset(set(all_metrics[-1].keys()))

        # Population metrics are non-negative
        last_m = all_metrics[-1]
        assert float(last_m["population_size"]) >= 0
        assert float(last_m["births_this_step"]) >= 0
        assert float(last_m["deaths_this_step"]) >= 0
        assert float(last_m["mean_energy"]) >= 0
        assert float(last_m["oldest_agent_age"]) >= 0

        # ---- 3. Evaluate with deterministic policy ----
        network = ActorCritic(
            hidden_dims=tuple(config.agent.hidden_dims),
            num_actions=6,
        )
        eval_key = jax.random.PRNGKey(99)
        eval_state = reset(eval_key, config)

        total_eval_reward = 0.0
        for t in range(config.env.max_steps):
            obs = get_observations(eval_state, config)
            obs_batched = obs[None, :, :]
            actions = get_deterministic_actions(network, runner_state.params, obs_batched)
            actions = actions[0]
            eval_state, rewards, done, info = step(eval_state, actions, config)
            total_eval_reward += float(jnp.sum(rewards))
            if bool(done):
                break

        # Reward is a finite number
        assert np.isfinite(total_eval_reward)

        # ---- 4. Render frames ----
        # Render current evaluation state
        frame = render_frame(eval_state, config)
        assert frame.ndim == 3
        assert frame.shape[2] == 3  # RGB
        assert frame.shape[0] >= 100  # at least 100px height
        assert frame.shape[1] >= 100  # at least 100px width
        assert frame.dtype == np.uint8

        # Record a short video episode
        frames = record_episode(network, runner_state.params, config, jax.random.PRNGKey(7))
        assert len(frames) > 0
        assert all(f.shape == frames[0].shape for f in frames)

        # ---- 5. Analyze field metrics ----
        field_state = eval_state.field_state
        entropy = field_entropy(field_state)
        structure = field_structure(field_state)
        mi = field_food_mi(field_state, eval_state.food_positions)

        assert entropy.shape == ()
        assert structure.shape == ()
        assert mi.shape == ()
        assert float(entropy) >= 0.0
        assert 0.0 <= float(structure) <= 1.0
        assert float(mi) >= 0.0

        # ---- 6. Emergence tracker ----
        tracker = EmergenceTracker(config)
        # Feed it field states from the training run
        first_env_field = jax.tree.map(lambda x: x[0], runner_state.env_state.field_state)
        tracker.update(first_env_field, step=0)

        em_metrics = tracker.get_metrics()
        assert "emergence/entropy" in em_metrics
        assert "emergence/structure" in em_metrics
        assert "emergence/num_events" in em_metrics

        summary = tracker.get_summary()
        assert "total_updates" in summary
        assert summary["total_updates"] == 1

        # ---- 7. Ablation test (small: 2 episodes) ----
        results = ablation_test(
            network=network,
            params=runner_state.params,
            config=config,
            num_episodes=2,
            seed=42,
        )
        assert "normal" in results
        assert "zeroed" in results
        assert "random" in results
        for cond_name, result in results.items():
            assert np.isfinite(result.mean_reward), f"Non-finite reward in {cond_name}"
            assert len(result.episode_rewards) == 2

        print(f"\nIntegration test passed!")
        print(f"  Training updates: {num_updates}")
        print(f"  Eval reward: {total_eval_reward:.2f}")
        print(f"  Field entropy: {float(entropy):.4f}")
        print(f"  Field structure: {float(structure):.4f}")
        print(f"  Field-food MI: {float(mi):.4f}")
        print(f"  Frames recorded: {len(frames)}")
        print(f"  Ablation: normal={results['normal'].mean_reward:.2f}, "
              f"zeroed={results['zeroed'].mean_reward:.2f}, "
              f"random={results['random'].mean_reward:.2f}")
        print(f"  Population: {float(last_m['population_size']):.1f}")
        print(f"  Births: {float(last_m['births_this_step'])}")
        print(f"  Deaths: {float(last_m['deaths_this_step'])}")

    def test_training_stability(self):
        """Test that training produces no NaN/Inf values over multiple updates."""
        from src.training.train import create_train_state, train_step

        config = _small_config()
        key = jax.random.PRNGKey(123)
        runner_state = create_train_state(config, key)

        # Run several training steps
        for i in range(5):
            runner_state, metrics = train_step(runner_state, config)

            # Check for NaN/Inf in all metrics
            for name, value in metrics.items():
                if isinstance(value, (float, jnp.ndarray)):
                    assert not jnp.isnan(value).any(), f"NaN in {name} at step {i}"
                    assert not jnp.isinf(value).any(), f"Inf in {name} at step {i}"

    def test_video_save(self):
        """Test that video recording and saving works end-to-end."""
        from src.agents.network import ActorCritic
        from src.training.train import create_train_state
        from src.utils.video import record_episode, save_video

        config = _small_config()
        key = jax.random.PRNGKey(77)
        runner_state = create_train_state(config, key)

        network = ActorCritic(
            hidden_dims=tuple(config.agent.hidden_dims),
            num_actions=6,
        )

        frames = record_episode(network, runner_state.params, config, jax.random.PRNGKey(0))
        assert len(frames) > 0

        # Save as MP4 to a temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "test_episode.mp4")
            save_video(frames, video_path, fps=10)
            assert os.path.exists(video_path)
            assert os.path.getsize(video_path) > 0


@pytest.mark.timeout(120)
class TestPhase2Integration:
    """Phase 2 integration tests: population dynamics, lineage, weight divergence."""

    def test_population_dynamics(self):
        """Test that population changes based on food availability and energy.

        Runs episodes under different food conditions:
        1. High food: population should sustain or grow
        2. No food: population should decline (agents starve)
        """
        from src.configs import Config
        from src.environment.env import reset, step
        from src.environment.obs import get_observations
        from src.agents.network import ActorCritic
        from src.agents.policy import get_deterministic_actions

        config = _small_config()
        config.env.max_steps = 50
        config.evolution.starting_energy = 50
        config.evolution.energy_per_step = 2
        config.evolution.food_energy = 30
        config.evolution.max_agents = 16
        config.evolution.reproduce_threshold = 80
        config.evolution.reproduce_cost = 40

        network = ActorCritic(
            hidden_dims=tuple(config.agent.hidden_dims),
            num_actions=6,
        )
        key = jax.random.PRNGKey(42)
        dummy_obs = jnp.zeros((obs_dim_val(config),))
        params = network.init(key, dummy_obs)

        # --- Scenario 1: No food (all food pre-collected) => population declines ---
        key, ep_key = jax.random.split(key)
        state = reset(ep_key, config)
        # Mark all food as collected so agents cannot eat
        state = state.replace(food_collected=jnp.ones(config.env.num_food, dtype=jnp.bool_))

        initial_pop = int(jnp.sum(state.agent_alive.astype(jnp.int32)))
        total_deaths = 0
        for t in range(config.env.max_steps):
            obs = get_observations(state, config)
            obs_batched = obs[None, :, :]
            actions = get_deterministic_actions(network, params, obs_batched)[0]
            state, rewards, done, info = step(state, actions, config)
            total_deaths += int(info["deaths_this_step"])
            # Keep food collected to prevent respawns from helping
            state = state.replace(food_collected=jnp.ones(config.env.num_food, dtype=jnp.bool_))
            if bool(done):
                break

        final_pop = int(jnp.sum(state.agent_alive.astype(jnp.int32)))
        # With no food and energy drain, agents should starve
        assert total_deaths > 0, "Expected deaths with no food"
        assert final_pop < initial_pop, "Population should decline with no food"

        # --- Scenario 2: Normal food => population should sustain ---
        key, ep_key2 = jax.random.split(key)
        # High food, high respawn to keep food available
        config2 = _small_config()
        config2.env.num_food = 20
        config2.env.max_steps = 50
        config2.env.food_respawn_prob = 1.0  # instant respawn
        config2.evolution.starting_energy = 100
        config2.evolution.energy_per_step = 1
        config2.evolution.food_energy = 50
        config2.evolution.max_agents = 16
        config2.evolution.reproduce_threshold = 150
        config2.evolution.reproduce_cost = 80

        state2 = reset(ep_key2, config2)
        pop_history = []
        for t in range(config2.env.max_steps):
            obs = get_observations(state2, config2)
            obs_batched = obs[None, :, :]
            actions = get_deterministic_actions(network, params, obs_batched)[0]
            state2, rewards, done, info = step(state2, actions, config2)
            pop = int(jnp.sum(state2.agent_alive.astype(jnp.int32)))
            pop_history.append(pop)
            if bool(done):
                break

        # With abundant food and energy, population should not go extinct
        final_pop2 = pop_history[-1]
        assert final_pop2 > 0, "Population should not go extinct with abundant food"

        print(f"\nPopulation dynamics test passed!")
        print(f"  No food: {initial_pop} -> {final_pop} (deaths={total_deaths})")
        print(f"  With food: pop history min={min(pop_history)}, max={max(pop_history)}, final={final_pop2}")

    def test_lineage_tracking_across_generations(self):
        """Test that lineage tracking works correctly across multiple generations.

        Runs an episode with reproduction conditions, feeds events to LineageTracker,
        and verifies the family tree structure.
        """
        from src.environment.env import reset, step
        from src.environment.obs import get_observations
        from src.agents.network import ActorCritic
        from src.analysis.lineage import LineageTracker

        config = _small_config()
        config.env.num_food = 20
        config.env.max_steps = 80
        config.env.food_respawn_prob = 1.0
        config.evolution.starting_energy = 140
        config.evolution.energy_per_step = 1
        config.evolution.food_energy = 50
        config.evolution.max_agents = 16
        config.evolution.reproduce_threshold = 150
        config.evolution.reproduce_cost = 60
        config.evolution.max_energy = 300

        network = ActorCritic(
            hidden_dims=tuple(config.agent.hidden_dims),
            num_actions=6,
        )
        key = jax.random.PRNGKey(7)
        dummy_obs = jnp.zeros((obs_dim_val(config),))
        params = network.init(key, dummy_obs)

        key, ep_key = jax.random.split(key)
        state = reset(ep_key, config)

        tracker = LineageTracker()
        # Register original agents
        num_agents = config.env.num_agents
        for i in range(num_agents):
            tracker.register_birth(i, parent_id=-1, step=0)

        prev_alive = set(range(num_agents))
        prev_ids = set(int(state.agent_ids[i]) for i in range(config.evolution.max_agents)
                       if bool(state.agent_alive[i]))

        total_births = 0
        total_deaths = 0

        for t in range(config.env.max_steps):
            obs = get_observations(state, config)
            obs_batched = obs[None, :, :]
            # Force reproduce action for all alive agents to encourage births
            max_agents = config.evolution.max_agents
            actions = jnp.full((max_agents,), 5, dtype=jnp.int32)
            state, rewards, done, info = step(state, actions, config)

            births = int(info["births_this_step"])
            deaths = int(info["deaths_this_step"])
            total_births += births
            total_deaths += deaths

            # Track new births and deaths
            current_alive_set = set()
            for i in range(max_agents):
                if bool(state.agent_alive[i]):
                    aid = int(state.agent_ids[i])
                    current_alive_set.add(aid)
                    if aid not in prev_ids:
                        # New agent born
                        parent_id = int(state.agent_parent_ids[i])
                        tracker.register_birth(aid, parent_id=parent_id, step=t + 1)
                        prev_ids.add(aid)

            for aid in prev_alive - current_alive_set:
                tracker.register_death(aid, step=t + 1)

            prev_alive = current_alive_set

            if bool(done):
                break

        # Verify lineage tracker
        summary = tracker.get_summary()
        assert summary["total_agents"] >= num_agents, "At least original agents should be tracked"
        assert summary["original_agents"] == num_agents

        # If any births happened, verify lineage structure
        if total_births > 0:
            assert summary["total_births"] > 0, "Tracker should record births"
            assert summary["max_depth"] >= 1, "At least 1 generation if births occurred"

            # Check dominant lineages
            dominant = tracker.get_dominant_lineages(top_k=3)
            assert len(dominant) > 0, "Should have at least one lineage"
            for ancestor_id, count in dominant:
                assert ancestor_id >= 0
                assert count >= 0

            # Verify family tree consistency
            for ancestor_id, count in dominant:
                tree = tracker.get_family_tree(ancestor_id)
                assert len(tree) == count
                depth = tracker.get_lineage_depth(ancestor_id)
                assert depth == 0  # originals have depth 0

        print(f"\nLineage tracking test passed!")
        print(f"  Total agents tracked: {summary['total_agents']}")
        print(f"  Births: {total_births}, Deaths: {total_deaths}")
        print(f"  Max depth: {summary['max_depth']}")
        print(f"  Dominant lineages: {tracker.get_dominant_lineages(top_k=3)}")

    def test_weight_divergence(self):
        """Test that per-agent params diverge over time due to mutation.

        Creates per-agent params, runs reproduction with mutation, and verifies
        that child params differ from parent params after mutation.
        """
        from src.agents.network import ActorCritic
        from src.agents.reproduction import mutate_params, mutate_agent_params

        config = _small_config()
        config.evolution.mutation_std = 0.1  # large std for visible divergence

        network = ActorCritic(
            hidden_dims=tuple(config.agent.hidden_dims),
            num_actions=6,
        )
        key = jax.random.PRNGKey(42)
        dummy_obs = jnp.zeros((obs_dim_val(config),))
        params = network.init(key, dummy_obs)

        max_agents = config.evolution.max_agents
        # Create per-agent params: (max_agents, ...) for each leaf
        per_agent_params = jax.tree.map(
            lambda leaf: jnp.broadcast_to(
                leaf[None], (max_agents,) + leaf.shape
            ).copy(),
            params,
        )

        # Initially all agents have identical params
        leaves_initial = jax.tree_util.tree_leaves(per_agent_params)
        for leaf in leaves_initial:
            # All agent slots should be identical to slot 0
            for i in range(1, min(4, max_agents)):
                assert jnp.allclose(leaf[0], leaf[i]), "Initial params should be identical"

        # Apply mutation: parent=0, child=1
        key, mut_key = jax.random.split(key)
        mutated_params = mutate_agent_params(
            per_agent_params, parent_idx=0, child_idx=1,
            key=mut_key, mutation_std=config.evolution.mutation_std,
        )

        # After mutation, child (slot 1) should differ from parent (slot 0)
        leaves_mutated = jax.tree_util.tree_leaves(mutated_params)
        any_different = False
        for leaf in leaves_mutated:
            if not jnp.allclose(leaf[0], leaf[1], atol=1e-6):
                any_different = True
                break
        assert any_different, "Mutated child params should differ from parent"

        # Slot 2 should still match slot 0 (untouched)
        for leaf in leaves_mutated:
            assert jnp.allclose(leaf[0], leaf[2]), "Untouched slots should remain identical"

        # Apply multi-generation mutations and check divergence increases
        current_params = mutated_params
        divergences = []
        for gen in range(5):
            key, gen_key = jax.random.split(key)
            # Each generation: parent is slot gen+1, child is slot gen+2
            parent_idx = min(gen + 1, max_agents - 2)
            child_idx = min(gen + 2, max_agents - 1)
            current_params = mutate_agent_params(
                current_params, parent_idx=parent_idx, child_idx=child_idx,
                key=gen_key, mutation_std=config.evolution.mutation_std,
            )
            # Measure divergence from original (slot 0) to latest child
            leaves = jax.tree_util.tree_leaves(current_params)
            total_diff = sum(
                float(jnp.mean(jnp.abs(leaf[0] - leaf[child_idx])))
                for leaf in leaves
            )
            divergences.append(total_diff)

        # Divergence should generally increase over generations
        # (not strictly monotonic due to randomness, but later > earlier)
        assert divergences[-1] > divergences[0], \
            f"Divergence should increase over generations: {divergences}"

        print(f"\nWeight divergence test passed!")
        print(f"  Divergences over generations: "
              + ", ".join(f"{d:.4f}" for d in divergences))

    def test_evolution_ablation(self):
        """Test 2x2 evolution ablation: field x evolution conditions."""
        from src.agents.network import ActorCritic
        from src.analysis.ablation import evolution_ablation_test

        config = _small_config()
        config.env.max_steps = 15

        network = ActorCritic(
            hidden_dims=tuple(config.agent.hidden_dims),
            num_actions=6,
        )
        key = jax.random.PRNGKey(42)
        dummy_obs = jnp.zeros((obs_dim_val(config),))
        params = network.init(key, dummy_obs)

        results = evolution_ablation_test(
            network=network,
            params=params,
            config=config,
            num_episodes=2,
            seed=42,
        )

        # All 4 conditions should be present
        expected_conditions = {"field+evolution", "field_only", "evolution_only", "neither"}
        assert expected_conditions == set(results.keys())

        # All results should have finite rewards and valid population stats
        for name, result in results.items():
            assert np.isfinite(result.mean_reward), f"Non-finite reward in {name}"
            assert result.final_population >= 0, f"Negative population in {name}"
            assert result.total_births >= 0, f"Negative births in {name}"
            assert result.total_deaths >= 0, f"Negative deaths in {name}"
            assert 0.0 <= result.survival_rate, f"Negative survival rate in {name}"

        # Evolution-disabled conditions should have no births
        assert results["field_only"].total_births == 0, \
            "No births expected with evolution disabled"
        assert results["neither"].total_births == 0, \
            "No births expected with evolution disabled"

        print(f"\nEvolution ablation test passed!")
        for name, r in results.items():
            print(f"  {name}: reward={r.mean_reward:.2f}, "
                  f"pop={r.final_population:.1f}, "
                  f"births={r.total_births:.0f}")


def obs_dim_val(config):
    """Helper to compute observation dimension."""
    from src.environment.obs import obs_dim
    return obs_dim(config)
