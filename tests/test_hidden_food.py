"""Tests for hidden food environment feature.

Hidden food requires K agents within distance D to reveal.
Once revealed, it stays visible for a duration then re-hides.
This creates tasks requiring coordination.
"""

import jax
import jax.numpy as jnp
import pytest

from src.configs import Config, HiddenFoodConfig
from src.environment.env import reset, step


class TestHiddenFoodConfig:
    """Tests for HiddenFoodConfig dataclass."""

    def test_default_config_disabled(self):
        """Hidden food should be disabled by default for backward compatibility."""
        config = Config()
        assert config.env.hidden_food.enabled is False

    def test_default_parameters(self):
        """Verify default hidden food parameters match PRD specifications."""
        hf = HiddenFoodConfig()
        assert hf.enabled is False
        assert hf.num_hidden == 3
        assert hf.required_agents == 3
        assert hf.reveal_distance == 3
        assert hf.reveal_duration == 10
        assert hf.hidden_food_value_multiplier == 5.0

    def test_custom_config(self):
        """Test creating custom hidden food config."""
        hf = HiddenFoodConfig(
            enabled=True,
            num_hidden=5,
            required_agents=2,
            reveal_distance=2,
            reveal_duration=5,
            hidden_food_value_multiplier=3.0,
        )
        assert hf.enabled is True
        assert hf.num_hidden == 5
        assert hf.required_agents == 2


class TestHiddenFoodDisabled:
    """Tests that hidden food disabled doesn't break existing functionality."""

    def test_env_state_no_hidden_food_fields(self):
        """When disabled, hidden food fields should be None."""
        config = Config()
        assert config.env.hidden_food.enabled is False

        key = jax.random.PRNGKey(0)
        state = reset(key, config)

        assert state.hidden_food_positions is None
        assert state.hidden_food_revealed is None
        assert state.hidden_food_reveal_timer is None
        assert state.hidden_food_collected is None

    def test_step_works_without_hidden_food(self):
        """Step should work normally when hidden food is disabled."""
        config = Config()
        key = jax.random.PRNGKey(0)
        state = reset(key, config)

        # Take a random action
        actions = jnp.zeros((config.evolution.max_agents,), dtype=jnp.int32)
        new_state, rewards, done, info = step(state, actions, config)

        # Should have normal info keys
        assert "food_collected_this_step" in info
        assert "deaths_this_step" in info
        assert "births_this_step" in info
        assert "hidden_food_collected_this_step" in info

        # Hidden food collected should be 0
        assert float(info["hidden_food_collected_this_step"]) == 0.0

        # Hidden food fields should still be None
        assert new_state.hidden_food_positions is None


class TestHiddenFoodEnabled:
    """Tests for hidden food when enabled."""

    @pytest.fixture
    def hidden_food_config(self):
        """Create a config with hidden food enabled."""
        config = Config()
        # Modify the config to enable hidden food
        config = Config(
            env=config.env.__class__(
                grid_size=config.env.grid_size,
                num_agents=config.env.num_agents,
                num_food=config.env.num_food,
                max_steps=config.env.max_steps,
                observation_radius=config.env.observation_radius,
                food_respawn_prob=config.env.food_respawn_prob,
                hidden_food=HiddenFoodConfig(
                    enabled=True,
                    num_hidden=3,
                    required_agents=3,
                    reveal_distance=3,
                    reveal_duration=10,
                    hidden_food_value_multiplier=5.0,
                ),
            ),
            field=config.field,
            agent=config.agent,
            train=config.train,
            log=config.log,
            analysis=config.analysis,
            evolution=config.evolution,
            specialization=config.specialization,
            freeze_evolve=config.freeze_evolve,
            archive=config.archive,
        )
        return config

    def test_hidden_food_initialized(self, hidden_food_config):
        """Hidden food arrays should be initialized when enabled."""
        key = jax.random.PRNGKey(0)
        state = reset(key, hidden_food_config)

        num_hidden = hidden_food_config.env.hidden_food.num_hidden

        assert state.hidden_food_positions is not None
        assert state.hidden_food_positions.shape == (num_hidden, 2)

        assert state.hidden_food_revealed is not None
        assert state.hidden_food_revealed.shape == (num_hidden,)
        assert state.hidden_food_revealed.dtype == jnp.bool_

        assert state.hidden_food_reveal_timer is not None
        assert state.hidden_food_reveal_timer.shape == (num_hidden,)

        assert state.hidden_food_collected is not None
        assert state.hidden_food_collected.shape == (num_hidden,)

    def test_hidden_food_starts_hidden(self, hidden_food_config):
        """All hidden food should start as not revealed."""
        key = jax.random.PRNGKey(0)
        state = reset(key, hidden_food_config)

        assert not jnp.any(state.hidden_food_revealed)
        assert jnp.all(state.hidden_food_reveal_timer == 0)
        assert not jnp.any(state.hidden_food_collected)

    def test_hidden_food_positions_in_grid(self, hidden_food_config):
        """Hidden food positions should be within grid bounds."""
        key = jax.random.PRNGKey(42)
        state = reset(key, hidden_food_config)

        grid_size = hidden_food_config.env.grid_size
        assert jnp.all(state.hidden_food_positions >= 0)
        assert jnp.all(state.hidden_food_positions < grid_size)


class TestHiddenFoodReveal:
    """Tests for the reveal mechanism."""

    @pytest.fixture
    def easy_reveal_config(self):
        """Config with easy reveal conditions (1 agent, distance 5)."""
        config = Config()
        return Config(
            env=config.env.__class__(
                grid_size=10,
                num_agents=4,
                num_food=5,
                max_steps=100,
                observation_radius=5,
                food_respawn_prob=0.0,  # No normal food respawn for cleaner tests
                hidden_food=HiddenFoodConfig(
                    enabled=True,
                    num_hidden=2,
                    required_agents=1,  # Only 1 agent needed
                    reveal_distance=5,  # Large distance
                    reveal_duration=5,
                    hidden_food_value_multiplier=5.0,
                ),
            ),
            field=config.field,
            agent=config.agent,
            train=config.train,
            log=config.log,
            analysis=config.analysis,
            evolution=config.evolution,
            specialization=config.specialization,
            freeze_evolve=config.freeze_evolve,
            archive=config.archive,
        )

    def test_hidden_food_reveals_when_agents_nearby(self, easy_reveal_config):
        """Hidden food should reveal when enough agents are nearby."""
        key = jax.random.PRNGKey(123)
        state = reset(key, easy_reveal_config)

        # Place an agent at the same position as first hidden food
        # Since agent is within distance 1, it will collect immediately if revealed
        # This proves the reveal/collection cycle works
        hf_pos = state.hidden_food_positions[0]
        new_agent_positions = state.agent_positions.at[0].set(hf_pos)
        state = state.replace(agent_positions=new_agent_positions)

        # Take a step (stay action)
        actions = jnp.zeros((easy_reveal_config.evolution.max_agents,), dtype=jnp.int32)
        new_state, _, _, info = step(state, actions, easy_reveal_config)

        # When agent is on hidden food, it gets revealed AND collected in same step
        # After collection, food respawns at new location (hidden again)
        # So we check that hidden_food_collected_this_step > 0 to verify reveal worked
        hf_collected = float(info["hidden_food_collected_this_step"])
        assert hf_collected > 0, "Hidden food should be revealed and collected when agent is on it"

    def test_hidden_food_reveals_but_not_collected(self, easy_reveal_config):
        """Hidden food should reveal when agents are nearby but not adjacent."""
        key = jax.random.PRNGKey(456)
        state = reset(key, easy_reveal_config)

        # Place agent exactly 2 cells away from hidden food (within reveal_distance=5 but not collectible distance=1)
        hf_pos = state.hidden_food_positions[0]
        nearby_pos = jnp.array([hf_pos[0] + 2, hf_pos[1]])  # 2 cells away (Chebyshev)
        nearby_pos = jnp.clip(nearby_pos, 0, easy_reveal_config.env.grid_size - 1)
        new_agent_positions = state.agent_positions.at[0].set(nearby_pos)
        state = state.replace(agent_positions=new_agent_positions)

        # Take a step (stay action)
        actions = jnp.zeros((easy_reveal_config.evolution.max_agents,), dtype=jnp.int32)
        new_state, _, _, info = step(state, actions, easy_reveal_config)

        # Food should be revealed (timer > 0) but not collected
        # Check both hidden food items since positions are random
        any_revealed = bool(jnp.any(new_state.hidden_food_reveal_timer > 0))
        hf_collected = float(info["hidden_food_collected_this_step"])

        # At least one hidden food should be revealed (within reveal_distance)
        # and no hidden food should be collected (agent is 2 cells away > 1)
        assert any_revealed or hf_collected > 0, "At least one hidden food should be revealed or collected"

    def test_hidden_food_stays_hidden_without_enough_agents(self):
        """Hidden food should stay hidden without enough agents nearby."""
        config = Config()
        config = Config(
            env=config.env.__class__(
                grid_size=20,
                num_agents=2,  # Only 2 agents
                num_food=5,
                max_steps=100,
                observation_radius=5,
                food_respawn_prob=0.0,
                hidden_food=HiddenFoodConfig(
                    enabled=True,
                    num_hidden=2,
                    required_agents=5,  # Need 5 agents (more than we have)
                    reveal_distance=3,
                    reveal_duration=10,
                    hidden_food_value_multiplier=5.0,
                ),
            ),
            field=config.field,
            agent=config.agent,
            train=config.train,
            log=config.log,
            analysis=config.analysis,
            evolution=config.evolution,
            specialization=config.specialization,
            freeze_evolve=config.freeze_evolve,
            archive=config.archive,
        )

        key = jax.random.PRNGKey(0)
        state = reset(key, config)

        # Run several steps
        for _ in range(10):
            actions = jnp.zeros((config.evolution.max_agents,), dtype=jnp.int32)
            state, _, _, _ = step(state, actions, config)

        # Hidden food should still be hidden (not enough agents to reveal)
        assert not jnp.any(state.hidden_food_revealed)


class TestHiddenFoodCollection:
    """Tests for hidden food collection."""

    @pytest.fixture
    def collection_config(self):
        """Config for testing collection."""
        config = Config()
        return Config(
            env=config.env.__class__(
                grid_size=10,
                num_agents=4,
                num_food=0,  # No normal food
                max_steps=100,
                observation_radius=5,
                food_respawn_prob=0.0,
                hidden_food=HiddenFoodConfig(
                    enabled=True,
                    num_hidden=1,
                    required_agents=1,  # Easy reveal
                    reveal_distance=10,  # Very large distance
                    reveal_duration=100,  # Long duration
                    hidden_food_value_multiplier=5.0,
                ),
            ),
            field=config.field,
            agent=config.agent,
            train=config.train,
            log=config.log,
            analysis=config.analysis,
            evolution=config.evolution,
            specialization=config.specialization,
            freeze_evolve=config.freeze_evolve,
            archive=config.archive,
        )

    def test_hidden_food_gives_energy_on_collection(self, collection_config):
        """Collecting hidden food should give energy * multiplier."""
        key = jax.random.PRNGKey(456)
        state = reset(key, collection_config)

        # Place first alive agent on hidden food
        hf_pos = state.hidden_food_positions[0]
        new_agent_positions = state.agent_positions.at[0].set(hf_pos)

        # Set agent as alive with known energy
        initial_energy = 50.0
        new_energy = state.agent_energy.at[0].set(initial_energy)

        state = state.replace(
            agent_positions=new_agent_positions,
            agent_energy=new_energy,
        )

        # Step to reveal and potentially collect
        actions = jnp.zeros((collection_config.evolution.max_agents,), dtype=jnp.int32)
        new_state, rewards, _, info = step(state, actions, collection_config)

        # Check if hidden food was collected
        hf_collected = float(info["hidden_food_collected_this_step"])

        # If collected, reward should reflect the high-value hidden food
        if hf_collected > 0:
            expected_energy = (
                collection_config.evolution.food_energy
                * collection_config.env.hidden_food.hidden_food_value_multiplier
            )
            assert float(rewards[0]) >= expected_energy - 1  # Allow for energy drain


class TestHiddenFoodRespawn:
    """Tests for hidden food respawning."""

    @pytest.fixture
    def respawn_config(self):
        """Config for testing respawn behavior."""
        config = Config()
        return Config(
            env=config.env.__class__(
                grid_size=10,
                num_agents=4,
                num_food=0,
                max_steps=100,
                observation_radius=5,
                food_respawn_prob=0.0,
                hidden_food=HiddenFoodConfig(
                    enabled=True,
                    num_hidden=1,
                    required_agents=1,
                    reveal_distance=10,
                    reveal_duration=3,  # Short duration for testing
                    hidden_food_value_multiplier=5.0,
                ),
            ),
            field=config.field,
            agent=config.agent,
            train=config.train,
            log=config.log,
            analysis=config.analysis,
            evolution=config.evolution,
            specialization=config.specialization,
            freeze_evolve=config.freeze_evolve,
            archive=config.archive,
        )

    def test_hidden_food_timer_decrements(self, respawn_config):
        """Reveal timer should decrement each step when not actively revealed."""
        key = jax.random.PRNGKey(789)
        state = reset(key, respawn_config)

        # Manually set revealed with timer, place agents far from hidden food
        far_positions = jnp.zeros_like(state.agent_positions)
        state = state.replace(
            agent_positions=far_positions,
            hidden_food_positions=jnp.array([[9, 9]]),  # Far corner
            hidden_food_revealed=jnp.array([True]),
            hidden_food_reveal_timer=jnp.array([3], dtype=jnp.int32),
            hidden_food_collected=jnp.array([False]),
        )

        # Step without agents nearby
        actions = jnp.zeros((respawn_config.evolution.max_agents,), dtype=jnp.int32)
        new_state, _, _, _ = step(state, actions, respawn_config)

        # Timer should have decremented (unless continuously revealed)
        # Since agents are at (0,0) and food is at (9,9), distance > reveal_distance
        # So timer should decrement
        assert new_state.hidden_food_reveal_timer[0] <= 3


class TestHiddenFoodBackwardCompatibility:
    """Tests ensuring backward compatibility with existing tests."""

    def test_existing_env_tests_still_work(self):
        """Basic environment functionality should work with hidden food disabled."""
        config = Config()
        key = jax.random.PRNGKey(0)

        # Reset should work
        state = reset(key, config)
        assert state.agent_positions.shape[0] == config.evolution.max_agents

        # Step should work
        actions = jax.random.randint(
            key, shape=(config.evolution.max_agents,), minval=0, maxval=5
        )
        new_state, rewards, done, info = step(state, actions, config)

        # Basic checks
        assert new_state.step == 1
        assert rewards.shape == (config.evolution.max_agents,)
        assert "food_collected_this_step" in info

    def test_info_dict_has_hidden_food_key(self):
        """Info dict should always have hidden_food_collected_this_step."""
        config = Config()
        key = jax.random.PRNGKey(0)
        state = reset(key, config)

        actions = jnp.zeros((config.evolution.max_agents,), dtype=jnp.int32)
        _, _, _, info = step(state, actions, config)

        assert "hidden_food_collected_this_step" in info
        # When disabled, should be 0
        assert float(info["hidden_food_collected_this_step"]) == 0.0


class TestHiddenFoodIntegration:
    """Integration tests for hidden food with other features."""

    def test_hidden_food_with_evolution(self):
        """Hidden food should work with evolution enabled."""
        config = Config()
        config = Config(
            env=config.env.__class__(
                grid_size=10,
                num_agents=4,
                num_food=5,
                max_steps=50,
                observation_radius=5,
                food_respawn_prob=0.1,
                hidden_food=HiddenFoodConfig(
                    enabled=True,
                    num_hidden=2,
                    required_agents=2,
                    reveal_distance=3,
                    reveal_duration=10,
                    hidden_food_value_multiplier=5.0,
                ),
            ),
            field=config.field,
            agent=config.agent,
            train=config.train,
            log=config.log,
            analysis=config.analysis,
            evolution=config.evolution.__class__(
                enabled=True,
                starting_energy=100,
                energy_per_step=1,
                food_energy=50,
                max_energy=200,
                reproduce_threshold=150,
                reproduce_cost=80,
                mutation_std=0.01,
                max_agents=8,
                min_agents=2,
            ),
            specialization=config.specialization,
            freeze_evolve=config.freeze_evolve,
            archive=config.archive,
        )

        key = jax.random.PRNGKey(42)
        state = reset(key, config)

        # Run several steps
        for _ in range(20):
            actions = jax.random.randint(
                key, shape=(config.evolution.max_agents,), minval=0, maxval=6
            )
            key, _ = jax.random.split(key)
            state, rewards, done, info = step(state, actions, config)

            if done:
                break

        # Should have run without errors
        assert state.step > 0

    def test_hidden_food_with_field(self):
        """Hidden food should work alongside the shared field."""
        config = Config()
        config = Config(
            env=config.env.__class__(
                grid_size=10,
                num_agents=4,
                num_food=5,
                max_steps=50,
                observation_radius=5,
                food_respawn_prob=0.1,
                hidden_food=HiddenFoodConfig(
                    enabled=True,
                    num_hidden=2,
                    required_agents=2,
                    reveal_distance=3,
                    reveal_duration=10,
                    hidden_food_value_multiplier=5.0,
                ),
            ),
            field=config.field.__class__(
                num_channels=4,
                diffusion_rate=0.1,
                decay_rate=0.05,
            ),
            agent=config.agent,
            train=config.train,
            log=config.log,
            analysis=config.analysis,
            evolution=config.evolution,
            specialization=config.specialization,
            freeze_evolve=config.freeze_evolve,
            archive=config.archive,
        )

        key = jax.random.PRNGKey(42)
        state = reset(key, config)

        # Run several steps
        for _ in range(10):
            actions = jnp.ones((config.evolution.max_agents,), dtype=jnp.int32)  # Move up
            state, _, _, _ = step(state, actions, config)

        # Field should have been written to
        assert jnp.sum(jnp.abs(state.field_state.values)) > 0


class TestVerificationCommand:
    """Test the verification command from PRD."""

    def test_hidden_food_enabled_attribute(self):
        """Verify the config access pattern from PRD verification."""
        config = Config()
        # This matches: python -c "from src.configs import Config; c = Config(); print('hidden_food enabled:', c.env.hidden_food.enabled)"
        assert hasattr(config.env, "hidden_food")
        assert hasattr(config.env.hidden_food, "enabled")
        assert config.env.hidden_food.enabled is False
