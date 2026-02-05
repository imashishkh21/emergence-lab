"""Tests for the biological pheromone system (nest, food carrying, delivery)."""

import jax
import jax.numpy as jnp
import pytest

from src.configs import Config
from src.environment.env import reset, step


def _make_config(**overrides) -> Config:
    """Create a test config with sensible defaults."""
    config = Config()
    config.env.grid_size = 20
    config.env.num_agents = 4
    config.env.num_food = 5
    config.env.max_steps = 100
    config.evolution.max_agents = 8
    config.evolution.starting_energy = 100
    config.evolution.food_energy = 50
    config.evolution.max_energy = 200
    config.evolution.reproduce_threshold = 9999  # prevent auto-reproduction
    config.evolution.energy_per_step = 0  # no drain for cleaner tests
    for k, v in overrides.items():
        parts = k.split(".")
        obj = config
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], v)
    return config


class TestNestSpawn:
    """Step 6: Agents spawn within nest area."""

    def test_agents_spawn_in_nest_area(self):
        config = _make_config()
        state = reset(jax.random.PRNGKey(0), config)
        nest_center = config.env.grid_size // 2
        nest_radius = config.nest.radius

        alive_positions = state.agent_positions[:config.env.num_agents]
        dr = jnp.abs(alive_positions[:, 0] - nest_center)
        dc = jnp.abs(alive_positions[:, 1] - nest_center)
        assert jnp.all(dr <= nest_radius), "Agents should spawn within nest radius (row)"
        assert jnp.all(dc <= nest_radius), "Agents should spawn within nest radius (col)"

    def test_territory_channel_initialized(self):
        config = _make_config()
        state = reset(jax.random.PRNGKey(0), config)
        nest_center = config.env.grid_size // 2
        border = config.nest.radius + 1

        # Ch1 should be 1.0 in territory area
        r_min = max(0, nest_center - border)
        r_max = min(config.env.grid_size, nest_center + border + 1)
        c_min = max(0, nest_center - border)
        c_max = min(config.env.grid_size, nest_center + border + 1)
        territory_patch = state.field_state.values[r_min:r_max, c_min:c_max, 1]
        assert jnp.allclose(territory_patch, 1.0), "Territory channel should be 1.0 in nest border area"

        # Ch1 outside territory area should be 0
        full_ch1 = state.field_state.values[:, :, 1]
        outside_mask = jnp.ones_like(full_ch1, dtype=jnp.bool_)
        outside_mask = outside_mask.at[r_min:r_max, c_min:c_max].set(False)
        assert jnp.allclose(full_ch1[outside_mask], 0.0), "Territory channel should be 0.0 outside nest area"

    def test_other_channels_zero_on_reset(self):
        config = _make_config()
        state = reset(jax.random.PRNGKey(0), config)
        for ch in [0, 2, 3]:
            assert jnp.allclose(state.field_state.values[:, :, ch], 0.0), (
                f"Channel {ch} should be all zeros on reset"
            )


class TestFoodCarrying:
    """Step 7: Food carrying mechanics."""

    def _place_agent_near_food(self, config):
        """Create a state with agent 0 next to food item 0."""
        state = reset(jax.random.PRNGKey(42), config)
        # Place agent 0 at (5, 5) and food 0 at (5, 5) (same cell = within Chebyshev 1)
        state = state.replace(
            agent_positions=state.agent_positions.at[0].set(jnp.array([5, 5])),
            food_positions=state.food_positions.at[0].set(jnp.array([5, 5])),
            food_collected=jnp.zeros_like(state.food_collected),
        )
        return state

    def test_food_pickup_sets_has_food(self):
        config = _make_config()
        state = self._place_agent_near_food(config)
        assert not state.has_food[0], "Agent should not have food initially"

        actions = jnp.zeros(config.evolution.max_agents, dtype=jnp.int32)
        new_state, _, _, _ = step(state, actions, config)
        assert new_state.has_food[0], "Agent should have food after pickup"

    def test_scout_sip_gives_5_percent_energy(self):
        config = _make_config()
        state = self._place_agent_near_food(config)
        initial_energy = float(state.agent_energy[0])

        actions = jnp.zeros(config.evolution.max_agents, dtype=jnp.int32)
        new_state, _, _, _ = step(state, actions, config)

        expected_sip = config.evolution.food_energy * config.nest.food_sip_fraction
        expected_energy = initial_energy + expected_sip
        assert jnp.isclose(new_state.agent_energy[0], expected_energy, atol=0.1), (
            f"Expected {expected_energy}, got {float(new_state.agent_energy[0])}"
        )

    def test_laden_agent_cannot_pick_up_more_food(self):
        config = _make_config()
        state = self._place_agent_near_food(config)
        # Make agent already carrying food
        state = state.replace(has_food=state.has_food.at[0].set(True))
        initial_energy = float(state.agent_energy[0])

        actions = jnp.zeros(config.evolution.max_agents, dtype=jnp.int32)
        new_state, _, _, _ = step(state, actions, config)
        # Energy should not increase (no sip)
        assert jnp.isclose(new_state.agent_energy[0], initial_energy, atol=0.1)

    def test_laden_cooldown_toggles(self):
        config = _make_config()
        state = reset(jax.random.PRNGKey(0), config)
        # Place agent 0 outside nest to avoid delivery clearing has_food
        state = state.replace(
            agent_positions=state.agent_positions.at[0].set(jnp.array([0, 0])),
            has_food=state.has_food.at[0].set(True),
            laden_cooldown=state.laden_cooldown.at[0].set(False),
        )

        actions = jnp.zeros(config.evolution.max_agents, dtype=jnp.int32)
        s1, _, _, _ = step(state, actions, config)
        # Cooldown should toggle to True
        assert s1.laden_cooldown[0], "Laden cooldown should toggle to True"

        s2, _, _, _ = step(s1, actions, config)
        # Cooldown should toggle back to False (if still carrying)
        if s2.has_food[0]:
            assert not s2.laden_cooldown[0], "Laden cooldown should toggle back to False"


class TestNestDelivery:
    """Step 8: Nest delivery mechanics."""

    def _make_delivery_state(self, config):
        """Create state with agent 0 carrying food, placed in nest."""
        state = reset(jax.random.PRNGKey(0), config)
        nest_center = config.env.grid_size // 2
        # Place agent 0 at nest center, carrying food
        state = state.replace(
            agent_positions=state.agent_positions.at[0].set(
                jnp.array([nest_center, nest_center])
            ),
            has_food=state.has_food.at[0].set(True),
            laden_cooldown=state.laden_cooldown.at[0].set(False),
        )
        return state

    def test_nest_delivery_gives_95_percent_energy(self):
        config = _make_config()
        state = self._make_delivery_state(config)
        initial_energy = float(state.agent_energy[0])

        actions = jnp.zeros(config.evolution.max_agents, dtype=jnp.int32)
        new_state, _, _, _ = step(state, actions, config)

        delivery_energy = config.evolution.food_energy * config.nest.food_delivery_fraction
        expected = initial_energy + delivery_energy
        assert jnp.isclose(new_state.agent_energy[0], expected, atol=0.1), (
            f"Expected energy {expected}, got {float(new_state.agent_energy[0])}"
        )

    def test_nest_delivery_resets_has_food(self):
        config = _make_config()
        state = self._make_delivery_state(config)
        assert state.has_food[0], "Agent should have food before delivery"

        actions = jnp.zeros(config.evolution.max_agents, dtype=jnp.int32)
        new_state, _, _, _ = step(state, actions, config)
        assert not new_state.has_food[0], "Agent should not have food after delivery"

    def test_nest_delivery_resets_laden_cooldown(self):
        config = _make_config()
        state = self._make_delivery_state(config)
        # Set cooldown to True
        state = state.replace(laden_cooldown=state.laden_cooldown.at[0].set(True))

        actions = jnp.zeros(config.evolution.max_agents, dtype=jnp.int32)
        new_state, _, _, _ = step(state, actions, config)
        assert not new_state.laden_cooldown[0], "Laden cooldown should reset on delivery"

    def test_no_delivery_outside_nest(self):
        config = _make_config()
        state = reset(jax.random.PRNGKey(0), config)
        # Place agent 0 far from nest, carrying food
        state = state.replace(
            agent_positions=state.agent_positions.at[0].set(jnp.array([0, 0])),
            has_food=state.has_food.at[0].set(True),
            laden_cooldown=state.laden_cooldown.at[0].set(False),
        )
        initial_energy = float(state.agent_energy[0])

        actions = jnp.zeros(config.evolution.max_agents, dtype=jnp.int32)
        new_state, _, _, _ = step(state, actions, config)

        # Should still have food (no delivery)
        assert new_state.has_food[0], "Agent outside nest should keep food"
        # Energy should not include delivery bonus
        assert jnp.isclose(new_state.agent_energy[0], initial_energy, atol=0.1)

    def test_delivery_reward_signal(self):
        config = _make_config()
        state = self._make_delivery_state(config)

        actions = jnp.zeros(config.evolution.max_agents, dtype=jnp.int32)
        _, rewards, _, _ = step(state, actions, config)

        expected_reward = config.evolution.food_energy * config.nest.delivery_reward_fraction
        assert jnp.isclose(rewards[0], expected_reward, atol=0.1), (
            f"Expected delivery reward {expected_reward}, got {float(rewards[0])}"
        )

    def test_no_delivery_without_food(self):
        config = _make_config()
        state = reset(jax.random.PRNGKey(0), config)
        nest_center = config.env.grid_size // 2
        # Place agent 0 at nest center but NOT carrying food
        state = state.replace(
            agent_positions=state.agent_positions.at[0].set(
                jnp.array([nest_center, nest_center])
            ),
            has_food=state.has_food.at[0].set(False),
        )

        actions = jnp.zeros(config.evolution.max_agents, dtype=jnp.int32)
        _, rewards, _, _ = step(state, actions, config)
        # No delivery reward (agent not carrying food)
        assert jnp.isclose(rewards[0], 0.0, atol=0.01)

    def test_delivery_at_nest_edge(self):
        """Agent at the edge of nest radius should still deliver."""
        config = _make_config()
        state = reset(jax.random.PRNGKey(0), config)
        nest_center = config.env.grid_size // 2
        nest_radius = config.nest.radius
        # Place at corner of nest area
        edge_pos = jnp.array([nest_center + nest_radius, nest_center + nest_radius])
        state = state.replace(
            agent_positions=state.agent_positions.at[0].set(edge_pos),
            has_food=state.has_food.at[0].set(True),
        )

        actions = jnp.zeros(config.evolution.max_agents, dtype=jnp.int32)
        new_state, rewards, _, _ = step(state, actions, config)
        assert not new_state.has_food[0], "Agent at nest edge should deliver"
        expected_reward = config.evolution.food_energy * config.nest.delivery_reward_fraction
        assert rewards[0] > 0, "Should get delivery reward at nest edge"

    def test_delivery_just_outside_nest(self):
        """Agent one cell outside nest radius should NOT deliver."""
        config = _make_config()
        state = reset(jax.random.PRNGKey(0), config)
        nest_center = config.env.grid_size // 2
        nest_radius = config.nest.radius
        # Place one cell outside nest
        outside_pos = jnp.array([nest_center + nest_radius + 1, nest_center])
        state = state.replace(
            agent_positions=state.agent_positions.at[0].set(outside_pos),
            has_food=state.has_food.at[0].set(True),
        )

        actions = jnp.zeros(config.evolution.max_agents, dtype=jnp.int32)
        new_state, _, _, _ = step(state, actions, config)
        assert new_state.has_food[0], "Agent outside nest should keep food"


class TestNestReproduction:
    """Step 9: Reproduction requires nest area."""

    def test_reproduction_requires_nest_area(self):
        """Agent in nest with enough energy should reproduce."""
        config = _make_config(**{
            "evolution.reproduce_threshold": 150,
            "evolution.reproduce_cost": 80,
        })
        state = reset(jax.random.PRNGKey(0), config)
        nest_center = config.env.grid_size // 2
        # Place agent 0 at nest center with high energy
        state = state.replace(
            agent_positions=state.agent_positions.at[0].set(
                jnp.array([nest_center, nest_center])
            ),
            agent_energy=state.agent_energy.at[0].set(160.0),
        )

        actions = jnp.zeros(config.evolution.max_agents, dtype=jnp.int32)
        _, _, _, info = step(state, actions, config)
        assert info["births_this_step"] >= 1, "Agent in nest with enough energy should reproduce"

    def test_no_reproduction_outside_nest(self):
        """Agent outside nest should NOT reproduce even with enough energy."""
        config = _make_config(**{
            "evolution.reproduce_threshold": 150,
            "evolution.reproduce_cost": 80,
        })
        state = reset(jax.random.PRNGKey(0), config)
        # Place agent 0 far from nest with high energy
        state = state.replace(
            agent_positions=state.agent_positions.at[0].set(jnp.array([0, 0])),
            agent_energy=state.agent_energy.at[0].set(200.0),
        )
        # Ensure other agents don't reproduce
        state = state.replace(
            agent_energy=state.agent_energy.at[1:].set(10.0),
        )

        actions = jnp.zeros(config.evolution.max_agents, dtype=jnp.int32)
        new_state, _, _, info = step(state, actions, config)
        # Agent 0 should not have reproduced
        assert jnp.isclose(new_state.agent_energy[0], 200.0, atol=0.1), (
            "Agent outside nest should not lose energy to reproduction"
        )

    def test_child_spawns_in_nest(self):
        """Child should spawn within the nest area."""
        config = _make_config(**{
            "evolution.reproduce_threshold": 150,
            "evolution.reproduce_cost": 80,
        })
        state = reset(jax.random.PRNGKey(0), config)
        nest_center = config.env.grid_size // 2
        nest_radius = config.nest.radius

        # Place agent 0 in nest with high energy, ensure others don't reproduce
        new_energy = state.agent_energy.at[0].set(160.0)
        new_energy = new_energy.at[1:].set(10.0)
        state = state.replace(
            agent_positions=state.agent_positions.at[0].set(
                jnp.array([nest_center, nest_center])
            ),
            agent_energy=new_energy,
        )

        actions = jnp.zeros(config.evolution.max_agents, dtype=jnp.int32)
        new_state, _, _, info = step(state, actions, config)

        if info["births_this_step"] > 0:
            # Find the new agent
            for i in range(config.evolution.max_agents):
                if new_state.agent_alive[i] and not state.agent_alive[i]:
                    child_pos = new_state.agent_positions[i]
                    dr = jnp.abs(child_pos[0] - nest_center)
                    dc = jnp.abs(child_pos[1] - nest_center)
                    assert dr <= nest_radius, f"Child row {child_pos[0]} outside nest"
                    assert dc <= nest_radius, f"Child col {child_pos[1]} outside nest"

    def test_dead_agent_clears_has_food(self):
        """When an agent dies, has_food should be cleared."""
        config = _make_config(**{
            "evolution.energy_per_step": 200,  # High drain to cause death
        })
        state = reset(jax.random.PRNGKey(0), config)
        # Set agent 0 with low energy and carrying food
        state = state.replace(
            agent_energy=state.agent_energy.at[0].set(1.0),
            has_food=state.has_food.at[0].set(True),
        )

        actions = jnp.zeros(config.evolution.max_agents, dtype=jnp.int32)
        new_state, _, _, _ = step(state, actions, config)
        # Agent 0 should have died
        assert not new_state.agent_alive[0], "Agent should have died from energy drain"
        assert not new_state.has_food[0], "Dead agent should not carry food"


class TestPerChannelDynamics:
    """Step 3: Per-channel diffusion and decay rates."""

    def test_ch0_decays_faster_than_ch1(self):
        """Ch0 (recruitment) should decay more than Ch1 (territory)."""
        from src.field.field import FieldState
        from src.field.dynamics import decay

        values = jnp.ones((10, 10, 4), dtype=jnp.float32)
        field = FieldState(values=values)
        rates = jnp.array([0.05, 0.0001, 0.0, 0.0])
        result = decay(field, rates)

        ch0_after = float(result.values[5, 5, 0])
        ch1_after = float(result.values[5, 5, 1])
        assert ch0_after < ch1_after, (
            f"Ch0 ({ch0_after}) should have decayed more than Ch1 ({ch1_after})"
        )

    def test_ch0_diffuses_wider_than_ch1(self):
        """Ch0 (recruitment) should spread more than Ch1 (territory)."""
        from src.field.field import FieldState
        from src.field.dynamics import diffuse

        values = jnp.zeros((10, 10, 4), dtype=jnp.float32)
        values = values.at[5, 5, 0].set(1.0)  # point source on Ch0
        values = values.at[5, 5, 1].set(1.0)  # point source on Ch1
        field = FieldState(values=values)
        rates = jnp.array([0.5, 0.01, 0.0, 0.0])
        result = diffuse(field, rates)

        # Neighbors should have higher values for Ch0 than Ch1
        ch0_neighbor = float(result.values[5, 6, 0])
        ch1_neighbor = float(result.values[5, 6, 1])
        assert ch0_neighbor > ch1_neighbor, (
            f"Ch0 neighbor ({ch0_neighbor}) should be higher than Ch1 neighbor ({ch1_neighbor})"
        )

    def test_reserved_channels_unchanged(self):
        """Ch2 and Ch3 with rate=0.0 should be unchanged after diffuse+decay."""
        from src.field.field import FieldState
        from src.field.dynamics import diffuse, decay

        values = jnp.zeros((10, 10, 4), dtype=jnp.float32)
        values = values.at[3, 3, 2].set(0.7)
        values = values.at[7, 2, 3].set(0.4)
        field = FieldState(values=values)

        diffusion_rates = jnp.array([0.5, 0.01, 0.0, 0.0])
        decay_rates = jnp.array([0.05, 0.0001, 0.0, 0.0])
        result = diffuse(field, diffusion_rates)
        result = decay(result, decay_rates)

        assert jnp.allclose(result.values[:, :, 2], values[:, :, 2]), "Ch2 should be unchanged"
        assert jnp.allclose(result.values[:, :, 3], values[:, :, 3]), "Ch3 should be unchanged"


class TestLadenMovement:
    """Step 7: Laden movement suppression."""

    def test_laden_agent_frozen_on_write_step(self):
        """Agent with has_food=True and laden_cooldown=True should not move."""
        config = _make_config()
        state = reset(jax.random.PRNGKey(0), config)
        # Place agent 0 at (3, 3) — away from nest to avoid delivery
        pos = jnp.array([3, 3])
        state = state.replace(
            agent_positions=state.agent_positions.at[0].set(pos),
            has_food=state.has_food.at[0].set(True),
            laden_cooldown=state.laden_cooldown.at[0].set(True),  # write phase = frozen
        )

        # Give action=UP (1), but agent should be frozen
        actions = jnp.zeros(config.evolution.max_agents, dtype=jnp.int32)
        actions = actions.at[0].set(1)  # UP
        new_state, _, _, _ = step(state, actions, config)

        assert jnp.array_equal(new_state.agent_positions[0], pos), (
            f"Laden agent on write step should stay at {pos}, got {new_state.agent_positions[0]}"
        )


class TestFieldWrites:
    """Step 10: Channel-specific field writes."""

    def test_territory_channel_written_by_all_agents(self):
        """All alive agents should write to Ch1 (territory)."""
        config = _make_config()
        state = reset(jax.random.PRNGKey(0), config)
        # Clear field to isolate write effects
        from src.field.field import FieldState
        state = state.replace(
            field_state=FieldState(
                values=jnp.zeros_like(state.field_state.values)
            ),
        )

        actions = jnp.zeros(config.evolution.max_agents, dtype=jnp.int32)
        new_state, _, _, _ = step(state, actions, config)

        # Ch1 should have nonzero values where agents are
        ch1 = new_state.field_state.values[:, :, 1]
        assert jnp.any(ch1 > 0), "Territory channel should have writes from agents"

    def test_recruitment_channel_only_by_laden_agents(self):
        """Only laden agents in write phase should write to Ch0 (recruitment)."""
        config = _make_config()
        state = reset(jax.random.PRNGKey(0), config)
        from src.field.field import FieldState
        state = state.replace(
            field_state=FieldState(
                values=jnp.zeros_like(state.field_state.values)
            ),
        )

        # No agents carrying food — Ch0 should remain zero
        actions = jnp.zeros(config.evolution.max_agents, dtype=jnp.int32)
        new_state, _, _, _ = step(state, actions, config)

        # Ch0 should have only diffusion/decay effects on zero = still zero
        ch0 = new_state.field_state.values[:, :, 0]
        assert jnp.allclose(ch0, 0.0, atol=1e-6), "Ch0 should be zero when no agents carry food"

    def test_recruitment_write_when_laden(self):
        """Laden agent in write phase should write to Ch0."""
        config = _make_config()
        state = reset(jax.random.PRNGKey(0), config)
        from src.field.field import FieldState
        state = state.replace(
            field_state=FieldState(
                values=jnp.zeros_like(state.field_state.values)
            ),
            # Agent 0 is carrying food, place outside nest to prevent delivery
            agent_positions=state.agent_positions.at[0].set(jnp.array([0, 0])),
            has_food=state.has_food.at[0].set(True),
            laden_cooldown=state.laden_cooldown.at[0].set(False),  # Will toggle to True = write phase
        )

        actions = jnp.zeros(config.evolution.max_agents, dtype=jnp.int32)
        new_state, _, _, _ = step(state, actions, config)

        # After toggle, laden_cooldown becomes True = write phase → Ch0 should have recruitment
        ch0 = new_state.field_state.values[:, :, 0]
        assert jnp.any(ch0 > 0), "Ch0 should have recruitment pheromone from laden agent"

    def test_channels_2_3_remain_zero(self):
        """Channels 2 and 3 should remain unused (zero)."""
        config = _make_config()
        state = reset(jax.random.PRNGKey(0), config)
        from src.field.field import FieldState
        state = state.replace(
            field_state=FieldState(
                values=jnp.zeros_like(state.field_state.values)
            ),
        )

        actions = jnp.zeros(config.evolution.max_agents, dtype=jnp.int32)
        new_state, _, _, _ = step(state, actions, config)
        assert jnp.allclose(new_state.field_state.values[:, :, 2], 0.0), "Ch2 should be zero"
        assert jnp.allclose(new_state.field_state.values[:, :, 3], 0.0), "Ch3 should be zero"

    def test_field_values_capped_to_1(self):
        """Field values should be capped to field_value_cap."""
        config = _make_config()
        state = reset(jax.random.PRNGKey(0), config)

        # Run several steps to accumulate writes
        actions = jnp.zeros(config.evolution.max_agents, dtype=jnp.int32)
        for _ in range(50):
            state, _, _, _ = step(state, actions, config)

        cap = config.field.field_value_cap
        assert jnp.all(state.field_state.values <= cap + 1e-6), (
            f"Field values should be <= {cap}"
        )


class TestPheromoneIntegration:
    """Full pheromone loop: pickup -> carry -> deliver -> trail -> reproduce."""

    def test_full_pheromone_cycle(self):
        """Walk through the complete pheromone loop with manual actions.

        Layout (grid_size=20, nest at center (10,10), radius=2):
        - Agent starts at (10,10)
        - Food placed at (10,15)
        - Agent walks east 4 steps to (10,14), picks up food (adjacent to (10,15))
        - Agent returns west with move/freeze alternation
        - Agent enters nest at (10,12), delivers food
        - Reproduction triggers (energy > threshold)
        """
        config = _make_config(**{
            "env.num_agents": 1,
            "env.num_food": 1,
            "env.food_respawn_prob": 0.0,
            "evolution.max_agents": 4,
            "evolution.starting_energy": 110,
            "evolution.food_energy": 50,
            "evolution.max_energy": 300,
            "evolution.reproduce_threshold": 150,
            "evolution.reproduce_cost": 50,
            "evolution.energy_per_step": 0,
        })
        state = reset(jax.random.PRNGKey(99), config)
        nest_center = config.env.grid_size // 2  # 10
        max_agents = config.evolution.max_agents
        from src.field.field import FieldState

        # Setup: agent at nest center, food 5 cells east, field cleared
        state = state.replace(
            agent_positions=state.agent_positions.at[0].set(
                jnp.array([nest_center, nest_center])
            ),
            food_positions=state.food_positions.at[0].set(
                jnp.array([nest_center, nest_center + 5])
            ),
            food_collected=jnp.zeros(1, dtype=jnp.bool_),
            field_state=FieldState(values=jnp.zeros_like(state.field_state.values)),
        )

        RIGHT, LEFT = 4, 3

        # --- Phase 1: Walk east toward food (4 steps) ---
        for _ in range(4):
            actions = jnp.zeros(max_agents, dtype=jnp.int32).at[0].set(RIGHT)
            state, _, _, _ = step(state, actions, config)

        # Agent at (10,14), food at (10,15) is chebyshev dist 1 -> picked up
        assert int(state.agent_positions[0][1]) == nest_center + 4
        assert state.has_food[0], "Food should be picked up (adjacent on step 4)"
        sip_energy = 110.0 + 50.0 * 0.05  # 112.5
        assert jnp.isclose(state.agent_energy[0], sip_energy, atol=0.5), (
            f"Sip energy: expected ~{sip_energy}, got {float(state.agent_energy[0])}"
        )
        assert state.laden_cooldown[0], "Cooldown should be True after pickup toggle"

        # --- Phase 2: Return west with move/write alternation ---
        # Step 5 (frozen): stays at (10,14)
        actions = jnp.zeros(max_agents, dtype=jnp.int32).at[0].set(LEFT)
        state, _, _, _ = step(state, actions, config)
        assert int(state.agent_positions[0][1]) == nest_center + 4, "Should be frozen"
        assert not state.laden_cooldown[0], "Cooldown toggled to False"

        # Step 6 (move + write Ch0): (10,14) -> (10,13)
        actions = jnp.zeros(max_agents, dtype=jnp.int32).at[0].set(LEFT)
        state, _, _, _ = step(state, actions, config)
        assert int(state.agent_positions[0][1]) == nest_center + 3, "Should move to (10,13)"

        # Step 7 (frozen): stays at (10,13)
        actions = jnp.zeros(max_agents, dtype=jnp.int32).at[0].set(LEFT)
        state, _, _, _ = step(state, actions, config)
        assert int(state.agent_positions[0][1]) == nest_center + 3, "Should be frozen"

        # Step 8 (move + delivery): (10,13) -> (10,12), in nest -> delivery
        actions = jnp.zeros(max_agents, dtype=jnp.int32).at[0].set(LEFT)
        state, rewards, _, info = step(state, actions, config)
        assert int(state.agent_positions[0][1]) == nest_center + 2, "Should move to nest edge"

        # --- Phase 3: Verify delivery ---
        assert not state.has_food[0], "Food should be delivered at nest"
        assert float(rewards[0]) > 0, "Should receive delivery reward"

        # --- Phase 4: Verify Ch0 recruitment trail ---
        ch0 = state.field_state.values[:, :, 0]
        assert float(ch0[nest_center, nest_center + 4]) > 0, (
            "Ch0 trail at pickup position (10,14)"
        )
        assert float(ch0[nest_center, nest_center + 3]) > 0, (
            "Ch0 trail at return position (10,13)"
        )

        # --- Phase 5: Verify reproduction (energy > threshold in nest) ---
        births = int(info["births_this_step"])
        assert births >= 1, (
            f"Agent with ~160 energy > 150 threshold in nest should reproduce, "
            f"got {births} births"
        )
        alive_count = int(jnp.sum(state.agent_alive))
        assert alive_count >= 2, f"Should have parent + child, got {alive_count}"
