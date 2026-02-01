"""Tests for specialization detection module."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np


class TestWeightDivergence:
    """US-001: Weight divergence metric tests."""

    def test_identical_agents_zero_divergence(self):
        """Agents with identical weights should have zero divergence."""
        from src.analysis.specialization import compute_weight_divergence

        # Create per-agent params where all agents are identical
        key = jax.random.PRNGKey(42)
        num_agents = 4
        single_weights = jax.random.normal(key, (64, 32))
        # Tile to all agents (all identical)
        params = {"layer": {"kernel": jnp.tile(single_weights[None], (num_agents, 1, 1))}}
        alive = np.ones(num_agents, dtype=bool)

        result = compute_weight_divergence(params, alive)

        assert result["mean_divergence"] == pytest.approx(0.0, abs=1e-6)
        assert result["max_divergence"] == pytest.approx(0.0, abs=1e-6)
        assert result["divergence_matrix"].shape == (num_agents, num_agents)

    def test_different_agents_positive_divergence(self):
        """Agents with different weights should have positive divergence."""
        from src.analysis.specialization import compute_weight_divergence

        num_agents = 4
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, num_agents)
        # Each agent gets independent random weights
        kernels = jnp.stack([jax.random.normal(k, (64, 32)) for k in keys])
        params = {"layer": {"kernel": kernels}}
        alive = np.ones(num_agents, dtype=bool)

        result = compute_weight_divergence(params, alive)

        assert result["mean_divergence"] > 0.0
        assert result["max_divergence"] >= result["mean_divergence"]
        assert result["divergence_matrix"].shape == (num_agents, num_agents)

    def test_divergence_matrix_symmetric(self):
        """Divergence matrix should be symmetric with zero diagonal."""
        from src.analysis.specialization import compute_weight_divergence

        num_agents = 3
        key = jax.random.PRNGKey(1)
        keys = jax.random.split(key, num_agents)
        kernels = jnp.stack([jax.random.normal(k, (32, 16)) for k in keys])
        params = {"dense": {"kernel": kernels}}
        alive = np.ones(num_agents, dtype=bool)

        result = compute_weight_divergence(params, alive)
        matrix = result["divergence_matrix"]

        # Symmetric
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-6)
        # Zero diagonal
        np.testing.assert_allclose(np.diag(matrix), 0.0, atol=1e-6)

    def test_alive_mask_filters_agents(self):
        """Only alive agents should appear in the divergence matrix."""
        from src.analysis.specialization import compute_weight_divergence

        num_agents = 5
        key = jax.random.PRNGKey(2)
        keys = jax.random.split(key, num_agents)
        kernels = jnp.stack([jax.random.normal(k, (16, 8)) for k in keys])
        params = {"layer": {"kernel": kernels}}
        alive = np.array([True, False, True, False, True])

        result = compute_weight_divergence(params, alive)

        # Only 3 alive agents
        assert result["divergence_matrix"].shape == (3, 3)
        np.testing.assert_array_equal(result["agent_indices"], [0, 2, 4])

    def test_single_alive_agent(self):
        """With only one alive agent, divergence should be zero."""
        from src.analysis.specialization import compute_weight_divergence

        num_agents = 4
        key = jax.random.PRNGKey(3)
        kernels = jax.random.normal(key, (num_agents, 32, 16))
        params = {"layer": {"kernel": kernels}}
        alive = np.array([False, True, False, False])

        result = compute_weight_divergence(params, alive)

        assert result["mean_divergence"] == 0.0
        assert result["max_divergence"] == 0.0

    def test_no_alive_agents(self):
        """With no alive agents, divergence should be zero."""
        from src.analysis.specialization import compute_weight_divergence

        num_agents = 3
        key = jax.random.PRNGKey(4)
        kernels = jax.random.normal(key, (num_agents, 16, 8))
        params = {"layer": {"kernel": kernels}}
        alive = np.zeros(num_agents, dtype=bool)

        result = compute_weight_divergence(params, alive)

        assert result["mean_divergence"] == 0.0
        assert result["max_divergence"] == 0.0

    def test_none_alive_mask_uses_all(self):
        """Passing alive_mask=None should use all agents."""
        from src.analysis.specialization import compute_weight_divergence

        num_agents = 3
        key = jax.random.PRNGKey(5)
        keys = jax.random.split(key, num_agents)
        kernels = jnp.stack([jax.random.normal(k, (32, 16)) for k in keys])
        params = {"dense": {"kernel": kernels}}

        result = compute_weight_divergence(params, alive_mask=None)

        assert result["divergence_matrix"].shape == (num_agents, num_agents)
        assert len(result["agent_indices"]) == num_agents

    def test_multi_leaf_params(self):
        """Divergence works with multi-leaf param pytrees (realistic network)."""
        from src.analysis.specialization import compute_weight_divergence

        num_agents = 3
        key = jax.random.PRNGKey(6)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        # Simulate realistic ActorCritic params with multiple layers
        params = {
            "params": {
                "Dense_0": {
                    "kernel": jax.random.normal(k1, (num_agents, 32, 64)),
                    "bias": jax.random.normal(k2, (num_agents, 64)),
                },
                "Dense_1": {
                    "kernel": jax.random.normal(k3, (num_agents, 64, 5)),
                    "bias": jax.random.normal(k4, (num_agents, 5)),
                },
            }
        }
        alive = np.ones(num_agents, dtype=bool)

        result = compute_weight_divergence(params, alive)

        assert result["mean_divergence"] > 0.0
        assert result["divergence_matrix"].shape == (num_agents, num_agents)

    def test_cosine_distance_range(self):
        """Cosine distance should be in [0, 2] range."""
        from src.analysis.specialization import compute_weight_divergence

        num_agents = 4
        key = jax.random.PRNGKey(7)
        keys = jax.random.split(key, num_agents)
        kernels = jnp.stack([jax.random.normal(k, (64, 32)) for k in keys])
        params = {"layer": {"kernel": kernels}}
        alive = np.ones(num_agents, dtype=bool)

        result = compute_weight_divergence(params, alive)

        assert result["mean_divergence"] >= 0.0
        assert result["max_divergence"] <= 2.0

    def test_flatten_agent_params(self):
        """Test that flatten_agent_params returns correct shape."""
        from src.analysis.specialization import flatten_agent_params

        num_agents = 3
        key = jax.random.PRNGKey(8)
        k1, k2 = jax.random.split(key)

        params = {
            "layer1": {"kernel": jax.random.normal(k1, (num_agents, 10, 5))},
            "layer2": {"kernel": jax.random.normal(k2, (num_agents, 5, 3))},
        }

        flat = flatten_agent_params(params, agent_idx=0)

        # Should be 10*5 + 5*3 = 65 elements
        assert flat.shape == (65,)
        assert isinstance(flat, np.ndarray)


class TestBehaviorFeatures:
    """US-002: Behavioral feature extraction tests."""

    def _make_trajectory(
        self,
        num_steps: int = 50,
        num_agents: int = 4,
        actions: np.ndarray | None = None,
        positions: np.ndarray | None = None,
        rewards: np.ndarray | None = None,
        alive_mask: np.ndarray | None = None,
        energy: np.ndarray | None = None,
        births: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Helper to create trajectory dicts with sensible defaults."""
        rng = np.random.RandomState(42)
        if actions is None:
            actions = rng.randint(0, 6, size=(num_steps, num_agents))
        if positions is None:
            positions = rng.randint(0, 32, size=(num_steps, num_agents, 2))
        if rewards is None:
            rewards = rng.uniform(0, 1, size=(num_steps, num_agents))
        if alive_mask is None:
            alive_mask = np.ones((num_steps, num_agents), dtype=bool)
        if energy is None:
            energy = rng.uniform(1, 10, size=(num_steps, num_agents))
        traj: dict[str, np.ndarray] = {
            "actions": actions,
            "positions": positions,
            "rewards": rewards,
            "alive_mask": alive_mask,
            "energy": energy,
        }
        if births is not None:
            traj["births"] = births
        return traj

    def test_output_shape(self):
        """Features should have shape (num_agents, 7)."""
        from src.analysis.specialization import extract_behavior_features

        traj = self._make_trajectory(num_steps=50, num_agents=4)
        features = extract_behavior_features(traj)
        assert features.shape == (4, 7)

    def test_all_features_finite(self):
        """All features should be finite (no NaN/inf)."""
        from src.analysis.specialization import extract_behavior_features

        traj = self._make_trajectory(num_steps=100, num_agents=6)
        features = extract_behavior_features(traj)
        assert np.all(np.isfinite(features))

    def test_dead_agent_zero_features(self):
        """An agent that is never alive should have all-zero features."""
        from src.analysis.specialization import extract_behavior_features

        alive = np.ones((50, 3), dtype=bool)
        alive[:, 2] = False  # Agent 2 never alive
        traj = self._make_trajectory(num_steps=50, num_agents=3, alive_mask=alive)
        features = extract_behavior_features(traj)
        np.testing.assert_array_equal(features[2], 0.0)

    def test_deterministic_agent_low_entropy(self):
        """An agent that always takes the same action should have 0 entropy."""
        from src.analysis.specialization import extract_behavior_features

        actions = np.zeros((100, 2), dtype=int)  # Both agents always action 0
        traj = self._make_trajectory(num_steps=100, num_agents=2, actions=actions)
        features = extract_behavior_features(traj)
        # Movement entropy should be 0 for deterministic agent
        assert features[0, 0] == pytest.approx(0.0, abs=1e-10)
        assert features[1, 0] == pytest.approx(0.0, abs=1e-10)

    def test_uniform_actions_high_entropy(self):
        """An agent with perfectly uniform actions should have entropy ~1."""
        from src.analysis.specialization import extract_behavior_features

        # Create actions that are exactly uniform (6 actions, 600 steps)
        actions = np.tile(np.arange(6), 100).reshape(-1, 1)  # (600, 1)
        traj = self._make_trajectory(num_steps=600, num_agents=1, actions=actions)
        features = extract_behavior_features(traj)
        # Entropy should be close to 1.0 (normalized)
        assert features[0, 0] == pytest.approx(1.0, abs=0.01)

    def test_food_collection_rate(self):
        """Food collection rate should equal total reward / alive steps."""
        from src.analysis.specialization import extract_behavior_features

        rewards = np.zeros((100, 2))
        rewards[:, 0] = 1.0  # Agent 0 gets reward every step
        rewards[:, 1] = 0.0  # Agent 1 gets nothing
        traj = self._make_trajectory(num_steps=100, num_agents=2, rewards=rewards)
        features = extract_behavior_features(traj)
        assert features[0, 1] == pytest.approx(1.0, abs=1e-6)
        assert features[1, 1] == pytest.approx(0.0, abs=1e-6)

    def test_stationary_agent_zero_distance(self):
        """Agent staying in one place should have 0 distance per step."""
        from src.analysis.specialization import extract_behavior_features

        positions = np.zeros((50, 1, 2), dtype=int)
        positions[:, 0] = [5, 5]  # Always at (5,5)
        traj = self._make_trajectory(num_steps=50, num_agents=1, positions=positions)
        features = extract_behavior_features(traj)
        # Distance per step should be 0
        assert features[0, 2] == pytest.approx(0.0, abs=1e-6)

    def test_moving_agent_positive_distance(self):
        """Agent that moves should have positive distance per step."""
        from src.analysis.specialization import extract_behavior_features

        # Agent alternates between (0,0) and (1,1)
        positions = np.zeros((100, 1, 2), dtype=int)
        for t in range(100):
            positions[t, 0] = [t % 2, t % 2]
        traj = self._make_trajectory(num_steps=100, num_agents=1, positions=positions)
        features = extract_behavior_features(traj)
        assert features[0, 2] > 0.0

    def test_reproduction_rate_from_births(self):
        """Reproduction rate uses births key when available."""
        from src.analysis.specialization import extract_behavior_features

        births = np.zeros((100, 2), dtype=bool)
        births[10, 0] = True
        births[50, 0] = True  # Agent 0 reproduces twice
        # Agent 1 never reproduces
        traj = self._make_trajectory(num_steps=100, num_agents=2, births=births)
        features = extract_behavior_features(traj)
        # 2 births / 100 steps * 100 = 2.0
        assert features[0, 3] == pytest.approx(2.0, abs=1e-6)
        assert features[1, 3] == pytest.approx(0.0, abs=1e-6)

    def test_reproduction_rate_from_actions(self):
        """Without births key, reproduction rate inferred from action 5."""
        from src.analysis.specialization import extract_behavior_features

        actions = np.zeros((100, 1), dtype=int)
        actions[10, 0] = 5
        actions[20, 0] = 5
        actions[30, 0] = 5  # 3 reproduce actions
        traj = self._make_trajectory(num_steps=100, num_agents=1, actions=actions)
        features = extract_behavior_features(traj)
        # 3 reproduce actions / 100 steps * 100 = 3.0
        assert features[0, 3] == pytest.approx(3.0, abs=1e-6)

    def test_mean_energy(self):
        """Mean energy should be the average energy while alive."""
        from src.analysis.specialization import extract_behavior_features

        energy = np.full((100, 1), 5.0)
        traj = self._make_trajectory(num_steps=100, num_agents=1, energy=energy)
        features = extract_behavior_features(traj)
        assert features[0, 4] == pytest.approx(5.0, abs=1e-6)

    def test_exploration_ratio(self):
        """Exploration ratio = unique cells / total steps."""
        from src.analysis.specialization import extract_behavior_features

        # Agent visits 10 unique cells over 100 steps
        positions = np.zeros((100, 1, 2), dtype=int)
        for t in range(100):
            positions[t, 0] = [t % 10, 0]  # 10 unique rows
        traj = self._make_trajectory(num_steps=100, num_agents=1, positions=positions)
        features = extract_behavior_features(traj)
        assert features[0, 5] == pytest.approx(10.0 / 100.0, abs=1e-6)

    def test_stay_fraction(self):
        """Stay fraction should count actions 0 and 5."""
        from src.analysis.specialization import extract_behavior_features

        actions = np.zeros((100, 1), dtype=int)
        actions[:50, 0] = 0  # 50 stay actions
        actions[50:60, 0] = 5  # 10 reproduce (also stays)
        actions[60:, 0] = 1  # 40 up actions
        traj = self._make_trajectory(num_steps=100, num_agents=1, actions=actions)
        features = extract_behavior_features(traj)
        # 60 stay-type actions / 100 = 0.6
        assert features[0, 6] == pytest.approx(0.6, abs=1e-6)

    def test_partial_alive_mask(self):
        """Features should only consider steps where agent is alive."""
        from src.analysis.specialization import extract_behavior_features

        alive = np.zeros((100, 1), dtype=bool)
        alive[:50, 0] = True  # Alive only first 50 steps
        energy = np.full((100, 1), 10.0)
        energy[50:, 0] = 0.0  # Dead = 0 energy
        traj = self._make_trajectory(
            num_steps=100, num_agents=1, alive_mask=alive, energy=energy
        )
        features = extract_behavior_features(traj)
        # Mean energy should be 10.0 (only alive steps counted)
        assert features[0, 4] == pytest.approx(10.0, abs=1e-6)

    def test_different_agents_different_features(self):
        """Agents with different behaviors should have different feature vectors."""
        from src.analysis.specialization import extract_behavior_features

        actions = np.zeros((200, 2), dtype=int)
        actions[:, 0] = 0  # Agent 0: always stay
        actions[:, 1] = np.tile([1, 2, 3, 4], 50)  # Agent 1: always moving

        positions = np.zeros((200, 2, 2), dtype=int)
        positions[:, 0] = [5, 5]  # Agent 0: stationary
        for t in range(200):
            positions[t, 1] = [t % 32, (t * 2) % 32]  # Agent 1: moving

        traj = self._make_trajectory(
            num_steps=200, num_agents=2, actions=actions, positions=positions
        )
        features = extract_behavior_features(traj)
        # Features should differ
        assert not np.allclose(features[0], features[1])


class TestTrajectoryRecording:
    """US-003: Trajectory recording tests."""

    def test_recorder_basic_shape(self):
        """TrajectoryRecorder produces correct shapes."""
        from src.analysis.trajectory import TrajectoryRecorder

        max_agents = 4
        recorder = TrajectoryRecorder(max_agents)

        num_steps = 10
        for _ in range(num_steps):
            recorder.record_step(
                positions=np.zeros((max_agents, 2), dtype=int),
                actions=np.zeros(max_agents, dtype=int),
                rewards=np.zeros(max_agents),
                alive_mask=np.ones(max_agents, dtype=bool),
                energy=np.ones(max_agents) * 50.0,
            )

        traj = recorder.get_trajectory()
        assert traj["actions"].shape == (num_steps, max_agents)
        assert traj["positions"].shape == (num_steps, max_agents, 2)
        assert traj["rewards"].shape == (num_steps, max_agents)
        assert traj["alive_mask"].shape == (num_steps, max_agents)
        assert traj["energy"].shape == (num_steps, max_agents)

    def test_recorder_num_steps(self):
        """num_steps property tracks recorded steps."""
        from src.analysis.trajectory import TrajectoryRecorder

        recorder = TrajectoryRecorder(max_agents=2)
        assert recorder.num_steps == 0

        for i in range(5):
            recorder.record_step(
                positions=np.zeros((2, 2)),
                actions=np.zeros(2, dtype=int),
                rewards=np.zeros(2),
                alive_mask=np.ones(2, dtype=bool),
                energy=np.ones(2),
            )
            assert recorder.num_steps == i + 1

    def test_recorder_empty_raises(self):
        """get_trajectory() raises ValueError when no steps recorded."""
        from src.analysis.trajectory import TrajectoryRecorder

        recorder = TrajectoryRecorder(max_agents=4)
        with pytest.raises(ValueError, match="No steps recorded"):
            recorder.get_trajectory()

    def test_recorder_optional_births(self):
        """Births key present only when births data provided."""
        from src.analysis.trajectory import TrajectoryRecorder

        max_agents = 3
        recorder = TrajectoryRecorder(max_agents)

        # Record without births
        recorder_no_births = TrajectoryRecorder(max_agents)
        recorder_no_births.record_step(
            positions=np.zeros((max_agents, 2)),
            actions=np.zeros(max_agents, dtype=int),
            rewards=np.zeros(max_agents),
            alive_mask=np.ones(max_agents, dtype=bool),
            energy=np.ones(max_agents),
        )
        traj_no = recorder_no_births.get_trajectory()
        assert "births" not in traj_no

        # Record with births
        recorder_with_births = TrajectoryRecorder(max_agents)
        recorder_with_births.record_step(
            positions=np.zeros((max_agents, 2)),
            actions=np.zeros(max_agents, dtype=int),
            rewards=np.zeros(max_agents),
            alive_mask=np.ones(max_agents, dtype=bool),
            energy=np.ones(max_agents),
            births=np.zeros(max_agents, dtype=bool),
        )
        traj_yes = recorder_with_births.get_trajectory()
        assert "births" in traj_yes
        assert traj_yes["births"].shape == (1, max_agents)

    def test_recorder_optional_field_values(self):
        """field_values key present only when field data provided."""
        from src.analysis.trajectory import TrajectoryRecorder

        max_agents = 2
        recorder = TrajectoryRecorder(max_agents)
        recorder.record_step(
            positions=np.zeros((max_agents, 2)),
            actions=np.zeros(max_agents, dtype=int),
            rewards=np.zeros(max_agents),
            alive_mask=np.ones(max_agents, dtype=bool),
            energy=np.ones(max_agents),
            field_values=np.ones(max_agents) * 0.5,
        )
        traj = recorder.get_trajectory()
        assert "field_values" in traj
        assert traj["field_values"].shape == (1, max_agents)
        np.testing.assert_allclose(traj["field_values"][0], 0.5)

    def test_recorder_data_values_preserved(self):
        """Recorded data values are accurately preserved."""
        from src.analysis.trajectory import TrajectoryRecorder

        max_agents = 2
        recorder = TrajectoryRecorder(max_agents)

        positions = np.array([[3, 7], [10, 15]])
        actions = np.array([1, 4])
        rewards = np.array([0.5, 1.0])
        alive = np.array([True, False])
        energy = np.array([80.0, 0.0])

        recorder.record_step(
            positions=positions,
            actions=actions,
            rewards=rewards,
            alive_mask=alive,
            energy=energy,
        )
        traj = recorder.get_trajectory()

        np.testing.assert_array_equal(traj["positions"][0], positions)
        np.testing.assert_array_equal(traj["actions"][0], actions)
        np.testing.assert_allclose(traj["rewards"][0], rewards)
        np.testing.assert_array_equal(traj["alive_mask"][0], alive)
        np.testing.assert_allclose(traj["energy"][0], energy)

    def test_recorder_accepts_jax_arrays(self):
        """TrajectoryRecorder accepts JAX arrays and converts to numpy."""
        from src.analysis.trajectory import TrajectoryRecorder

        max_agents = 3
        recorder = TrajectoryRecorder(max_agents)
        recorder.record_step(
            positions=jnp.zeros((max_agents, 2), dtype=jnp.int32),
            actions=jnp.zeros(max_agents, dtype=jnp.int32),
            rewards=jnp.zeros(max_agents),
            alive_mask=jnp.ones(max_agents, dtype=jnp.bool_),
            energy=jnp.ones(max_agents),
        )
        traj = recorder.get_trajectory()
        # All values should be numpy arrays
        for key in ("actions", "positions", "rewards", "alive_mask", "energy"):
            assert isinstance(traj[key], np.ndarray), f"{key} should be numpy"

    def test_recorder_compatible_with_behavior_features(self):
        """Trajectory from recorder works with extract_behavior_features."""
        from src.analysis.specialization import extract_behavior_features
        from src.analysis.trajectory import TrajectoryRecorder

        max_agents = 4
        rng = np.random.RandomState(42)
        recorder = TrajectoryRecorder(max_agents)

        for _ in range(50):
            recorder.record_step(
                positions=rng.randint(0, 20, size=(max_agents, 2)),
                actions=rng.randint(0, 6, size=max_agents),
                rewards=rng.uniform(0, 1, size=max_agents),
                alive_mask=np.ones(max_agents, dtype=bool),
                energy=rng.uniform(10, 100, size=max_agents),
            )

        traj = recorder.get_trajectory()
        features = extract_behavior_features(traj)
        assert features.shape == (max_agents, 7)
        assert np.all(np.isfinite(features))

    def test_record_episode_runs(self):
        """record_episode runs without error and returns valid trajectory."""
        from src.analysis.trajectory import record_episode
        from src.agents.network import ActorCritic
        from src.configs import Config

        config = Config()
        config.env.max_steps = 10
        config.env.grid_size = 10
        config.env.num_agents = 4
        config.env.num_food = 5
        config.evolution.max_agents = 8

        network = ActorCritic(
            hidden_dims=config.agent.hidden_dims,
            num_actions=6,
        )

        # Initialize params
        from src.environment.obs import obs_dim

        key = jax.random.PRNGKey(42)
        key, init_key = jax.random.split(key)
        dummy_obs = jnp.zeros(obs_dim(config))
        params = network.init(init_key, dummy_obs)

        traj = record_episode(network, params, config, key)

        # Check required keys
        for k in ("actions", "positions", "rewards", "alive_mask", "energy"):
            assert k in traj, f"Missing key: {k}"

        # Check shapes: T <= max_steps, A = max_agents
        max_agents = config.evolution.max_agents
        t = traj["actions"].shape[0]
        assert 1 <= t <= config.env.max_steps
        assert traj["actions"].shape == (t, max_agents)
        assert traj["positions"].shape == (t, max_agents, 2)
        assert traj["rewards"].shape == (t, max_agents)
        assert traj["alive_mask"].shape == (t, max_agents)
        assert traj["energy"].shape == (t, max_agents)

    def test_record_episode_has_births_and_field(self):
        """record_episode includes births and field_values."""
        from src.analysis.trajectory import record_episode
        from src.agents.network import ActorCritic
        from src.configs import Config
        from src.environment.obs import obs_dim

        config = Config()
        config.env.max_steps = 10
        config.env.grid_size = 10
        config.env.num_agents = 4
        config.env.num_food = 5
        config.evolution.max_agents = 8

        network = ActorCritic(
            hidden_dims=config.agent.hidden_dims,
            num_actions=6,
        )
        key = jax.random.PRNGKey(99)
        key, init_key = jax.random.split(key)
        params = network.init(init_key, jnp.zeros(obs_dim(config)))

        traj = record_episode(network, params, config, key)

        assert "births" in traj
        assert "field_values" in traj
        t = traj["actions"].shape[0]
        max_agents = config.evolution.max_agents
        assert traj["births"].shape == (t, max_agents)
        assert traj["field_values"].shape == (t, max_agents)

    def test_record_episode_deterministic_flag(self):
        """Deterministic mode produces consistent trajectories."""
        from src.analysis.trajectory import record_episode
        from src.agents.network import ActorCritic
        from src.configs import Config
        from src.environment.obs import obs_dim

        config = Config()
        config.env.max_steps = 5
        config.env.grid_size = 10
        config.env.num_agents = 3
        config.env.num_food = 5
        config.evolution.max_agents = 8

        network = ActorCritic(
            hidden_dims=config.agent.hidden_dims,
            num_actions=6,
        )
        key = jax.random.PRNGKey(7)
        key, init_key = jax.random.split(key)
        params = network.init(init_key, jnp.zeros(obs_dim(config)))

        traj1 = record_episode(network, params, config, key, deterministic=True)
        traj2 = record_episode(network, params, config, key, deterministic=True)

        # Same seed + deterministic => identical trajectories
        np.testing.assert_array_equal(traj1["actions"], traj2["actions"])
        np.testing.assert_array_equal(traj1["positions"], traj2["positions"])

    def test_record_episode_compatible_with_features(self):
        """Full pipeline: record_episode -> extract_behavior_features."""
        from src.analysis.specialization import extract_behavior_features
        from src.analysis.trajectory import record_episode
        from src.agents.network import ActorCritic
        from src.configs import Config
        from src.environment.obs import obs_dim

        config = Config()
        config.env.max_steps = 20
        config.env.grid_size = 10
        config.env.num_agents = 4
        config.env.num_food = 5
        config.evolution.max_agents = 8

        network = ActorCritic(
            hidden_dims=config.agent.hidden_dims,
            num_actions=6,
        )
        key = jax.random.PRNGKey(123)
        key, init_key = jax.random.split(key)
        params = network.init(init_key, jnp.zeros(obs_dim(config)))

        traj = record_episode(network, params, config, key)
        features = extract_behavior_features(traj)

        assert features.shape == (config.evolution.max_agents, 7)
        assert np.all(np.isfinite(features))

    def test_record_episode_alive_mask_reflects_state(self):
        """alive_mask in trajectory should reflect actual agent alive status."""
        from src.analysis.trajectory import record_episode
        from src.agents.network import ActorCritic
        from src.configs import Config
        from src.environment.obs import obs_dim

        config = Config()
        config.env.max_steps = 10
        config.env.grid_size = 10
        config.env.num_agents = 4
        config.env.num_food = 5
        config.evolution.max_agents = 8

        network = ActorCritic(
            hidden_dims=config.agent.hidden_dims,
            num_actions=6,
        )
        key = jax.random.PRNGKey(55)
        key, init_key = jax.random.split(key)
        params = network.init(init_key, jnp.zeros(obs_dim(config)))

        traj = record_episode(network, params, config, key)

        # First step: first num_agents should be alive, rest dead
        num_agents = config.env.num_agents
        max_agents = config.evolution.max_agents
        assert np.all(traj["alive_mask"][0, :num_agents])  # first agents alive
        assert not np.any(traj["alive_mask"][0, num_agents:])  # rest dead
