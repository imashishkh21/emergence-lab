"""Trajectory recording for agent behavior analysis.

Records per-agent, per-step data during evaluation episodes for use
in specialization detection and behavioral clustering.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from src.agents.network import ActorCritic
from src.agents.policy import sample_actions
from src.configs import Config
from src.environment.env import reset, step
from src.environment.obs import get_observations
from src.field.ops import read_local


class TrajectoryRecorder:
    """Records per-step trajectory data during an episode.

    Captures agent positions, actions, field reads/writes, energy levels,
    and rewards at each timestep. The resulting trajectory dict is
    compatible with ``extract_behavior_features()`` from
    ``src.analysis.specialization``.

    Usage::

        recorder = TrajectoryRecorder(max_agents=32)
        recorder.record_step(
            positions=..., actions=..., rewards=...,
            alive_mask=..., energy=..., births=...,
            field_values=...,
        )
        trajectory = recorder.get_trajectory()
    """

    def __init__(self, max_agents: int) -> None:
        self.max_agents = max_agents
        self._positions: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._rewards: list[np.ndarray] = []
        self._alive_mask: list[np.ndarray] = []
        self._energy: list[np.ndarray] = []
        self._births: list[np.ndarray] = []
        self._field_values: list[np.ndarray] = []

    def record_step(
        self,
        positions: np.ndarray | jnp.ndarray,
        actions: np.ndarray | jnp.ndarray,
        rewards: np.ndarray | jnp.ndarray,
        alive_mask: np.ndarray | jnp.ndarray,
        energy: np.ndarray | jnp.ndarray,
        births: np.ndarray | jnp.ndarray | None = None,
        field_values: np.ndarray | jnp.ndarray | None = None,
    ) -> None:
        """Record data from one environment step.

        Args:
            positions: Agent positions, shape (max_agents, 2).
            actions: Agent actions, shape (max_agents,).
            rewards: Agent rewards, shape (max_agents,).
            alive_mask: Boolean alive mask, shape (max_agents,).
            energy: Agent energy levels, shape (max_agents,).
            births: Boolean birth events, shape (max_agents,). Optional.
            field_values: Field values at agent positions, shape (max_agents,)
                or (max_agents, C). Optional.
        """
        self._positions.append(np.asarray(positions))
        self._actions.append(np.asarray(actions))
        self._rewards.append(np.asarray(rewards))
        self._alive_mask.append(np.asarray(alive_mask, dtype=bool))
        self._energy.append(np.asarray(energy))

        if births is not None:
            self._births.append(np.asarray(births, dtype=bool))

        if field_values is not None:
            self._field_values.append(np.asarray(field_values))

    @property
    def num_steps(self) -> int:
        """Number of steps recorded so far."""
        return len(self._actions)

    def get_trajectory(self) -> dict[str, np.ndarray]:
        """Return recorded data as a trajectory dict.

        Returns:
            Dict with keys:
                - 'actions': (T, max_agents) int
                - 'positions': (T, max_agents, 2) int
                - 'rewards': (T, max_agents) float
                - 'alive_mask': (T, max_agents) bool
                - 'energy': (T, max_agents) float
                - 'births': (T, max_agents) bool (if any births recorded)
                - 'field_values': (T, max_agents, ...) float (if any recorded)
        """
        if self.num_steps == 0:
            raise ValueError("No steps recorded. Call record_step() first.")

        trajectory: dict[str, np.ndarray] = {
            "actions": np.stack(self._actions, axis=0),
            "positions": np.stack(self._positions, axis=0),
            "rewards": np.stack(self._rewards, axis=0),
            "alive_mask": np.stack(self._alive_mask, axis=0),
            "energy": np.stack(self._energy, axis=0),
        }

        if self._births:
            trajectory["births"] = np.stack(self._births, axis=0)

        if self._field_values:
            trajectory["field_values"] = np.stack(self._field_values, axis=0)

        return trajectory


def record_episode(
    network: ActorCritic,
    params: dict[str, Any],
    config: Config,
    key: jax.Array | None = None,
    *,
    deterministic: bool = False,
) -> dict[str, np.ndarray]:
    """Record a full episode trajectory for analysis.

    Runs one episode using the given network and params, recording
    per-agent data at each step. The returned trajectory dict is
    compatible with ``extract_behavior_features()``.

    Args:
        network: ActorCritic network module.
        params: Network parameters (shared or per-agent).
        config: Master configuration object.
        key: PRNG key for environment reset and action sampling.
            If None, uses seed 0.
        deterministic: If True, use greedy (argmax) actions instead of
            sampling. Default False (stochastic for realistic behavior).

    Returns:
        Trajectory dict with keys:
            - 'actions': (T, max_agents)
            - 'positions': (T, max_agents, 2)
            - 'rewards': (T, max_agents)
            - 'alive_mask': (T, max_agents)
            - 'energy': (T, max_agents)
            - 'births': (T, max_agents)
            - 'field_values': (T, max_agents) — mean field value at position
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    key, reset_key = jax.random.split(key)
    state = reset(reset_key, config)
    max_agents = config.evolution.max_agents

    recorder = TrajectoryRecorder(max_agents)

    for _ in range(config.env.max_steps):
        # Get observations
        obs = get_observations(state, config)  # (max_agents, obs_dim)

        # Read field values at agent positions (mean across channels)
        field_at_pos = read_local(
            state.field_state, state.agent_positions, radius=0
        )  # (max_agents, C)
        field_mean = np.asarray(jnp.mean(field_at_pos, axis=-1))  # (max_agents,)

        # Add batch dimension for policy: (1, max_agents, obs_dim)
        obs_batched = obs[None, :, :]

        # Get actions
        if deterministic:
            from src.agents.policy import get_deterministic_actions

            actions = get_deterministic_actions(network, params, obs_batched)
            actions = actions[0]  # remove batch dim
        else:
            key, action_key = jax.random.split(key)
            actions, _log_probs, _values, _entropy = sample_actions(
                network, params, obs_batched, action_key
            )
            actions = actions[0]  # remove batch dim

        # Snapshot pre-step alive mask for recording
        pre_alive = state.agent_alive

        # Step environment
        state, rewards, done, info = step(state, actions, config)

        # Detect births: agents that are alive now but weren't before
        births = np.asarray(state.agent_alive & ~pre_alive)

        # Record step
        recorder.record_step(
            positions=state.agent_positions,
            actions=actions,
            rewards=rewards,
            alive_mask=pre_alive,
            energy=state.agent_energy,
            births=births,
            field_values=field_mean,
        )

        if bool(done):
            break

    return recorder.get_trajectory()
