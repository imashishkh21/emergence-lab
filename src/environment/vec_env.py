"""Vectorized environment using jax.vmap for parallel simulation."""

from typing import Any

import jax
import jax.numpy as jnp

from src.configs import Config
from src.environment.env import reset, step
from src.environment.state import EnvState


class VecEnv:
    """Vectorized environment that runs multiple environments in parallel.

    Uses jax.vmap to parallelize reset and step across num_envs independent
    environments sharing the same config.

    Batch shapes are (num_envs, num_agents, ...) for all outputs.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.num_envs = config.train.num_envs

        # vmap reset over different PRNG keys, config is shared (not mapped)
        self._vmapped_reset = jax.vmap(
            lambda key: reset(key, self.config)
        )

        # vmap step over (state, actions), config is shared (not mapped)
        self._vmapped_step = jax.vmap(
            lambda state, actions: step(state, actions, self.config)
        )

    def reset(self, key: jax.Array) -> EnvState:
        """Reset all environments in parallel.

        Args:
            key: A single PRNG key that will be split into num_envs keys.

        Returns:
            Batched EnvState with leading dimension num_envs.
        """
        keys = jax.random.split(key, self.num_envs)
        return self._vmapped_reset(keys)

    def step(
        self, states: EnvState, actions: jnp.ndarray
    ) -> tuple[EnvState, jnp.ndarray, jnp.ndarray, dict[str, Any]]:
        """Step all environments in parallel.

        Args:
            states: Batched EnvState with leading dimension num_envs.
            actions: Actions for all envs, shape (num_envs, num_agents).

        Returns:
            Tuple of (new_states, rewards, dones, info) with leading num_envs dim.
        """
        return self._vmapped_step(states, actions)
