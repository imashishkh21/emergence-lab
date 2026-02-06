"""Rollout collection for PPO training."""

from typing import Any

import flax.struct
import jax
import jax.numpy as jnp

from src.agents.network import ActorCritic
from src.agents.policy import sample_actions
from src.configs import Config
from src.environment.obs import get_observations
from src.environment.state import EnvState
from src.environment.vec_env import VecEnv


@flax.struct.dataclass
class RunnerState:
    """State carried across rollout steps and training iterations.

    Attributes:
        params: Network parameters (pytree).
        opt_state: Optimizer state (pytree).
        env_state: Batched environment state (num_envs, ...).
        last_obs: Last observations, shape (num_envs, num_agents, obs_dim).
        key: PRNG key.
    """
    params: Any
    opt_state: Any
    env_state: EnvState
    last_obs: jnp.ndarray
    key: jax.Array


def collect_rollout(
    runner_state: RunnerState,
    network: ActorCritic,
    vec_env: VecEnv,
    config: Config,
) -> tuple[RunnerState, dict[str, jnp.ndarray]]:
    """Collect a rollout of num_steps transitions from the vectorized environment.

    Uses jax.lax.scan for efficient, JIT-compatible trajectory collection.

    Args:
        runner_state: Current runner state with params, env_state, etc.
        network: ActorCritic network module.
        vec_env: Vectorized environment.
        config: Master configuration.

    Returns:
        Tuple of (new_runner_state, batch) where batch is a dict containing:
            - obs: (num_steps, num_envs, max_agents, obs_dim)
            - actions: (num_steps, num_envs, max_agents)
            - rewards: (num_steps, num_envs, max_agents)
            - dones: (num_steps, num_envs)
            - values: (num_steps, num_envs, max_agents)
            - log_probs: (num_steps, num_envs, max_agents)
            - alive_mask: (num_steps, num_envs, max_agents)
    """
    num_steps = config.train.num_steps

    def _step_fn(
        carry: RunnerState,
        _unused: None,
    ) -> tuple[RunnerState, dict[str, jnp.ndarray]]:
        rs = carry

        # Sample actions from current observations
        key, action_key = jax.random.split(rs.key)

        # Pass gate_bias if adaptive gate is enabled
        gate_bias = rs.env_state.agent_gate_bias  # None or (num_envs, max_agents, C)
        actions, log_probs, values, _entropy, gate = sample_actions(
            network, rs.params, rs.last_obs, action_key, gate_bias
        )

        # Step the environment
        env_state, rewards, dones, info = vec_env.step(rs.env_state, actions)

        # Get new observations
        new_obs = jax.vmap(lambda s: get_observations(s, config))(env_state)

        # Store transition data (including alive mask for dead agent masking)
        # alive_mask is from the state *before* the step, matching the obs
        alive_mask = rs.env_state.agent_alive  # (num_envs, max_agents)
        transition = {
            'obs': rs.last_obs,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'values': values,
            'log_probs': log_probs,
            'alive_mask': alive_mask,
            'births_this_step': info['births_this_step'],  # (num_envs,)
            'deaths_this_step': info['deaths_this_step'],  # (num_envs,)
            'gate': gate,  # (num_envs, max_agents, num_channels)
        }

        # Update runner state
        new_rs = RunnerState(
            params=rs.params,
            opt_state=rs.opt_state,
            env_state=env_state,
            last_obs=new_obs,
            key=key,
        )

        return new_rs, transition

    # Scan over num_steps
    new_runner_state, batch = jax.lax.scan(
        _step_fn,
        runner_state,
        None,
        length=num_steps,
    )

    return new_runner_state, batch
