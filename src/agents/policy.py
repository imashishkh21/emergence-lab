"""Action sampling and policy utilities for agent decision-making."""

import jax
import jax.numpy as jnp

from src.agents.network import ActorCritic


def sample_actions(
    network: ActorCritic,
    params: dict,
    obs: jnp.ndarray,
    key: jnp.ndarray,
    gate_bias: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample actions from the policy for all agents across all environments.

    Args:
        network: ActorCritic network module.
        params: Network parameters.
        obs: Observations of shape (num_envs, num_agents, obs_dim).
        key: PRNG key for sampling.
        gate_bias: Optional per-agent gate bias of shape (num_envs, num_agents, num_channels).
            Used when adaptive_gate is enabled to provide evolved gate preferences.

    Returns:
        Tuple of (actions, log_probs, values, entropy, gate):
            - actions: shape (num_envs, num_agents) integer actions
            - log_probs: shape (num_envs, num_agents) log probability of chosen actions
            - values: shape (num_envs, num_agents) critic value estimates
            - entropy: shape (num_envs, num_agents) policy entropy
            - gate: shape (num_envs, num_agents, num_channels) field gate values
    """
    num_envs, num_agents, obs_dim_val = obs.shape

    # Flatten envs and agents into a single batch dimension
    flat_obs = obs.reshape(num_envs * num_agents, obs_dim_val)

    # Flatten gate_bias if provided
    if gate_bias is not None:
        num_channels = gate_bias.shape[-1]
        flat_gate_bias = gate_bias.reshape(num_envs * num_agents, num_channels)
        # Vectorized forward pass with gate_bias
        batched_apply = jax.vmap(network.apply, in_axes=(None, 0, None, 0))
        out = batched_apply(params, flat_obs, None, flat_gate_bias)
    else:
        # Vectorized forward pass without gate_bias
        batched_apply = jax.vmap(network.apply, in_axes=(None, 0))
        out = batched_apply(params, flat_obs)

    logits = out[0]  # (num_envs * num_agents, num_actions)
    values = jnp.asarray(out[1])  # (num_envs * num_agents,)
    gate = jnp.asarray(out[2])  # (num_envs * num_agents, num_channels)

    # Categorical sampling
    # Compute log probabilities from logits
    log_probs_all = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(log_probs_all)

    # Sample actions
    keys = jax.random.split(key, num_envs * num_agents)
    actions = jax.vmap(lambda k, p: jax.random.categorical(k, jnp.log(p)))(
        keys, probs
    )

    # Gather log probs for chosen actions
    log_probs = jnp.take_along_axis(
        log_probs_all, actions[:, None], axis=-1
    ).squeeze(-1)

    # Compute entropy: -sum(p * log(p))
    entropy = -jnp.sum(probs * log_probs_all, axis=-1)

    # Reshape back to (num_envs, num_agents, ...)
    actions = actions.reshape(num_envs, num_agents)
    log_probs = log_probs.reshape(num_envs, num_agents)
    values = values.reshape(num_envs, num_agents)
    entropy = entropy.reshape(num_envs, num_agents)
    num_gate_channels = gate.shape[-1]
    gate = gate.reshape(num_envs, num_agents, num_gate_channels)

    return actions, log_probs, values, entropy, gate


def get_deterministic_actions(
    network: ActorCritic,
    params: dict,
    obs: jnp.ndarray,
    gate_bias: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Get deterministic (greedy) actions from the policy.

    Args:
        network: ActorCritic network module.
        params: Network parameters.
        obs: Observations of shape (num_envs, num_agents, obs_dim).
        gate_bias: Optional per-agent gate bias of shape (num_envs, num_agents, num_channels).

    Returns:
        Tuple of (actions, gate):
            - actions: shape (num_envs, num_agents) integer actions (argmax of logits)
            - gate: shape (num_envs, num_agents, num_channels) field gate values
    """
    num_envs, num_agents, obs_dim_val = obs.shape

    # Flatten envs and agents
    flat_obs = obs.reshape(num_envs * num_agents, obs_dim_val)

    # Flatten gate_bias if provided
    if gate_bias is not None:
        num_channels = gate_bias.shape[-1]
        flat_gate_bias = gate_bias.reshape(num_envs * num_agents, num_channels)
        batched_apply = jax.vmap(network.apply, in_axes=(None, 0, None, 0))
        out = batched_apply(params, flat_obs, None, flat_gate_bias)
    else:
        batched_apply = jax.vmap(network.apply, in_axes=(None, 0))
        out = batched_apply(params, flat_obs)

    logits: jnp.ndarray = out[0]
    gate: jnp.ndarray = jnp.asarray(out[2])

    # Greedy action selection
    actions = jnp.argmax(logits, axis=-1)

    num_gate_channels = gate.shape[-1]
    return (
        actions.reshape(num_envs, num_agents),
        gate.reshape(num_envs, num_agents, num_gate_channels),
    )
