"""PPO (Proximal Policy Optimization) loss function."""

import jax
import jax.numpy as jnp
import flax.linen as nn


def ppo_loss(
    network: nn.Module,
    params: dict,
    batch: dict[str, jnp.ndarray],
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Compute PPO clipped surrogate loss.

    Args:
        network: ActorCritic network module.
        params: Network parameters.
        batch: Dictionary containing:
            - obs: (batch_size, obs_dim)
            - actions: (batch_size,) integer actions
            - log_probs: (batch_size,) old log probabilities
            - advantages: (batch_size,)
            - returns: (batch_size,) target returns for value function
        clip_eps: PPO clipping epsilon.
        vf_coef: Value function loss coefficient.
        ent_coef: Entropy bonus coefficient.

    Returns:
        (loss, metrics) where loss is scalar and metrics is a dict of scalars.
    """
    obs = batch['obs']
    actions = batch['actions']
    old_log_probs = batch['log_probs']
    advantages = batch['advantages']
    returns = batch['returns']

    # Forward pass
    out = network.apply(params, obs)
    logits = out[0]
    values = jnp.asarray(out[1])

    # Compute new log probabilities and entropy from categorical distribution
    log_softmax = jax.nn.log_softmax(logits, axis=-1)
    new_log_probs = jnp.take_along_axis(
        log_softmax, actions[..., None].astype(jnp.int32), axis=-1
    ).squeeze(-1)

    # Entropy: -sum(p * log(p))
    probs = jax.nn.softmax(logits, axis=-1)
    entropy = -jnp.sum(probs * log_softmax, axis=-1).mean()

    # Policy loss: clipped surrogate objective
    log_ratio = new_log_probs - old_log_probs
    ratio = jnp.exp(log_ratio)

    # Clipped surrogate
    surr1 = ratio * advantages
    surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -jnp.minimum(surr1, surr2).mean()

    # Value loss: MSE
    value_loss = jnp.square(values - returns).mean()

    # Approximate KL divergence (for monitoring)
    approx_kl = ((ratio - 1.0) - log_ratio).mean()

    # Clip fraction (for monitoring)
    clip_fraction = jnp.mean(jnp.abs(ratio - 1.0) > clip_eps)

    # Total loss
    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

    metrics = {
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'entropy': entropy,
        'approx_kl': approx_kl,
        'clip_fraction': clip_fraction,
    }

    return loss, metrics
