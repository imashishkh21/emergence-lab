"""PPO (Proximal Policy Optimization) loss function."""

import jax
import jax.numpy as jnp
import flax.linen as nn


def _masked_mean(x: jnp.ndarray, mask: jnp.ndarray | None) -> jnp.ndarray:
    """Compute mean of x, optionally weighted by a boolean mask.

    When mask is provided, computes sum(x * mask) / max(sum(mask), 1)
    so that dead-agent entries are excluded from the average.
    """
    if mask is None:
        return jnp.mean(x)
    mask_f = mask.astype(jnp.float32)
    return jnp.sum(x * mask_f) / jnp.maximum(jnp.sum(mask_f), 1.0)


def ppo_loss(
    network: nn.Module,
    params: dict,
    batch: dict[str, jnp.ndarray],
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Compute PPO clipped surrogate loss.

    If the batch contains an 'alive_mask' key, dead agents are excluded
    from all loss terms via masked averaging.

    Args:
        network: ActorCritic network module.
        params: Network parameters.
        batch: Dictionary containing:
            - obs: (batch_size, obs_dim)
            - actions: (batch_size,) integer actions
            - log_probs: (batch_size,) old log probabilities
            - advantages: (batch_size,)
            - returns: (batch_size,) target returns for value function
            - alive_mask: (batch_size,) optional boolean mask (True = alive)
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
    mask = batch.get('alive_mask', None)

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
    per_sample_entropy = -jnp.sum(probs * log_softmax, axis=-1)
    entropy = _masked_mean(per_sample_entropy, mask)

    # Policy loss: clipped surrogate objective
    log_ratio = new_log_probs - old_log_probs
    ratio = jnp.exp(log_ratio)

    # Clipped surrogate
    surr1 = ratio * advantages
    surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -_masked_mean(jnp.minimum(surr1, surr2), mask)

    # Value loss: MSE
    value_loss = _masked_mean(jnp.square(values - returns), mask)

    # Approximate KL divergence (for monitoring)
    approx_kl = _masked_mean((ratio - 1.0) - log_ratio, mask)

    # Clip fraction (for monitoring)
    clip_fraction = _masked_mean((jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32), mask)

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
