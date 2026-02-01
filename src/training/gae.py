"""Generalized Advantage Estimation (GAE) for PPO training."""

import jax
import jax.numpy as jnp


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Generalized Advantage Estimation.

    Args:
        rewards: shape (T,) — rewards at each timestep
        values: shape (T+1,) — value estimates including bootstrap value
        dones: shape (T,) — episode termination flags (True = done)
        gamma: discount factor
        gae_lambda: GAE lambda for bias-variance tradeoff

    Returns:
        advantages: shape (T,)
        returns: shape (T,) — advantages + values[:T]
    """
    T = rewards.shape[0]  # noqa: N806

    # TD residuals: delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
    not_dones = 1.0 - dones.astype(jnp.float32)
    deltas = rewards + gamma * values[1:] * not_dones - values[:T]

    def _scan_fn(
        gae: jnp.ndarray,
        inp: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        delta, not_done = inp
        gae = delta + gamma * gae_lambda * not_done * gae
        return gae, gae

    # Reverse scan: iterate from T-1 down to 0
    _, advantages = jax.lax.scan(
        _scan_fn,
        jnp.zeros(()),  # initial carry: gae = 0
        (deltas, not_dones),
        reverse=True,
    )

    returns = advantages + values[:T]
    return advantages, returns
