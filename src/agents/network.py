"""Actor-Critic neural network for agent policies."""

from typing import Sequence

import jax.numpy as jnp
import flax.linen as nn


class ActorCritic(nn.Module):
    """MLP actor-critic network with shared backbone.

    Architecture:
        - Shared MLP backbone with LayerNorm + Tanh activations
        - Actor head: linear layer outputting logits for each action
        - Critic head: linear layer outputting scalar value estimate

    Initialization:
        - Hidden layers: orthogonal with scale sqrt(2)
        - Actor head: orthogonal with scale 0.01
        - Critic head: orthogonal with scale 1.0
    """

    hidden_dims: Sequence[int] = (64, 64)
    num_actions: int = 6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass.

        Args:
            x: Observation vector of shape (obs_dim,).

        Returns:
            Tuple of (action_logits, value):
                - action_logits: shape (num_actions,)
                - value: shape () scalar
        """
        # Shared backbone
        for dim in self.hidden_dims:
            x = nn.Dense(
                dim,
                kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)),
                bias_init=nn.initializers.zeros,
            )(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)

        # Actor head
        logits = nn.Dense(
            self.num_actions,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.zeros,
        )(x)

        # Critic head
        value = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.zeros,
        )(x)
        value = jnp.squeeze(value, axis=-1)

        return logits, value
