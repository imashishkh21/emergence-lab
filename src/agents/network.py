"""Actor-Critic neural network for agent policies."""

from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


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


class AgentSpecificActorCritic(nn.Module):
    """Actor-Critic with shared encoder and per-agent output heads.

    Architecture:
        - Shared MLP encoder (same weights for all agents)
        - Per-agent actor + critic heads (different weights per agent)
        - Forward pass selects head based on agent_id

    This allows agents to share perception (encoder) while developing
    individual decision-making strategies (heads) — like siblings with
    the same eyes but different personalities.

    All n_agents heads are evaluated, and the correct head's output is
    selected via array indexing (JIT-compatible). Gradients only flow
    through the selected head's parameters.
    """

    hidden_dims: Sequence[int] = (64, 64)
    num_actions: int = 6
    n_agents: int = 32

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, agent_id: jnp.ndarray | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass.

        Args:
            x: Observation vector of shape (obs_dim,).
            agent_id: Scalar integer identifying which agent head to use.
                If None, uses head 0 (backward compatible with ActorCritic).

        Returns:
            Tuple of (action_logits, value):
                - action_logits: shape (num_actions,)
                - value: shape () scalar
        """
        # Shared encoder backbone
        for dim in self.hidden_dims:
            x = nn.Dense(
                dim,
                kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)),
                bias_init=nn.initializers.zeros,
            )(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)

        # Per-agent actor heads: each agent gets its own Dense layer
        # Compute all heads, then select by agent_id
        all_logits = []
        all_values = []
        for i in range(self.n_agents):
            head_logits = nn.Dense(
                self.num_actions,
                kernel_init=nn.initializers.orthogonal(0.01),
                bias_init=nn.initializers.zeros,
                name=f"actor_head_{i}",
            )(x)
            head_value = nn.Dense(
                1,
                kernel_init=nn.initializers.orthogonal(1.0),
                bias_init=nn.initializers.zeros,
                name=f"critic_head_{i}",
            )(x)
            all_logits.append(head_logits)
            all_values.append(jnp.squeeze(head_value, axis=-1))

        # Stack: (n_agents, num_actions) and (n_agents,)
        stacked_logits = jnp.stack(all_logits, axis=0)
        stacked_values = jnp.stack(all_values, axis=0)

        if agent_id is None:
            # Default to head 0 for backward compatibility
            return stacked_logits[0], stacked_values[0]

        # Select the correct head via indexing (JIT-compatible)
        safe_id = jnp.clip(agent_id, 0, self.n_agents - 1).astype(jnp.int32)
        logits = stacked_logits[safe_id]
        value = stacked_values[safe_id]

        return logits, value
