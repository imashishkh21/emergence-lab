"""Actor-Critic neural network for agent policies."""

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp


class ActorCritic(nn.Module):
    """MLP actor-critic network with shared backbone.

    Architecture:
        - Optional adaptive field gate (modulates field observation per channel)
        - Optional agent identity embedding (concatenated to observation)
        - Shared MLP backbone with LayerNorm + Tanh activations
        - Actor head: linear layer outputting logits for each action
        - Critic head: linear layer outputting scalar value estimate
        - Optional gate head: outputs per-channel field gate values

    Initialization:
        - Hidden layers: orthogonal with scale sqrt(2)
        - Actor head: orthogonal with scale 0.01
        - Critic head: orthogonal with scale 1.0
        - Gate head: orthogonal with scale 0.1
    """

    hidden_dims: Sequence[int] = (64, 64)
    num_actions: int = 5
    agent_embed_dim: int = 0
    n_agents: int = 32
    adaptive_gate: bool = False
    num_field_channels: int = 4
    evolutionary_gate_only: bool = False

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        agent_id: jnp.ndarray | None = None,
        gate_bias: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward pass.

        Args:
            x: Observation vector of shape (obs_dim,).
            agent_id: Optional scalar integer for agent identity embedding.
                Only used when agent_embed_dim > 0.
            gate_bias: Optional per-agent gate bias of shape (num_field_channels,).
                Added to gate logits before sigmoid. Used for evolved gate preferences.

        Returns:
            Tuple of (action_logits, value, gate):
                - action_logits: shape (num_actions,)
                - value: shape () scalar
                - gate: shape (num_field_channels,) sigmoid gate values in [0, 1]
        """
        # Adaptive field gate: compute gate from non-field observations
        # and apply to field portion before backbone processing.
        # Obs structure: pos(2) + energy(1) + has_food(1) + compass(2) +
        #                field_spatial(5*C) + field_temporal(C) + food(15)
        if self.adaptive_gate:
            c = self.num_field_channels
            non_field_end = 6  # pos + energy + has_food + compass
            field_spatial_end = non_field_end + 5 * c
            field_temporal_end = field_spatial_end + c

            # Split observation
            non_field_pre = x[..., :non_field_end]
            field_spatial = x[..., non_field_end:field_spatial_end]
            field_temporal = x[..., field_spatial_end:field_temporal_end]
            food_obs = x[..., field_temporal_end:]

            if self.evolutionary_gate_only and gate_bias is not None:
                # Evolution-only gate: skip Dense head, use evolved bias directly
                gate = nn.sigmoid(gate_bias)  # (num_field_channels,)
            else:
                # Compute gate from non-field observations (avoids circularity)
                gate_input = jnp.concatenate([non_field_pre, food_obs], axis=-1)
                gate_hidden = nn.Dense(
                    16,
                    kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)),
                    bias_init=nn.initializers.zeros,
                    name="gate_hidden",
                )(gate_input)
                gate_hidden = nn.tanh(gate_hidden)
                gate_logits = nn.Dense(
                    c,
                    kernel_init=nn.initializers.orthogonal(0.1),
                    bias_init=nn.initializers.zeros,
                    name="gate_head",
                )(gate_hidden)

                # Add per-agent bias if provided (for evolved gate preferences)
                if gate_bias is not None:
                    gate_logits = gate_logits + gate_bias

                gate = nn.sigmoid(gate_logits)  # (num_field_channels,)

            # Apply gate per-channel using tile (handles any batch shape)
            # field_spatial layout: [N0,S0,E0,W0,C0, N1,S1,E1,W1,C1, ...]
            # gate shape: (c,) or (batch, c) -> tile to (5*c,) or (batch, 5*c)
            gate_tiled = jnp.tile(gate, 5)  # (5*c,)
            gated_spatial = field_spatial * gate_tiled
            gated_temporal = field_temporal * gate

            # Reconstruct observation with gated field
            x = jnp.concatenate(
                [non_field_pre, gated_spatial, gated_temporal, food_obs], axis=-1
            )
        else:
            # No gating: return dummy zeros for gate
            gate = jnp.zeros(self.num_field_channels)

        # Agent identity embedding (optional)
        if self.agent_embed_dim > 0 and agent_id is not None:
            embedding = nn.Embed(
                num_embeddings=self.n_agents,
                features=self.agent_embed_dim,
                name="agent_embedding",
            )(agent_id)
            x = jnp.concatenate([x, embedding], axis=-1)

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

        return logits, value, gate


class AgentSpecificActorCritic(nn.Module):
    """Actor-Critic with shared encoder and per-agent output heads.

    Architecture:
        - Optional agent identity embedding (concatenated to observation)
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
    num_actions: int = 5
    n_agents: int = 32
    agent_embed_dim: int = 0

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, agent_id: jnp.ndarray | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass.

        Args:
            x: Observation vector of shape (obs_dim,).
            agent_id: Scalar integer identifying which agent head to use
                and (when agent_embed_dim > 0) which embedding to look up.
                If None, uses head 0 (backward compatible with ActorCritic).

        Returns:
            Tuple of (action_logits, value):
                - action_logits: shape (num_actions,)
                - value: shape () scalar
        """
        # Agent identity embedding (optional)
        if self.agent_embed_dim > 0 and agent_id is not None:
            safe_embed_id = jnp.clip(
                agent_id, 0, self.n_agents - 1
            ).astype(jnp.int32)
            embedding = nn.Embed(
                num_embeddings=self.n_agents,
                features=self.agent_embed_dim,
                name="agent_embedding",
            )(safe_embed_id)
            x = jnp.concatenate([x, embedding], axis=-1)

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
