"""Specialization detection: weight divergence, behavioral clustering, and species detection."""

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def flatten_agent_params(agent_params: Any, agent_idx: int) -> np.ndarray:
    """Extract and flatten one agent's parameters into a 1D vector.

    Args:
        agent_params: Per-agent parameters pytree where each leaf has
            leading dimension (max_agents, ...).
        agent_idx: Index of the agent to extract.

    Returns:
        1D numpy array of all concatenated weight values for that agent.
    """
    leaves = jax.tree_util.tree_leaves(agent_params)
    flat_parts = [np.asarray(leaf[agent_idx]).ravel() for leaf in leaves]
    return np.concatenate(flat_parts)


def compute_weight_divergence(
    agent_params: Any,
    alive_mask: np.ndarray | jnp.ndarray | None = None,
) -> dict[str, Any]:
    """Compute pairwise weight divergence between agents.

    Measures how different agents' neural network weights have become
    using cosine distance between flattened weight vectors.

    Args:
        agent_params: Per-agent parameters pytree where each leaf has
            leading dimension (max_agents, ...).
        alive_mask: Boolean array of shape (max_agents,) indicating
            which agents are alive. If None, all agents are assumed alive.

    Returns:
        Dict with keys:
            - 'mean_divergence': Mean pairwise cosine distance.
            - 'max_divergence': Maximum pairwise cosine distance.
            - 'divergence_matrix': Full pairwise cosine distance matrix
              of shape (n_alive, n_alive).
            - 'agent_indices': Indices of alive agents used.
    """
    leaves = jax.tree_util.tree_leaves(agent_params)
    max_agents = leaves[0].shape[0]

    if alive_mask is None:
        alive_mask = np.ones(max_agents, dtype=bool)
    alive_mask = np.asarray(alive_mask, dtype=bool)

    agent_indices = np.where(alive_mask)[0]
    n_alive = len(agent_indices)

    if n_alive < 2:
        return {
            "mean_divergence": 0.0,
            "max_divergence": 0.0,
            "divergence_matrix": np.zeros((n_alive, n_alive)),
            "agent_indices": agent_indices,
        }

    # Flatten all alive agents' params into weight vectors
    weight_vectors = np.array(
        [flatten_agent_params(agent_params, idx) for idx in agent_indices]
    )

    # Compute pairwise cosine distance
    # cosine_distance = 1 - cosine_similarity
    norms = np.linalg.norm(weight_vectors, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.maximum(norms, 1e-8)
    normalized = weight_vectors / norms

    # Cosine similarity matrix
    similarity_matrix = normalized @ normalized.T
    # Clip to [-1, 1] for numerical stability
    similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)

    # Cosine distance
    divergence_matrix = 1.0 - similarity_matrix

    # Extract upper triangle (excluding diagonal) for summary stats
    triu_indices = np.triu_indices(n_alive, k=1)
    pairwise_distances = divergence_matrix[triu_indices]

    return {
        "mean_divergence": float(np.mean(pairwise_distances)),
        "max_divergence": float(np.max(pairwise_distances)),
        "divergence_matrix": divergence_matrix,
        "agent_indices": agent_indices,
    }
