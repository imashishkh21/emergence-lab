"""Specialization detection: weight divergence, behavioral clustering, and species detection."""

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import entropy as scipy_entropy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


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


def _movement_entropy(actions: np.ndarray, num_actions: int = 6) -> float:
    """Compute entropy of an agent's action distribution.

    Higher entropy = more random/exploratory movement.
    Lower entropy = more deterministic/specialized behavior.

    Args:
        actions: 1D array of integer actions taken by one agent.
        num_actions: Total number of possible actions.

    Returns:
        Normalized entropy in [0, 1]. 0 = always same action, 1 = uniform.
    """
    if len(actions) == 0:
        return 0.0
    counts = np.bincount(actions.astype(int), minlength=num_actions)
    probs = counts / counts.sum()
    max_entropy = np.log(num_actions)
    if max_entropy == 0:
        return 0.0
    return float(scipy_entropy(probs) / max_entropy)


def _distance_traveled(positions: np.ndarray) -> float:
    """Compute total Manhattan distance traveled by an agent.

    Args:
        positions: Array of shape (T, 2) with (row, col) positions per step.

    Returns:
        Total distance traveled (sum of step-to-step Manhattan distances).
    """
    if len(positions) < 2:
        return 0.0
    diffs = np.abs(np.diff(positions, axis=0))
    return float(np.sum(diffs))


def _exploration_ratio(positions: np.ndarray) -> float:
    """Compute ratio of unique cells visited to total steps.

    Args:
        positions: Array of shape (T, 2) with (row, col) positions per step.

    Returns:
        Ratio in [0, 1]. 1 = every step visits a new cell.
    """
    if len(positions) == 0:
        return 0.0
    unique_cells = len(set(map(tuple, positions)))
    return float(unique_cells / len(positions))


def extract_behavior_features(trajectory: dict[str, np.ndarray]) -> np.ndarray:
    """Extract behavioral feature vectors from agent trajectory data.

    Takes a trajectory dict (per-agent data over time) and produces a
    feature vector per agent suitable for clustering.

    Args:
        trajectory: Dict with the following keys (all numpy arrays):
            - 'actions': (T, num_agents) int actions per step
            - 'positions': (T, num_agents, 2) agent positions per step
            - 'rewards': (T, num_agents) rewards per step
            - 'alive_mask': (T, num_agents) bool, whether agent was alive
            - 'energy': (T, num_agents) energy levels per step
            Optional:
            - 'births': (T, num_agents) bool, whether agent reproduced
            - 'field_values': (T, num_agents) float, field value at agent pos

    Returns:
        Feature array of shape (num_agents, num_features) where features are:
            [0] movement_entropy      - action distribution entropy [0, 1]
            [1] food_collection_rate  - food per alive step
            [2] distance_per_step     - avg Manhattan distance per alive step
            [3] reproduction_rate     - reproductions per 100 alive steps
            [4] mean_energy           - average energy while alive
            [5] exploration_ratio     - unique cells / total steps
            [6] action_stay_fraction  - fraction of stay actions (action 0 or 5)
    """
    actions = np.asarray(trajectory["actions"])       # (T, A)
    positions = np.asarray(trajectory["positions"])    # (T, A, 2)
    rewards = np.asarray(trajectory["rewards"])        # (T, A)
    alive_mask = np.asarray(trajectory["alive_mask"])  # (T, A)
    energy = np.asarray(trajectory["energy"])          # (T, A)

    num_steps, num_agents = actions.shape
    num_features = 7

    features = np.zeros((num_agents, num_features), dtype=np.float64)

    for a in range(num_agents):
        alive_steps = alive_mask[:, a].astype(bool)
        n_alive = int(np.sum(alive_steps))

        if n_alive == 0:
            # Dead the whole time — all zeros
            continue

        agent_actions = actions[alive_steps, a]
        agent_positions = positions[alive_steps, a]
        agent_rewards = rewards[alive_steps, a]
        agent_energy = energy[alive_steps, a]

        # [0] Movement entropy
        features[a, 0] = _movement_entropy(agent_actions)

        # [1] Food collection rate (reward is proportional to food collected)
        features[a, 1] = float(np.sum(agent_rewards)) / n_alive

        # [2] Distance per step
        total_dist = _distance_traveled(agent_positions)
        features[a, 2] = total_dist / n_alive

        # [3] Reproduction rate (per 100 steps)
        if "births" in trajectory:
            births = np.asarray(trajectory["births"])
            agent_births = births[alive_steps, a]
            features[a, 3] = float(np.sum(agent_births)) / n_alive * 100.0
        else:
            # Infer from action 5 (reproduce)
            reproduce_actions = (agent_actions == 5).sum()
            features[a, 3] = float(reproduce_actions) / n_alive * 100.0

        # [4] Mean energy
        features[a, 4] = float(np.mean(agent_energy))

        # [5] Exploration ratio
        features[a, 5] = _exploration_ratio(agent_positions)

        # [6] Stay fraction (actions 0=stay or 5=reproduce, both don't move)
        stay_actions = ((agent_actions == 0) | (agent_actions == 5)).sum()
        features[a, 6] = float(stay_actions) / n_alive

    return features


def cluster_agents(
    behavior_features: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
) -> dict[str, Any]:
    """Cluster agents by behavioral features using K-means.

    Features are standardized (zero mean, unit variance) before clustering
    to ensure all features contribute equally regardless of scale.

    Args:
        behavior_features: Array of shape (num_agents, num_features).
        n_clusters: Number of clusters to form.
        random_state: Random seed for K-means reproducibility.

    Returns:
        Dict with keys:
            - 'labels': Cluster label per agent, shape (num_agents,).
            - 'centroids': Cluster centers in standardized space,
              shape (n_clusters, num_features).
            - 'silhouette': Silhouette score in [-1, 1]. Higher = better
              separated clusters. Returns 0.0 if clustering is degenerate
              (fewer unique points than clusters, or only 1 effective cluster).
            - 'n_clusters': Actual number of clusters used.
    """
    features = np.asarray(behavior_features, dtype=np.float64)
    n_samples, n_features = features.shape

    # Clamp n_clusters to number of unique sample points
    unique_rows = np.unique(features, axis=0)
    effective_k = min(n_clusters, len(unique_rows))

    if effective_k < 2:
        # Degenerate: all identical or single agent — one cluster
        return {
            "labels": np.zeros(n_samples, dtype=int),
            "centroids": features[:1].copy() if n_samples > 0 else np.empty((0, n_features)),
            "silhouette": 0.0,
            "n_clusters": 1,
        }

    # Standardize features
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    km = KMeans(n_clusters=effective_k, random_state=random_state, n_init=10)
    labels = km.fit_predict(scaled)

    # Silhouette requires at least 2 distinct labels AND n_samples > n_labels
    n_unique_labels = len(set(labels))
    if n_unique_labels < 2 or n_samples <= n_unique_labels:
        sil = 0.0
    else:
        sil = float(silhouette_score(scaled, labels))

    return {
        "labels": labels,
        "centroids": km.cluster_centers_,
        "silhouette": sil,
        "n_clusters": effective_k,
    }


def find_optimal_clusters(
    behavior_features: np.ndarray,
    max_k: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Find the optimal number of behavioral clusters using silhouette score.

    Tries k=2..max_k and picks the k with the highest silhouette score.
    If the data has fewer than 3 unique points, returns k=1 (or k=2 if
    exactly 2 unique points).

    Args:
        behavior_features: Array of shape (num_agents, num_features).
        max_k: Maximum number of clusters to try (inclusive).

    Returns:
        Dict with keys:
            - 'optimal_k': Best number of clusters.
            - 'labels': Cluster labels at optimal k, shape (num_agents,).
            - 'centroids': Cluster centers at optimal k.
            - 'silhouette': Best silhouette score.
            - 'silhouette_scores': Dict mapping k -> silhouette score for all
              tested k values.
    """
    features = np.asarray(behavior_features, dtype=np.float64)
    n_samples = features.shape[0]

    unique_rows = np.unique(features, axis=0)
    n_unique = len(unique_rows)

    # Need at least 2 unique points and n_samples >= 2 for meaningful clustering
    if n_unique < 2 or n_samples < 2:
        result = cluster_agents(features, n_clusters=1, random_state=random_state)
        return {
            "optimal_k": 1,
            "labels": result["labels"],
            "centroids": result["centroids"],
            "silhouette": 0.0,
            "silhouette_scores": {},
        }

    # Try k from 2 to min(max_k, n_unique)
    upper_k = min(max_k, n_unique)
    silhouette_scores: dict[int, float] = {}
    best_k = 2
    best_sil = -1.0
    best_result: dict[str, Any] | None = None

    for k in range(2, upper_k + 1):
        result = cluster_agents(features, n_clusters=k, random_state=random_state)
        sil = result["silhouette"]
        silhouette_scores[k] = sil
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_result = result

    assert best_result is not None
    return {
        "optimal_k": best_k,
        "labels": best_result["labels"],
        "centroids": best_result["centroids"],
        "silhouette": best_sil,
        "silhouette_scores": silhouette_scores,
    }
