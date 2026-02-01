"""Specialization detection: weight divergence, behavioral clustering, and species detection."""

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import entropy as scipy_entropy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.configs import Config


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


def specialization_score(
    behavior_features: np.ndarray,
    agent_params: Any | None = None,
    alive_mask: np.ndarray | None = None,
    w_silhouette: float = 0.5,
    w_divergence: float = 0.25,
    w_variance: float = 0.25,
) -> dict[str, Any]:
    """Compute a single 0-1 specialization score for a population.

    Combines three signals:
    - Silhouette score from optimal clustering (how well-separated are clusters)
    - Weight divergence between agents (how different are their neural networks)
    - Behavioral variance (how spread out are feature vectors)

    All three components are normalized to [0, 1] and combined with weights.

    Args:
        behavior_features: Array of shape (num_agents, num_features).
        agent_params: Optional per-agent parameters pytree for weight divergence.
            If None, the weight divergence component is set to 0 and its weight
            is redistributed equally to silhouette and variance.
        alive_mask: Optional boolean mask for weight divergence computation.
        w_silhouette: Weight for silhouette component (default 0.5).
        w_divergence: Weight for weight divergence component (default 0.25).
        w_variance: Weight for behavioral variance component (default 0.25).

    Returns:
        Dict with keys:
            - 'score': Float in [0, 1]. 0 = all identical, 1 = fully specialized.
            - 'silhouette_component': Silhouette score mapped to [0, 1].
            - 'divergence_component': Mean weight divergence mapped to [0, 1].
            - 'variance_component': Behavioral variance mapped to [0, 1].
            - 'optimal_k': Number of clusters found.
    """
    features = np.asarray(behavior_features, dtype=np.float64)
    n_samples = features.shape[0]

    # --- Silhouette component ---
    # find_optimal_clusters returns silhouette in [-1, 1].
    # Negative values indicate mis-clustered data; clamp to [0, 1].
    # 0 = no meaningful clusters, 1 = perfectly separated.
    if n_samples < 2:
        sil_component = 0.0
        optimal_k = 1
    else:
        clustering = find_optimal_clusters(features)
        sil_component = float(np.clip(clustering["silhouette"], 0.0, 1.0))
        optimal_k = clustering["optimal_k"]

    # --- Weight divergence component ---
    if agent_params is not None:
        div_result = compute_weight_divergence(agent_params, alive_mask)
        # Cosine distance is in [0, 2]; normalize to [0, 1]
        div_component = float(np.clip(div_result["mean_divergence"] / 2.0, 0.0, 1.0))
    else:
        div_component = 0.0

    # --- Behavioral variance component ---
    # Use mean feature-wise variance, normalized via sigmoid-like mapping
    if n_samples < 2:
        var_component = 0.0
    else:
        # Standardize features first so variance is comparable across features
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        # Mean variance of standardized features is 1.0 for "normal" data
        # Use tanh to map to [0, 1]: tanh(mean_var) gives ~0.76 for mean_var=1.0
        mean_var = float(np.mean(np.var(scaled, axis=0)))
        var_component = float(np.tanh(mean_var))

    # --- Combine components ---
    if agent_params is None:
        # Redistribute divergence weight
        effective_w_sil = w_silhouette + w_divergence / 2.0
        effective_w_var = w_variance + w_divergence / 2.0
        score = effective_w_sil * sil_component + effective_w_var * var_component
    else:
        score = (
            w_silhouette * sil_component
            + w_divergence * div_component
            + w_variance * var_component
        )

    # Clamp to [0, 1]
    score = float(np.clip(score, 0.0, 1.0))

    return {
        "score": score,
        "silhouette_component": sil_component,
        "divergence_component": div_component,
        "variance_component": var_component,
        "optimal_k": optimal_k,
    }


def novelty_score(
    agent_features: np.ndarray,
    archive: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """Compute novelty scores using k-nearest neighbor distance in behavior space.

    Implements the novelty metric from Lehman & Stanley (2011):
        novelty(x) = (1/k) * sum(dist(x, neighbor_i) for i in range(k))

    Args:
        agent_features: Array of shape (num_agents, num_features) — current
            population's behavioral feature vectors.
        archive: Array of shape (archive_size, num_features) — previously
            observed behavioral feature vectors. Can include the current
            population as well.
        k: Number of nearest neighbors to use (default 5).

    Returns:
        Array of shape (num_agents,) with novelty scores. Higher = more novel.
        Returns zeros if archive has fewer than k entries.
    """
    agent_features = np.asarray(agent_features, dtype=np.float64)
    archive = np.asarray(archive, dtype=np.float64)

    n_agents = agent_features.shape[0]
    n_archive = archive.shape[0]

    if n_archive == 0 or k <= 0:
        return np.zeros(n_agents, dtype=np.float64)

    # Effective k: can't use more neighbors than available in archive
    effective_k = min(k, n_archive)

    # Compute pairwise Euclidean distances between agents and archive
    distances = cdist(agent_features, archive, metric="euclidean")

    # For each agent, find k-nearest neighbors in archive
    # np.partition is O(n) vs O(n log n) for full sort
    scores = np.zeros(n_agents, dtype=np.float64)
    for i in range(n_agents):
        d = distances[i]
        # Partition to find k smallest distances
        if effective_k >= len(d):
            knn_dists = d
        else:
            knn_indices = np.argpartition(d, effective_k)[:effective_k]
            knn_dists = d[knn_indices]
        scores[i] = float(np.mean(knn_dists))

    return scores


def analyze_field_usage(
    trajectories: dict[str, np.ndarray],
    cluster_labels: np.ndarray,
) -> dict[str, Any]:
    """Analyze how different behavioral clusters use the shared field.

    Computes per-cluster field usage statistics to identify "writers"
    (agents that contribute to field presence in new areas) vs "readers"
    (agents that navigate toward high-field areas).

    Since all alive agents write to the field automatically, the distinction
    comes from:
    - **Write patterns**: How much agents move (spreading field deposits) vs
      staying (concentrating deposits).
    - **Read patterns**: How strongly agent movement correlates with field
      values (do they follow the field?).

    Args:
        trajectories: Trajectory dict with keys:
            - 'actions': (T, num_agents) int
            - 'positions': (T, num_agents, 2) int
            - 'alive_mask': (T, num_agents) bool
            Required for full analysis:
            - 'field_values': (T, num_agents) float — field value at position
        cluster_labels: Cluster label per agent, shape (num_agents,).

    Returns:
        Dict with keys:
            - 'per_cluster': Dict mapping cluster_id -> {
                'write_frequency': float — fraction of steps agents are alive
                    (alive agents auto-write),
                'mean_field_value': float — mean field value at agent positions,
                'field_value_std': float — std of field values at positions,
                'movement_rate': float — fraction of steps with position change,
                'spatial_spread': float — mean unique positions per agent,
                'field_action_correlation': float — correlation between field
                    value and subsequent movement (higher = more field-guided),
              }
            - 'cluster_roles': Dict mapping cluster_id -> str
                ('writer', 'reader', or 'balanced') based on heuristics.
            - 'num_clusters': int — number of clusters analyzed.
    """
    actions = np.asarray(trajectories["actions"])           # (T, A)
    positions = np.asarray(trajectories["positions"])       # (T, A, 2)
    alive_mask = np.asarray(trajectories["alive_mask"])     # (T, A)
    has_field = "field_values" in trajectories
    if has_field:
        field_values = np.asarray(trajectories["field_values"])  # (T, A)

    cluster_labels = np.asarray(cluster_labels)
    num_agents = actions.shape[1]
    num_steps = actions.shape[0]

    unique_clusters = sorted(set(int(c) for c in cluster_labels))
    per_cluster: dict[int, dict[str, float]] = {}
    cluster_roles: dict[int, str] = {}

    for cid in unique_clusters:
        agent_indices = np.where(cluster_labels == cid)[0]

        # Collect per-agent stats, then average across cluster
        write_freqs: list[float] = []
        mean_fields: list[float] = []
        field_stds: list[float] = []
        movement_rates: list[float] = []
        spatial_spreads: list[float] = []
        field_action_corrs: list[float] = []

        for a in agent_indices:
            alive_steps = alive_mask[:, a].astype(bool)
            n_alive = int(np.sum(alive_steps))

            if n_alive == 0:
                write_freqs.append(0.0)
                mean_fields.append(0.0)
                field_stds.append(0.0)
                movement_rates.append(0.0)
                spatial_spreads.append(0.0)
                field_action_corrs.append(0.0)
                continue

            # Write frequency: fraction of total steps this agent is alive
            write_freqs.append(n_alive / num_steps)

            # Field values at agent positions (when alive)
            if has_field:
                agent_field = field_values[alive_steps, a]
                mean_fields.append(float(np.mean(agent_field)))
                field_stds.append(float(np.std(agent_field)))
            else:
                mean_fields.append(0.0)
                field_stds.append(0.0)

            # Movement rate: fraction of alive steps where position changed
            agent_pos = positions[:, a, :]  # (T, 2)
            alive_pos = agent_pos[alive_steps]  # (n_alive, 2)
            if n_alive >= 2:
                moved = np.any(np.diff(alive_pos, axis=0) != 0, axis=1)
                movement_rates.append(float(np.mean(moved)))
            else:
                movement_rates.append(0.0)

            # Spatial spread: unique positions / alive steps
            unique_pos = len(set(map(tuple, alive_pos)))
            spatial_spreads.append(unique_pos / n_alive)

            # Field-action correlation: do higher field values predict staying?
            # Correlation between field_value[t] and whether agent moved at step t+1
            if has_field and n_alive >= 3:
                agent_field_alive = field_values[alive_steps, a]
                # Did the agent move on the next alive step?
                moved_arr = np.concatenate(
                    [np.any(np.diff(alive_pos, axis=0) != 0, axis=1).astype(float),
                     [0.0]]
                )
                # Correlation between field value and movement
                if np.std(agent_field_alive) > 1e-10 and np.std(moved_arr) > 1e-10:
                    corr = float(np.corrcoef(agent_field_alive, moved_arr)[0, 1])
                    if np.isfinite(corr):
                        field_action_corrs.append(corr)
                    else:
                        field_action_corrs.append(0.0)
                else:
                    field_action_corrs.append(0.0)
            else:
                field_action_corrs.append(0.0)

        # Average across agents in cluster
        stats: dict[str, float] = {
            "write_frequency": float(np.mean(write_freqs)) if write_freqs else 0.0,
            "mean_field_value": float(np.mean(mean_fields)) if mean_fields else 0.0,
            "field_value_std": float(np.mean(field_stds)) if field_stds else 0.0,
            "movement_rate": float(np.mean(movement_rates)) if movement_rates else 0.0,
            "spatial_spread": float(np.mean(spatial_spreads)) if spatial_spreads else 0.0,
            "field_action_correlation": (
                float(np.mean(field_action_corrs)) if field_action_corrs else 0.0
            ),
        }
        per_cluster[cid] = stats

        # Classify cluster role based on heuristics
        cluster_roles[cid] = _classify_cluster_role(stats)

    return {
        "per_cluster": per_cluster,
        "cluster_roles": cluster_roles,
        "num_clusters": len(unique_clusters),
    }


def _classify_cluster_role(stats: dict[str, float]) -> str:
    """Classify a cluster as 'writer', 'reader', or 'balanced'.

    Heuristic based on movement rate and field correlation:
    - Writers: high movement (spread field deposits), low field dependence
    - Readers: low movement (exploit known areas), high field values
    - Balanced: neither extreme

    Args:
        stats: Per-cluster statistics dict from analyze_field_usage.

    Returns:
        One of 'writer', 'reader', or 'balanced'.
    """
    high_movement = stats["movement_rate"] > 0.6
    low_movement = stats["movement_rate"] < 0.4
    high_field = stats["mean_field_value"] > 0.5

    if high_movement and not high_field:
        return "writer"
    elif low_movement and high_field:
        return "reader"
    else:
        return "balanced"


@dataclass
class SpecializationEvent:
    """A detected specialization event (sudden divergence increase)."""

    step: int
    metric_name: str
    old_value: float
    new_value: float
    z_score: float

    def __str__(self) -> str:
        direction = "increase" if self.new_value > self.old_value else "decrease"
        return (
            f"[step {self.step}] Specialization event in {self.metric_name}: "
            f"{self.old_value:.4f} -> {self.new_value:.4f} "
            f"({direction}, z={self.z_score:.2f})"
        )


class SpecializationTracker:
    """Tracks specialization metrics during training and detects specialization events.

    Monitors weight divergence between agents over training iterations.
    Detects "specialization events" when divergence increases suddenly
    (beyond ``z_threshold`` standard deviations from the rolling mean).

    Attributes:
        config: Master configuration.
        history: Dict mapping metric name to lists of (step, value) pairs.
        events: List of detected SpecializationEvent instances.
        step_count: Number of update calls made.
    """

    def __init__(
        self,
        config: Config,
        window_size: int = 20,
        z_threshold: float = 3.0,
    ) -> None:
        self.config = config
        self.window_size = window_size
        self.z_threshold = z_threshold

        self.history: dict[str, list[float]] = {
            "weight_divergence": [],
            "max_divergence": [],
            "num_alive": [],
        }
        self.steps: list[int] = []
        self.events: list[SpecializationEvent] = []
        self.step_count: int = 0

    def update(
        self,
        agent_params: Any,
        alive_mask: np.ndarray | jnp.ndarray,
        step: int,
    ) -> list[SpecializationEvent]:
        """Compute specialization metrics and check for events.

        Args:
            agent_params: Per-agent parameters pytree where each leaf has
                leading dimension ``(max_agents, ...)``. Should be params
                for a single environment (not batched over envs).
            alive_mask: Boolean array of shape ``(max_agents,)`` indicating
                which agents are alive.
            step: Current training step.

        Returns:
            List of SpecializationEvent instances detected at this step
            (empty if no events detected).
        """
        alive_mask = np.asarray(alive_mask, dtype=bool)
        num_alive = int(np.sum(alive_mask))

        # Compute weight divergence
        div_result = compute_weight_divergence(agent_params, alive_mask)
        mean_div = div_result["mean_divergence"]
        max_div = div_result["max_divergence"]

        new_values = {
            "weight_divergence": mean_div,
            "max_divergence": max_div,
            "num_alive": float(num_alive),
        }

        new_events: list[SpecializationEvent] = []

        for name, value in new_values.items():
            hist = self.history[name]

            # Check for specialization event (need enough history)
            if len(hist) >= self.window_size:
                recent = hist[-self.window_size:]
                mean = float(np.mean(recent))
                std = float(np.std(recent))
                if std < 1e-8:
                    std = 1e-8
                z_score = abs(value - mean) / std

                if z_score > self.z_threshold:
                    event = SpecializationEvent(
                        step=step,
                        metric_name=name,
                        old_value=mean,
                        new_value=value,
                        z_score=z_score,
                    )
                    self.events.append(event)
                    new_events.append(event)

            hist.append(value)

        self.steps.append(step)
        self.step_count += 1
        return new_events

    def get_metrics(self) -> dict[str, float]:
        """Return current metric values for logging.

        Returns:
            Dict with keys like ``'specialization/weight_divergence'``,
            ``'specialization/max_divergence'``, etc.
        """
        metrics: dict[str, float] = {}
        for name, hist in self.history.items():
            if len(hist) > 0:
                metrics[f"specialization/{name}"] = hist[-1]
        metrics["specialization/num_events"] = float(len(self.events))
        return metrics

    def get_summary(self) -> dict[str, Any]:
        """Return a summary of specialization tracking.

        Returns:
            Dict with overall statistics and event list.
        """
        summary: dict[str, Any] = {
            "total_updates": self.step_count,
            "total_events": len(self.events),
            "events": [str(e) for e in self.events],
        }
        for name, hist in self.history.items():
            if len(hist) > 0:
                summary[f"{name}_final"] = hist[-1]
                summary[f"{name}_mean"] = float(np.mean(hist))
                summary[f"{name}_std"] = float(np.std(hist))
        return summary
