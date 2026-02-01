"""Field analysis metrics for measuring emergence properties."""

import jax.numpy as jnp

from src.field.field import FieldState


def field_entropy(field: FieldState) -> jnp.ndarray:
    """Compute spatial entropy of the field.

    Treats the absolute field values as a probability distribution
    over spatial locations (after normalization). Higher entropy means
    more uniform distribution; lower entropy means concentrated values.

    Averages entropy across channels.

    Args:
        field: FieldState with values of shape (H, W, C).

    Returns:
        Scalar entropy value.
    """
    values = jnp.abs(field.values)  # (H, W, C)
    # Sum over spatial dims per channel to get normalization constants
    channel_sums = jnp.sum(values, axis=(0, 1), keepdims=True)  # (1, 1, C)
    # Avoid division by zero: if a channel is all zeros, use uniform
    channel_sums = jnp.where(channel_sums == 0, 1.0, channel_sums)
    # Normalize to get probability distribution per channel
    probs = values / channel_sums  # (H, W, C)
    # Compute entropy per channel: -sum(p * log(p))
    log_probs = jnp.where(probs > 0, jnp.log(probs + 1e-10), 0.0)
    entropy_per_channel = -jnp.sum(probs * log_probs, axis=(0, 1))  # (C,)
    # Average across channels
    return jnp.mean(entropy_per_channel)


def field_structure(field: FieldState) -> jnp.ndarray:
    """Measure spatial autocorrelation of the field.

    Computes the average correlation between each cell and its
    immediate neighbors (up, down, left, right). Higher values
    indicate more spatial structure (smooth gradients), lower
    values indicate random noise.

    Averages across channels.

    Args:
        field: FieldState with values of shape (H, W, C).

    Returns:
        Scalar autocorrelation value in [0, 1].
    """
    values = field.values  # (H, W, C)
    # Shifted versions for 4 cardinal neighbors
    up = jnp.roll(values, -1, axis=0)
    down = jnp.roll(values, 1, axis=0)
    left = jnp.roll(values, -1, axis=1)
    right = jnp.roll(values, 1, axis=1)

    # Mean of neighbor values
    neighbor_mean = (up + down + left + right) / 4.0  # (H, W, C)

    # Pearson correlation between values and neighbor mean, per channel
    def _channel_correlation(v: jnp.ndarray, n: jnp.ndarray) -> jnp.ndarray:
        """Compute correlation between flattened v and n."""
        v_flat = v.ravel()
        n_flat = n.ravel()
        v_mean = jnp.mean(v_flat)
        n_mean = jnp.mean(n_flat)
        v_centered = v_flat - v_mean
        n_centered = n_flat - n_mean
        numerator = jnp.sum(v_centered * n_centered)
        denominator = jnp.sqrt(jnp.sum(v_centered ** 2) * jnp.sum(n_centered ** 2) + 1e-10)
        return numerator / denominator

    # Compute per-channel correlation
    num_channels = values.shape[2]
    correlations = jnp.stack([
        _channel_correlation(values[:, :, c], neighbor_mean[:, :, c])
        for c in range(num_channels)
    ])
    # Average across channels, clamp to [0, 1]
    mean_corr = jnp.mean(correlations)
    return jnp.clip(mean_corr, 0.0, 1.0)


def field_food_mi(field: FieldState, food_positions: jnp.ndarray) -> jnp.ndarray:
    """Estimate mutual information between field values and food positions.

    Compares the mean field value at food positions vs. non-food positions.
    Uses a simple binning-based MI approximation:
      MI = H(field) + H(food_indicator) - H(field, food_indicator)

    where H is entropy computed over discretized bins.

    Args:
        field: FieldState with values of shape (H, W, C).
        food_positions: Array of shape (num_food, 2) with (row, col) positions.

    Returns:
        Scalar mutual information estimate (non-negative).
    """
    values = field.values  # (H, W, C)
    h, w, c = values.shape

    # Sum across channels to get a single scalar per cell
    field_sum = jnp.sum(values, axis=2)  # (H, W)

    # Discretize field values into bins
    num_bins = 10
    f_min = jnp.min(field_sum)
    f_max = jnp.max(field_sum)
    f_range = jnp.where(f_max - f_min == 0, 1.0, f_max - f_min)
    # Normalize to [0, 1]
    normalized = (field_sum - f_min) / f_range
    # Bin indices: [0, num_bins-1]
    bin_indices = jnp.clip((normalized * num_bins).astype(jnp.int32), 0, num_bins - 1)

    # Create food indicator grid: 1 at food positions, 0 elsewhere
    food_indicator = jnp.zeros((h, w), dtype=jnp.int32)
    food_rows = jnp.clip(food_positions[:, 0], 0, h - 1)
    food_cols = jnp.clip(food_positions[:, 1], 0, w - 1)
    food_indicator = food_indicator.at[food_rows, food_cols].set(1)

    # Flatten everything
    bins_flat = bin_indices.ravel()  # (H*W,)
    food_flat = food_indicator.ravel()  # (H*W,)
    total = h * w

    # H(field_bins): entropy of field bin distribution
    bin_counts = jnp.zeros(num_bins, dtype=jnp.float32)
    for b in range(num_bins):
        bin_counts = bin_counts.at[b].set(jnp.sum(bins_flat == b).astype(jnp.float32))
    bin_probs = bin_counts / total
    h_field = -jnp.sum(jnp.where(bin_probs > 0, bin_probs * jnp.log(bin_probs + 1e-10), 0.0))

    # H(food_indicator): entropy of food presence
    food_count = jnp.sum(food_flat).astype(jnp.float32)
    p_food = food_count / total
    p_nofood = 1.0 - p_food
    h_food = -jnp.where(p_food > 0, p_food * jnp.log(p_food + 1e-10), 0.0) - \
             jnp.where(p_nofood > 0, p_nofood * jnp.log(p_nofood + 1e-10), 0.0)

    # H(field_bins, food_indicator): joint entropy
    joint_counts = jnp.zeros((num_bins, 2), dtype=jnp.float32)
    for b in range(num_bins):
        mask = (bins_flat == b)
        joint_counts = joint_counts.at[b, 0].set(jnp.sum(mask & (food_flat == 0)).astype(jnp.float32))
        joint_counts = joint_counts.at[b, 1].set(jnp.sum(mask & (food_flat == 1)).astype(jnp.float32))
    joint_probs = joint_counts / total
    h_joint = -jnp.sum(jnp.where(joint_probs > 0, joint_probs * jnp.log(joint_probs + 1e-10), 0.0))

    # MI = H(field) + H(food) - H(field, food)
    mi = h_field + h_food - h_joint
    return jnp.clip(mi, 0.0)  # Ensure non-negative
