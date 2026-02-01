"""Field read/write operations for agent-field interaction."""

import jax
import jax.numpy as jnp

from src.field.field import FieldState


def write_local(field: FieldState, positions: jnp.ndarray, values: jnp.ndarray) -> FieldState:
    """Write values to the field at agent positions.

    Adds the given values at each agent's (row, col) position.

    Args:
        field: Current field state.
        positions: Agent positions with shape (N, 2) as (row, col) integers.
        values: Values to write with shape (N, C) where C = num_channels.

    Returns:
        New FieldState with values added at agent positions.
    """
    new_values = field.values.at[positions[:, 0], positions[:, 1]].add(values)
    return FieldState(values=new_values)


def read_local(field: FieldState, positions: jnp.ndarray, radius: int) -> jnp.ndarray:
    """Read local field values around each agent position.

    For radius=0, returns the field values at each position: shape (N, C).
    For radius>0, returns a flattened (2*radius+1)x(2*radius+1) neighborhood
    per agent: shape (N, (2*radius+1)^2 * C).

    Out-of-bounds reads are clamped to the field edges.

    Args:
        field: Current field state.
        positions: Agent positions with shape (N, 2) as (row, col) integers.
        radius: Read radius. 0 means just the cell, r>0 means (2r+1)x(2r+1) patch.

    Returns:
        Local field values. Shape depends on radius.
    """
    h, w, _c = field.values.shape

    if radius == 0:
        return field.values[positions[:, 0], positions[:, 1]]

    # Build offset grid for the neighborhood
    offsets = jnp.arange(-radius, radius + 1)
    # (patch_size, patch_size, 2) grid of (dr, dc) offsets
    dr, dc = jnp.meshgrid(offsets, offsets, indexing='ij')
    dr = dr.ravel()  # (patch_size^2,)
    dc = dc.ravel()  # (patch_size^2,)

    # Compute absolute positions for each agent and each offset
    # positions: (N, 2), dr/dc: (P,) where P = (2*radius+1)^2
    rows = positions[:, 0:1] + dr[None, :]  # (N, P)
    cols = positions[:, 1:2] + dc[None, :]  # (N, P)

    # Clamp to valid range
    rows = jnp.clip(rows, 0, h - 1)
    cols = jnp.clip(cols, 0, w - 1)

    # Gather values: field.values[rows, cols] -> (N, P, C)
    patch_values = field.values[rows, cols]  # (N, P, C)

    # Flatten to (N, P * C)
    n = positions.shape[0]
    return patch_values.reshape(n, -1)
