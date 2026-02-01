"""FieldState dataclass and factory function for the shared field."""

import flax.struct
import jax.numpy as jnp


@flax.struct.dataclass
class FieldState:
    """State of the shared field.

    The field is a 2D grid with multiple channels that agents
    read from and write to. It has its own dynamics (diffusion, decay).

    Attributes:
        values: Field values with shape (H, W, C) where H=height,
                W=width, C=num_channels.
    """
    values: jnp.ndarray  # (H, W, C)


def create_field(height: int, width: int, channels: int) -> FieldState:
    """Create a new FieldState initialized to zeros.

    Args:
        height: Grid height.
        width: Grid width.
        channels: Number of field channels.

    Returns:
        Initialized FieldState with zero values.
    """
    values = jnp.zeros((height, width, channels), dtype=jnp.float32)
    return FieldState(values=values)
