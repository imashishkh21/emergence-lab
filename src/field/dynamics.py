"""Field dynamics: diffusion and decay for the shared field."""

import jax.numpy as jnp

from src.field.field import FieldState


def diffuse(field: FieldState, rate: float) -> FieldState:
    """Apply diffusion to the field using a 3x3 Gaussian-like blur.

    Blends the original field with a blurred version controlled by rate.
    rate=0 means no diffusion, rate=1 means full blur.

    Args:
        field: Current field state.
        rate: Diffusion rate in [0, 1].

    Returns:
        New FieldState with diffused values.
    """
    values = field.values  # (H, W, C)

    # 3x3 Gaussian kernel (unnormalized)
    kernel = jnp.array([
        [1.0, 2.0, 1.0],
        [2.0, 4.0, 2.0],
        [1.0, 2.0, 1.0],
    ]) / 16.0  # Normalize to sum=1

    # Apply convolution per channel using depthwise approach
    # Pad the spatial dims with 'same' padding (reflect to conserve mass better)
    padded = jnp.pad(values, ((1, 1), (1, 1), (0, 0)), mode='edge')

    # Manual 3x3 convolution via shifted sums
    blurred = jnp.zeros_like(values)
    for di in range(3):
        for dj in range(3):
            blurred = blurred + kernel[di, dj] * padded[di:di + values.shape[0],
                                                         dj:dj + values.shape[1], :]

    # Blend original with blurred
    new_values = (1.0 - rate) * values + rate * blurred
    return FieldState(values=new_values)


def decay(field: FieldState, rate: float) -> FieldState:
    """Apply exponential decay to field values.

    Args:
        field: Current field state.
        rate: Decay rate in [0, 1]. Values are multiplied by (1 - rate).

    Returns:
        New FieldState with decayed values.
    """
    new_values = field.values * (1.0 - rate)
    return FieldState(values=new_values)


def step_field(field: FieldState, diffusion_rate: float, decay_rate: float) -> FieldState:
    """Apply one timestep of field dynamics: diffusion then decay.

    Args:
        field: Current field state.
        diffusion_rate: Rate for diffusion blur.
        decay_rate: Rate for value decay.

    Returns:
        New FieldState after dynamics step.
    """
    field = diffuse(field, diffusion_rate)
    field = decay(field, decay_rate)
    return field
