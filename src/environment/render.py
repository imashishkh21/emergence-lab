"""Environment rendering: produces RGB images of the simulation state."""

import numpy as np
import jax.numpy as jnp

from src.configs import Config
from src.environment.state import EnvState


# Agent colors (up to 16 distinct agents; cycles if more)
_AGENT_COLORS = [
    (31, 119, 180),   # blue
    (255, 127, 14),   # orange
    (44, 160, 44),    # green
    (214, 39, 40),    # red
    (148, 103, 189),  # purple
    (140, 86, 75),    # brown
    (227, 119, 194),  # pink
    (127, 127, 127),  # gray
    (188, 189, 34),   # olive
    (23, 190, 207),   # cyan
    (174, 199, 232),  # light blue
    (255, 187, 120),  # light orange
    (152, 223, 138),  # light green
    (255, 152, 150),  # light red
    (197, 176, 213),  # light purple
    (196, 156, 148),  # light brown
]

# Food color (bright green)
_FOOD_COLOR = (50, 205, 50)

# Grid line color (light gray)
_GRID_COLOR = (200, 200, 200)

# Background color (white)
_BG_COLOR = (255, 255, 255)


def render_frame(state: EnvState, config: Config, pixel_size: int = 0) -> np.ndarray:
    """Render the current environment state as an RGB image.

    Shows grid lines, agents as colored circles, food as green dots,
    and the field as a heatmap overlay (sum across channels).

    Args:
        state: Current environment state.
        config: Master configuration object.
        pixel_size: Pixels per grid cell. If 0, auto-computed to ensure
            the image is at least 400x400.

    Returns:
        RGB image as uint8 numpy array with shape (H, W, 3).
    """
    grid_size = config.env.grid_size

    # Ensure at least 400x400 pixels
    if pixel_size <= 0:
        pixel_size = max(400 // grid_size, 20)
    img_size = grid_size * pixel_size

    # Create white background
    img = np.full((img_size, img_size, 3), _BG_COLOR, dtype=np.uint8)

    # --- 1. Field heatmap overlay ---
    field_vals = np.asarray(state.field_state.values)  # (H, W, C)
    field_sum = field_vals.sum(axis=-1)  # (H, W)

    field_max = field_sum.max()
    if field_max > 0:
        field_norm = field_sum / field_max  # normalize to [0, 1]
    else:
        field_norm = field_sum

    # Map to a blue-to-red heatmap, blended with background
    for r in range(grid_size):
        for c in range(grid_size):
            val = float(field_norm[r, c])
            if val > 1e-6:
                # Interpolate: blue (low) -> red (high)
                red = int(255 * val)
                blue = int(255 * (1.0 - val))
                green = 0
                alpha = min(val * 0.6, 0.6)  # partial transparency

                y0 = r * pixel_size
                y1 = (r + 1) * pixel_size
                x0 = c * pixel_size
                x1 = (c + 1) * pixel_size

                # Alpha blend with background
                patch = img[y0:y1, x0:x1].astype(np.float32)
                overlay = np.array([red, green, blue], dtype=np.float32)
                img[y0:y1, x0:x1] = (
                    patch * (1 - alpha) + overlay * alpha
                ).astype(np.uint8)

    # --- 2. Grid lines ---
    for i in range(grid_size + 1):
        pos = i * pixel_size
        # Horizontal lines
        if pos < img_size:
            img[pos, :] = _GRID_COLOR
        # Vertical lines
        if pos < img_size:
            img[:, pos] = _GRID_COLOR

    # --- 3. Food as green dots ---
    food_positions = np.asarray(state.food_positions)  # (num_food, 2)
    food_collected = np.asarray(state.food_collected)  # (num_food,)
    food_radius = max(pixel_size // 6, 2)

    for i in range(food_positions.shape[0]):
        if food_collected[i]:
            continue
        row, col = int(food_positions[i, 0]), int(food_positions[i, 1])
        cy = row * pixel_size + pixel_size // 2
        cx = col * pixel_size + pixel_size // 2
        _draw_circle(img, cy, cx, food_radius, _FOOD_COLOR)

    # --- 4. Agents as colored circles (only alive agents) ---
    agent_positions = np.asarray(state.agent_positions)  # (max_agents, 2)
    agent_alive = np.asarray(state.agent_alive)  # (max_agents,)
    agent_radius = max(pixel_size // 3, 3)

    for i in range(agent_positions.shape[0]):
        if not agent_alive[i]:
            continue
        row, col = int(agent_positions[i, 0]), int(agent_positions[i, 1])
        cy = row * pixel_size + pixel_size // 2
        cx = col * pixel_size + pixel_size // 2
        color = _AGENT_COLORS[i % len(_AGENT_COLORS)]
        _draw_circle(img, cy, cx, agent_radius, color)

    return img


def _draw_circle(
    img: np.ndarray,
    cy: int,
    cx: int,
    radius: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a filled circle on the image (in-place).

    Args:
        img: RGB image array (H, W, 3).
        cy: Center y coordinate.
        cx: Center x coordinate.
        radius: Circle radius in pixels.
        color: RGB color tuple.
    """
    h, w = img.shape[:2]
    y_min = max(cy - radius, 0)
    y_max = min(cy + radius + 1, h)
    x_min = max(cx - radius, 0)
    x_max = min(cx + radius + 1, w)

    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            if (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2:
                img[y, x] = color
