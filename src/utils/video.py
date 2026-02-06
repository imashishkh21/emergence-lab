"""Video recording utilities for episode visualization."""

import jax
import numpy as np

from src.agents.network import ActorCritic
from src.agents.policy import get_deterministic_actions
from src.configs import Config
from src.environment.env import reset, step
from src.environment.obs import get_observations
from src.environment.render import render_frame


def record_episode(
    network: ActorCritic,
    params: dict,
    config: Config,
    key: jax.Array | None = None,
) -> list[np.ndarray]:
    """Record an episode by running the policy and rendering each frame.

    Runs one full episode (up to max_steps) using deterministic (greedy)
    actions from the given network/params. Renders each timestep as an
    RGB frame.

    Args:
        network: ActorCritic network module.
        params: Network parameters.
        config: Master configuration object.
        key: PRNG key for environment reset. If None, uses seed 0.

    Returns:
        List of RGB frames as uint8 numpy arrays (H, W, 3).
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    # Reset environment for a single episode
    state = reset(key, config)
    frames: list[np.ndarray] = []

    # Render initial frame
    frames.append(render_frame(state, config))

    for _ in range(config.env.max_steps):
        # Get observations — shape (num_agents, obs_dim)
        obs = get_observations(state, config)

        # Add batch dimension for policy: (1, num_agents, obs_dim)
        obs_batched = obs[None, :, :]

        # Get deterministic actions: returns (actions, gate) tuple
        # actions shape: (1, num_agents) -> (num_agents,)
        actions, _gate = get_deterministic_actions(network, params, obs_batched)
        actions = actions[0]  # remove batch dim

        # Step environment
        state, _rewards, done, _info = step(state, actions, config)

        # Render frame
        frames.append(render_frame(state, config))

        # Stop if episode is done
        if bool(done):
            break

    return frames


def save_video(frames: list[np.ndarray], path: str, fps: int = 30) -> None:
    """Save a list of RGB frames as an MP4 video file.

    Args:
        frames: List of RGB images as uint8 numpy arrays (H, W, 3).
        path: Output file path (should end with .mp4).
        fps: Frames per second for playback.
    """
    import imageio

    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
