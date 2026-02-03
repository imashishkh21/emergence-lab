"""MAPPO Baseline: Multi-Agent PPO with centralized critic.

This baseline implements CTDE (Centralized Training, Decentralized Execution):
- Decentralized actors: Each agent uses its own local observation for action selection
- Centralized critic: Value function takes concatenated observations of ALL agents

Key characteristics:
    - Field disabled: write_strength=0, decay_rate=1.0 (no field communication)
    - Evolution disabled: population stays at initial num_agents
    - Shared actor weights: all agents use the same actor policy
    - Centralized critic: value function sees all agents' observations
    - Value normalization: running mean/std of returns for stable learning

Reference: Yu et al. (2022), "The Surprising Effectiveness of PPO in Cooperative
Multi-Agent Games" (MAPPO)
"""

from dataclasses import replace
from typing import Any, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.agents.network import ActorCritic
from src.agents.policy import get_deterministic_actions, sample_actions
from src.configs import Config
from src.environment.env import reset, step
from src.environment.obs import get_observations, obs_dim


class CentralizedCritic(nn.Module):
    """Centralized critic that takes concatenated observations of all agents.

    Architecture:
        - Input: concatenated observations of all agents (flattened)
        - MLP backbone with LayerNorm + Tanh activations
        - Output: single value estimate per agent (or one global value)

    The centralized critic provides a more accurate value estimate during
    training by seeing the full state (all agents' observations), while
    the decentralized actors only see their local observation.
    """

    hidden_dims: Sequence[int] = (128, 128)
    n_agents: int = 8

    @nn.compact
    def __call__(self, all_obs: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            all_obs: Concatenated observations of all agents.
                Shape: (n_agents * obs_dim,) for single-agent value
                or (n_agents, obs_dim) which gets flattened internally.

        Returns:
            Value estimates of shape (n_agents,) - one value per agent.
        """
        # Flatten if needed: (n_agents, obs_dim) -> (n_agents * obs_dim,)
        if all_obs.ndim == 2:
            x = all_obs.flatten()
        else:
            x = all_obs

        # Shared backbone
        for dim in self.hidden_dims:
            x = nn.Dense(
                dim,
                kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)),
                bias_init=nn.initializers.zeros,
            )(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)

        # Output: one value per agent
        values = nn.Dense(
            self.n_agents,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.zeros,
        )(x)

        return values


class RunningMeanStd:
    """Running mean and standard deviation for value normalization.

    Tracks running statistics with Welford's algorithm for numerical stability.
    Used to normalize returns for more stable value function learning.
    """

    def __init__(self, epsilon: float = 1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon  # Small count to avoid division by zero

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with a batch of values."""
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = x.size

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: float, batch_var: float, batch_count: int
    ) -> None:
        """Update from batch mean/var/count using parallel algorithm."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count

        self.mean = new_mean
        self.var = m_2 / tot_count
        self.count = tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize values using running statistics."""
        result: np.ndarray = (x - self.mean) / np.sqrt(self.var + 1e-8)
        return result


def mappo_config(base_config: Config | None = None) -> Config:
    """Create a config for MAPPO baseline.

    MAPPO disables the field (centralized critic handles coordination)
    and evolution (population stays constant).

    Args:
        base_config: Optional base config to modify. If None, uses default Config.

    Returns:
        Config with field zeroed and evolution disabled.
    """
    if base_config is None:
        base_config = Config()

    # Disable field: agents write nothing, field decays instantly
    field_config = replace(
        base_config.field,
        write_strength=0.0,
        decay_rate=1.0,  # Full decay each step = field is always zeros
    )

    # Disable evolution: no births, deaths, or reproduction
    evolution_config = replace(
        base_config.evolution,
        enabled=False,
        starting_energy=1000000,
        energy_per_step=0,
        max_energy=1000000,
        reproduce_threshold=10000000,
        max_agents=base_config.env.num_agents,
        min_agents=base_config.env.num_agents,
    )

    return replace(
        base_config,
        field=field_config,
        evolution=evolution_config,
    )


def create_mappo_network(config: Config) -> ActorCritic:
    """Create the decentralized actor network for MAPPO.

    Uses the standard ActorCritic for the actor (shared weights across agents).
    The critic is separate (CentralizedCritic).

    Args:
        config: Configuration object.

    Returns:
        ActorCritic network module (actor only, value head unused).
    """
    return ActorCritic(
        hidden_dims=config.agent.hidden_dims,
        num_actions=6,
        agent_embed_dim=config.agent.agent_embed_dim,
        n_agents=config.env.num_agents,
    )


def create_centralized_critic(config: Config) -> CentralizedCritic:
    """Create the centralized critic network for MAPPO.

    Args:
        config: Configuration object.

    Returns:
        CentralizedCritic network module.
    """
    # Use larger hidden dims for centralized critic (more capacity)
    hidden_dims = tuple(dim * 2 for dim in config.agent.hidden_dims)
    return CentralizedCritic(
        hidden_dims=hidden_dims,
        n_agents=config.env.num_agents,
    )


def init_mappo_params(
    actor: ActorCritic,
    critic: CentralizedCritic,
    config: Config,
    key: jax.Array,
) -> tuple[Any, Any]:
    """Initialize MAPPO actor and critic parameters.

    Args:
        actor: ActorCritic network module (decentralized actor).
        critic: CentralizedCritic network module.
        config: Configuration object.
        key: JAX PRNG key for initialization.

    Returns:
        Tuple of (actor_params, critic_params).
    """
    key1, key2 = jax.random.split(key)

    # Initialize actor
    observation_dim = obs_dim(config)
    dummy_obs = jnp.zeros((observation_dim,))
    actor_params = actor.init(key1, dummy_obs)

    # Initialize critic (takes concatenated observations)
    total_obs_dim = observation_dim * config.env.num_agents
    dummy_all_obs = jnp.zeros((total_obs_dim,))
    critic_params = critic.init(key2, dummy_all_obs)

    return actor_params, critic_params


def _masked_mean(x: jnp.ndarray, mask: jnp.ndarray | None) -> jnp.ndarray:
    """Compute mean of x, optionally weighted by a boolean mask."""
    if mask is None:
        return jnp.mean(x)
    mask_f = mask.astype(jnp.float32)
    return jnp.sum(x * mask_f) / jnp.maximum(jnp.sum(mask_f), 1.0)


def mappo_loss(
    actor: nn.Module,
    critic: nn.Module,
    actor_params: dict,
    critic_params: dict,
    batch: dict[str, jnp.ndarray],
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Compute MAPPO loss with centralized critic.

    Args:
        actor: ActorCritic network module (only actor part used).
        critic: CentralizedCritic network module.
        actor_params: Actor network parameters.
        critic_params: Critic network parameters.
        batch: Dictionary containing:
            - obs: (batch_size, obs_dim) individual agent observations
            - all_obs: (batch_size, n_agents * obs_dim) concatenated observations
            - actions: (batch_size,) integer actions
            - log_probs: (batch_size,) old log probabilities
            - advantages: (batch_size,)
            - returns: (batch_size,) target returns for value function
            - alive_mask: (batch_size,) optional boolean mask (True = alive)
            - agent_indices: (batch_size,) which agent each sample belongs to
        clip_eps: PPO clipping epsilon.
        vf_coef: Value function loss coefficient.
        ent_coef: Entropy bonus coefficient.

    Returns:
        (loss, metrics) where loss is scalar and metrics is a dict of scalars.
    """
    obs = batch["obs"]
    all_obs = batch["all_obs"]
    actions = batch["actions"]
    old_log_probs = batch["log_probs"]
    advantages = batch["advantages"]
    returns = batch["returns"]
    mask = batch.get("alive_mask", None)
    agent_indices = batch.get("agent_indices", None)

    # Actor forward pass (individual observations)
    # Vectorized forward pass over batch
    def actor_forward(single_obs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        out = actor.apply(actor_params, single_obs)
        return out[0], out[1]  # type: ignore[return-value]

    logits, _ = jax.vmap(actor_forward)(obs)

    # Critic forward pass (centralized observations)
    # Vectorized forward pass over batch
    def critic_forward(single_all_obs: jnp.ndarray) -> jnp.ndarray:
        out = critic.apply(critic_params, single_all_obs)
        return out  # type: ignore[return-value]

    values_all = critic_forward(all_obs[0])[None, :]  # Get shape right
    values_all = jax.vmap(critic_forward)(all_obs)

    # Select the value for the correct agent if agent_indices is provided
    if agent_indices is not None:
        # Gather values for each agent's sample
        batch_indices = jnp.arange(values_all.shape[0])
        values = values_all[batch_indices, agent_indices.astype(jnp.int32)]
    else:
        # If no agent indices, assume single agent or average
        values = values_all[:, 0]

    # Compute new log probabilities and entropy
    log_softmax = jax.nn.log_softmax(logits, axis=-1)
    new_log_probs = jnp.take_along_axis(
        log_softmax, actions[..., None].astype(jnp.int32), axis=-1
    ).squeeze(-1)

    # Entropy: -sum(p * log(p))
    probs = jax.nn.softmax(logits, axis=-1)
    per_sample_entropy = -jnp.sum(probs * log_softmax, axis=-1)
    entropy = _masked_mean(per_sample_entropy, mask)

    # Policy loss: clipped surrogate objective
    log_ratio = new_log_probs - old_log_probs
    ratio = jnp.exp(log_ratio)

    surr1 = ratio * advantages
    surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -_masked_mean(jnp.minimum(surr1, surr2), mask)

    # Value loss: MSE
    value_loss = _masked_mean(jnp.square(values - returns), mask)

    # Approximate KL divergence
    approx_kl = _masked_mean((ratio - 1.0) - log_ratio, mask)

    # Clip fraction
    clip_fraction = _masked_mean(
        (jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32), mask
    )

    # Total loss
    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

    metrics = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
    }

    return loss, metrics


def run_mappo_episode(
    actor: ActorCritic,
    critic: CentralizedCritic,
    actor_params: Any,
    critic_params: Any,
    config: Config,
    key: jax.Array,
    deterministic: bool = False,
) -> dict[str, Any]:
    """Run a single episode using MAPPO.

    Args:
        actor: ActorCritic network module (decentralized actor).
        critic: CentralizedCritic network module.
        actor_params: Actor network parameters.
        critic_params: Critic network parameters.
        config: Configuration (should be from mappo_config()).
        key: JAX PRNG key.
        deterministic: If True, use greedy actions instead of sampling.

    Returns:
        Standardized result dict with:
            - total_reward: Sum of all rewards across all agents and steps
            - food_collected: Total food items collected in the episode
            - final_population: Number of alive agents at episode end
            - per_agent_rewards: List of total reward per agent over the episode
    """
    # Initialize environment
    key, reset_key = jax.random.split(key)
    state = reset(reset_key, config)

    # Track metrics
    total_reward = 0.0
    total_food = 0.0
    num_agents = config.env.num_agents
    per_agent_rewards = np.zeros(num_agents)

    # Run episode
    for _ in range(config.env.max_steps):
        # Get observations (add batch dimension for single env)
        obs = get_observations(state, config)
        obs_batch = obs[None, :, :]  # (1, max_agents, obs_dim)

        # Sample or get deterministic actions
        key, action_key = jax.random.split(key)
        if deterministic:
            actions = get_deterministic_actions(actor, actor_params, obs_batch)
        else:
            actions, _, _, _ = sample_actions(actor, actor_params, obs_batch, action_key)

        # Remove batch dimension
        actions = actions[0]  # (max_agents,)

        # Step environment
        state, rewards, done, info = step(state, actions, config)

        # Accumulate metrics
        rewards_np = np.array(rewards)
        total_reward += float(np.sum(rewards_np))
        total_food += float(info["food_collected_this_step"])
        per_agent_rewards[:num_agents] += rewards_np[:num_agents]

        if done:
            break

    # Final population (should be same as initial for MAPPO)
    final_population = int(np.sum(np.array(state.agent_alive)))

    return {
        "total_reward": total_reward,
        "food_collected": total_food,
        "final_population": final_population,
        "per_agent_rewards": per_agent_rewards.tolist(),
    }


def evaluate_mappo(
    actor: ActorCritic,
    critic: CentralizedCritic,
    actor_params: Any,
    critic_params: Any,
    config: Config,
    n_episodes: int,
    seed: int,
    deterministic: bool = False,
) -> dict[str, Any]:
    """Evaluate MAPPO over multiple episodes.

    Args:
        actor: ActorCritic network module (decentralized actor).
        critic: CentralizedCritic network module.
        actor_params: Actor network parameters.
        critic_params: Critic network parameters.
        config: Configuration (should be from mappo_config()).
        n_episodes: Number of episodes to run.
        seed: Random seed for reproducibility.
        deterministic: If True, use greedy actions instead of sampling.

    Returns:
        Aggregated result dict with:
            - total_reward: Mean total reward across episodes
            - total_reward_std: Standard deviation of total reward
            - food_collected: Mean food collected across episodes
            - food_collected_std: Standard deviation of food collected
            - final_population: Mean final population across episodes
            - per_agent_rewards: Mean reward per agent across episodes
            - episode_rewards: List of total rewards per episode
            - episode_food: List of food collected per episode
            - n_episodes: Number of episodes run
    """
    key = jax.random.PRNGKey(seed)

    episode_rewards = []
    episode_food = []
    episode_populations = []
    all_per_agent_rewards = []

    for _ in range(n_episodes):
        key, episode_key = jax.random.split(key)
        result = run_mappo_episode(
            actor,
            critic,
            actor_params,
            critic_params,
            config,
            episode_key,
            deterministic=deterministic,
        )
        episode_rewards.append(result["total_reward"])
        episode_food.append(result["food_collected"])
        episode_populations.append(result["final_population"])
        all_per_agent_rewards.append(result["per_agent_rewards"])

    # Compute statistics
    rewards_array = np.array(episode_rewards)
    food_array = np.array(episode_food)
    pop_array = np.array(episode_populations)
    per_agent_array = np.array(all_per_agent_rewards)

    return {
        "total_reward": float(np.mean(rewards_array)),
        "total_reward_std": float(np.std(rewards_array)),
        "food_collected": float(np.mean(food_array)),
        "food_collected_std": float(np.std(food_array)),
        "final_population": float(np.mean(pop_array)),
        "per_agent_rewards": np.mean(per_agent_array, axis=0).tolist(),
        "episode_rewards": episode_rewards,
        "episode_food": episode_food,
        "n_episodes": n_episodes,
    }


def create_mappo_train_state(
    actor: ActorCritic,
    critic: CentralizedCritic,
    config: Config,
    key: jax.Array,
    learning_rate: float = 3e-4,
) -> tuple[Any, Any, Any, Any]:
    """Create training state for MAPPO.

    Args:
        actor: ActorCritic network module.
        critic: CentralizedCritic network module.
        config: Configuration object.
        key: JAX PRNG key.
        learning_rate: Learning rate for both actor and critic.

    Returns:
        Tuple of (actor_params, critic_params, actor_opt_state, critic_opt_state).
    """
    actor_params, critic_params = init_mappo_params(actor, critic, config, key)

    # Create optimizers
    actor_optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate),
    )
    critic_optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate),
    )

    actor_opt_state = actor_optimizer.init(actor_params)
    critic_opt_state = critic_optimizer.init(critic_params)

    return actor_params, critic_params, actor_opt_state, critic_opt_state
