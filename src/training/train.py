"""Training step and loop for PPO with shared field dynamics."""

import os
import pickle
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from src.agents.network import ActorCritic
from src.agents.policy import sample_actions
from src.analysis.emergence import EmergenceTracker
from src.analysis.specialization import SpecializationTracker
from src.configs import Config, TrainingMode
from src.environment.obs import get_observations, obs_dim
from src.environment.vec_env import VecEnv
from src.training.gae import compute_gae
from src.training.ppo import ppo_loss
from src.training.rollout import RunnerState, collect_rollout
from src.utils.logging import finish_wandb, init_wandb, log_metrics


def _compute_specialization_bonuses(
    agent_params: Any,
    alive_mask: jnp.ndarray,
    diversity_bonus: float,
    niche_pressure: float,
) -> jnp.ndarray:
    """Compute per-agent reward bonuses for specialization incentives.

    Operates on a single environment's per-agent params.

    Args:
        agent_params: Per-agent params pytree, each leaf shape (max_agents, ...).
        alive_mask: Boolean (max_agents,) indicating alive agents.
        diversity_bonus: Scale for diversity reward (mean cosine distance).
        niche_pressure: Scale for niche penalty (penalize nearest-neighbor similarity).

    Returns:
        Per-agent bonus array of shape (max_agents,).
    """
    # Flatten each agent's params into a single vector
    leaves = jax.tree_util.tree_leaves(agent_params)
    flat_parts = [leaf.reshape(leaf.shape[0], -1) for leaf in leaves]
    weight_matrix = jnp.concatenate(flat_parts, axis=-1)  # (max_agents, D)

    max_agents = weight_matrix.shape[0]
    alive_f = alive_mask.astype(jnp.float32)

    # Compute pairwise cosine distances: 1 - cos_sim(a, b)
    norms = jnp.linalg.norm(weight_matrix, axis=-1, keepdims=True)  # (max_agents, 1)
    norms = jnp.maximum(norms, 1e-8)
    normalized = weight_matrix / norms  # (max_agents, D)
    cos_sim = normalized @ normalized.T  # (max_agents, max_agents)
    cos_dist = 1.0 - cos_sim  # (max_agents, max_agents)

    # Mask: only consider alive pairs
    pair_mask = alive_f[:, None] * alive_f[None, :]  # (max_agents, max_agents)
    # Zero out self-pairs
    self_mask = 1.0 - jnp.eye(max_agents)
    pair_mask = pair_mask * self_mask

    bonuses = jnp.zeros(max_agents)

    # Diversity bonus: mean cosine distance to all other alive agents
    if diversity_bonus != 0.0:
        pair_count = jnp.maximum(jnp.sum(pair_mask, axis=-1), 1.0)  # (max_agents,)
        mean_dist = jnp.sum(cos_dist * pair_mask, axis=-1) / pair_count
        bonuses = bonuses + diversity_bonus * mean_dist * alive_f

    # Niche pressure: penalize agents too close to nearest neighbor
    if niche_pressure != 0.0:
        # Set non-pair entries to large distance so they don't affect min
        large_dist = jnp.where(pair_mask > 0, cos_dist, 1e6)
        min_dist = jnp.min(large_dist, axis=-1)  # (max_agents,)
        # Penalty: niche_pressure * (1 - min_dist) when min_dist < 1
        # More similar neighbors → higher penalty
        penalty = niche_pressure * jnp.maximum(1.0 - min_dist, 0.0)
        bonuses = bonuses - penalty * alive_f

    return bonuses


def create_train_state(config: Config, key: jax.Array) -> RunnerState:
    """Initialize all training state: network, optimizer, environments.

    Args:
        config: Master configuration.
        key: PRNG key.

    Returns:
        Initialized RunnerState ready for training.
    """
    key, init_key, env_key = jax.random.split(key, 3)

    # Create vectorized environment
    vec_env = VecEnv(config)

    # Reset environments
    env_state = vec_env.reset(env_key)

    # Compute observation dimension and create network
    observation_dim = obs_dim(config)
    num_actions = 6  # 0-4 movement + 5 reproduce
    network = ActorCritic(
        hidden_dims=tuple(config.agent.hidden_dims),
        num_actions=num_actions,
    )

    # Initialize network parameters
    dummy_obs = jnp.zeros((observation_dim,))
    params = network.init(init_key, dummy_obs)

    # Initialize per-agent params: replicate shared params to (num_envs, max_agents, ...)
    # Each leaf gets leading (num_envs, max_agents) dimensions for the batched state
    if config.evolution.enabled:
        max_agents = config.evolution.max_agents
        num_envs = config.train.num_envs
        per_agent_params = jax.tree.map(
            lambda leaf: jnp.broadcast_to(
                leaf[None, None], (num_envs, max_agents) + leaf.shape
            ).copy(),
            params,
        )
        # Set per-agent params on the batched env state
        env_state = env_state.replace(agent_params=per_agent_params)  # type: ignore[attr-defined]

    # Create optimizer with gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.train.max_grad_norm),
        optax.adam(config.train.learning_rate, eps=1e-5),
    )
    opt_state = optimizer.init(params)

    # Get initial observations
    last_obs = jax.vmap(lambda s: get_observations(s, config))(env_state)

    return RunnerState(
        params=params,
        opt_state=opt_state,
        env_state=env_state,
        last_obs=last_obs,
        key=key,
    )


def train_step(
    runner_state: RunnerState,
    config: Config,
) -> tuple[RunnerState, dict[str, Any]]:
    """Execute one PPO training iteration.

    Steps:
        1. Collect a rollout of num_steps transitions
        2. Compute bootstrap value for GAE
        3. Compute GAE advantages and returns
        4. Flatten rollout into training batch
        5. Run multiple epochs of minibatch PPO updates
        6. Return updated runner state and aggregated metrics

    Args:
        runner_state: Current training state.
        config: Master configuration.

    Returns:
        (new_runner_state, metrics) where metrics is a dict of scalar values.
    """
    vec_env = VecEnv(config)
    num_actions = 6  # 0-4 movement + 5 reproduce
    network = ActorCritic(
        hidden_dims=tuple(config.agent.hidden_dims),
        num_actions=num_actions,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.train.max_grad_norm),
        optax.adam(config.train.learning_rate, eps=1e-5),
    )

    # 1. Collect rollout
    runner_state, batch = collect_rollout(runner_state, network, vec_env, config)

    # batch shapes: (num_steps, num_envs, max_agents, ...) or (num_steps, num_envs)
    # rewards: (num_steps, num_envs, max_agents)
    # dones: (num_steps, num_envs)
    # values: (num_steps, num_envs, max_agents)
    # obs: (num_steps, num_envs, max_agents, obs_dim)
    # actions: (num_steps, num_envs, max_agents)
    # log_probs: (num_steps, num_envs, max_agents)
    # alive_mask: (num_steps, num_envs, max_agents)

    # 2. Compute bootstrap value for the last observation
    # We need value of the final state for GAE computation
    # Use sample_actions to get the value (we ignore the sampled action)
    key, bootstrap_key = jax.random.split(runner_state.key)
    _, _, bootstrap_values, _ = sample_actions(
        network, runner_state.params, runner_state.last_obs, bootstrap_key
    )
    # bootstrap_values: (num_envs, max_agents)

    # 3. Compute GAE per environment per agent
    # Mask dead agents: zero rewards and values so GAE produces 0 advantages
    alive_mask = batch['alive_mask']  # (T, num_envs, max_agents)
    alive_f = alive_mask.astype(jnp.float32)
    masked_rewards = batch['rewards'] * alive_f

    # Apply specialization incentives (diversity bonus / niche pressure)
    div_bonus = config.specialization.diversity_bonus
    niche_p = config.specialization.niche_pressure
    if (div_bonus != 0.0 or niche_p != 0.0) and runner_state.env_state.agent_params is not None:
        # Compute per-agent bonuses for each environment using current params
        per_env_params = runner_state.env_state.agent_params
        per_env_alive = runner_state.env_state.agent_alive
        # vmap over environments
        spec_bonuses = jax.vmap(
            lambda p, a: _compute_specialization_bonuses(p, a, div_bonus, niche_p)
        )(per_env_params, per_env_alive)  # (num_envs, max_agents)
        # Add bonus to every timestep's rewards
        masked_rewards = masked_rewards + spec_bonuses[None, :, :] * alive_f

    masked_values = batch['values'] * alive_f

    # Bootstrap values also masked by final alive state
    final_alive = runner_state.env_state.agent_alive  # (num_envs, max_agents)
    masked_bootstrap = bootstrap_values * final_alive.astype(jnp.float32)

    # We need values with shape (T+1, num_envs, max_agents) for GAE
    all_values = jnp.concatenate(
        [masked_values, masked_bootstrap[None, :, :]], axis=0
    )  # (T+1, num_envs, max_agents)

    # Expand dones to match agent dimension: (T, num_envs) -> (T, num_envs, max_agents)
    dones_expanded = jnp.broadcast_to(
        batch['dones'][:, :, None],
        masked_rewards.shape,
    )

    # Vectorize GAE over (env, agent) pairs
    def _gae_single(rewards: jnp.ndarray, values: jnp.ndarray, dones: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return compute_gae(rewards, values, dones, config.train.gamma, config.train.gae_lambda)

    # Reshape to map over env*agent: (T, num_envs * max_agents)
    num_steps = masked_rewards.shape[0]
    num_envs = masked_rewards.shape[1]
    max_agents = masked_rewards.shape[2]

    rewards_flat = masked_rewards.reshape(num_steps, num_envs * max_agents)
    values_flat = all_values.reshape(num_steps + 1, num_envs * max_agents)
    dones_flat = dones_expanded.reshape(num_steps, num_envs * max_agents)

    # vmap over the env*agent dimension (axis 1)
    advantages_flat, returns_flat = jax.vmap(
        _gae_single, in_axes=(1, 1, 1), out_axes=(1, 1)
    )(rewards_flat, values_flat, dones_flat)
    # advantages_flat: (T, num_envs * max_agents)
    # returns_flat: (T, num_envs * max_agents)

    advantages = advantages_flat.reshape(num_steps, num_envs, max_agents)
    returns = returns_flat.reshape(num_steps, num_envs, max_agents)

    # 4. Flatten rollout into training batch
    # Flatten (num_steps, num_envs, max_agents) -> (batch_size,)
    batch_size = num_steps * num_envs * max_agents
    obs_dim_val = batch['obs'].shape[-1]

    flat_batch = {
        'obs': batch['obs'].reshape(batch_size, obs_dim_val),
        'actions': batch['actions'].reshape(batch_size),
        'log_probs': batch['log_probs'].reshape(batch_size),
        'advantages': advantages.reshape(batch_size),
        'returns': returns.reshape(batch_size),
        'alive_mask': alive_mask.reshape(batch_size),
    }

    # 5. Run multiple epochs of minibatch updates
    num_epochs = config.train.num_epochs
    minibatch_size = config.train.minibatch_size
    num_minibatches = batch_size // minibatch_size if batch_size >= minibatch_size else 1

    def _epoch_step(
        carry: tuple[Any, Any, jax.Array],
        _unused: None,
    ) -> tuple[tuple[Any, Any, jax.Array], dict[str, jnp.ndarray]]:
        params, opt_state, epoch_key = carry

        # Shuffle the batch
        epoch_key, shuffle_key = jax.random.split(epoch_key)
        perm = jax.random.permutation(shuffle_key, batch_size)

        shuffled_batch = {
            k: v[perm] for k, v in flat_batch.items()
        }

        # Process minibatches
        def _minibatch_step(
            mb_carry: tuple[Any, Any],
            mb_idx: jnp.ndarray,
        ) -> tuple[tuple[Any, Any], dict[str, jnp.ndarray]]:
            mb_params, mb_opt_state = mb_carry

            start = mb_idx * minibatch_size
            mb = {
                k: jax.lax.dynamic_slice_in_dim(v, start, minibatch_size, axis=0)
                for k, v in shuffled_batch.items()
            }

            # Normalize advantages within minibatch (using alive mask)
            mb_adv = mb['advantages']
            mb_mask = mb.get('alive_mask', None)
            if mb_mask is not None:
                mask_f = mb_mask.astype(jnp.float32)
                mask_sum = jnp.maximum(jnp.sum(mask_f), 1.0)
                adv_mean = jnp.sum(mb_adv * mask_f) / mask_sum
                adv_var = jnp.sum(jnp.square(mb_adv - adv_mean) * mask_f) / mask_sum
                mb_adv = (mb_adv - adv_mean) / (jnp.sqrt(adv_var) + 1e-8)
            else:
                mb_adv = (mb_adv - jnp.mean(mb_adv)) / (jnp.std(mb_adv) + 1e-8)
            mb = {**mb, 'advantages': mb_adv}

            # Compute loss and gradients
            grad_fn = jax.value_and_grad(ppo_loss, argnums=1, has_aux=True)
            (loss, metrics), grads = grad_fn(
                network, mb_params, mb,
                config.train.clip_eps, config.train.vf_coef, config.train.ent_coef,
            )

            # Apply optimizer update
            updates, new_opt_state = optimizer.update(grads, mb_opt_state, mb_params)
            new_params = optax.apply_updates(mb_params, updates)

            metrics['total_loss'] = loss
            return (new_params, new_opt_state), metrics

        # Scan over minibatches
        (params, opt_state), epoch_metrics = jax.lax.scan(
            _minibatch_step,
            (params, opt_state),
            jnp.arange(num_minibatches),
        )

        # Average metrics over minibatches
        avg_metrics = jax.tree.map(lambda x: jnp.mean(x), epoch_metrics)
        return (params, opt_state, epoch_key), avg_metrics

    # Scan over epochs
    (new_params, new_opt_state, key), all_epoch_metrics = jax.lax.scan(
        _epoch_step,
        (runner_state.params, runner_state.opt_state, key),
        None,
        length=num_epochs,
    )

    # Average metrics over epochs
    metrics = jax.tree.map(lambda x: jnp.mean(x), all_epoch_metrics)

    # Add rollout-level metrics (masked to alive agents only)
    alive_f = alive_mask.astype(jnp.float32)
    alive_count = jnp.maximum(jnp.sum(alive_f), 1.0)
    metrics['mean_reward'] = jnp.sum(batch['rewards'] * alive_f) / alive_count
    metrics['mean_value'] = jnp.sum(batch['values'] * alive_f) / alive_count
    metrics['mean_advantage'] = jnp.sum(advantages * alive_f) / alive_count
    metrics['mean_return'] = jnp.sum(returns * alive_f) / alive_count

    # --- Population metrics ---
    # Population size: mean across rollout steps and envs
    # alive_mask: (T, num_envs, max_agents) -> sum over agents -> (T, num_envs)
    pop_per_step = jnp.sum(alive_f, axis=-1)  # (T, num_envs)
    metrics['population_size'] = jnp.mean(pop_per_step)

    # Births and deaths aggregated over the rollout
    # batch['births_this_step']: (T, num_envs), batch['deaths_this_step']: (T, num_envs)
    metrics['births_this_step'] = jnp.sum(batch['births_this_step'])
    metrics['deaths_this_step'] = jnp.sum(batch['deaths_this_step'])

    # Energy stats of alive agents (snapshot from final env state)
    final_env = runner_state.env_state
    final_alive = final_env.agent_alive  # (num_envs, max_agents)
    final_alive_f = final_alive.astype(jnp.float32)
    final_energy = final_env.agent_energy  # (num_envs, max_agents)
    final_alive_count = jnp.maximum(jnp.sum(final_alive_f), 1.0)
    metrics['mean_energy'] = jnp.sum(final_energy * final_alive_f) / final_alive_count
    # For max/min, mask dead agents with -inf/+inf respectively
    metrics['max_energy'] = jnp.max(jnp.where(final_alive, final_energy, -jnp.inf))
    metrics['min_energy'] = jnp.min(jnp.where(final_alive, final_energy, jnp.inf))

    # Oldest agent age (steps alive) — snapshot from final env state
    # age = current_step - birth_step for alive agents
    final_step = final_env.step  # (num_envs,) scalar per env
    final_birth = final_env.agent_birth_step  # (num_envs, max_agents)
    agent_ages = jnp.where(
        final_alive,
        final_step[:, None] - final_birth,
        jnp.int32(0),
    )
    metrics['oldest_agent_age'] = jnp.max(agent_ages)

    # Sync per-agent params: broadcast updated shared params to all alive agents
    env_state = runner_state.env_state
    if env_state.agent_params is not None:
        num_envs = config.train.num_envs
        max_agents_val = config.evolution.max_agents
        # Replicate shared params to (num_envs, max_agents, ...) for all slots
        updated_per_agent = jax.tree.map(
            lambda leaf: jnp.broadcast_to(
                leaf[None, None], (num_envs, max_agents_val) + leaf.shape
            ).copy(),
            new_params,
        )
        # Keep existing per-agent params for dead slots (they may be reused on spawn)
        # For alive slots, use the updated shared params
        current_alive = env_state.agent_alive  # (num_envs, max_agents)

        def _sync_params(updated: jnp.ndarray, existing: jnp.ndarray) -> jnp.ndarray:
            """Broadcast alive mask to match param leaf shape and select."""
            # Both updated and existing: (num_envs, max_agents, ...)
            # current_alive: (num_envs, max_agents)
            ndim_extra = existing.ndim - current_alive.ndim
            mask = current_alive
            for _ in range(ndim_extra):
                mask = mask[..., None]
            return jnp.where(mask, updated, existing)

        new_agent_params = jax.tree.map(
            _sync_params, updated_per_agent, env_state.agent_params
        )
        env_state = env_state.replace(agent_params=new_agent_params)  # type: ignore[attr-defined]

    new_runner_state = RunnerState(
        params=new_params,
        opt_state=new_opt_state,
        env_state=env_state,
        last_obs=runner_state.last_obs,
        key=key,
    )

    return new_runner_state, metrics


def evolve_step(
    runner_state: RunnerState,
    config: Config,
) -> tuple[RunnerState, dict[str, Any]]:
    """Execute one evolution-only iteration (no gradient updates).

    Collects a rollout (agents act using frozen policy, environment
    handles reproduction/mutation/death), but does NOT compute losses
    or update shared parameters. Only per-agent params change via
    reproduction and mutation in the environment step.

    Args:
        runner_state: Current training state.
        config: Master configuration.

    Returns:
        (new_runner_state, metrics) with rollout-level metrics only.
    """
    vec_env = VecEnv(config)
    num_actions = 6
    network = ActorCritic(
        hidden_dims=tuple(config.agent.hidden_dims),
        num_actions=num_actions,
    )

    # Collect rollout (agents act, environment evolves)
    runner_state, batch = collect_rollout(runner_state, network, vec_env, config)

    # Compute rollout-level metrics (no loss/gradient metrics)
    alive_mask = batch['alive_mask']
    alive_f = alive_mask.astype(jnp.float32)
    alive_count = jnp.maximum(jnp.sum(alive_f), 1.0)

    metrics: dict[str, Any] = {}
    metrics['mean_reward'] = jnp.sum(batch['rewards'] * alive_f) / alive_count
    metrics['mean_value'] = jnp.sum(batch['values'] * alive_f) / alive_count

    # Population metrics
    pop_per_step = jnp.sum(alive_f, axis=-1)
    metrics['population_size'] = jnp.mean(pop_per_step)
    metrics['births_this_step'] = jnp.sum(batch['births_this_step'])
    metrics['deaths_this_step'] = jnp.sum(batch['deaths_this_step'])

    # Energy stats
    final_env = runner_state.env_state
    final_alive = final_env.agent_alive
    final_alive_f = final_alive.astype(jnp.float32)
    final_energy = final_env.agent_energy
    final_alive_count = jnp.maximum(jnp.sum(final_alive_f), 1.0)
    metrics['mean_energy'] = jnp.sum(final_energy * final_alive_f) / final_alive_count
    metrics['max_energy'] = jnp.max(jnp.where(final_alive, final_energy, -jnp.inf))
    metrics['min_energy'] = jnp.min(jnp.where(final_alive, final_energy, jnp.inf))

    # Oldest agent age
    final_step = final_env.step
    final_birth = final_env.agent_birth_step
    agent_ages = jnp.where(
        final_alive,
        final_step[:, None] - final_birth,
        jnp.int32(0),
    )
    metrics['oldest_agent_age'] = jnp.max(agent_ages)

    # Set zero placeholders for gradient metrics (so logging doesn't break)
    metrics['total_loss'] = jnp.float32(0.0)
    metrics['policy_loss'] = jnp.float32(0.0)
    metrics['value_loss'] = jnp.float32(0.0)
    metrics['entropy'] = jnp.float32(0.0)
    metrics['clip_fraction'] = jnp.float32(0.0)

    # No param sync during evolve — per-agent params diverge freely
    return runner_state, metrics


def train(config: Config) -> RunnerState:
    """Main training entry point.

    Initializes environment, network, optimizer, and runner state,
    then runs the PPO training loop with progress bar and metric logging.

    Args:
        config: Master configuration.

    Returns:
        Final RunnerState after training.
    """
    key = jax.random.PRNGKey(config.train.seed)

    print("=" * 60)
    print("Emergence Lab — Phase 4: Research Microscope")
    print("=" * 60)
    print(f"Grid: {config.env.grid_size}x{config.env.grid_size}")
    print(f"Agents: {config.env.num_agents} (max: {config.evolution.max_agents})")
    print(f"Food: {config.env.num_food}")
    print(f"Evolution: {'enabled' if config.evolution.enabled else 'disabled'}")
    print(f"Training mode: {config.train.training_mode.value}")
    if config.train.training_mode == TrainingMode.FREEZE_EVOLVE:
        print(f"  Gradient steps: {config.freeze_evolve.gradient_steps}")
        print(f"  Evolve steps: {config.freeze_evolve.evolve_steps}")
        print(f"  Evolve mutation boost: {config.freeze_evolve.evolve_mutation_boost}x")
    if config.specialization.diversity_bonus != 0.0 or config.specialization.niche_pressure != 0.0:
        print(f"Specialization: diversity_bonus={config.specialization.diversity_bonus}, "
              f"niche_pressure={config.specialization.niche_pressure}")
        if config.specialization.layer_mutation_rates:
            print(f"  Layer mutation rates: {config.specialization.layer_mutation_rates}")
    print(f"Field channels: {config.field.num_channels}")
    print(f"Num envs: {config.train.num_envs}")
    print(f"Total steps: {config.train.total_steps}")
    print(f"Steps per rollout: {config.train.num_steps}")
    print(f"Seed: {config.train.seed}")
    print("=" * 60)

    # Steps per training iteration = num_envs * num_steps * max_agents
    # Uses max_agents since the batch has (max_agents,) agent dimension
    steps_per_iter = (
        config.train.num_envs * config.train.num_steps * config.evolution.max_agents
    )
    num_iterations = config.train.total_steps // steps_per_iter
    if num_iterations < 1:
        num_iterations = 1

    print(f"Steps per iteration: {steps_per_iter}")
    print(f"Number of iterations: {num_iterations}")
    print(f"Log interval: {config.log.log_interval} steps")
    print()

    # Initialize W&B if enabled
    if config.log.wandb:
        print("Initializing W&B...")
        init_wandb(config)
        print("W&B initialized.")

    # Initialize training state
    print("Initializing training state...")
    runner_state = create_train_state(config, key)
    print("Training state initialized.")

    # Resume from checkpoint if specified
    if config.train.resume_from is not None:
        print(f"Loading checkpoint from {config.train.resume_from}...")
        with open(config.train.resume_from, "rb") as f:
            checkpoint_data = pickle.load(f)

        if isinstance(checkpoint_data, dict) and "agent_params" in checkpoint_data:
            loaded_params = checkpoint_data["params"]
        else:
            loaded_params = checkpoint_data

        # Replace shared params and reinitialize optimizer
        runner_state = runner_state.replace(  # type: ignore[attr-defined]
            params=loaded_params,
            opt_state=optax.chain(
                optax.clip_by_global_norm(config.train.max_grad_norm),
                optax.adam(config.train.learning_rate, eps=1e-5),
            ).init(loaded_params),
        )

        # If evolution is enabled, replicate loaded params to per-agent params
        if config.evolution.enabled:
            max_agents = config.evolution.max_agents
            num_envs = config.train.num_envs
            per_agent_params = jax.tree.map(
                lambda leaf: jnp.broadcast_to(
                    leaf[None, None], (num_envs, max_agents) + leaf.shape
                ).copy(),
                loaded_params,
            )
            env_state = runner_state.env_state.replace(  # type: ignore[attr-defined]
                agent_params=per_agent_params,
            )
            runner_state = runner_state.replace(  # type: ignore[attr-defined]
                env_state=env_state,
            )

        # Recompute observations with new params
        last_obs = jax.vmap(lambda s: get_observations(s, config))(
            runner_state.env_state
        )
        runner_state = runner_state.replace(  # type: ignore[attr-defined]
            last_obs=last_obs,
        )
        print("Checkpoint loaded.")

    # Initialize emergence tracker
    emergence_tracker = EmergenceTracker(config)

    # Initialize specialization tracker
    specialization_tracker = SpecializationTracker(config)

    # JIT-compile train_step and evolve_step with config captured in closure
    training_mode = config.train.training_mode
    use_evolve = training_mode in (TrainingMode.EVOLVE, TrainingMode.FREEZE_EVOLVE)

    print("JIT compiling train_step...")

    @jax.jit
    def jit_train_step(rs: RunnerState) -> tuple[RunnerState, dict[str, Any]]:
        return train_step(rs, config)

    if use_evolve:
        import dataclasses as _dc

        # Create config with boosted mutation for evolve phases
        boosted_mutation_std = (
            config.evolution.mutation_std * config.freeze_evolve.evolve_mutation_boost
        )
        evolve_config = _dc.replace(
            config,
            evolution=_dc.replace(
                config.evolution,
                mutation_std=boosted_mutation_std,
            ),
        )

        print(
            f"JIT compiling evolve_step "
            f"(mutation_std={boosted_mutation_std:.4f})..."
        )

        @jax.jit
        def jit_evolve_step(rs: RunnerState) -> tuple[RunnerState, dict[str, Any]]:
            return evolve_step(rs, evolve_config)

    # Warm up JIT with first iteration
    t0 = time.time()
    if training_mode == TrainingMode.EVOLVE:
        runner_state, metrics = jit_evolve_step(runner_state)
    else:
        runner_state, metrics = jit_train_step(runner_state)
    jax.block_until_ready(metrics)
    jit_time = time.time() - t0
    print(f"JIT compilation done ({jit_time:.1f}s)")
    print()

    total_env_steps = steps_per_iter  # Already did one iteration

    # Freeze-evolve phase tracking
    if training_mode == TrainingMode.FREEZE_EVOLVE:
        fe_cfg = config.freeze_evolve
        # Cycle length in env steps
        cycle_length = fe_cfg.gradient_steps + fe_cfg.evolve_steps
        current_phase = TrainingMode.GRADIENT  # Start with gradient phase
        phase_step_counter = steps_per_iter  # Steps in current phase
        phase_transitions: list[tuple[int, str]] = []  # (step, phase_name)
        print(f"Freeze-Evolve: starting GRADIENT phase "
              f"(cycle: {fe_cfg.gradient_steps}G + {fe_cfg.evolve_steps}E)")
    else:
        current_phase = training_mode

    # Training loop
    pbar = tqdm(
        range(1, num_iterations),
        desc="Training",
        unit="iter",
        file=sys.stderr,
    )

    for iteration in pbar:
        # Determine which step function to use
        if training_mode == TrainingMode.EVOLVE:
            runner_state, metrics = jit_evolve_step(runner_state)
        elif training_mode == TrainingMode.FREEZE_EVOLVE:
            if current_phase == TrainingMode.GRADIENT:
                runner_state, metrics = jit_train_step(runner_state)
            else:
                runner_state, metrics = jit_evolve_step(runner_state)
        else:
            runner_state, metrics = jit_train_step(runner_state)

        total_env_steps += steps_per_iter

        # Freeze-evolve phase switching
        if training_mode == TrainingMode.FREEZE_EVOLVE:
            phase_step_counter += steps_per_iter
            if current_phase == TrainingMode.GRADIENT:
                if phase_step_counter >= fe_cfg.gradient_steps:
                    current_phase = TrainingMode.EVOLVE
                    phase_step_counter = 0
                    phase_transitions.append((total_env_steps, "EVOLVE"))
                    tqdm.write(
                        f"FREEZE-EVOLVE: Switching to EVOLVE phase at step {total_env_steps}"
                    )
            else:
                if phase_step_counter >= fe_cfg.evolve_steps:
                    current_phase = TrainingMode.GRADIENT
                    phase_step_counter = 0
                    phase_transitions.append((total_env_steps, "GRADIENT"))
                    tqdm.write(
                        f"FREEZE-EVOLVE: Switching to GRADIENT phase at step {total_env_steps}"
                    )

        # Log metrics at intervals
        if total_env_steps % config.log.log_interval < steps_per_iter:
            # Extract scalar values from JAX arrays for display
            reward = float(metrics["mean_reward"])
            loss = float(metrics["total_loss"])
            entropy = float(metrics["entropy"])
            pop = float(metrics["population_size"])

            postfix: dict[str, Any] = {
                "reward": f"{reward:.4f}",
                "loss": f"{loss:.4f}",
                "entropy": f"{entropy:.4f}",
                "pop": f"{pop:.1f}",
                "steps": total_env_steps,
            }
            if training_mode == TrainingMode.FREEZE_EVOLVE:
                phase_label = "G" if current_phase == TrainingMode.GRADIENT else "E"
                postfix["phase"] = phase_label
            pbar.set_postfix(**postfix)

            # Log to W&B
            if config.log.wandb:
                if training_mode == TrainingMode.FREEZE_EVOLVE:
                    metrics["freeze_evolve/phase"] = jnp.float32(
                        1.0 if current_phase == TrainingMode.GRADIENT else 0.0
                    )
                log_metrics(metrics, step=total_env_steps)

            # Check for NaN/Inf (only in gradient mode — evolve has zero loss)
            is_gradient = (
                training_mode == TrainingMode.GRADIENT
                or (training_mode == TrainingMode.FREEZE_EVOLVE
                    and current_phase == TrainingMode.GRADIENT)
            )
            if is_gradient and (jnp.isnan(loss) or jnp.isinf(loss)):
                print(f"\nWARNING: NaN/Inf detected at step {total_env_steps}!")
                break

        # Emergence tracking at configured interval
        if total_env_steps % config.analysis.emergence_check_interval < steps_per_iter:
            # Use field from the first environment for metrics
            first_env_field = jax.tree.map(lambda x: x[0], runner_state.env_state.field_state)
            new_events = emergence_tracker.update(first_env_field, step=total_env_steps)
            for event in new_events:
                tqdm.write(f"EMERGENCE: {event}")

            # Log emergence metrics to W&B
            if config.log.wandb:
                emergence_metrics = emergence_tracker.get_metrics()
                log_metrics(emergence_metrics, step=total_env_steps)

        # Specialization tracking at configured interval
        if (
            config.evolution.enabled
            and total_env_steps % config.analysis.specialization_check_interval
            < steps_per_iter
        ):
            # Use per-agent params from the first environment
            first_env_agent_params = jax.tree.map(
                lambda x: x[0], runner_state.env_state.agent_params
            )
            first_env_alive = np.asarray(
                runner_state.env_state.agent_alive[0]
            )
            spec_events = specialization_tracker.update(
                first_env_agent_params, first_env_alive, step=total_env_steps
            )
            for spec_event in spec_events:
                tqdm.write(f"SPECIALIZATION: {spec_event}")

            # Log specialization metrics to W&B
            if config.log.wandb:
                spec_metrics = specialization_tracker.get_metrics()
                log_metrics(spec_metrics, step=total_env_steps)

    pbar.close()

    # Print emergence summary
    emergence_summary = emergence_tracker.get_summary()
    if emergence_summary["total_events"] > 0:
        print()
        print("Emergence Events Detected:")
        for event_str in emergence_summary["events"]:
            print(f"  {event_str}")

    # Print specialization summary
    if config.evolution.enabled:
        spec_summary = specialization_tracker.get_summary()
        if spec_summary["total_updates"] > 0:
            print()
            print("Specialization Tracking Summary:")
            print(f"  Updates: {spec_summary['total_updates']}")
            if "weight_divergence_final" in spec_summary:
                print(
                    f"  Final weight divergence: "
                    f"{spec_summary['weight_divergence_final']:.6f}"
                )
                print(
                    f"  Mean weight divergence: "
                    f"{spec_summary['weight_divergence_mean']:.6f}"
                )
            if spec_summary["total_events"] > 0:
                print(f"  Specialization events: {spec_summary['total_events']}")
                for event_str in spec_summary["events"]:
                    print(f"    {event_str}")

    # Print freeze-evolve phase transition summary
    if training_mode == TrainingMode.FREEZE_EVOLVE and phase_transitions:
        print()
        print("Freeze-Evolve Phase Transitions:")
        for step_num, phase_name in phase_transitions:
            print(f"  Step {step_num}: → {phase_name}")
        print(f"  Total transitions: {len(phase_transitions)}")

    print()
    print("=" * 60)
    print("Training complete!")
    print(f"Total env steps: {total_env_steps}")
    print(f"Training mode: {config.train.training_mode.value}")
    print("Final metrics:")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {float(v):.6f}")
    print("=" * 60)

    # Save checkpoint
    checkpoint_dir = config.log.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "params.pkl")
    if (
        config.evolution.enabled
        and runner_state.env_state.agent_params is not None
    ):
        # Structured checkpoint with per-agent data
        checkpoint_data: dict = {
            "params": runner_state.params,
            "agent_params": jax.tree_util.tree_map(
                lambda x: x[0], runner_state.env_state.agent_params
            ),
            "alive_mask": runner_state.env_state.agent_alive[0],
        }
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint_data, f)
    else:
        # Raw Flax variables dict (backward compatible)
        with open(checkpoint_path, "wb") as f:
            pickle.dump(runner_state.params, f)
    print(f"Checkpoint saved to {checkpoint_path}")

    # Finish W&B run
    if config.log.wandb:
        finish_wandb()

    final_state: RunnerState = runner_state
    return final_state


def main() -> None:
    """CLI entry point using tyro for argument parsing."""
    import tyro

    config = tyro.cli(Config)
    train(config)


if __name__ == "__main__":
    main()
