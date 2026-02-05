"""Parallel multi-seed training for running experiments on TPU.

This module provides infrastructure to run N seeds in parallel using jax.vmap,
enabling efficient statistical studies on TPU (e.g., Google Colab TPU v5e).

Usage:
    from src.training.parallel_train import ParallelTrainer

    trainer = ParallelTrainer(
        config=config,
        num_seeds=5,
        seed_ids=[0, 1, 2, 3, 4],
        checkpoint_dir="/content/drive/MyDrive/emergence-lab",
    )
    trainer.train(num_iterations=1000, checkpoint_interval_minutes=30)
"""

import copy
import os
import pickle
import time
from dataclasses import dataclass
from typing import Any

import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.agents.network import ActorCritic
from src.agents.policy import sample_actions
from src.configs import Config, TrainingMode
from src.environment.obs import get_observations, obs_dim
from src.environment.vec_env import VecEnv
from src.training.gae import compute_gae
from src.training.ppo import ppo_loss
from src.training.rollout import RunnerState, collect_rollout


@flax.struct.dataclass
class ParallelRunnerState:
    """State for parallel multi-seed training.

    Each leaf has an additional leading dimension for the seed index.

    Attributes:
        params: Network parameters, shape (num_seeds, ...).
        opt_state: Optimizer state, shape (num_seeds, ...).
        env_state: Batched environment state, shape (num_seeds, num_envs, ...).
        last_obs: Last observations, shape (num_seeds, num_envs, max_agents, obs_dim).
        keys: PRNG keys, shape (num_seeds, 2).
    """
    params: Any
    opt_state: Any
    env_state: Any
    last_obs: jnp.ndarray
    keys: jax.Array


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    seed_id: int
    step: int
    timestamp: float
    path: str


def _jax_to_numpy(tree: Any) -> Any:
    """Convert all JAX arrays in a pytree to numpy arrays."""
    return jax.tree_util.tree_map(
        lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x,
        tree,
    )


def _numpy_to_jax(tree: Any) -> Any:
    """Convert all numpy arrays in a pytree to JAX arrays."""
    return jax.tree_util.tree_map(
        lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x,
        tree,
    )


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
    norms = jnp.linalg.norm(weight_matrix, axis=-1, keepdims=True)
    norms = jnp.maximum(norms, 1e-8)
    normalized = weight_matrix / norms
    cos_sim = normalized @ normalized.T
    cos_dist = 1.0 - cos_sim

    # Mask: only consider alive pairs
    pair_mask = alive_f[:, None] * alive_f[None, :]
    self_mask = 1.0 - jnp.eye(max_agents)
    pair_mask = pair_mask * self_mask

    bonuses = jnp.zeros(max_agents)

    # Diversity bonus: mean cosine distance to all other alive agents
    if diversity_bonus != 0.0:
        pair_count = jnp.maximum(jnp.sum(pair_mask, axis=-1), 1.0)
        mean_dist = jnp.sum(cos_dist * pair_mask, axis=-1) / pair_count
        bonuses = bonuses + diversity_bonus * mean_dist * alive_f

    # Niche pressure: penalize agents too close to nearest neighbor
    if niche_pressure != 0.0:
        large_dist = jnp.where(pair_mask > 0, cos_dist, 1e6)
        min_dist = jnp.min(large_dist, axis=-1)
        penalty = niche_pressure * jnp.maximum(1.0 - min_dist, 0.0)
        bonuses = bonuses - penalty * alive_f

    return bonuses


def sample_actions_per_agent(
    network: ActorCritic,
    per_agent_params: Any,
    obs: jnp.ndarray,
    key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample actions where each agent uses its OWN per-agent params.

    Unlike sample_actions() which uses shared params for all agents,
    this function gives each agent its own network weights. This is
    essential for evolve phases where mutation should produce different
    behaviors.

    Args:
        network: ActorCritic network module.
        per_agent_params: Per-agent params pytree, leaves shape (num_envs, max_agents, ...).
        obs: Observations of shape (num_envs, max_agents, obs_dim).
        key: PRNG key for sampling.

    Returns:
        Tuple of (actions, log_probs, values, entropy), each (num_envs, max_agents).
    """
    num_envs, max_agents, obs_dim_val = obs.shape

    # Flatten envs and agents into single batch dimension
    flat_obs = obs.reshape(num_envs * max_agents, obs_dim_val)
    flat_params = jax.tree.map(
        lambda x: x.reshape(num_envs * max_agents, *x.shape[2:]),
        per_agent_params,
    )

    # Key difference: in_axes=(0, 0) — each element gets its OWN params
    batched_apply = jax.vmap(network.apply, in_axes=(0, 0))
    out = batched_apply(flat_params, flat_obs)
    logits = out[0]  # (num_envs * max_agents, num_actions)
    values = jnp.asarray(out[1])  # (num_envs * max_agents,)

    # Categorical sampling (same as sample_actions)
    log_probs_all = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(log_probs_all)

    keys = jax.random.split(key, num_envs * max_agents)
    actions = jax.vmap(lambda k, p: jax.random.categorical(k, jnp.log(p)))(
        keys, probs
    )

    log_probs = jnp.take_along_axis(
        log_probs_all, actions[:, None], axis=-1
    ).squeeze(-1)

    entropy = -jnp.sum(probs * log_probs_all, axis=-1)

    # Reshape back to (num_envs, max_agents)
    actions = actions.reshape(num_envs, max_agents)
    log_probs = log_probs.reshape(num_envs, max_agents)
    values = values.reshape(num_envs, max_agents)
    entropy = entropy.reshape(num_envs, max_agents)

    return actions, log_probs, values, entropy


def create_single_seed_state(
    config: Config,
    key: jax.Array,
) -> RunnerState:
    """Initialize training state for a single seed.

    This is a pure function suitable for vmapping.

    Args:
        config: Master configuration.
        key: PRNG key for this seed.

    Returns:
        Initialized RunnerState for this seed.
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

    # Initialize per-agent params if evolution is enabled
    if config.evolution.enabled:
        max_agents = config.evolution.max_agents
        num_envs = config.train.num_envs
        per_agent_params = jax.tree.map(
            lambda leaf: jnp.broadcast_to(
                leaf[None, None], (num_envs, max_agents) + leaf.shape
            ).copy(),
            params,
        )
        env_state = env_state.replace(agent_params=per_agent_params)

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


def single_seed_train_step(
    runner_state: RunnerState,
    config: Config,
) -> tuple[RunnerState, dict[str, Any]]:
    """Execute one PPO training iteration for a single seed.

    This is a pure function suitable for vmapping.

    Args:
        runner_state: Current training state for this seed.
        config: Master configuration.

    Returns:
        (new_runner_state, metrics) tuple.
    """
    vec_env = VecEnv(config)
    num_actions = 6
    network = ActorCritic(
        hidden_dims=tuple(config.agent.hidden_dims),
        num_actions=num_actions,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.train.max_grad_norm),
        optax.adam(config.train.learning_rate, eps=1e-5),
    )

    # Collect rollout
    runner_state, batch = collect_rollout(runner_state, network, vec_env, config)

    # Compute bootstrap value
    key, bootstrap_key = jax.random.split(runner_state.key)
    _, _, bootstrap_values, _ = sample_actions(
        network, runner_state.params, runner_state.last_obs, bootstrap_key
    )

    # Compute GAE
    alive_mask = batch['alive_mask']
    alive_f = alive_mask.astype(jnp.float32)
    masked_rewards = batch['rewards'] * alive_f
    masked_values = batch['values'] * alive_f

    # Add specialization bonuses if configured
    div_bonus = config.specialization.diversity_bonus
    niche_p = config.specialization.niche_pressure
    if (div_bonus > 0 or niche_p > 0) and runner_state.env_state.agent_params is not None:
        # Compute bonuses per env using the per-agent params
        # agent_params leaves: (num_envs, max_agents, ...)
        def _bonus_single_env(env_agent_params, env_alive):
            return _compute_specialization_bonuses(
                env_agent_params, env_alive, div_bonus, niche_p,
            )

        # Extract per-env agent params and alive masks
        env_alive_mask = runner_state.env_state.agent_alive  # (num_envs, max_agents)
        spec_bonuses = jax.vmap(_bonus_single_env)(
            runner_state.env_state.agent_params, env_alive_mask,
        )  # (num_envs, max_agents)
        # Broadcast over timesteps: (1, num_envs, max_agents)
        masked_rewards = masked_rewards + spec_bonuses[None, :, :]

    final_alive = runner_state.env_state.agent_alive
    masked_bootstrap = bootstrap_values * final_alive.astype(jnp.float32)

    all_values = jnp.concatenate(
        [masked_values, masked_bootstrap[None, :, :]], axis=0
    )

    dones_expanded = jnp.broadcast_to(
        batch['dones'][:, :, None],
        masked_rewards.shape,
    )

    def _gae_single(rewards: jnp.ndarray, values: jnp.ndarray, dones: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return compute_gae(rewards, values, dones, config.train.gamma, config.train.gae_lambda)

    num_steps = masked_rewards.shape[0]
    num_envs = masked_rewards.shape[1]
    max_agents = masked_rewards.shape[2]

    rewards_flat = masked_rewards.reshape(num_steps, num_envs * max_agents)
    values_flat = all_values.reshape(num_steps + 1, num_envs * max_agents)
    dones_flat = dones_expanded.reshape(num_steps, num_envs * max_agents)

    advantages_flat, returns_flat = jax.vmap(
        _gae_single, in_axes=(1, 1, 1), out_axes=(1, 1)
    )(rewards_flat, values_flat, dones_flat)

    advantages = advantages_flat.reshape(num_steps, num_envs, max_agents)
    returns = returns_flat.reshape(num_steps, num_envs, max_agents)

    # Flatten for PPO
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

    # PPO epochs and minibatches
    num_epochs = config.train.num_epochs
    minibatch_size = config.train.minibatch_size
    num_minibatches = batch_size // minibatch_size if batch_size >= minibatch_size else 1

    def _epoch_step(
        carry: tuple[Any, Any, jax.Array],
        _unused: None,
    ) -> tuple[tuple[Any, Any, jax.Array], dict[str, jnp.ndarray]]:
        params, opt_state, epoch_key = carry

        epoch_key, shuffle_key = jax.random.split(epoch_key)
        perm = jax.random.permutation(shuffle_key, batch_size)

        shuffled_batch = {k: v[perm] for k, v in flat_batch.items()}

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

            # Normalize advantages
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

            updates, new_opt_state = optimizer.update(grads, mb_opt_state, mb_params)
            new_params = optax.apply_updates(mb_params, updates)

            metrics['total_loss'] = loss
            return (new_params, new_opt_state), metrics

        (params, opt_state), epoch_metrics = jax.lax.scan(
            _minibatch_step,
            (params, opt_state),
            jnp.arange(num_minibatches),
        )

        avg_metrics = jax.tree.map(lambda x: jnp.mean(x), epoch_metrics)
        return (params, opt_state, epoch_key), avg_metrics

    (new_params, new_opt_state, key), all_epoch_metrics = jax.lax.scan(
        _epoch_step,
        (runner_state.params, runner_state.opt_state, key),
        None,
        length=num_epochs,
    )

    metrics = jax.tree.map(lambda x: jnp.mean(x), all_epoch_metrics)

    # Add rollout metrics
    alive_count = jnp.maximum(jnp.sum(alive_f), 1.0)
    metrics['mean_reward'] = jnp.sum(batch['rewards'] * alive_f) / alive_count
    metrics['mean_value'] = jnp.sum(batch['values'] * alive_f) / alive_count

    pop_per_step = jnp.sum(alive_f, axis=-1)
    metrics['population_size'] = jnp.mean(pop_per_step)
    metrics['births_this_step'] = jnp.sum(batch['births_this_step'])
    metrics['deaths_this_step'] = jnp.sum(batch['deaths_this_step'])

    # Sync per-agent params
    env_state = runner_state.env_state
    if env_state.agent_params is not None:
        num_envs_val = config.train.num_envs
        max_agents_val = config.evolution.max_agents
        updated_per_agent = jax.tree.map(
            lambda leaf: jnp.broadcast_to(
                leaf[None, None], (num_envs_val, max_agents_val) + leaf.shape
            ).copy(),
            new_params,
        )
        current_alive = env_state.agent_alive

        def _sync_params(updated: jnp.ndarray, existing: jnp.ndarray) -> jnp.ndarray:
            ndim_extra = existing.ndim - current_alive.ndim
            mask = current_alive
            for _ in range(ndim_extra):
                mask = mask[..., None]
            return jnp.where(mask, updated, existing)

        new_agent_params = jax.tree.map(
            _sync_params, updated_per_agent, env_state.agent_params
        )
        env_state = env_state.replace(agent_params=new_agent_params)

    new_runner_state = RunnerState(
        params=new_params,
        opt_state=new_opt_state,
        env_state=env_state,
        last_obs=runner_state.last_obs,
        key=key,
    )

    return new_runner_state, metrics


def collect_evolve_rollout(
    runner_state: RunnerState,
    network: ActorCritic,
    vec_env: VecEnv,
    config: Config,
) -> tuple[RunnerState, dict[str, jnp.ndarray]]:
    """Collect a rollout using per-agent params for action selection.

    Unlike collect_rollout() which uses shared params, this uses each agent's
    own per-agent params stored in env_state.agent_params. This is essential
    for evolve phases where mutation should produce different behaviors.

    Args:
        runner_state: Current runner state (agent_params must be set).
        network: ActorCritic network module.
        vec_env: Vectorized environment.
        config: Master configuration.

    Returns:
        Tuple of (new_runner_state, batch) with same structure as collect_rollout.
    """
    num_steps = config.train.num_steps

    def _step_fn(
        carry: RunnerState,
        _unused: None,
    ) -> tuple[RunnerState, dict[str, jnp.ndarray]]:
        rs = carry

        # Sample actions using PER-AGENT PARAMS (not shared params)
        key, action_key = jax.random.split(rs.key)
        actions, log_probs, values, _entropy = sample_actions_per_agent(
            network, rs.env_state.agent_params, rs.last_obs, action_key
        )

        # Step the environment
        env_state, rewards, dones, info = vec_env.step(rs.env_state, actions)

        # Get new observations
        new_obs = jax.vmap(lambda s: get_observations(s, config))(env_state)

        # Store transition data
        alive_mask = rs.env_state.agent_alive
        transition = {
            'obs': rs.last_obs,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'values': values,
            'log_probs': log_probs,
            'alive_mask': alive_mask,
            'births_this_step': info['births_this_step'],
            'deaths_this_step': info['deaths_this_step'],
        }

        new_rs = RunnerState(
            params=rs.params,
            opt_state=rs.opt_state,
            env_state=env_state,
            last_obs=new_obs,
            key=key,
        )

        return new_rs, transition

    new_runner_state, batch = jax.lax.scan(
        _step_fn,
        runner_state,
        None,
        length=num_steps,
    )

    return new_runner_state, batch


def single_seed_evolve_step(
    runner_state: RunnerState,
    config: Config,
) -> tuple[RunnerState, dict[str, Any]]:
    """Execute one evolution-only iteration for a single seed.

    Per-agent action selection (each agent uses its own params), no PPO
    gradient updates, no param sync. Mutation happens naturally through
    reproduction in env.step().

    Args:
        runner_state: Current training state for this seed.
        config: Master configuration.

    Returns:
        (new_runner_state, metrics) tuple.
    """
    vec_env = VecEnv(config)
    num_actions = 6
    network = ActorCritic(
        hidden_dims=tuple(config.agent.hidden_dims),
        num_actions=num_actions,
    )

    # Collect rollout using PER-AGENT PARAMS
    runner_state, batch = collect_evolve_rollout(
        runner_state, network, vec_env, config
    )

    # Compute metrics only (no PPO loss)
    alive_f = batch['alive_mask'].astype(jnp.float32)
    alive_count = jnp.maximum(jnp.sum(alive_f), 1.0)
    metrics = {
        'mean_reward': jnp.sum(batch['rewards'] * alive_f) / alive_count,
        'mean_value': jnp.sum(batch['values'] * alive_f) / alive_count,
        'population_size': jnp.mean(jnp.sum(alive_f, axis=-1)),
        'births_this_step': jnp.sum(batch['births_this_step']),
        'deaths_this_step': jnp.sum(batch['deaths_this_step']),
        # Zero placeholders for gradient metrics
        'total_loss': jnp.float32(0.0),
        'policy_loss': jnp.float32(0.0),
        'value_loss': jnp.float32(0.0),
        'entropy': jnp.float32(0.0),
        'clip_fraction': jnp.float32(0.0),
    }

    # NO PPO loss, NO param update, NO param sync
    # Per-agent params diverge freely via mutation in env.step()
    return runner_state, metrics


class ParallelTrainer:
    """Parallel multi-seed trainer for TPU.

    Runs N seeds in parallel using jax.vmap, with periodic checkpointing
    to Google Drive (or local filesystem).

    Example:
        trainer = ParallelTrainer(
            config=config,
            num_seeds=5,
            seed_ids=[0, 1, 2, 3, 4],
            checkpoint_dir="/content/drive/MyDrive/emergence-lab",
        )
        trainer.train(num_iterations=1000, checkpoint_interval_minutes=30)
    """

    def __init__(
        self,
        config: Config,
        num_seeds: int = 5,
        seed_ids: list[int] | None = None,
        checkpoint_dir: str = "checkpoints/parallel",
        master_seed: int = 42,
    ):
        """Initialize the parallel trainer.

        Args:
            config: Master configuration.
            num_seeds: Number of seeds to run in parallel.
            seed_ids: List of seed IDs for checkpoint naming. If None, uses [0, 1, ...].
            checkpoint_dir: Base directory for checkpoints. Each seed gets a subdirectory.
            master_seed: Master PRNG seed for generating per-seed keys.
        """
        self.config = config
        self.num_seeds = num_seeds
        self.seed_ids = seed_ids if seed_ids is not None else list(range(num_seeds))
        self.checkpoint_dir = checkpoint_dir
        self.master_seed = master_seed

        # Disable W&B for parallel training (too many runs)
        self.config.log.wandb = False

        # Create checkpoint directories
        for seed_id in self.seed_ids:
            seed_dir = os.path.join(checkpoint_dir, f"seed_{seed_id}")
            os.makedirs(seed_dir, exist_ok=True)

        # Initialize state
        self._parallel_state: ParallelRunnerState | None = None
        self._current_step = 0
        self._last_checkpoint_time = 0.0

        # JIT-compiled functions (initialized on first train call)
        self._jit_parallel_step: Any = None
        self._jit_parallel_evolve_step: Any = None

        # Phase tracking for FREEZE_EVOLVE mode
        self._current_phase = TrainingMode.GRADIENT
        self._phase_iter_counter = 0

    def _initialize_state(self) -> ParallelRunnerState:
        """Initialize parallel state for all seeds."""
        master_key = jax.random.PRNGKey(self.master_seed)
        seed_keys = jax.random.split(master_key, self.num_seeds)

        # Create state for each seed using vmap
        # Note: We can't directly vmap create_single_seed_state because VecEnv
        # isn't vmappable. Instead, we initialize each seed sequentially then stack.
        states = []
        for i, key in enumerate(seed_keys):
            print(f"  Initializing seed {self.seed_ids[i]}...")
            state = create_single_seed_state(self.config, key)
            states.append(state)

        # Stack into parallel state
        parallel_state = ParallelRunnerState(
            params=jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *[s.params for s in states]),
            opt_state=jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *[s.opt_state for s in states]),
            env_state=jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *[s.env_state for s in states]),
            last_obs=jnp.stack([s.last_obs for s in states], axis=0),
            keys=jnp.stack([s.key for s in states], axis=0),
        )

        return parallel_state

    def _compile_parallel_step(self) -> None:
        """JIT-compile the parallel train step."""
        config = self.config

        def _parallel_train_step(parallel_state: ParallelRunnerState) -> tuple[ParallelRunnerState, dict[str, Any]]:
            """Execute one training iteration for all seeds in parallel."""

            def _single_step(runner_state: RunnerState) -> tuple[RunnerState, dict[str, Any]]:
                return single_seed_train_step(runner_state, config)

            # Unpack parallel state into per-seed RunnerStates
            def _make_runner_state(i: int) -> RunnerState:
                return RunnerState(
                    params=jax.tree.map(lambda x: x[i], parallel_state.params),
                    opt_state=jax.tree.map(lambda x: x[i], parallel_state.opt_state),
                    env_state=jax.tree.map(lambda x: x[i], parallel_state.env_state),
                    last_obs=parallel_state.last_obs[i],
                    key=parallel_state.keys[i],
                )

            # Run train step for each seed (vmapped)
            # We need to restructure for vmap: create a batched RunnerState
            batched_runner = RunnerState(
                params=parallel_state.params,
                opt_state=parallel_state.opt_state,
                env_state=parallel_state.env_state,
                last_obs=parallel_state.last_obs,
                key=parallel_state.keys,
            )

            # Vmap over the seed dimension (axis 0)
            vmapped_step = jax.vmap(_single_step, in_axes=(RunnerState(
                params=0, opt_state=0, env_state=0, last_obs=0, key=0
            ),))

            new_batched, all_metrics = vmapped_step(batched_runner)

            # Pack back into ParallelRunnerState
            new_parallel_state = ParallelRunnerState(
                params=new_batched.params,
                opt_state=new_batched.opt_state,
                env_state=new_batched.env_state,
                last_obs=new_batched.last_obs,
                keys=new_batched.key,
            )

            return new_parallel_state, all_metrics

        print("JIT compiling parallel train step...")
        t0 = time.time()
        self._jit_parallel_step = jax.jit(_parallel_train_step)

        # Warm up with a single iteration (save/restore state to avoid side effects)
        if self._parallel_state is not None:
            saved_state = self._parallel_state
            self._parallel_state, metrics = self._jit_parallel_step(self._parallel_state)
            jax.block_until_ready(metrics)
            self._parallel_state = saved_state

        print(f"JIT compilation done ({time.time() - t0:.1f}s)")

        # Compile evolve step if needed
        training_mode = config.train.training_mode
        if training_mode in (TrainingMode.FREEZE_EVOLVE, TrainingMode.EVOLVE):
            # Create config with boosted mutation for evolve phases
            evolve_config = copy.deepcopy(config)
            evolve_config.evolution.mutation_std = (
                config.evolution.mutation_std * config.freeze_evolve.evolve_mutation_boost
            )

            def _parallel_evolve_step(
                parallel_state: ParallelRunnerState,
            ) -> tuple[ParallelRunnerState, dict[str, Any]]:
                """Execute one evolve iteration for all seeds in parallel."""

                def _single_evolve(
                    runner_state: RunnerState,
                ) -> tuple[RunnerState, dict[str, Any]]:
                    return single_seed_evolve_step(runner_state, evolve_config)

                batched_runner = RunnerState(
                    params=parallel_state.params,
                    opt_state=parallel_state.opt_state,
                    env_state=parallel_state.env_state,
                    last_obs=parallel_state.last_obs,
                    key=parallel_state.keys,
                )

                vmapped_step = jax.vmap(
                    _single_evolve,
                    in_axes=(RunnerState(
                        params=0, opt_state=0, env_state=0, last_obs=0, key=0
                    ),),
                )

                new_batched, all_metrics = vmapped_step(batched_runner)

                new_parallel_state = ParallelRunnerState(
                    params=new_batched.params,
                    opt_state=new_batched.opt_state,
                    env_state=new_batched.env_state,
                    last_obs=new_batched.last_obs,
                    keys=new_batched.key,
                )

                return new_parallel_state, all_metrics

            print("JIT compiling parallel evolve step...")
            t0 = time.time()
            self._jit_parallel_evolve_step = jax.jit(_parallel_evolve_step)

            # Warm up evolve step (save/restore state to avoid side effects)
            if self._parallel_state is not None:
                saved_state = self._parallel_state
                self._parallel_state, metrics = self._jit_parallel_evolve_step(
                    self._parallel_state
                )
                jax.block_until_ready(metrics)
                self._parallel_state = saved_state

            print(f"Evolve JIT compilation done ({time.time() - t0:.1f}s)")

    def _sync_shared_to_agents(self) -> None:
        """GRADIENT -> EVOLVE: Copy shared params to all alive agents' per-agent params.

        Gives evolution the latest learned behavior as starting point.
        Dead agent slots keep their existing params.
        """
        if self._parallel_state is None:
            return

        ps = self._parallel_state
        config = self.config
        num_envs = config.train.num_envs
        max_agents = config.evolution.max_agents

        # For each seed: broadcast shared_params to (num_envs, max_agents, ...)
        # Then only overwrite alive agents' slots
        def _sync_one_seed(params, env_state):
            updated = jax.tree.map(
                lambda leaf: jnp.broadcast_to(
                    leaf[None, None], (num_envs, max_agents) + leaf.shape
                ).copy(),
                params,
            )
            alive = env_state.agent_alive  # (num_envs, max_agents)

            def _merge(new, old):
                ndim_extra = old.ndim - alive.ndim
                mask = alive
                for _ in range(ndim_extra):
                    mask = mask[..., None]
                return jnp.where(mask, new, old)

            synced = jax.tree.map(_merge, updated, env_state.agent_params)
            return env_state.replace(agent_params=synced)

        # vmap over seeds
        new_env_state = jax.vmap(_sync_one_seed)(ps.params, ps.env_state)
        self._parallel_state = ps.replace(env_state=new_env_state)

    def _reverse_sync_agents_to_shared(self) -> None:
        """EVOLVE -> GRADIENT: Average alive agents' params into shared params.

        Preserves evolutionary discoveries as the starting point for PPO.
        Computes mean of alive agents' per-agent params across all envs.
        """
        if self._parallel_state is None:
            return

        ps = self._parallel_state

        def _mean_one_seed(env_state):
            alive = env_state.agent_alive  # (num_envs, max_agents)
            alive_f = alive.astype(jnp.float32)
            # Total alive across all envs
            total_alive = jnp.maximum(jnp.sum(alive_f), 1.0)

            def _mean_leaf(leaf):
                # leaf shape: (num_envs, max_agents, ...)
                # Expand alive mask to match leaf shape
                ndim_extra = leaf.ndim - alive_f.ndim
                mask = alive_f
                for _ in range(ndim_extra):
                    mask = mask[..., None]
                masked = leaf * mask
                # Sum over envs and agents dimensions
                summed = jnp.sum(masked, axis=(0, 1))
                return summed / total_alive

            return jax.tree.map(_mean_leaf, env_state.agent_params)

        # vmap over seeds
        new_params = jax.vmap(_mean_one_seed)(ps.env_state)
        self._parallel_state = ps.replace(params=new_params)

    def save_checkpoints(self, step: int) -> list[str]:
        """Save checkpoints for all seeds.

        Args:
            step: Current training step.

        Returns:
            List of saved checkpoint paths.
        """
        if self._parallel_state is None:
            raise RuntimeError("No state to save. Call train() first.")

        saved_paths = []

        for i, seed_id in enumerate(self.seed_ids):
            seed_dir = os.path.join(self.checkpoint_dir, f"seed_{seed_id}")
            ckpt_path = os.path.join(seed_dir, f"step_{step}.pkl")

            # Extract this seed's state
            seed_state = {
                "params": jax.tree.map(lambda x: x[i], self._parallel_state.params),
                "opt_state": jax.tree.map(lambda x: x[i], self._parallel_state.opt_state),
                "agent_params": jax.tree.map(
                    lambda x: x[i], self._parallel_state.env_state.agent_params
                ) if self._parallel_state.env_state.agent_params is not None else None,
                "prng_key": self._parallel_state.keys[i],
                "step": step,
                "seed_id": seed_id,
                "config": self.config,
                "phase": self._current_phase.value,
                "phase_iter_counter": self._phase_iter_counter,
            }

            # Convert to numpy for pickling
            serializable = _jax_to_numpy(seed_state)

            # Save checkpoint
            tmp_path = ckpt_path + ".tmp"
            with open(tmp_path, "wb") as f:
                pickle.dump(serializable, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, ckpt_path)

            # Update latest symlink
            latest_path = os.path.join(seed_dir, "latest.pkl")
            if os.path.exists(latest_path) or os.path.islink(latest_path):
                os.remove(latest_path)
            os.symlink(os.path.basename(ckpt_path), latest_path)

            saved_paths.append(ckpt_path)

        self._last_checkpoint_time = time.time()
        return saved_paths

    def load_checkpoints(self) -> bool:
        """Load checkpoints for all seeds if they exist.

        Returns:
            True if all checkpoints were loaded successfully, False otherwise.
        """
        states = []
        min_step = float('inf')

        for seed_id in self.seed_ids:
            seed_dir = os.path.join(self.checkpoint_dir, f"seed_{seed_id}")
            latest_path = os.path.join(seed_dir, "latest.pkl")

            if not os.path.exists(latest_path):
                print(f"No checkpoint found for seed {seed_id}")
                return False

            try:
                with open(latest_path, "rb") as f:
                    data = pickle.load(f)

                # Convert numpy back to JAX
                data = _numpy_to_jax(data)
                states.append(data)

                step = data.get("step", 0)
                min_step = min(min_step, step)
                print(f"  Loaded seed {seed_id} at step {step}")

            except Exception as e:
                print(f"Failed to load checkpoint for seed {seed_id}: {e}")
                return False

        # Reconstruct parallel state from loaded states
        # First create full env states by resetting and replacing params
        print("Reconstructing environment states...")
        master_key = jax.random.PRNGKey(self.master_seed + 1000)
        seed_keys = jax.random.split(master_key, self.num_seeds)

        full_states = []
        for i, (state_data, key) in enumerate(zip(states, seed_keys)):
            # Create fresh env state for structure
            fresh_state = create_single_seed_state(self.config, key)

            # Replace params with loaded params
            if state_data.get("agent_params") is not None:
                env_state = fresh_state.env_state.replace(
                    agent_params=state_data["agent_params"]
                )
            else:
                env_state = fresh_state.env_state

            # Get observations with loaded state
            last_obs = jax.vmap(lambda s: get_observations(s, self.config))(env_state)

            runner_state = RunnerState(
                params=state_data["params"],
                opt_state=state_data["opt_state"],
                env_state=env_state,
                last_obs=last_obs,
                key=state_data.get("prng_key", key),
            )
            full_states.append(runner_state)

        # Stack into parallel state
        self._parallel_state = ParallelRunnerState(
            params=jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *[s.params for s in full_states]),
            opt_state=jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *[s.opt_state for s in full_states]),
            env_state=jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *[s.env_state for s in full_states]),
            last_obs=jnp.stack([s.last_obs for s in full_states], axis=0),
            keys=jnp.stack([s.key for s in full_states], axis=0),
        )

        self._current_step = int(min_step)

        # Restore phase state from checkpoint (use first seed's data)
        loaded_phase = states[0].get("phase", "gradient")
        loaded_counter = states[0].get("phase_iter_counter", 0)
        try:
            self._current_phase = TrainingMode(loaded_phase)
        except ValueError:
            self._current_phase = TrainingMode.GRADIENT
        self._phase_iter_counter = int(loaded_counter)

        print(f"Resumed all seeds from step {self._current_step}")
        if self.config.train.training_mode == TrainingMode.FREEZE_EVOLVE:
            print(f"  Phase: {self._current_phase.value}, phase_iter_counter: {self._phase_iter_counter}")
        return True

    def train(
        self,
        num_iterations: int,
        checkpoint_interval_minutes: float = 30.0,
        resume: bool = True,
        print_interval: int = 10,
    ) -> dict[str, list[float]]:
        """Run parallel training.

        Args:
            num_iterations: Number of training iterations to run.
            checkpoint_interval_minutes: Save checkpoints every N minutes.
            resume: If True, attempt to resume from existing checkpoints.
            print_interval: Print progress every N iterations.

        Returns:
            Dictionary mapping metric names to lists of per-seed values.
        """
        print("=" * 60)
        print("Parallel Multi-Seed Training")
        print("=" * 60)
        print(f"Seeds: {self.seed_ids}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        print(f"Checkpoint interval: {checkpoint_interval_minutes} minutes")
        print()

        # Try to resume
        resumed = False
        if resume:
            print("Attempting to resume from checkpoints...")
            resumed = self.load_checkpoints()

        # Initialize if not resumed
        if not resumed:
            print("Initializing fresh state...")
            self._parallel_state = self._initialize_state()
            self._current_step = 0

        # Compile parallel step
        self._compile_parallel_step()

        # Steps per iteration (agent-steps: num_envs * num_steps * max_agents)
        num_envs = self.config.train.num_envs
        num_steps = self.config.train.num_steps
        steps_per_iter = num_envs * num_steps * self.config.evolution.max_agents

        # Phase iteration counts for FREEZE_EVOLVE
        # gradient_steps/evolve_steps are in agent-steps (same units as total_steps)
        training_mode = self.config.train.training_mode
        gradient_iters = max(
            1, self.config.freeze_evolve.gradient_steps // steps_per_iter
        )
        evolve_iters = max(
            1, self.config.freeze_evolve.evolve_steps // steps_per_iter
        )

        print()
        print(f"Steps per iteration: {steps_per_iter}")
        print(f"Starting from step: {self._current_step}")
        if training_mode == TrainingMode.FREEZE_EVOLVE:
            print(f"FREEZE_EVOLVE: gradient_iters={gradient_iters}, evolve_iters={evolve_iters}")
        elif training_mode == TrainingMode.EVOLVE:
            print("EVOLVE mode: pure evolution, no gradient updates")
        print()

        # Training loop
        self._last_checkpoint_time = time.time()
        checkpoint_interval_seconds = checkpoint_interval_minutes * 60

        all_metrics: dict[str, list[list[float]]] = {
            "mean_reward": [],
            "total_loss": [],
            "population_size": [],
        }

        t_start = time.time()

        for iteration in range(num_iterations):
            # Select step function based on training mode and phase
            if training_mode == TrainingMode.FREEZE_EVOLVE:
                if self._current_phase == TrainingMode.GRADIENT:
                    self._parallel_state, metrics = self._jit_parallel_step(
                        self._parallel_state
                    )
                else:
                    self._parallel_state, metrics = self._jit_parallel_evolve_step(
                        self._parallel_state
                    )

                self._phase_iter_counter += 1

                # Phase transitions
                if (
                    self._current_phase == TrainingMode.GRADIENT
                    and self._phase_iter_counter >= gradient_iters
                ):
                    # GRADIENT -> EVOLVE: copy shared params to agent_params
                    self._sync_shared_to_agents()
                    self._current_phase = TrainingMode.EVOLVE
                    self._phase_iter_counter = 0
                    print(
                        f"\n  [Phase] GRADIENT -> EVOLVE at step {self._current_step}\n"
                    )

                elif (
                    self._current_phase == TrainingMode.EVOLVE
                    and self._phase_iter_counter >= evolve_iters
                ):
                    # EVOLVE -> GRADIENT: reverse sync (mean agent_params -> shared)
                    self._reverse_sync_agents_to_shared()
                    self._current_phase = TrainingMode.GRADIENT
                    self._phase_iter_counter = 0
                    print(
                        f"\n  [Phase] EVOLVE -> GRADIENT at step {self._current_step}\n"
                    )

            elif training_mode == TrainingMode.EVOLVE:
                self._parallel_state, metrics = self._jit_parallel_evolve_step(
                    self._parallel_state
                )

            else:  # GRADIENT (default)
                self._parallel_state, metrics = self._jit_parallel_step(
                    self._parallel_state
                )

            self._current_step += steps_per_iter

            # Record metrics for all seeds
            for key in all_metrics:
                if key in metrics:
                    # metrics[key] has shape (num_seeds,)
                    all_metrics[key].append([float(v) for v in metrics[key]])

            # Print progress
            if iteration % print_interval == 0 or iteration == num_iterations - 1:
                mean_rewards = [float(v) for v in metrics["mean_reward"]]
                losses = [float(v) for v in metrics["total_loss"]]
                elapsed = time.time() - t_start
                iter_per_sec = (iteration + 1) / elapsed if elapsed > 0 else 0

                phase_str = ""
                if training_mode == TrainingMode.FREEZE_EVOLVE:
                    phase_str = f" [{self._current_phase.value}]"

                print(
                    f"Iter {iteration:>5d}/{num_iterations} | "
                    f"Step {self._current_step:>10d}{phase_str} | "
                    f"Reward: {np.mean(mean_rewards):.4f} +/- {np.std(mean_rewards):.4f} | "
                    f"Loss: {np.mean(losses):.4f} | "
                    f"{iter_per_sec:.2f} it/s"
                )

            # Checkpoint if interval elapsed
            elapsed_since_ckpt = time.time() - self._last_checkpoint_time
            if elapsed_since_ckpt >= checkpoint_interval_seconds:
                print(f"\nSaving checkpoints at step {self._current_step}...")
                paths = self.save_checkpoints(self._current_step)
                print(f"Saved {len(paths)} checkpoints\n")

        # Final checkpoint
        print(f"\nSaving final checkpoints at step {self._current_step}...")
        paths = self.save_checkpoints(self._current_step)
        print(f"Saved {len(paths)} checkpoints")

        # Aggregate metrics
        final_metrics = {
            key: [vals[-1][i] for i in range(self.num_seeds)]
            for key, vals in all_metrics.items()
            if vals
        }

        print()
        print("=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Total steps: {self._current_step}")
        print(f"Total time: {time.time() - t_start:.1f}s")

        return final_metrics

    def get_seed_checkpoint_path(self, seed_id: int) -> str:
        """Get the path to the latest checkpoint for a seed.

        Args:
            seed_id: The seed ID.

        Returns:
            Path to the latest checkpoint.
        """
        return os.path.join(self.checkpoint_dir, f"seed_{seed_id}", "latest.pkl")


def train_parallel(
    config: Config,
    num_seeds: int = 5,
    seed_ids: list[int] | None = None,
    checkpoint_dir: str = "checkpoints/parallel",
    num_iterations: int = 1000,
    checkpoint_interval_minutes: float = 30.0,
    resume: bool = True,
    master_seed: int = 42,
) -> dict[str, list[float]]:
    """Convenience function for parallel training.

    Args:
        config: Master configuration.
        num_seeds: Number of seeds to run in parallel.
        seed_ids: List of seed IDs for checkpoint naming.
        checkpoint_dir: Base directory for checkpoints.
        num_iterations: Number of training iterations.
        checkpoint_interval_minutes: Checkpoint interval in minutes.
        resume: Whether to resume from existing checkpoints.
        master_seed: Master PRNG seed.

    Returns:
        Dictionary mapping metric names to lists of per-seed values.
    """
    trainer = ParallelTrainer(
        config=config,
        num_seeds=num_seeds,
        seed_ids=seed_ids,
        checkpoint_dir=checkpoint_dir,
        master_seed=master_seed,
    )

    return trainer.train(
        num_iterations=num_iterations,
        checkpoint_interval_minutes=checkpoint_interval_minutes,
        resume=resume,
    )


if __name__ == "__main__":
    import argparse
    import tyro

    parser = argparse.ArgumentParser(description="Parallel multi-seed training")
    parser.add_argument("--num-seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--iterations", type=int, default=100, help="Training iterations")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/parallel", help="Checkpoint directory")
    parser.add_argument("--checkpoint-interval", type=float, default=30.0, help="Checkpoint interval in minutes")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoints")

    args = parser.parse_args()

    config = Config()

    train_parallel(
        config=config,
        num_seeds=args.num_seeds,
        checkpoint_dir=args.checkpoint_dir,
        num_iterations=args.iterations,
        checkpoint_interval_minutes=args.checkpoint_interval,
        resume=not args.no_resume,
    )
