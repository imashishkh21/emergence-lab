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
from src.configs import Config
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

        # Warm up with a single iteration
        if self._parallel_state is not None:
            self._parallel_state, metrics = self._jit_parallel_step(self._parallel_state)
            jax.block_until_ready(metrics)

        print(f"JIT compilation done ({time.time() - t0:.1f}s)")

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
        print(f"Resumed all seeds from step {self._current_step}")
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

        # Steps per iteration
        steps_per_iter = (
            self.config.train.num_envs
            * self.config.train.num_steps
            * self.config.evolution.max_agents
        )

        print()
        print(f"Steps per iteration: {steps_per_iter}")
        print(f"Starting from step: {self._current_step}")
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
            # Run one parallel iteration
            self._parallel_state, metrics = self._jit_parallel_step(self._parallel_state)
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

                print(
                    f"Iter {iteration:>5d}/{num_iterations} | "
                    f"Step {self._current_step:>10d} | "
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
