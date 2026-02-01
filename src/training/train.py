"""Training step and loop for PPO with shared field dynamics."""

import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from src.agents.network import ActorCritic
from src.agents.policy import sample_actions
from src.configs import Config
from src.environment.obs import get_observations, obs_dim
from src.environment.vec_env import VecEnv
from src.training.gae import compute_gae
from src.training.ppo import ppo_loss
from src.training.rollout import RunnerState, collect_rollout
from src.analysis.emergence import EmergenceTracker
from src.utils.logging import finish_wandb, init_wandb, log_metrics


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
    network = ActorCritic(
        hidden_dims=tuple(config.agent.hidden_dims),
        num_actions=5,
    )

    # Initialize network parameters
    dummy_obs = jnp.zeros((observation_dim,))
    params = network.init(init_key, dummy_obs)

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
    network = ActorCritic(
        hidden_dims=tuple(config.agent.hidden_dims),
        num_actions=5,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.train.max_grad_norm),
        optax.adam(config.train.learning_rate, eps=1e-5),
    )

    # 1. Collect rollout
    runner_state, batch = collect_rollout(runner_state, network, vec_env, config)

    # batch shapes: (num_steps, num_envs, num_agents, ...) or (num_steps, num_envs)
    # rewards: (num_steps, num_envs, num_agents)
    # dones: (num_steps, num_envs)
    # values: (num_steps, num_envs, num_agents)
    # obs: (num_steps, num_envs, num_agents, obs_dim)
    # actions: (num_steps, num_envs, num_agents)
    # log_probs: (num_steps, num_envs, num_agents)

    # 2. Compute bootstrap value for the last observation
    # We need value of the final state for GAE computation
    # Use sample_actions to get the value (we ignore the sampled action)
    key, bootstrap_key = jax.random.split(runner_state.key)
    _, _, bootstrap_values, _ = sample_actions(
        network, runner_state.params, runner_state.last_obs, bootstrap_key
    )
    # bootstrap_values: (num_envs, num_agents)

    # 3. Compute GAE per environment per agent
    # We need values with shape (T+1, num_envs, num_agents) for GAE
    all_values = jnp.concatenate(
        [batch['values'], bootstrap_values[None, :, :]], axis=0
    )  # (T+1, num_envs, num_agents)

    # Expand dones to match agent dimension: (T, num_envs) -> (T, num_envs, num_agents)
    dones_expanded = jnp.broadcast_to(
        batch['dones'][:, :, None],
        batch['rewards'].shape,
    )

    # Vectorize GAE over (env, agent) pairs
    def _gae_single(rewards: jnp.ndarray, values: jnp.ndarray, dones: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return compute_gae(rewards, values, dones, config.train.gamma, config.train.gae_lambda)

    # Reshape to map over env*agent: (T, num_envs * num_agents)
    num_steps = batch['rewards'].shape[0]
    num_envs = batch['rewards'].shape[1]
    num_agents = batch['rewards'].shape[2]

    rewards_flat = batch['rewards'].reshape(num_steps, num_envs * num_agents)
    values_flat = all_values.reshape(num_steps + 1, num_envs * num_agents)
    dones_flat = dones_expanded.reshape(num_steps, num_envs * num_agents)

    # vmap over the env*agent dimension (axis 1)
    advantages_flat, returns_flat = jax.vmap(
        _gae_single, in_axes=(1, 1, 1), out_axes=(1, 1)
    )(rewards_flat, values_flat, dones_flat)
    # advantages_flat: (T, num_envs * num_agents)
    # returns_flat: (T, num_envs * num_agents)

    advantages = advantages_flat.reshape(num_steps, num_envs, num_agents)
    returns = returns_flat.reshape(num_steps, num_envs, num_agents)

    # 4. Flatten rollout into training batch
    # Flatten (num_steps, num_envs, num_agents) -> (batch_size,)
    batch_size = num_steps * num_envs * num_agents
    obs_dim_val = batch['obs'].shape[-1]

    flat_batch = {
        'obs': batch['obs'].reshape(batch_size, obs_dim_val),
        'actions': batch['actions'].reshape(batch_size),
        'log_probs': batch['log_probs'].reshape(batch_size),
        'advantages': advantages.reshape(batch_size),
        'returns': returns.reshape(batch_size),
    }

    # 5. Run multiple epochs of minibatch updates
    num_epochs = config.train.num_epochs
    minibatch_size = config.train.minibatch_size
    # Ensure minibatch_size doesn't exceed batch_size
    effective_minibatch_size = jnp.minimum(minibatch_size, batch_size)
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

            # Normalize advantages within minibatch
            mb_adv = mb['advantages']
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

    # Add rollout-level metrics
    metrics['mean_reward'] = jnp.mean(batch['rewards'])
    metrics['mean_value'] = jnp.mean(batch['values'])
    metrics['mean_advantage'] = jnp.mean(advantages)
    metrics['mean_return'] = jnp.mean(returns)

    new_runner_state = RunnerState(
        params=new_params,
        opt_state=new_opt_state,
        env_state=runner_state.env_state,
        last_obs=runner_state.last_obs,
        key=key,
    )

    return new_runner_state, metrics


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
    print("Emergence Lab — Phase 1: Digital Petri Dish")
    print("=" * 60)
    print(f"Grid: {config.env.grid_size}x{config.env.grid_size}")
    print(f"Agents: {config.env.num_agents}, Food: {config.env.num_food}")
    print(f"Field channels: {config.field.num_channels}")
    print(f"Num envs: {config.train.num_envs}")
    print(f"Total steps: {config.train.total_steps}")
    print(f"Steps per rollout: {config.train.num_steps}")
    print(f"Seed: {config.train.seed}")
    print("=" * 60)

    # Steps per training iteration = num_envs * num_steps * num_agents
    steps_per_iter = (
        config.train.num_envs * config.train.num_steps * config.env.num_agents
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

    # Initialize emergence tracker
    emergence_tracker = EmergenceTracker(config)

    # JIT-compile train_step with config captured in closure
    print("JIT compiling train_step...")

    @jax.jit
    def jit_train_step(rs: RunnerState) -> tuple[RunnerState, dict[str, Any]]:
        return train_step(rs, config)

    # Warm up JIT with first iteration
    t0 = time.time()
    runner_state, metrics = jit_train_step(runner_state)
    # Block until computation completes
    jax.block_until_ready(metrics)
    jit_time = time.time() - t0
    print(f"JIT compilation done ({jit_time:.1f}s)")
    print()

    total_env_steps = steps_per_iter  # Already did one iteration

    # Training loop
    pbar = tqdm(
        range(1, num_iterations),
        desc="Training",
        unit="iter",
        file=sys.stderr,
    )

    for iteration in pbar:
        runner_state, metrics = jit_train_step(runner_state)
        total_env_steps += steps_per_iter

        # Log metrics at intervals
        if total_env_steps % config.log.log_interval < steps_per_iter:
            # Extract scalar values from JAX arrays for display
            reward = float(metrics["mean_reward"])
            loss = float(metrics["total_loss"])
            entropy = float(metrics["entropy"])
            value = float(metrics["mean_value"])

            pbar.set_postfix(
                reward=f"{reward:.4f}",
                loss=f"{loss:.4f}",
                entropy=f"{entropy:.4f}",
                value=f"{value:.4f}",
                steps=total_env_steps,
            )

            # Log to W&B
            if config.log.wandb:
                log_metrics(metrics, step=total_env_steps)

            # Check for NaN/Inf
            if jnp.isnan(loss) or jnp.isinf(loss):
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

    pbar.close()

    # Print emergence summary
    emergence_summary = emergence_tracker.get_summary()
    if emergence_summary["total_events"] > 0:
        print()
        print("Emergence Events Detected:")
        for event_str in emergence_summary["events"]:
            print(f"  {event_str}")

    print()
    print("=" * 60)
    print("Training complete!")
    print(f"Total env steps: {total_env_steps}")
    print(f"Final metrics:")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {float(v):.6f}")
    print("=" * 60)

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
