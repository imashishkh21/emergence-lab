"""Ablation test to evaluate whether the shared field and evolution matter.

Tests field conditions:
    1. Normal: field operates normally (diffusion + decay + agent writes)
    2. Zeroed: field values are zeroed every step (agents see no field info)
    3. Random: field values are randomized every step (agents see noise)

Evolution ablation compares 4 conditions (2x2 field x evolution):
    1. field + evolution: both active
    2. field only: field active, evolution disabled (fixed population)
    3. evolution only: field zeroed, evolution active
    4. neither: field zeroed, evolution disabled

Specialization ablation compares 3 weight conditions:
    1. divergent: normal trained per-agent weights (allow specialization)
    2. uniform: all agents cloned to have mean weights (force uniformity)
    3. random_weights: agents get random perturbations of mean weights

Compares mean rewards and population dynamics across conditions.
"""

from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from src.agents.network import ActorCritic
from src.agents.policy import get_deterministic_actions
from src.configs import Config
from src.environment.env import reset, step
from src.environment.obs import get_observations, obs_dim
from src.environment.state import EnvState
from src.field.field import FieldState, create_field

FieldCondition = Literal["normal", "zeroed", "random"]


@dataclass
class AblationResult:
    """Results from one ablation condition.

    Attributes:
        condition: Name of the ablation condition.
        mean_reward: Mean total episode reward across episodes.
        std_reward: Standard deviation of total episode rewards.
        episode_rewards: Per-episode total rewards.
        final_population: Mean final alive population across episodes.
        total_births: Mean total births across episodes.
        total_deaths: Mean total deaths across episodes.
        survival_rate: Mean fraction of original agents still alive at end.
    """
    condition: str
    mean_reward: float
    std_reward: float
    episode_rewards: list[float]
    final_population: float = 0.0
    total_births: float = 0.0
    total_deaths: float = 0.0
    survival_rate: float = 0.0


@dataclass
class _EpisodeStats:
    """Internal episode statistics."""
    total_reward: float
    final_population: int
    total_births: int
    total_deaths: int
    survival_rate: float


def _run_episode(
    network: ActorCritic,
    params: dict,
    config: Config,
    key: jax.Array,
    condition: FieldCondition,
) -> float:
    """Run a single episode under the given field condition.

    Args:
        network: ActorCritic network module.
        params: Network parameters.
        config: Master configuration.
        key: PRNG key for environment reset.
        condition: One of "normal", "zeroed", "random".

    Returns:
        Total episode reward (summed across all agents and steps).
    """
    stats = _run_episode_full(network, params, config, key, condition, evolution=True)
    return stats.total_reward


def _run_episode_full(
    network: ActorCritic,
    params: dict,
    config: Config,
    key: jax.Array,
    condition: FieldCondition,
    evolution: bool = True,
) -> _EpisodeStats:
    """Run a single episode under the given field and evolution conditions.

    Args:
        network: ActorCritic network module.
        params: Network parameters.
        config: Master configuration.
        key: PRNG key for environment reset.
        condition: One of "normal", "zeroed", "random".
        evolution: If False, disable evolution (keep energy high, no death/reproduction).

    Returns:
        Episode statistics including reward and population dynamics.
    """
    key, reset_key = jax.random.split(key)
    state = reset(reset_key, config)
    total_reward = 0.0
    total_births = 0
    total_deaths = 0
    num_agents = config.env.num_agents

    for t in range(config.env.max_steps):
        # Apply field ablation before observation
        if condition == "zeroed":
            zero_field = create_field(
                config.env.grid_size,
                config.env.grid_size,
                config.field.num_channels,
            )
            state = _replace_field(state, zero_field)
        elif condition == "random":
            key, noise_key = jax.random.split(key)
            noise_values = jax.random.uniform(
                noise_key,
                shape=(config.env.grid_size, config.env.grid_size, config.field.num_channels),
            )
            random_field = FieldState(values=noise_values)
            state = _replace_field(state, random_field)

        # When evolution is disabled, keep all agents at high energy
        # so they never die and never reproduce (energy < threshold after reset)
        if not evolution:
            state = _reset_energy(state, config)

        # Get observations
        obs = get_observations(state, config)

        # Add batch dim: (1, max_agents, obs_dim)
        obs_batched = obs[None, :, :]

        # Get deterministic actions: (1, max_agents) -> (max_agents,)
        actions = get_deterministic_actions(network, params, obs_batched)
        actions = actions[0]

        # When evolution is disabled, mask out reproduce actions
        if not evolution:
            actions = jnp.where(actions == 5, 0, actions)

        # Step environment
        state, rewards, done, info = step(state, actions, config)

        # Sum rewards across agents for this step
        total_reward += float(jnp.sum(rewards))
        total_births += int(info["births_this_step"])
        total_deaths += int(info["deaths_this_step"])

        if bool(done):
            break

    final_population = int(jnp.sum(state.agent_alive.astype(jnp.int32)))
    survival_rate = final_population / max(num_agents, 1)

    return _EpisodeStats(
        total_reward=total_reward,
        final_population=final_population,
        total_births=total_births,
        total_deaths=total_deaths,
        survival_rate=survival_rate,
    )


def _reset_energy(state: EnvState, config: Config) -> EnvState:
    """Reset alive agents' energy to starting_energy to prevent death/reproduction.

    This effectively disables evolution by keeping energy at a fixed level
    that is above 0 (no death) but below reproduce_threshold (no reproduction).
    """
    # Set energy to starting_energy for alive agents (below reproduce_threshold)
    fixed_energy = jnp.where(
        state.agent_alive,
        jnp.float32(config.evolution.starting_energy),
        state.agent_energy,
    )
    return EnvState(
        agent_positions=state.agent_positions,
        food_positions=state.food_positions,
        food_collected=state.food_collected,
        field_state=state.field_state,
        step=state.step,
        key=state.key,
        agent_energy=fixed_energy,
        agent_alive=state.agent_alive,
        agent_ids=state.agent_ids,
        agent_parent_ids=state.agent_parent_ids,
        next_agent_id=state.next_agent_id,
        agent_birth_step=state.agent_birth_step,
        agent_params=state.agent_params,
    )


def _replace_field(state: EnvState, field_state: FieldState) -> EnvState:
    """Return a copy of state with a replaced field_state."""
    return EnvState(
        agent_positions=state.agent_positions,
        food_positions=state.food_positions,
        food_collected=state.food_collected,
        field_state=field_state,
        step=state.step,
        key=state.key,
        agent_energy=state.agent_energy,
        agent_alive=state.agent_alive,
        agent_ids=state.agent_ids,
        agent_parent_ids=state.agent_parent_ids,
        next_agent_id=state.next_agent_id,
        agent_birth_step=state.agent_birth_step,
        agent_params=state.agent_params,
    )


def ablation_test(
    network: ActorCritic,
    params: dict,
    config: Config,
    num_episodes: int = 20,
    seed: int = 0,
) -> dict[str, AblationResult]:
    """Run ablation test comparing normal, zeroed, and random field conditions.

    Args:
        network: ActorCritic network module.
        params: Network parameters.
        config: Master configuration.
        num_episodes: Number of episodes per condition.
        seed: Base random seed.

    Returns:
        Dictionary mapping condition name to AblationResult.
    """
    conditions: list[FieldCondition] = ["normal", "zeroed", "random"]
    results: dict[str, AblationResult] = {}

    for condition in conditions:
        episode_rewards: list[float] = []
        base_key = jax.random.PRNGKey(seed)

        for ep in range(num_episodes):
            ep_key = jax.random.fold_in(base_key, ep)
            reward = _run_episode(network, params, config, ep_key, condition)
            episode_rewards.append(reward)

        rewards_arr = np.array(episode_rewards)
        results[condition] = AblationResult(
            condition=condition,
            mean_reward=float(np.mean(rewards_arr)),
            std_reward=float(np.std(rewards_arr)),
            episode_rewards=episode_rewards,
        )

    return results


def evolution_ablation_test(
    network: ActorCritic,
    params: dict,
    config: Config,
    num_episodes: int = 20,
    seed: int = 0,
) -> dict[str, AblationResult]:
    """Run 2x2 ablation comparing field and evolution conditions.

    Tests 4 conditions:
        1. field + evolution: both active (normal operation)
        2. field only: field active, evolution disabled (fixed population)
        3. evolution only: field zeroed, evolution active
        4. neither: field zeroed, evolution disabled

    Args:
        network: ActorCritic network module.
        params: Network parameters.
        config: Master configuration.
        num_episodes: Number of episodes per condition.
        seed: Base random seed.

    Returns:
        Dictionary mapping condition name to AblationResult with population stats.
    """
    # 2x2 grid: (field_condition, evolution_enabled)
    conditions: list[tuple[str, FieldCondition, bool]] = [
        ("field+evolution", "normal", True),
        ("field_only", "normal", False),
        ("evolution_only", "zeroed", True),
        ("neither", "zeroed", False),
    ]
    results: dict[str, AblationResult] = {}

    for name, field_cond, evo_enabled in conditions:
        stats_list: list[_EpisodeStats] = []
        base_key = jax.random.PRNGKey(seed)

        for ep in range(num_episodes):
            ep_key = jax.random.fold_in(base_key, ep)
            stats = _run_episode_full(
                network, params, config, ep_key, field_cond, evolution=evo_enabled
            )
            stats_list.append(stats)

        rewards = [s.total_reward for s in stats_list]
        rewards_arr = np.array(rewards)
        populations = np.array([s.final_population for s in stats_list])
        births = np.array([s.total_births for s in stats_list])
        deaths = np.array([s.total_deaths for s in stats_list])
        survivals = np.array([s.survival_rate for s in stats_list])

        results[name] = AblationResult(
            condition=name,
            mean_reward=float(np.mean(rewards_arr)),
            std_reward=float(np.std(rewards_arr)),
            episode_rewards=list(rewards),
            final_population=float(np.mean(populations)),
            total_births=float(np.mean(births)),
            total_deaths=float(np.mean(deaths)),
            survival_rate=float(np.mean(survivals)),
        )

    return results


def print_ablation_results(results: dict[str, AblationResult]) -> None:
    """Pretty-print ablation test results."""
    print("=" * 60)
    print("Ablation Test Results")
    print("=" * 60)
    print(f"{'Condition':<12} {'Mean Reward':>12} {'Std':>10} {'Episodes':>10}")
    print("-" * 44)
    for name in ["normal", "zeroed", "random"]:
        if name in results:
            r = results[name]
            print(f"{r.condition:<12} {r.mean_reward:>12.2f} {r.std_reward:>10.2f} {len(r.episode_rewards):>10}")
    print("=" * 60)

    # Statistical comparison
    if "normal" in results and "zeroed" in results:
        diff = results["normal"].mean_reward - results["zeroed"].mean_reward
        print(f"\nNormal - Zeroed field gap: {diff:+.2f}")
    if "normal" in results and "random" in results:
        diff = results["normal"].mean_reward - results["random"].mean_reward
        print(f"Normal - Random field gap: {diff:+.2f}")


def print_evolution_ablation_results(results: dict[str, AblationResult]) -> None:
    """Pretty-print evolution ablation test results with population dynamics."""
    print("=" * 80)
    print("Evolution Ablation Test Results")
    print("=" * 80)
    header = (
        f"{'Condition':<18} {'Reward':>10} {'Std':>8} "
        f"{'Pop':>6} {'Births':>8} {'Deaths':>8} {'Survival':>10}"
    )
    print(header)
    print("-" * 80)
    for name in ["field+evolution", "field_only", "evolution_only", "neither"]:
        if name in results:
            r = results[name]
            print(
                f"{r.condition:<18} {r.mean_reward:>10.2f} {r.std_reward:>8.2f} "
                f"{r.final_population:>6.1f} {r.total_births:>8.1f} "
                f"{r.total_deaths:>8.1f} {r.survival_rate:>9.1%}"
            )
    print("=" * 80)

    # Field effect (with evolution)
    if "field+evolution" in results and "evolution_only" in results:
        diff = results["field+evolution"].mean_reward - results["evolution_only"].mean_reward
        print(f"\nField effect (with evolution): {diff:+.2f}")
    # Field effect (without evolution)
    if "field_only" in results and "neither" in results:
        diff = results["field_only"].mean_reward - results["neither"].mean_reward
        print(f"Field effect (no evolution):   {diff:+.2f}")
    # Evolution effect (with field)
    if "field+evolution" in results and "field_only" in results:
        diff = results["field+evolution"].mean_reward - results["field_only"].mean_reward
        print(f"Evolution effect (with field):  {diff:+.2f}")
    # Evolution effect (without field)
    if "evolution_only" in results and "neither" in results:
        diff = results["evolution_only"].mean_reward - results["neither"].mean_reward
        print(f"Evolution effect (no field):    {diff:+.2f}")


@dataclass
class SpecializationAblationResult:
    """Results from one specialization ablation condition.

    Attributes:
        condition: Name of the weight condition.
        mean_reward: Mean total episode reward across episodes.
        std_reward: Standard deviation of total episode rewards.
        episode_rewards: Per-episode total rewards.
        final_population: Mean final alive population across episodes.
        total_births: Mean total births across episodes.
        total_deaths: Mean total deaths across episodes.
        survival_rate: Mean fraction of original agents still alive at end.
        food_collected: Mean total food collected across episodes.
        population_stability: Std of alive population at each step, averaged.
    """
    condition: str
    mean_reward: float
    std_reward: float
    episode_rewards: list[float]
    final_population: float = 0.0
    total_births: float = 0.0
    total_deaths: float = 0.0
    survival_rate: float = 0.0
    food_collected: float = 0.0
    population_stability: float = 0.0


@dataclass
class _SpecEpisodeStats:
    """Internal episode statistics for specialization ablation."""
    total_reward: float
    final_population: int
    total_births: int
    total_deaths: int
    survival_rate: float
    food_collected: float
    population_over_time: list[int]


def _make_uniform_params(agent_params: Any, alive_mask: jnp.ndarray) -> Any:
    """Create uniform per-agent params by averaging alive agents' weights.

    All agent slots get the mean of the alive agents' weights, preserving
    the pytree structure.

    Args:
        agent_params: Per-agent params pytree, each leaf (max_agents, ...).
        alive_mask: Boolean array (max_agents,).

    Returns:
        New agent_params where all slots have the mean of alive agents' weights.
    """
    alive_mask_float = alive_mask.astype(jnp.float32)
    n_alive = jnp.sum(alive_mask_float)
    n_alive = jnp.maximum(n_alive, 1.0)  # avoid division by zero

    def _average_leaf(leaf: jnp.ndarray) -> jnp.ndarray:
        # leaf shape: (max_agents, ...)
        # Compute mean across alive agents
        # Expand alive_mask to broadcast with leaf dims
        mask = alive_mask_float
        for _ in range(leaf.ndim - 1):
            mask = mask[..., None]
        weighted_sum = jnp.sum(leaf * mask, axis=0)  # (...)
        mean_weights = weighted_sum / n_alive
        # Broadcast mean to all agent slots
        return jnp.broadcast_to(mean_weights[None], leaf.shape)

    return jax.tree_util.tree_map(_average_leaf, agent_params)


def _make_random_params(
    agent_params: Any,
    alive_mask: jnp.ndarray,
    key: jax.Array,
    noise_std: float = 0.1,
) -> Any:
    """Create randomized per-agent params by adding noise to mean weights.

    Starts from the mean of alive agents' weights, then adds i.i.d.
    Gaussian noise independently to each agent's copy.

    Args:
        agent_params: Per-agent params pytree, each leaf (max_agents, ...).
        alive_mask: Boolean array (max_agents,).
        key: PRNG key for noise generation.
        noise_std: Standard deviation of Gaussian noise.

    Returns:
        New agent_params where each agent has mean weights + random noise.
    """
    uniform = _make_uniform_params(agent_params, alive_mask)

    leaves, treedef = jax.tree_util.tree_flatten(uniform)
    keys = jax.random.split(key, len(leaves))

    noisy_leaves = []
    for leaf, k in zip(leaves, keys):
        noise = jax.random.normal(k, shape=leaf.shape) * noise_std
        noisy_leaves.append(leaf + noise)

    return jax.tree_util.tree_unflatten(treedef, noisy_leaves)


def _replace_agent_params(state: EnvState, agent_params: Any) -> EnvState:
    """Return a copy of state with replaced agent_params."""
    return EnvState(
        agent_positions=state.agent_positions,
        food_positions=state.food_positions,
        food_collected=state.food_collected,
        field_state=state.field_state,
        step=state.step,
        key=state.key,
        agent_energy=state.agent_energy,
        agent_alive=state.agent_alive,
        agent_ids=state.agent_ids,
        agent_parent_ids=state.agent_parent_ids,
        next_agent_id=state.next_agent_id,
        agent_birth_step=state.agent_birth_step,
        agent_params=agent_params,
    )


WeightCondition = Literal["divergent", "uniform", "random_weights"]


def _run_specialization_episode(
    network: ActorCritic,
    params: dict,
    config: Config,
    key: jax.Array,
    weight_condition: WeightCondition,
    trained_agent_params: Any,
    trained_alive_mask: jnp.ndarray,
) -> _SpecEpisodeStats:
    """Run a single episode under a weight condition for specialization ablation.

    Args:
        network: ActorCritic network module.
        params: Shared network parameters (used as fallback).
        config: Master configuration.
        key: PRNG key.
        weight_condition: How to set agent weights.
        trained_agent_params: The trained per-agent params to use/modify.
        trained_alive_mask: Alive mask from training.

    Returns:
        Episode statistics including food collected and population over time.
    """
    key, reset_key, noise_key = jax.random.split(key, 3)
    state = reset(reset_key, config)

    # Apply weight condition to the initial state
    if trained_agent_params is not None:
        if weight_condition == "divergent":
            state = _replace_agent_params(state, trained_agent_params)
        elif weight_condition == "uniform":
            uniform_params = _make_uniform_params(
                trained_agent_params, trained_alive_mask
            )
            state = _replace_agent_params(state, uniform_params)
        elif weight_condition == "random_weights":
            random_params = _make_random_params(
                trained_agent_params, trained_alive_mask, noise_key
            )
            state = _replace_agent_params(state, random_params)

    total_reward = 0.0
    total_births = 0
    total_deaths = 0
    food_collected = 0
    num_agents = config.env.num_agents
    population_over_time: list[int] = []

    for t in range(config.env.max_steps):
        obs = get_observations(state, config)
        obs_batched = obs[None, :, :]
        actions = get_deterministic_actions(network, params, obs_batched)
        actions = actions[0]

        pre_food = int(jnp.sum(state.food_collected.astype(jnp.int32)))

        state, rewards, done, info = step(state, actions, config)

        post_food = int(jnp.sum(state.food_collected.astype(jnp.int32)))
        food_this_step = max(0, post_food - pre_food)
        food_collected += food_this_step

        total_reward += float(jnp.sum(rewards))
        total_births += int(info["births_this_step"])
        total_deaths += int(info["deaths_this_step"])
        population_over_time.append(
            int(jnp.sum(state.agent_alive.astype(jnp.int32)))
        )

        if bool(done):
            break

    final_population = int(jnp.sum(state.agent_alive.astype(jnp.int32)))
    survival_rate = final_population / max(num_agents, 1)

    return _SpecEpisodeStats(
        total_reward=total_reward,
        final_population=final_population,
        total_births=total_births,
        total_deaths=total_deaths,
        survival_rate=survival_rate,
        food_collected=float(food_collected),
        population_over_time=population_over_time,
    )


def specialization_ablation_test(
    network: ActorCritic,
    params: dict,
    config: Config,
    trained_agent_params: Any,
    trained_alive_mask: jnp.ndarray,
    num_episodes: int = 20,
    seed: int = 0,
) -> dict[str, SpecializationAblationResult]:
    """Run specialization ablation comparing weight diversity conditions.

    Tests 3 conditions:
        1. divergent: trained per-agent weights (allow specialization)
        2. uniform: all agents cloned to mean weights (force uniformity)
        3. random_weights: agents get random perturbations of mean weights

    Args:
        network: ActorCritic network module.
        params: Shared network parameters.
        config: Master configuration.
        trained_agent_params: Per-agent params from training.
        trained_alive_mask: Alive mask from training.
        num_episodes: Number of episodes per condition.
        seed: Base random seed.

    Returns:
        Dictionary mapping condition name to SpecializationAblationResult.
    """
    conditions: list[WeightCondition] = ["divergent", "uniform", "random_weights"]
    results: dict[str, SpecializationAblationResult] = {}

    for condition in conditions:
        stats_list: list[_SpecEpisodeStats] = []
        base_key = jax.random.PRNGKey(seed)

        for ep in range(num_episodes):
            ep_key = jax.random.fold_in(base_key, ep)
            stats = _run_specialization_episode(
                network, params, config, ep_key, condition,
                trained_agent_params, trained_alive_mask,
            )
            stats_list.append(stats)

        rewards = [s.total_reward for s in stats_list]
        rewards_arr = np.array(rewards)
        populations = np.array([s.final_population for s in stats_list])
        births = np.array([s.total_births for s in stats_list])
        deaths = np.array([s.total_deaths for s in stats_list])
        survivals = np.array([s.survival_rate for s in stats_list])
        foods = np.array([s.food_collected for s in stats_list])

        # Population stability: mean of per-episode std of population over time
        pop_stabilities = []
        for s in stats_list:
            if len(s.population_over_time) > 1:
                pop_stabilities.append(float(np.std(s.population_over_time)))
            else:
                pop_stabilities.append(0.0)

        results[condition] = SpecializationAblationResult(
            condition=condition,
            mean_reward=float(np.mean(rewards_arr)),
            std_reward=float(np.std(rewards_arr)),
            episode_rewards=list(rewards),
            final_population=float(np.mean(populations)),
            total_births=float(np.mean(births)),
            total_deaths=float(np.mean(deaths)),
            survival_rate=float(np.mean(survivals)),
            food_collected=float(np.mean(foods)),
            population_stability=float(np.mean(pop_stabilities)),
        )

    return results


def print_specialization_ablation_results(
    results: dict[str, SpecializationAblationResult],
) -> None:
    """Pretty-print specialization ablation test results."""
    print("=" * 90)
    print("Specialization Ablation Test Results")
    print("=" * 90)
    header = (
        f"{'Condition':<18} {'Reward':>10} {'Std':>8} "
        f"{'Food':>8} {'Pop':>6} {'Births':>8} {'Deaths':>8} "
        f"{'Survival':>10} {'PopStab':>8}"
    )
    print(header)
    print("-" * 90)
    for name in ["divergent", "uniform", "random_weights"]:
        if name in results:
            r = results[name]
            print(
                f"{r.condition:<18} {r.mean_reward:>10.2f} {r.std_reward:>8.2f} "
                f"{r.food_collected:>8.1f} {r.final_population:>6.1f} "
                f"{r.total_births:>8.1f} {r.total_deaths:>8.1f} "
                f"{r.survival_rate:>9.1%} {r.population_stability:>8.2f}"
            )
    print("=" * 90)

    # Comparisons
    if "divergent" in results and "uniform" in results:
        div_r = results["divergent"]
        uni_r = results["uniform"]
        reward_diff = div_r.mean_reward - uni_r.mean_reward
        food_diff = div_r.food_collected - uni_r.food_collected
        surv_diff = div_r.survival_rate - uni_r.survival_rate
        print(f"\nDivergent vs Uniform:")
        print(f"  Reward gap:   {reward_diff:+.2f}")
        print(f"  Food gap:     {food_diff:+.1f}")
        print(f"  Survival gap: {surv_diff:+.1%}")
        if reward_diff > 0 and food_diff > 0:
            print("  -> Specialization (divergent weights) HELPS performance")
        elif reward_diff < 0 and food_diff < 0:
            print("  -> Uniform weights perform BETTER (specialization not beneficial)")
        else:
            print("  -> Mixed results — specialization may help some metrics but not others")

    if "divergent" in results and "random_weights" in results:
        div_r = results["divergent"]
        rnd_r = results["random_weights"]
        reward_diff = div_r.mean_reward - rnd_r.mean_reward
        print(f"\nDivergent vs Random:")
        print(f"  Reward gap:   {reward_diff:+.2f}")
        if reward_diff > 0:
            print("  -> Trained divergence is better than random — specialization is learned")
        else:
            print("  -> Random weights perform similarly — divergence may be noise")


def main() -> None:
    """CLI entry point for ablation testing.

    Usage:
        python -m src.analysis.ablation --checkpoint=path/to/params.pkl
        python -m src.analysis.ablation --checkpoint=path/to/params.pkl --num-episodes=50
    """
    import argparse
    import pickle

    parser = argparse.ArgumentParser(description="Run ablation test on trained model")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to saved parameters (pickle file)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (optional, uses defaults if not provided)",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=20,
        help="Number of episodes per condition",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed",
    )
    args = parser.parse_args()

    # Load config
    if args.config is not None:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    # Load checkpoint params
    with open(args.checkpoint, "rb") as f:
        params = pickle.load(f)

    # Create network
    observation_dim = obs_dim(config)
    network = ActorCritic(
        hidden_dims=tuple(config.agent.hidden_dims),
        num_actions=6,
    )

    # Verify params by running a dummy forward pass
    dummy_obs = jnp.zeros((observation_dim,))
    network.apply(params, dummy_obs)

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Running ablation test with {args.num_episodes} episodes per condition...")
    print()

    results = ablation_test(
        network=network,
        params=params,
        config=config,
        num_episodes=args.num_episodes,
        seed=args.seed,
    )

    print_ablation_results(results)


if __name__ == "__main__":
    main()
