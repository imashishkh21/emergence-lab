"""Baselines comparison experiment runner.

Runs all methods (Ours, IPPO, ACO-Fixed, ACO-Hybrid, MAPPO) across environment
configs and seeds with the same compute budget for fair comparison.

Methods:
    - ours: Full emergence system (field + evolution)
    - ippo: Independent PPO with no field
    - aco_fixed: Ant Colony Optimization with hardcoded rules (no NN)
    - aco_hybrid: ACO rules for field writes + NN for movement
    - mappo: Multi-Agent PPO with centralized critic

Reference: Agarwal et al. (2021), "Deep RL at the Edge of the Statistical Precipice"
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np

from src.agents.network import ActorCritic
from src.agents.policy import sample_actions
from src.baselines.aco_fixed import (
    aco_config,
    create_aco_hybrid_network,
    evaluate_aco_fixed,
    evaluate_aco_hybrid,
    init_aco_hybrid_params,
)
from src.baselines.ippo import (
    create_ippo_network,
    evaluate_ippo,
    init_ippo_params,
    ippo_config,
)
from src.baselines.mappo import (
    create_centralized_critic,
    create_mappo_network,
    evaluate_mappo,
    init_mappo_params,
    mappo_config,
)
from src.configs import Config
from src.environment.env import reset, step
from src.environment.obs import get_observations
from src.experiments.configs import get_env_config
from src.experiments.runner import (
    ExperimentConfig,
    ExperimentResult,
    run_experiment,
)

# Type for method names
MethodName = Literal["ours", "ippo", "aco_fixed", "aco_hybrid", "mappo"]
ALL_METHODS: list[MethodName] = ["ours", "ippo", "aco_fixed", "aco_hybrid", "mappo"]


@dataclass
class BaselinesComparisonResult:
    """Aggregated results from comparing all baseline methods.

    Attributes:
        env_config_name: Environment configuration used.
        n_seeds: Number of seeds per method.
        n_episodes: Episodes per seed for evaluation.

        # Per-method results
        method_results: Dict mapping method name -> ExperimentResult.

        # Rankings and comparisons
        rankings: List of dicts with method rankings by IQM reward.
        best_method: Method with highest IQM reward.

        # Metadata
        methods_run: List of method names actually run.
        paired_seeds: Whether paired seeds were used.
        seed_offset: Base seed offset.
    """

    env_config_name: str
    n_seeds: int
    n_episodes: int

    # Per-method results
    method_results: dict[str, ExperimentResult] = field(default_factory=dict)

    # Rankings
    rankings: list[dict[str, Any]] = field(default_factory=list)
    best_method: str | None = None

    # Metadata
    methods_run: list[str] = field(default_factory=list)
    paired_seeds: bool = True
    seed_offset: int = 0

    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            f"BaselinesComparisonResult({self.env_config_name})",
            f"  Methods: {self.methods_run}",
            f"  Seeds: {self.n_seeds}, Episodes per seed: {self.n_episodes}",
            f"  Best method: {self.best_method}",
            "",
            "  Rankings (by IQM reward):",
        ]

        for ranking in self.rankings:
            method = ranking["method"]
            iqm = ranking["iqm_reward"]
            mean = ranking["mean_reward"]
            ci = ranking["ci"]
            lines.append(
                f"    {ranking['rank']}. {method}: IQM={iqm:.2f}, "
                f"Mean={mean:.2f}, CI=({ci[0]:.2f}, {ci[1]:.2f})"
            )

        return "\n".join(lines)


def create_ours_network(config: Config) -> ActorCritic:
    """Create the network for our full emergence system.

    Args:
        config: Configuration object.

    Returns:
        ActorCritic network module.
    """
    return ActorCritic(
        hidden_dims=config.agent.hidden_dims,
        num_actions=config.agent.num_actions,
        agent_embed_dim=config.agent.agent_embed_dim,
        n_agents=config.evolution.max_agents,
    )


def init_ours_params(
    network: ActorCritic,
    config: Config,
    key: jax.Array,
) -> Any:
    """Initialize parameters for our full emergence system.

    Args:
        network: ActorCritic network module.
        config: Configuration object.
        key: JAX PRNG key.

    Returns:
        Initialized network parameters.
    """
    from src.environment.obs import obs_dim

    observation_dim = obs_dim(config)
    dummy_obs = jnp.zeros((observation_dim,))
    params = network.init(key, dummy_obs)
    return params


def run_ours_episode(
    network: ActorCritic,
    params: Any,
    config: Config,
    key: jax.Array,
    deterministic: bool = False,
) -> dict[str, Any]:
    """Run a single episode using our full emergence system.

    Uses field and evolution as configured.

    Args:
        network: ActorCritic network module.
        params: Network parameters.
        config: Configuration (full emergence config).
        key: JAX PRNG key.
        deterministic: If True, use greedy actions.

    Returns:
        Standardized result dict.
    """
    key, reset_key = jax.random.split(key)
    state = reset(reset_key, config)

    total_reward = 0.0
    total_food = 0.0
    max_agents = config.evolution.max_agents
    per_agent_rewards = np.zeros(max_agents)

    for _ in range(config.env.max_steps):
        obs = get_observations(state, config)
        obs_batch = obs[None, :, :]

        key, action_key = jax.random.split(key)
        if deterministic:
            from src.agents.policy import get_deterministic_actions

            actions = get_deterministic_actions(network, params, obs_batch)
        else:
            actions, _, _, _ = sample_actions(network, params, obs_batch, action_key)

        actions = actions[0]

        state, rewards, done, info = step(state, actions, config)

        rewards_np = np.array(rewards)
        total_reward += float(np.sum(rewards_np))
        total_food += float(info["food_collected_this_step"])
        per_agent_rewards += rewards_np

        if done:
            break

    final_population = int(np.sum(np.array(state.agent_alive)))

    return {
        "total_reward": total_reward,
        "food_collected": total_food,
        "final_population": final_population,
        "per_agent_rewards": per_agent_rewards.tolist(),
    }


def evaluate_ours(
    network: ActorCritic,
    params: Any,
    config: Config,
    n_episodes: int,
    seed: int,
    deterministic: bool = False,
) -> dict[str, Any]:
    """Evaluate our full emergence system over multiple episodes.

    Args:
        network: ActorCritic network module.
        params: Network parameters.
        config: Configuration.
        n_episodes: Number of episodes to run.
        seed: Random seed.
        deterministic: If True, use greedy actions.

    Returns:
        Aggregated result dict.
    """
    key = jax.random.PRNGKey(seed)

    episode_rewards = []
    episode_food = []
    episode_populations = []
    all_per_agent_rewards = []

    for _ in range(n_episodes):
        key, episode_key = jax.random.split(key)
        result = run_ours_episode(
            network, params, config, episode_key, deterministic=deterministic
        )
        episode_rewards.append(result["total_reward"])
        episode_food.append(result["food_collected"])
        episode_populations.append(result["final_population"])
        all_per_agent_rewards.append(result["per_agent_rewards"])

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


def create_method_runner(
    method: MethodName,
    env_config: Config,
    n_episodes: int,
    params: Any | None = None,
) -> Callable[[int], dict[str, Any]]:
    """Create a method runner function for the experiment harness.

    Args:
        method: Method name ("ours", "ippo", etc.).
        env_config: Environment configuration.
        n_episodes: Number of episodes per seed.
        params: Optional pre-trained parameters (for "ours" method).

    Returns:
        Callable that takes a seed and returns a result dict.
    """
    if method == "ours":
        # Full emergence system with field and evolution
        network = create_ours_network(env_config)
        key = jax.random.PRNGKey(42)
        if params is None:
            params = init_ours_params(network, env_config, key)

        def ours_runner(seed: int) -> dict[str, Any]:
            result = evaluate_ours(
                network, params, env_config, n_episodes, seed, deterministic=False
            )
            return {
                "total_reward": result["total_reward"],
                "food_collected": result["food_collected"],
                "final_population": result["final_population"],
                "per_agent_rewards": result["per_agent_rewards"],
            }

        return ours_runner

    elif method == "ippo":
        # IPPO: no field, no evolution
        config = ippo_config(env_config)
        network = create_ippo_network(config)
        key = jax.random.PRNGKey(42)
        ippo_params = init_ippo_params(network, config, key)

        def ippo_runner(seed: int) -> dict[str, Any]:
            result = evaluate_ippo(
                network, ippo_params, config, n_episodes, seed, deterministic=False
            )
            return {
                "total_reward": result["total_reward"],
                "food_collected": result["food_collected"],
                "final_population": result["final_population"],
                "per_agent_rewards": result["per_agent_rewards"],
            }

        return ippo_runner

    elif method == "aco_fixed":
        # ACO-Fixed: no neural network, pure hardcoded pheromone rules
        config = aco_config(env_config)

        def aco_fixed_runner(seed: int) -> dict[str, Any]:
            result = evaluate_aco_fixed(config, n_episodes, seed)
            return {
                "total_reward": result["total_reward"],
                "food_collected": result["food_collected"],
                "final_population": result["final_population"],
                "per_agent_rewards": result["per_agent_rewards"],
            }

        return aco_fixed_runner

    elif method == "aco_hybrid":
        # ACO-Hybrid: NN for movement, hardcoded field writes
        config = aco_config(env_config)
        network = create_aco_hybrid_network(config)
        key = jax.random.PRNGKey(42)
        aco_params = init_aco_hybrid_params(network, config, key)

        def aco_hybrid_runner(seed: int) -> dict[str, Any]:
            result = evaluate_aco_hybrid(
                network, aco_params, config, n_episodes, seed, deterministic=False
            )
            return {
                "total_reward": result["total_reward"],
                "food_collected": result["food_collected"],
                "final_population": result["final_population"],
                "per_agent_rewards": result["per_agent_rewards"],
            }

        return aco_hybrid_runner

    elif method == "mappo":
        # MAPPO: centralized critic, decentralized actors
        config = mappo_config(env_config)
        actor = create_mappo_network(config)
        critic = create_centralized_critic(config)
        key = jax.random.PRNGKey(42)
        actor_params, critic_params = init_mappo_params(actor, critic, config, key)

        def mappo_runner(seed: int) -> dict[str, Any]:
            result = evaluate_mappo(
                actor,
                critic,
                actor_params,
                critic_params,
                config,
                n_episodes,
                seed,
                deterministic=False,
            )
            return {
                "total_reward": result["total_reward"],
                "food_collected": result["food_collected"],
                "final_population": result["final_population"],
                "per_agent_rewards": result["per_agent_rewards"],
            }

        return mappo_runner

    else:
        raise ValueError(f"Unknown method: {method}")


def run_baselines_comparison(
    env_config_name: str,
    methods: list[MethodName] | None = None,
    n_seeds: int = 20,
    n_episodes: int = 10,
    paired_seeds: bool = True,
    seed_offset: int = 0,
    ours_params: Any | None = None,
    verbose: bool = False,
) -> BaselinesComparisonResult:
    """Run full baselines comparison for a single environment config.

    Args:
        env_config_name: Environment configuration name ("standard", etc.).
        methods: List of methods to run. Default: all methods.
        n_seeds: Number of seeds per method. Default 20 (DR-4 gold standard).
        n_episodes: Episodes per seed for evaluation. Default 10.
        paired_seeds: If True, use same seeds for all methods.
        seed_offset: Base seed offset.
        ours_params: Optional pre-trained params for "ours" method.
        verbose: If True, print progress.

    Returns:
        BaselinesComparisonResult with all method results and rankings.
    """
    if methods is None:
        methods = list(ALL_METHODS)

    env_config = get_env_config(env_config_name)

    if verbose:
        print(f"\nRunning baselines comparison: {env_config_name}")
        print(f"  Methods: {methods}")
        print(f"  Seeds: {n_seeds}, Episodes: {n_episodes}")
        print(f"  Paired seeds: {paired_seeds}, Offset: {seed_offset}")

    # Run each method
    method_results: dict[str, ExperimentResult] = {}

    for method in methods:
        if verbose:
            print(f"\n  Running {method}...")

        # Create experiment config
        exp_config = ExperimentConfig(
            method_name=method,
            n_seeds=n_seeds,
            env_config_name=env_config_name,
            paired_seeds=paired_seeds,
            seed_offset=seed_offset,
            n_episodes=n_episodes,
            save_per_seed_results=True,
        )

        # Create method runner
        # Pass ours_params only for "ours" method
        method_params = ours_params if method == "ours" else None
        runner = create_method_runner(method, env_config, n_episodes, method_params)

        # Run experiment
        result = run_experiment(exp_config, runner, verbose=verbose)
        method_results[method] = result

        if verbose:
            print(f"    {method}: IQM={result.iqm_reward:.2f}, Mean={result.mean_reward:.2f}")

    # Compute rankings
    sorted_methods: list[str] = sorted(
        method_results.keys(),
        key=lambda m: method_results[m].iqm_reward,
        reverse=True,
    )

    rankings = []
    for rank, method_name in enumerate(sorted_methods, start=1):
        r = method_results[method_name]
        rankings.append({
            "rank": rank,
            "method": method_name,
            "iqm_reward": r.iqm_reward,
            "mean_reward": r.mean_reward,
            "ci": (r.ci_lower, r.ci_upper),
        })

    best_method = sorted_methods[0] if sorted_methods else None

    return BaselinesComparisonResult(
        env_config_name=env_config_name,
        n_seeds=n_seeds,
        n_episodes=n_episodes,
        method_results=method_results,
        rankings=rankings,
        best_method=best_method,
        methods_run=list(methods),
        paired_seeds=paired_seeds,
        seed_offset=seed_offset,
    )


def run_all_baselines_comparisons(
    env_config_names: list[str] | None = None,
    methods: list[MethodName] | None = None,
    n_seeds: int = 20,
    n_episodes: int = 10,
    paired_seeds: bool = True,
    seed_offset: int = 0,
    ours_params: Any | None = None,
    verbose: bool = False,
) -> dict[str, BaselinesComparisonResult]:
    """Run baselines comparison across multiple environment configs.

    Args:
        env_config_names: List of env config names. Default: all 3.
        methods: List of methods to run. Default: all methods.
        n_seeds: Number of seeds per method.
        n_episodes: Episodes per seed.
        paired_seeds: If True, use same seeds for all methods.
        seed_offset: Base seed offset.
        ours_params: Optional pre-trained params for "ours" method.
        verbose: If True, print progress.

    Returns:
        Dict mapping env_config_name -> BaselinesComparisonResult.
    """
    if env_config_names is None:
        from src.experiments.configs import list_env_configs
        env_config_names = list_env_configs()

    if methods is None:
        methods = list(ALL_METHODS)

    results: dict[str, BaselinesComparisonResult] = {}

    for env_config_name in env_config_names:
        result = run_baselines_comparison(
            env_config_name=env_config_name,
            methods=methods,
            n_seeds=n_seeds,
            n_episodes=n_episodes,
            paired_seeds=paired_seeds,
            seed_offset=seed_offset,
            ours_params=ours_params,
            verbose=verbose,
        )
        results[env_config_name] = result

    return results


def save_baselines_result(
    result: BaselinesComparisonResult,
    path: str | Path,
) -> None:
    """Save BaselinesComparisonResult to pickle file.

    Args:
        result: BaselinesComparisonResult to save.
        path: File path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(result, f)


def load_baselines_result(path: str | Path) -> BaselinesComparisonResult:
    """Load BaselinesComparisonResult from pickle file.

    Args:
        path: File path.

    Returns:
        Loaded BaselinesComparisonResult.

    Raises:
        FileNotFoundError: If file doesn't exist.
        TypeError: If loaded object is wrong type.
    """
    path = Path(path)

    with open(path, "rb") as f:
        result = pickle.load(f)

    if not isinstance(result, BaselinesComparisonResult):
        raise TypeError(f"Expected BaselinesComparisonResult, got {type(result)}")

    return result


def print_baselines_comparison(result: BaselinesComparisonResult) -> None:
    """Print a formatted summary of baselines comparison.

    Args:
        result: BaselinesComparisonResult to print.
    """
    print("\n" + "=" * 70)
    print(f"Baselines Comparison: {result.env_config_name}")
    print("=" * 70)
    print(f"Methods: {', '.join(result.methods_run)}")
    print(f"Seeds: {result.n_seeds}, Episodes per seed: {result.n_episodes}")
    print(f"Paired seeds: {result.paired_seeds}")
    print()

    # Rankings table
    print("Rankings (by IQM Reward):")
    print("-" * 70)
    print(f"{'Rank':<6} {'Method':<15} {'IQM':<12} {'Mean ± Std':<20} {'95% CI':<18}")
    print("-" * 70)

    for ranking in result.rankings:
        method = ranking["method"]
        r = result.method_results[method]
        iqm_str = f"{ranking['iqm_reward']:.2f}"
        mean_std = f"{r.mean_reward:.2f} ± {r.std_reward:.2f}"
        ci_str = f"({r.ci_lower:.2f}, {r.ci_upper:.2f})"
        print(f"{ranking['rank']:<6} {method:<15} {iqm_str:<12} {mean_std:<20} {ci_str:<18}")

    print("-" * 70)
    print(f"Best method: {result.best_method}")
    print()

    # Food collected comparison
    print("Food Collected:")
    print("-" * 50)
    for ranking in result.rankings:
        method = ranking["method"]
        r = result.method_results[method]
        print(f"  {method:<15}: {r.mean_food:.2f} ± {r.std_food:.2f}")
    print()


def compare_baselines_results(
    results: dict[str, BaselinesComparisonResult],
) -> dict[str, Any]:
    """Compare results across multiple environment configs.

    Args:
        results: Dict mapping env_config_name -> BaselinesComparisonResult.

    Returns:
        Summary dict with cross-environment comparisons.
    """
    if not results:
        return {"summary": "No results to compare"}

    # Aggregate wins per method
    method_wins: dict[str, int] = {}
    method_iqms: dict[str, list[float]] = {}

    all_methods: set[str] = set()
    for result in results.values():
        all_methods.update(result.methods_run)

    for method in all_methods:
        method_wins[method] = 0
        method_iqms[method] = []

    for env_name, result in results.items():
        if result.best_method:
            method_wins[result.best_method] = method_wins.get(result.best_method, 0) + 1

        for method, exp_result in result.method_results.items():
            method_iqms[method].append(exp_result.iqm_reward)

    # Compute average IQM across environments
    method_avg_iqm = {
        method: float(np.mean(iqms)) if iqms else 0.0
        for method, iqms in method_iqms.items()
    }

    # Overall ranking
    overall_ranking = sorted(
        method_avg_iqm.keys(),
        key=lambda m: method_avg_iqm[m],
        reverse=True,
    )

    return {
        "env_configs": list(results.keys()),
        "method_wins": method_wins,
        "method_avg_iqm": method_avg_iqm,
        "overall_ranking": overall_ranking,
        "overall_best": overall_ranking[0] if overall_ranking else None,
    }
