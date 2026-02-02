"""Weight inheritance and mutation for agent reproduction."""

from typing import Any, Sequence

import jax


def mutate_params(
    params: Any, key: jax.Array, mutation_std: float
) -> Any:
    """Add Gaussian noise to all weight arrays in a parameter pytree.

    Args:
        params: Network parameters (pytree of arrays).
        key: PRNG key for noise generation.
        mutation_std: Standard deviation of Gaussian noise.

    Returns:
        New parameter pytree with noise added to all leaves.
    """
    leaves, treedef = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(key, len(leaves))
    noisy_leaves = [
        leaf + mutation_std * jax.random.normal(k, leaf.shape)
        for leaf, k in zip(leaves, keys)
    ]
    return jax.tree_util.tree_unflatten(treedef, noisy_leaves)


def copy_agent_params(
    all_params: Any, parent_idx: int, child_idx: int
) -> Any:
    """Copy parameters from one agent slot to another within per-agent params.

    Per-agent params have shape (max_agents, ...) for each leaf.
    This copies the parent's slice to the child's slot.

    Args:
        all_params: Per-agent parameters where each leaf has leading
            dimension max_agents.
        parent_idx: Index of the parent agent slot.
        child_idx: Index of the child agent slot.

    Returns:
        Updated per-agent params with child slot set to parent's values.
    """
    return jax.tree.map(
        lambda leaf: leaf.at[child_idx].set(leaf[parent_idx]),
        all_params,
    )


def mutate_agent_params(
    all_params: Any,
    parent_idx: int,
    child_idx: int,
    key: jax.Array,
    mutation_std: float,
) -> Any:
    """Copy parent params to child slot and add Gaussian mutation.

    Combines copy_agent_params and mutation in one operation.

    Args:
        all_params: Per-agent parameters (each leaf has leading max_agents dim).
        parent_idx: Index of the parent agent slot.
        child_idx: Index of the child agent slot.
        key: PRNG key for noise generation.
        mutation_std: Standard deviation of Gaussian noise.

    Returns:
        Updated per-agent params with child slot set to mutated parent values.
    """
    leaves, treedef = jax.tree_util.tree_flatten(all_params)
    keys = jax.random.split(key, len(leaves))
    mutated_leaves = []
    for leaf, k in zip(leaves, keys):
        parent_vals = leaf[parent_idx]
        noise = mutation_std * jax.random.normal(k, parent_vals.shape)
        mutated_leaves.append(leaf.at[child_idx].set(parent_vals + noise))
    return jax.tree_util.tree_unflatten(treedef, mutated_leaves)


def compute_per_leaf_mutation_rates(
    params: Any,
    default_std: float,
    layer_mutation_rates: dict[str, float] | None,
) -> tuple[float, ...]:
    """Compute a per-leaf mutation rate tuple from layer name overrides.

    Args:
        params: Parameter pytree (used to get leaf paths).
        default_std: Default mutation standard deviation for unmatched leaves.
        layer_mutation_rates: Optional mapping from layer name substrings to
            mutation std values. If a leaf's path contains a matching substring,
            that rate is used instead of default_std.

    Returns:
        Tuple of floats, one per pytree leaf, in the same order as
        ``jax.tree_util.tree_leaves(params)``.
    """
    if layer_mutation_rates is None:
        leaves = jax.tree_util.tree_leaves(params)
        return tuple(default_std for _ in leaves)

    leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(params)
    rates = []
    for path, _leaf in leaves_with_path:
        path_str = "/".join(str(p) for p in path)
        matched_rate = default_std
        for pattern, rate in layer_mutation_rates.items():
            if pattern in path_str:
                matched_rate = rate
                break
        rates.append(matched_rate)
    return tuple(rates)


def mutate_agent_params_layered(
    all_params: Any,
    parent_idx: int,
    child_idx: int,
    key: jax.Array,
    per_leaf_rates: Sequence[float],
) -> Any:
    """Copy parent params to child slot with per-leaf mutation rates.

    Like ``mutate_agent_params`` but each leaf gets its own mutation
    standard deviation, enabling different mutation rates for different
    network layers.

    Args:
        all_params: Per-agent parameters (each leaf has leading max_agents dim).
        parent_idx: Index of the parent agent slot.
        child_idx: Index of the child agent slot.
        key: PRNG key for noise generation.
        per_leaf_rates: Sequence of mutation stds, one per pytree leaf.

    Returns:
        Updated per-agent params with child slot set to mutated parent values.
    """
    leaves, treedef = jax.tree_util.tree_flatten(all_params)
    keys = jax.random.split(key, len(leaves))
    mutated_leaves = []
    for leaf, k, std in zip(leaves, keys, per_leaf_rates):
        parent_vals = leaf[parent_idx]
        noise = std * jax.random.normal(k, parent_vals.shape)
        mutated_leaves.append(leaf.at[child_idx].set(parent_vals + noise))
    return jax.tree_util.tree_unflatten(treedef, mutated_leaves)
