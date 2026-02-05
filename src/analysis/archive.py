"""MAP-Elites Behavioral Archive for maintaining behaviorally diverse agents.

Implements a grid-based archive (Mouret & Clune 2015) that stores the
best-fitness agent params for each cell in a 2D behavioral descriptor
space. The two axes are:

- **Axis 0 (exploration):** movement_entropy in [0, 1] — how random is
  the agent's movement? 0 = always same action, 1 = uniform random.
- **Axis 1 (exploitation):** field_write_frequency in [0, 1] — how much
  does the agent mark territory? Fraction of steps the agent is alive
  (alive agents auto-write to the field).

The archive supports:
- ``add()`` to insert agents, keeping only the highest-fitness occupant
  per cell.
- ``sample()`` to draw diverse parents for reproduction.
- ``as_feature_array()`` to export all occupied cells' descriptors for
  use with ``novelty_score()`` or other analysis functions.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ArchiveEntry:
    """A single entry in the behavioral archive.

    Attributes:
        params: Agent parameters (numpy arrays or pytree leaves).
        fitness: Fitness score (higher is better).
        descriptor: 2D behavioral descriptor, shape ``(2,)``.
    """

    params: Any
    fitness: float
    descriptor: np.ndarray


class BehavioralArchive:
    """MAP-Elites grid archive for behaviorally diverse agents.

    The archive is a 2D grid where each cell is indexed by a
    discretized behavioral descriptor. Each cell stores at most one
    agent — the one with the highest fitness observed for that cell.

    Args:
        grid_size: Number of cells along each axis (total cells =
            grid_size ** 2). Default 100.
        seed: Random seed for sampling. Default 42.
    """

    def __init__(self, grid_size: int = 100, seed: int = 42) -> None:
        if grid_size < 1:
            raise ValueError(f"grid_size must be >= 1, got {grid_size}")
        self.grid_size = grid_size
        self._rng = np.random.RandomState(seed)
        # Sparse storage: only occupied cells are stored
        self._cells: dict[tuple[int, int], ArchiveEntry] = {}

    @property
    def size(self) -> int:
        """Number of occupied cells in the archive."""
        return len(self._cells)

    @property
    def capacity(self) -> int:
        """Total number of cells in the archive grid."""
        return self.grid_size * self.grid_size

    def _descriptor_to_cell(self, descriptor: np.ndarray) -> tuple[int, int]:
        """Map a continuous 2D descriptor in [0, 1]^2 to grid cell indices.

        Values are clipped to [0, 1] before discretization. The cell
        index along each axis is ``floor(value * grid_size)``, clamped
        to ``[0, grid_size - 1]``.

        Args:
            descriptor: Array of shape ``(2,)`` with values in [0, 1].

        Returns:
            Tuple ``(row, col)`` of integer cell indices.
        """
        d = np.asarray(descriptor, dtype=np.float64).ravel()
        if d.shape[0] != 2:
            raise ValueError(
                f"descriptor must have 2 elements, got {d.shape[0]}"
            )
        clipped = np.clip(d, 0.0, 1.0)
        indices = np.floor(clipped * self.grid_size).astype(int)
        # Clamp to valid range (edge case: descriptor == 1.0 exactly)
        indices = np.clip(indices, 0, self.grid_size - 1)
        return (int(indices[0]), int(indices[1]))

    def add(
        self,
        params: Any,
        fitness: float,
        descriptor: np.ndarray,
    ) -> bool:
        """Add an agent to the archive if it improves the cell's fitness.

        If the cell is empty, the agent is inserted unconditionally.
        If the cell is occupied, the agent replaces the occupant only
        if its fitness is strictly higher.

        Args:
            params: Agent parameters to store. Will be stored as-is
                (caller is responsible for copying if needed).
            fitness: Fitness score (higher is better).
            descriptor: 2D behavioral descriptor, shape ``(2,)``,
                values in [0, 1].

        Returns:
            ``True`` if the agent was inserted (new cell or improvement),
            ``False`` if the existing occupant was better.
        """
        cell = self._descriptor_to_cell(descriptor)
        entry = ArchiveEntry(
            params=params,
            fitness=float(fitness),
            descriptor=np.asarray(descriptor, dtype=np.float64).ravel(),
        )

        if cell not in self._cells:
            self._cells[cell] = entry
            return True

        if fitness > self._cells[cell].fitness:
            self._cells[cell] = entry
            return True

        return False

    def sample(self, n: int) -> list[ArchiveEntry]:
        """Sample ``n`` entries uniformly at random from occupied cells.

        Sampling is with replacement when ``n > size``, or without
        replacement when ``n <= size``.

        Args:
            n: Number of entries to sample.

        Returns:
            List of ``ArchiveEntry`` instances. Empty list if the
            archive is empty.

        Raises:
            ValueError: If ``n < 0``.
        """
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        if self.size == 0 or n == 0:
            return []

        entries = list(self._cells.values())

        if n <= len(entries):
            indices = self._rng.choice(len(entries), size=n, replace=False)
        else:
            indices = self._rng.choice(len(entries), size=n, replace=True)

        return [entries[i] for i in indices]

    def as_feature_array(self) -> np.ndarray:
        """Return descriptors of all occupied cells as a 2D array.

        Useful for computing novelty scores against the archive.

        Returns:
            Array of shape ``(size, 2)`` with behavioral descriptors.
            Empty array of shape ``(0, 2)`` if the archive is empty.
        """
        if self.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        return np.array(
            [entry.descriptor for entry in self._cells.values()],
            dtype=np.float64,
        )

    def get_cell(
        self, descriptor: np.ndarray
    ) -> ArchiveEntry | None:
        """Look up the entry at the cell corresponding to a descriptor.

        Args:
            descriptor: 2D behavioral descriptor, shape ``(2,)``.

        Returns:
            The ``ArchiveEntry`` at that cell, or ``None`` if empty.
        """
        cell = self._descriptor_to_cell(descriptor)
        return self._cells.get(cell)

    def clear(self) -> None:
        """Remove all entries from the archive."""
        self._cells.clear()

    def coverage(self) -> float:
        """Fraction of archive cells that are occupied.

        Returns:
            Float in [0, 1]. 0 = empty, 1 = every cell has an agent.
        """
        return self.size / self.capacity

    def fitness_stats(self) -> dict[str, float]:
        """Summary statistics of fitness across occupied cells.

        Returns:
            Dict with ``'mean'``, ``'std'``, ``'min'``, ``'max'``
            fitness values. All 0.0 if archive is empty.
        """
        if self.size == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        fitnesses = np.array(
            [e.fitness for e in self._cells.values()], dtype=np.float64
        )
        return {
            "mean": float(np.mean(fitnesses)),
            "std": float(np.std(fitnesses)),
            "min": float(np.min(fitnesses)),
            "max": float(np.max(fitnesses)),
        }

    def __repr__(self) -> str:
        return (
            f"BehavioralArchive(grid_size={self.grid_size}, "
            f"occupied={self.size}/{self.capacity}, "
            f"coverage={self.coverage():.4f})"
        )


def extract_descriptors(
    trajectory: dict[str, np.ndarray],
    alive_mask: np.ndarray | None = None,
    num_actions: int = 5,
) -> np.ndarray:
    """Extract 2D behavioral descriptors from a trajectory.

    Computes the two MAP-Elites descriptor axes:
    - **Axis 0 — movement_entropy:** Normalized entropy of the action
      distribution. 0 = always same action, 1 = uniform random.
    - **Axis 1 — field_write_frequency:** Fraction of total steps the
      agent was alive (alive agents auto-write to the field).

    Args:
        trajectory: Trajectory dict with at least:
            - ``'actions'``: ``(T, num_agents)`` int
            - ``'alive_mask'``: ``(T, num_agents)`` bool
        alive_mask: Optional per-agent alive mask of shape
            ``(num_agents,)`` for filtering. If ``None``, descriptors
            are computed for all agents.

    Returns:
        Array of shape ``(num_agents, 2)`` with descriptor values
        clipped to [0, 1]. Dead agents (never alive in trajectory)
        get ``[0, 0]``.
    """
    from scipy.stats import entropy as scipy_entropy

    actions = np.asarray(trajectory["actions"])           # (T, A)
    alive = np.asarray(trajectory["alive_mask"])           # (T, A)
    num_steps, num_agents = actions.shape

    descriptors = np.zeros((num_agents, 2), dtype=np.float64)

    for a in range(num_agents):
        agent_alive = alive[:, a].astype(bool)
        n_alive = int(np.sum(agent_alive))

        if n_alive == 0:
            continue

        # Axis 0: movement entropy
        agent_actions = actions[agent_alive, a]
        counts = np.bincount(agent_actions.astype(int), minlength=num_actions)
        probs = counts / counts.sum()
        max_entropy = np.log(num_actions)
        if max_entropy > 0:
            descriptors[a, 0] = float(scipy_entropy(probs) / max_entropy)

        # Axis 1: field write frequency (fraction of total steps alive)
        descriptors[a, 1] = n_alive / num_steps

    # Clip to [0, 1]
    descriptors = np.clip(descriptors, 0.0, 1.0)

    return descriptors
