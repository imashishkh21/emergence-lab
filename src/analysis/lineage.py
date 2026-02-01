"""Lineage tracking for evolutionary dynamics.

Records birth/death events and computes family tree statistics
for analyzing evolutionary pressure and dominant lineages.
"""

from dataclasses import dataclass
from dataclasses import field as dataclass_field


@dataclass
class AgentRecord:
    """Record of a single agent's lifecycle.

    Attributes:
        agent_id: Unique identifier for this agent.
        parent_id: ID of parent agent, or -1 if original.
        birth_step: Step at which the agent was born (0 for originals).
        death_step: Step at which the agent died, or -1 if still alive.
        children: List of child agent IDs.
    """

    agent_id: int
    parent_id: int
    birth_step: int
    death_step: int = -1
    children: list[int] = dataclass_field(default_factory=list)

    @property
    def is_alive(self) -> bool:
        """Whether the agent is still alive."""
        return self.death_step == -1

    def lifespan(self, current_step: int | None = None) -> int:
        """Return the number of steps the agent has been alive.

        Args:
            current_step: Current simulation step. Used if agent is still
                alive. If None and agent is alive, returns 0.

        Returns:
            Number of steps from birth to death (or current step).
        """
        if self.death_step >= 0:
            return self.death_step - self.birth_step
        if current_step is not None:
            return current_step - self.birth_step
        return 0


class LineageTracker:
    """Tracks agent lineages across evolutionary simulation.

    Records births, deaths, and parent-child relationships to enable
    analysis of evolutionary dynamics like dominant lineages and
    generational depth.

    Attributes:
        records: Dict mapping agent_id to AgentRecord.
        current_step: The last step that was recorded.
    """

    def __init__(self) -> None:
        self.records: dict[int, AgentRecord] = {}
        self.current_step: int = 0

    def register_birth(
        self, agent_id: int, parent_id: int, step: int
    ) -> None:
        """Register a new agent birth.

        Args:
            agent_id: Unique ID of the new agent.
            parent_id: ID of the parent agent (-1 for originals).
            step: Simulation step of the birth.
        """
        self.records[agent_id] = AgentRecord(
            agent_id=agent_id,
            parent_id=parent_id,
            birth_step=step,
        )
        if parent_id >= 0 and parent_id in self.records:
            self.records[parent_id].children.append(agent_id)
        self.current_step = max(self.current_step, step)

    def register_death(self, agent_id: int, step: int) -> None:
        """Register an agent death.

        Args:
            agent_id: ID of the agent that died.
            step: Simulation step of the death.
        """
        if agent_id in self.records:
            self.records[agent_id].death_step = step
        self.current_step = max(self.current_step, step)

    def get_lineage_depth(self, agent_id: int) -> int:
        """Get generational depth from the original ancestor.

        Args:
            agent_id: ID of the agent to query.

        Returns:
            Number of generations from the original ancestor.
            Returns 0 for original agents, 1 for their children, etc.
            Returns -1 if agent_id is not found.
        """
        depth = 0
        current = agent_id
        while current in self.records:
            parent = self.records[current].parent_id
            if parent < 0:
                return depth
            depth += 1
            current = parent
        return -1

    def get_family_tree(self, agent_id: int) -> list[int]:
        """Get all descendants of an agent (breadth-first).

        Args:
            agent_id: ID of the root agent.

        Returns:
            List of all descendant agent IDs (not including the root).
            Returns empty list if agent has no descendants or is not found.
        """
        if agent_id not in self.records:
            return []

        descendants: list[int] = []
        queue = list(self.records[agent_id].children)
        while queue:
            child_id = queue.pop(0)
            descendants.append(child_id)
            if child_id in self.records:
                queue.extend(self.records[child_id].children)
        return descendants

    def get_dominant_lineages(self, top_k: int = 5) -> list[tuple[int, int]]:
        """Get lineages with the most total descendants.

        Finds original ancestors (parent_id == -1) and counts all their
        descendants. Returns the top_k lineages sorted by descendant count.

        Args:
            top_k: Number of top lineages to return.

        Returns:
            List of (ancestor_id, descendant_count) tuples, sorted by
            descendant count descending.
        """
        originals = [
            aid
            for aid, rec in self.records.items()
            if rec.parent_id == -1
        ]

        lineage_sizes: list[tuple[int, int]] = []
        for ancestor_id in originals:
            descendants = self.get_family_tree(ancestor_id)
            lineage_sizes.append((ancestor_id, len(descendants)))

        lineage_sizes.sort(key=lambda x: x[1], reverse=True)
        return lineage_sizes[:top_k]

    def get_summary(self) -> dict[str, object]:
        """Get summary statistics of all tracked lineages.

        Returns:
            Dict with keys:
            - total_agents: Total agents ever registered.
            - alive_agents: Number of currently alive agents.
            - dead_agents: Number of dead agents.
            - total_births: Agents born from reproduction (parent_id >= 0).
            - original_agents: Agents with no parent (parent_id == -1).
            - max_depth: Maximum generational depth across all agents.
            - dominant_lineages: Top 5 lineages by descendant count.
        """
        total = len(self.records)
        alive = sum(1 for r in self.records.values() if r.is_alive)
        dead = total - alive
        births = sum(
            1 for r in self.records.values() if r.parent_id >= 0
        )
        originals = total - births

        max_depth = 0
        for agent_id in self.records:
            depth = self.get_lineage_depth(agent_id)
            if depth > max_depth:
                max_depth = depth

        return {
            "total_agents": total,
            "alive_agents": alive,
            "dead_agents": dead,
            "total_births": births,
            "original_agents": originals,
            "max_depth": max_depth,
            "dominant_lineages": self.get_dominant_lineages(),
        }
