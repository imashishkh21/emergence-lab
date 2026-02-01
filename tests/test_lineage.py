"""Tests for lineage tracking."""

import pytest

from src.analysis.lineage import AgentRecord, LineageTracker


class TestAgentRecord:
    """Tests for AgentRecord dataclass."""

    def test_default_alive(self):
        """New agent record defaults to alive (death_step=-1)."""
        rec = AgentRecord(agent_id=0, parent_id=-1, birth_step=0)
        assert rec.is_alive is True
        assert rec.death_step == -1

    def test_dead_after_death_step(self):
        """Agent is dead after death_step is set."""
        rec = AgentRecord(agent_id=0, parent_id=-1, birth_step=0, death_step=10)
        assert rec.is_alive is False

    def test_lifespan_dead(self):
        """Lifespan of dead agent is death_step - birth_step."""
        rec = AgentRecord(agent_id=0, parent_id=-1, birth_step=5, death_step=15)
        assert rec.lifespan() == 10

    def test_lifespan_alive_with_current_step(self):
        """Lifespan of alive agent uses current_step."""
        rec = AgentRecord(agent_id=0, parent_id=-1, birth_step=5)
        assert rec.lifespan(current_step=20) == 15

    def test_lifespan_alive_no_current_step(self):
        """Lifespan of alive agent without current_step returns 0."""
        rec = AgentRecord(agent_id=0, parent_id=-1, birth_step=5)
        assert rec.lifespan() == 0

    def test_empty_children(self):
        """New agent has empty children list."""
        rec = AgentRecord(agent_id=0, parent_id=-1, birth_step=0)
        assert rec.children == []


class TestLineageTrackerBirth:
    """Tests for birth registration."""

    def test_register_original_agent(self):
        """Original agents have parent_id=-1 and depth 0."""
        tracker = LineageTracker()
        tracker.register_birth(agent_id=0, parent_id=-1, step=0)
        assert 0 in tracker.records
        assert tracker.records[0].parent_id == -1
        assert tracker.records[0].birth_step == 0

    def test_register_offspring(self):
        """Offspring is registered with correct parent_id."""
        tracker = LineageTracker()
        tracker.register_birth(agent_id=0, parent_id=-1, step=0)
        tracker.register_birth(agent_id=1, parent_id=0, step=10)
        assert tracker.records[1].parent_id == 0
        assert tracker.records[1].birth_step == 10

    def test_parent_gets_child_link(self):
        """Registering offspring adds child to parent's children list."""
        tracker = LineageTracker()
        tracker.register_birth(agent_id=0, parent_id=-1, step=0)
        tracker.register_birth(agent_id=1, parent_id=0, step=10)
        assert 1 in tracker.records[0].children

    def test_multiple_children(self):
        """Parent can have multiple children."""
        tracker = LineageTracker()
        tracker.register_birth(agent_id=0, parent_id=-1, step=0)
        tracker.register_birth(agent_id=1, parent_id=0, step=5)
        tracker.register_birth(agent_id=2, parent_id=0, step=10)
        assert tracker.records[0].children == [1, 2]

    def test_current_step_updated(self):
        """current_step tracks the latest recorded step."""
        tracker = LineageTracker()
        tracker.register_birth(agent_id=0, parent_id=-1, step=0)
        assert tracker.current_step == 0
        tracker.register_birth(agent_id=1, parent_id=0, step=50)
        assert tracker.current_step == 50


class TestLineageTrackerDeath:
    """Tests for death registration."""

    def test_register_death(self):
        """Death sets death_step and marks agent as dead."""
        tracker = LineageTracker()
        tracker.register_birth(agent_id=0, parent_id=-1, step=0)
        tracker.register_death(agent_id=0, step=20)
        assert tracker.records[0].death_step == 20
        assert tracker.records[0].is_alive is False

    def test_death_unknown_agent(self):
        """Death of unregistered agent is a no-op (no crash)."""
        tracker = LineageTracker()
        tracker.register_death(agent_id=999, step=10)
        assert 999 not in tracker.records

    def test_death_updates_current_step(self):
        """Death updates current_step."""
        tracker = LineageTracker()
        tracker.register_birth(agent_id=0, parent_id=-1, step=0)
        tracker.register_death(agent_id=0, step=100)
        assert tracker.current_step == 100


class TestLineageDepth:
    """Tests for get_lineage_depth."""

    def test_original_agent_depth_zero(self):
        """Original agents have depth 0."""
        tracker = LineageTracker()
        tracker.register_birth(agent_id=0, parent_id=-1, step=0)
        assert tracker.get_lineage_depth(0) == 0

    def test_child_depth_one(self):
        """Direct children have depth 1."""
        tracker = LineageTracker()
        tracker.register_birth(agent_id=0, parent_id=-1, step=0)
        tracker.register_birth(agent_id=1, parent_id=0, step=5)
        assert tracker.get_lineage_depth(1) == 1

    def test_grandchild_depth_two(self):
        """Grandchildren have depth 2."""
        tracker = LineageTracker()
        tracker.register_birth(agent_id=0, parent_id=-1, step=0)
        tracker.register_birth(agent_id=1, parent_id=0, step=5)
        tracker.register_birth(agent_id=2, parent_id=1, step=10)
        assert tracker.get_lineage_depth(2) == 2

    def test_deep_lineage(self):
        """Multi-generation depth is computed correctly."""
        tracker = LineageTracker()
        tracker.register_birth(agent_id=0, parent_id=-1, step=0)
        for i in range(1, 6):
            tracker.register_birth(agent_id=i, parent_id=i - 1, step=i * 10)
        assert tracker.get_lineage_depth(5) == 5

    def test_unknown_agent_returns_negative(self):
        """Unknown agent returns -1."""
        tracker = LineageTracker()
        assert tracker.get_lineage_depth(999) == -1


class TestFamilyTree:
    """Tests for get_family_tree."""

    def test_no_descendants(self):
        """Agent with no children returns empty list."""
        tracker = LineageTracker()
        tracker.register_birth(agent_id=0, parent_id=-1, step=0)
        assert tracker.get_family_tree(0) == []

    def test_direct_children(self):
        """Agent returns its direct children."""
        tracker = LineageTracker()
        tracker.register_birth(agent_id=0, parent_id=-1, step=0)
        tracker.register_birth(agent_id=1, parent_id=0, step=5)
        tracker.register_birth(agent_id=2, parent_id=0, step=10)
        tree = tracker.get_family_tree(0)
        assert set(tree) == {1, 2}

    def test_multi_generation(self):
        """Family tree includes grandchildren and beyond."""
        tracker = LineageTracker()
        tracker.register_birth(agent_id=0, parent_id=-1, step=0)
        tracker.register_birth(agent_id=1, parent_id=0, step=5)
        tracker.register_birth(agent_id=2, parent_id=1, step=10)
        tracker.register_birth(agent_id=3, parent_id=2, step=15)
        tree = tracker.get_family_tree(0)
        assert set(tree) == {1, 2, 3}

    def test_branching_tree(self):
        """Family tree follows multiple branches."""
        tracker = LineageTracker()
        tracker.register_birth(agent_id=0, parent_id=-1, step=0)
        tracker.register_birth(agent_id=1, parent_id=0, step=5)
        tracker.register_birth(agent_id=2, parent_id=0, step=5)
        tracker.register_birth(agent_id=3, parent_id=1, step=10)
        tracker.register_birth(agent_id=4, parent_id=2, step=10)
        tree = tracker.get_family_tree(0)
        assert set(tree) == {1, 2, 3, 4}

    def test_unknown_agent(self):
        """Unknown agent returns empty list."""
        tracker = LineageTracker()
        assert tracker.get_family_tree(999) == []

    def test_subtree(self):
        """Family tree from a non-root agent returns only its descendants."""
        tracker = LineageTracker()
        tracker.register_birth(agent_id=0, parent_id=-1, step=0)
        tracker.register_birth(agent_id=1, parent_id=0, step=5)
        tracker.register_birth(agent_id=2, parent_id=0, step=5)
        tracker.register_birth(agent_id=3, parent_id=1, step=10)
        tree = tracker.get_family_tree(1)
        assert tree == [3]


class TestDominantLineages:
    """Tests for get_dominant_lineages."""

    def test_single_lineage(self):
        """Single ancestor with children."""
        tracker = LineageTracker()
        tracker.register_birth(agent_id=0, parent_id=-1, step=0)
        tracker.register_birth(agent_id=1, parent_id=0, step=5)
        tracker.register_birth(agent_id=2, parent_id=0, step=10)
        result = tracker.get_dominant_lineages(top_k=5)
        assert result == [(0, 2)]

    def test_multiple_lineages_sorted(self):
        """Multiple lineages sorted by descendant count."""
        tracker = LineageTracker()
        # Lineage 0: 3 descendants
        tracker.register_birth(agent_id=0, parent_id=-1, step=0)
        tracker.register_birth(agent_id=10, parent_id=0, step=5)
        tracker.register_birth(agent_id=11, parent_id=0, step=5)
        tracker.register_birth(agent_id=12, parent_id=10, step=10)
        # Lineage 1: 1 descendant
        tracker.register_birth(agent_id=1, parent_id=-1, step=0)
        tracker.register_birth(agent_id=20, parent_id=1, step=5)
        result = tracker.get_dominant_lineages(top_k=5)
        assert result[0] == (0, 3)
        assert result[1] == (1, 1)

    def test_top_k_limits(self):
        """top_k limits the number of returned lineages."""
        tracker = LineageTracker()
        for i in range(10):
            tracker.register_birth(agent_id=i, parent_id=-1, step=0)
        result = tracker.get_dominant_lineages(top_k=3)
        assert len(result) == 3

    def test_no_agents(self):
        """Empty tracker returns empty list."""
        tracker = LineageTracker()
        result = tracker.get_dominant_lineages()
        assert result == []

    def test_no_descendants(self):
        """Lineages with zero descendants still appear."""
        tracker = LineageTracker()
        tracker.register_birth(agent_id=0, parent_id=-1, step=0)
        result = tracker.get_dominant_lineages()
        assert result == [(0, 0)]


class TestSummary:
    """Tests for get_summary."""

    def test_empty_tracker(self):
        """Summary for empty tracker."""
        tracker = LineageTracker()
        summary = tracker.get_summary()
        assert summary["total_agents"] == 0
        assert summary["alive_agents"] == 0
        assert summary["dead_agents"] == 0
        assert summary["total_births"] == 0
        assert summary["original_agents"] == 0
        assert summary["max_depth"] == 0

    def test_full_scenario(self):
        """Summary for a multi-generation scenario."""
        tracker = LineageTracker()
        # 2 original agents
        tracker.register_birth(agent_id=0, parent_id=-1, step=0)
        tracker.register_birth(agent_id=1, parent_id=-1, step=0)
        # Agent 0 has child 2
        tracker.register_birth(agent_id=2, parent_id=0, step=10)
        # Agent 2 has child 3 (depth 2)
        tracker.register_birth(agent_id=3, parent_id=2, step=20)
        # Agent 1 dies
        tracker.register_death(agent_id=1, step=15)

        summary = tracker.get_summary()
        assert summary["total_agents"] == 4
        assert summary["alive_agents"] == 3  # 0, 2, 3 alive
        assert summary["dead_agents"] == 1  # 1 dead
        assert summary["total_births"] == 2  # agents 2 and 3
        assert summary["original_agents"] == 2  # agents 0 and 1
        assert summary["max_depth"] == 2  # agent 3

    def test_summary_dominant_lineages(self):
        """Summary includes dominant lineage info."""
        tracker = LineageTracker()
        tracker.register_birth(agent_id=0, parent_id=-1, step=0)
        tracker.register_birth(agent_id=1, parent_id=0, step=5)
        summary = tracker.get_summary()
        assert "dominant_lineages" in summary
        assert summary["dominant_lineages"] == [(0, 1)]
