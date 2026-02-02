# Phase 6: Graph Abstraction + Harder Environments

## Goal

Generalize the engine from a 2D grid to arbitrary graphs, then prove emergence holds on harder environments running on that generalized architecture.

Two parts, done in order:
- **6A: Graph Abstraction** — make the engine work on any graph, not just a spatial grid
- **6B: Harder Environments on Graphs** — test emergence under real pressure

---

## Part 6A: Graph Abstraction

### Why This Matters

Right now agents live on a 20x20 grid. That's fine for research, but every future product needs graphs:
- Code scanning: functions connected by calls/imports
- Network security: endpoints connected by traffic
- Supply chain: warehouses connected by routes
- Research: papers connected by citations

If the engine only works on grids, it can only power grid-based experiments. Graph abstraction is the KEY engineering milestone that unlocks productization.

### What Changes

The core idea: a grid IS a graph (each cell is a node, neighbors are edges). So we're generalizing, not rewriting.

| Component | Grid (Current) | Graph (New) |
|-----------|---------------|-------------|
| Environment | 2D array, (H, W) | Adjacency matrix, (N, N) sparse |
| Agent position | (x, y) coordinate | Node ID (integer) |
| Movement | Up/down/left/right/stay | Move to neighbor node / stay |
| Field | (H, W, C) tensor | (N, C) tensor — C channels per node |
| Field diffusion | 3x3 Gaussian shifted sums | Message passing along edges |
| Food | (x, y) coordinates | Node IDs |
| Observations | Local field patch + nearest food | Neighbor field values + nearest food |

### What Stays the Same

- Agent neural networks (ActorCritic architecture)
- Evolution (birth, death, reproduction, mutation)
- PPO training loop
- Specialization tracking
- All analysis tools
- Dashboard (just needs a graph layout renderer instead of grid renderer)

### Success Criteria (6A)

- Engine accepts any graph (adjacency matrix + node features)
- Grid environment still works as a special case (backwards compatible)
- All Phase 5 experiments reproduce on grid-as-graph (same results)
- Field diffusion works along arbitrary edges
- At least one non-grid graph tested (e.g., random graph, small-world graph)

---

## Part 6B: Harder Environments on Graphs

### Why This Matters

Simple foraging on a grid might be too easy. Agents don't NEED to cooperate to find food — cooperation just helps a bit. Harder environments force stronger collective behavior, which means stronger emergence signal, which means a stronger paper.

Running these on graphs (not grids) means the results are more general and more product-relevant.

### Environment 1: Predator

A predator node moves through the graph. Agents near the predator die (or lose large energy). Forces two possible emergent strategies:
- **Swarming**: agents cluster together for safety (predator can only eat one at a time)
- **Dispersal**: agents spread out so the predator can only catch a few

Which strategy emerges? Does the field encode predator location as a warning signal? Do scout agents evolve to track the predator and warn others through the field?

### Environment 2: Multi-Food Types

Multiple food types on the graph. Each agent can only carry one type at a time. Collecting a matched pair (type A + type B at same node) gives a big bonus. Forces:
- **Specialization**: some agents collect type A, others collect type B
- **Trading/meeting**: agents must converge at the same node with complementary types
- **Field as coordination**: field encodes where type A and type B deposits are

This is the strongest test of specialization — agents MUST differentiate to succeed.

### Environment 3: Resource Scarcity Cycles

Food availability follows cycles — abundant for N steps, then scarce for N steps. Forces:
- **Adaptation**: agents must switch strategies when conditions change
- **Field memory**: during scarcity, the field retains knowledge from the abundant phase
- **Hoarding vs sharing**: do agents evolve to stockpile energy during abundance?

Tests whether the collective adapts faster than individuals to changing conditions.

### Success Criteria (6B)

- Emergence metrics (synergy, transfer entropy, superlinear scaling) are STRONGER on harder environments than on simple foraging
- Predator environment produces measurable swarming or dispersal strategies
- Multi-food environment produces clear specialization (different species for different food types)
- Scarcity cycles show faster collective adaptation than individual adaptation
- All results on graph-based environments (not grids)

---

## Paper Opportunity

6A + 6B together make the second paper much stronger than Phase 5 alone:

**Phase 5 paper:** "Learnable Stigmergy: Emergent Communication through Differentiable Pheromone Fields"
- Proves emergence exists on simple foraging environments

**Phase 6 paper (or expanded Phase 5 paper):** "Emergent Collective Intelligence on Arbitrary Graphs Across Diverse Environment Complexities"
- Proves emergence generalizes to any graph structure
- Proves emergence gets stronger under harder conditions
- Directly supports the product thesis (engine works on any domain)

---

## What This Phase Builds On

- Phase 5 metrics (synergy, causal emergence, baselines) — reuse all of them
- Phase 5 ablation framework — reuse for new environments
- Phase 4B checkpointing — essential for long graph-based training runs
- Phase 4 dashboard — extend with graph layout renderer

## What Comes After

Phase 7: LLM Field Integration — replace the numerical field with a local LLM. The graph abstraction from 6A makes this natural: nodes already carry features, now they carry source code. The LLM reads the code and responds semantically. This is the product breakthrough.
