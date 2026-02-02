# Phase 5: Prove Emergence (Paper-Ready)

## Goal

Definitively prove that the collective is smarter than the sum of its parts. Design experiments that mathematically demonstrate emergence, then write the paper.

This phase is about PROOF, not building. No new products, no new features. Just rigorous science.

---

## Part A — Metrics (The Math)

### 1. Synergy / Partial Information Decomposition (PID)
Measure information that ONLY exists in the combination of agents, not in any individual. This is the crown jewel metric — if synergy is high, emergence is mathematically proven.

### 2. Causal Emergence (Hoel)
Show that the macro-level (swarm behavior) is more informative and predictive than the micro-level (individual agent behavior). Erik Hoel's framework for quantifying when coarse-graining reveals more structure.

### 3. Transfer Entropy at Scale
Already built in Phase 4 (US-010). Run it at 10M+ steps with statistical significance (N=30+ runs). Measure information flowing between agents through the field.

---

## Part A.5 — Baseline Comparisons

Reviewers will ask "how does this compare to existing multi-agent methods?" — so we answer preemptively.

### 1. vs Fixed Pheromone Rules (Classic Swarm)
Hardcoded deposit/follow rules like traditional ant colony optimization. Tests the core claim: "learnable stigmergy beats hand-designed stigmergy." Cheapest baseline to implement, most directly relevant.

### 2. vs MAPPO (Modern MARL)
Multi-Agent PPO — shared critic, independent actors. The standard modern multi-agent RL approach. Tests: "why not just use standard MARL?" Can use JaxMARL library to reduce implementation effort.

### 3. vs QMIX (If Time Allows)
Value decomposition baseline. Tests: "what about centralized training with decentralized execution?" Nice to have but MAPPO already covers the MARL comparison.

All baselines run on the same environment, same compute budget, same metrics.

---

## Part B — Killer Experiments (The Proof)

### 1. Hidden Resources
Resources that are only visible when multiple agents are nearby. A solo agent literally cannot solve this — it can never see the food. Only a coordinated group can. If the swarm finds these, that's undeniable proof of collective intelligence.

### 2. Superlinear Scaling
N agents should do better than N times what 1 agent does. If 64 agents collect more than 64x what a solo agent collects, that's superlinear — the whole is greater than the sum of its parts. This is the mathematical definition of emergence.

### 3. Stigmergy Ablation at Scale
Full system vs frozen field vs wiped field. N=30 runs per condition, 10M steps each, with confidence intervals and statistical significance tests. This is the rigorous, publishable version of the Phase 1 ablation that showed Normal > Zeroed > Random.

---

## Part C — Paper

**Target conferences:**
- NeurIPS 2026 (deadline ~May 2026) — tight but possible
- AAMAS 2027 (deadline ~Oct 2026) — more realistic
- ALIFE 2027 — natural fit for the emergence angle

**Paper title idea:**
"Learnable Stigmergy: Emergent Communication through Differentiable Pheromone Fields"

---

## Success Criteria

- Prove with metrics that swarm solves tasks individuals cannot
- High synergy score (information only exists in the collective)
- Clear ablation showing field is essential (not just parallelism)
- Beat both classic swarm (fixed pheromones) AND modern MARL (MAPPO) baselines
- All results with N=30+ runs and statistical significance
- Paper draft ready for submission

---

## What This Phase Builds On

- Transfer Entropy: already implemented (Phase 4 US-010)
- Division of Labor Index: already implemented (Phase 4 US-011)
- Phase Transition Detection: already implemented (Phase 4 US-012)
- Specialization Tracking: already implemented (Phase 3)
- Field Ablation (basic): already done (Phase 1), needs scaling up
- Live Dashboard: already built (Phase 4), useful for visual demos in the paper

## Why We're Confident (70-80%)

| Evidence | What It Means |
|----------|---------------|
| 8.4x field advantage (Phase 3) | Field carries REAL information |
| Random field hurts agents | Agents use it as presence detector — pure stigmergy |
| Gradient homogenization identified | We KNOW the failure mode and are fixing it |
| Phase 4 fixes architecture | Agent-specific heads + freeze-evolve = real divergence |

We're not hoping for emergence — we've already seen proto-emergence and identified what was blocking full divergence. Phase 4 removes the block. Phase 5 measures the result.

---

## Full Roadmap Context

```
Phase 4   → Research Microscope (dashboard, metrics, fix architecture)
Phase 4B  → Kaggle Infrastructure (checkpointing, long runs)
Phase 5   → Prove Emergence (this phase — paper on simple environments)
Phase 6A  → Graph Abstraction (generalize engine from grid to any graph)
Phase 6B  → Harder Environments ON GRAPHS (predator, multi-food, scarcity)
Phase 7   → LLM Field Integration (the product breakthrough)
```

Phase 6B comes AFTER 6A so harder environments run on the generalized architecture. This makes a stronger paper claim: "emergent collective intelligence on arbitrary graphs, demonstrated across diverse environment complexities."

## Key References

- Williams & Beer (2010) — Partial Information Decomposition
- Hoel et al. (2013) — Causal Emergence, effective information
- Lehman & Stanley (2011) — Novelty Search
- Rashid et al. (2018) — QMIX
- Yu et al. (2022) — MAPPO
- Dorigo et al. (2000) — Ant Colony Optimization (fixed pheromone baseline)
