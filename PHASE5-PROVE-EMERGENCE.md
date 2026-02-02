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

---

## Deep Research Findings

Comprehensive research conducted across four parallel investigations. Everything below is what we found — the tools, methods, formulas, libraries, pitfalls, and paper standards needed to execute Phase 5.

---

### DR-1: Synergy / Partial Information Decomposition (PID)

#### What PID Is

PID (Williams & Beer, 2010) decomposes the mutual information that multiple source variables provide about a target into four atoms:

```
I(X1, X2 ; Y) = Redundancy + Unique(X1) + Unique(X2) + Synergy
```

- **Redundancy**: Info about Y that EITHER X1 or X2 can provide alone (overlapping)
- **Unique(X1)**: Info only X1 provides
- **Unique(X2)**: Info only X2 provides
- **Synergy**: Info about Y that is ONLY available when you observe X1 AND X2 together — neither provides it alone

Synergy > 0 is the mathematical definition of "the whole is greater than the sum of its parts."

#### The Math (Williams & Beer I_min)

Step 1 — Specific information for a particular outcome y:

```
I_spec(Y=y ; X1) = sum_x1 p(x1|y) * log[ p(y|x1) / p(y) ]
```

Step 2 — Redundancy via minimum specific information:

```
I_min(Y; X1, X2) = sum_y p(y) * min[ I_spec(Y=y; X1), I_spec(Y=y; X2) ]
```

Step 3 — Derive all other atoms:

```
Unique(X1) = I(Y; X1) - I_min
Unique(X2) = I(Y; X2) - I_min
Synergy    = I(Y; X1, X2) - I(Y; X1) - I(Y; X2) + I_min
```

#### Canonical Example: XOR Gate (Pure Synergy)

Y = X1 XOR X2, with X1 and X2 independent uniform bits:
- I(Y; X1) = 0 (X1 alone tells you nothing)
- I(Y; X2) = 0 (X2 alone tells you nothing)
- I(Y; X1, X2) = 1 bit (together they perfectly determine Y)
- Result: Redundancy = 0, Unique = 0, **Synergy = 1 bit**

#### Known Limitation

The "two-bit copy problem" — I_min gives wrong results for Y = (X1, X2). Alternative measures exist: I_BROJA (Bertschinger), I_ccs (Ince), I_mmi (Barrett). The `dit` library supports all of them.

#### How to Apply PID to Our System

**Option A: Pairwise Agent Synergy (recommended, following Riedl 2025)**

```
Sources:  X_i,t = behavior of agent i at time t (action, position, energy)
          X_j,t = behavior of agent j at time t
Target:   T_ij,t+L = joint future state of agents i,j at time t+L

If Synergy > 0: agents coordinate in ways traceable to neither agent alone.
This IS emergence.
```

**Option B: Field as Mediating Variable (most natural for stigmergy)**

```
Sources:  X_t = agent actions/states at time t (discretized)
          F_t = field state at time t (discretized field features)
Target:   Y_t+1 = collective outcome at time t+1 (total food, population fitness)

Synergy tells you: how much of tomorrow's outcome can only be predicted
by knowing BOTH agent states AND field state jointly?
```

**Option C: Field Ablation with PID**

```
Compute Synergy for: normal field, zeroed field, random field
If Syn(normal) >> Syn(zeroed) >> Syn(random): the field IS the medium.
```

#### Variable Design for Our System

1. Agent behavior vector: action (already discrete, 6 values)
2. Field features: reduce (H,W,C) to low-dim summary (field entropy, structure score, field-food MI, or PCA)
3. Collective outcome target: total food collected in next N steps — discretize into K=2 bins via quantile binning
4. Discretization: K=2 quantile bins (crude but robust, keeps sample requirements manageable)

#### Computational Scaling

| Variables | Atoms | Feasibility |
|-----------|-------|-------------|
| 2 sources | 4 atoms | Trivial |
| 3 sources | 18 atoms | Feasible |
| 4 sources | 166 atoms | Expensive |
| n sources | Dedekind numbers | Intractable for n > 5 |

For 32 agents, full PID is impossible. Solutions:
- Compute pairwise PID over agent pairs (as Riedl 2025 does)
- Use O-information as a scalar summary (scales polynomially)
- Aggregate agents into species first, then PID over 2-3 group-level variables

#### Sample Size Requirements

With K=2 bins and 2 sources + 1 target: joint states = 8. Need ~500-1000 timesteps minimum.

Always compare synergy against null distributions:
- Row shuffle: permute each agent's time series independently (breaks cross-agent coordination)
- Column shuffle: block-shuffle time indices jointly (breaks temporal dependencies)
- If synergy is not above both surrogates at p < 0.05, it's spurious.

#### Simpler/Cheaper Alternatives

**O-Information (recommended practical alternative)**

```
Omega(X1, ..., Xn) = TC - DTC

TC  = sum_i H(Xi) - H(X1, ..., Xn)           (Total Correlation)
DTC = H(X1, ..., Xn) - sum_i H(Xi | X_{-i})  (Dual Total Correlation)

Omega > 0 → redundancy dominates
Omega < 0 → SYNERGY dominates (emergence!)
Omega = 0 → balance
```

Key advantages:
- Scales as O(n) in the number of variables
- Works with continuous data via Gaussian copula
- The `hoi` library is JAX-native — fits our stack perfectly
- No combinatorial explosion

**Causal Emergence criterion (Rosas et al. 2020)**

```
Psi(V) = I(V_t ; V_{t+1}) - sum_i I(X_i,t ; V_{t+1})

Psi > 0 → macro variable V is causally emergent
```

V could be: mean field intensity, total population fitness, cluster centroid positions. Single scalar comparison — much cheaper than full PID.

**Interaction Information (simplest, cruder)**

```
II(X1; X2; Y) = I(X1; Y) + I(X2; Y) - I(X1, X2; Y)

Negative II → synergy dominates
Positive II → redundancy dominates
Limitation: confounds the two
```

Good for initial screening before investing in full PID.

#### Python Libraries

| Library | What It Does | Install | Notes |
|---------|-------------|---------|-------|
| `dit` | Full PID computation (discrete). Supports I_min, I_BROJA, I_MMI, I_CCS | `pip install dit` | Gold standard for discrete PID |
| `pidpy` | Simpler PID interface with bias correction | `pip install pidpy` | Good for quick synergy checks |
| `hoi` | O-information, TC, DTC for continuous data. Built on JAX. | `pip install hoi` | Best fit for our JAX codebase |
| `THOI` | GPU-accelerated higher-order interactions (PyTorch) | `pip install thoi` | For large-scale batch processing |
| `frites` | Neuroscience-oriented info theory with PID | `pip install frites` | Alternative continuous estimator |
| `IDTxl` | Information Dynamics Toolkit, implements I_ccs | See docs | Efficient estimation |

#### Implementation Strategy

**Phase 1 (days): Quick win with O-information**
- Use `hoi` library (JAX-native)
- Compute O-information over agent behavioral features per timestep
- Negative O-info = synergistic emergence
- Compare: normal field vs zeroed vs random
- If O-info(normal) << O-info(zeroed): proof of field-mediated emergence

**Phase 2 (week): Pairwise PID with ablation**
- Use `dit` for discrete PID
- Variables: agent_action (discrete), field_summary (binned), future_food (binned)
- Compute for all agent pairs, report median synergy
- Surrogate test: shuffle agent identities, shuffle time

**Phase 3 (week): Causal emergence Psi**
- Define macro variable V = mean field intensity
- Compute Psi = I(V_t; V_{t+1}) - sum_i I(X_i,t; V_{t+1})
- Psi > 0 proves causal emergence of the field

#### Key Finding: The Gap in the Literature

**No published work (as of early 2026) directly applies PID/synergy to stigmergic multi-agent RL with shared fields.** The Honda paper (2022) uses simple grid-world foraging but not RL or learned fields. Riedl (2025) uses LLMs, not swarm agents. Rosas (2020) applied to Game of Life and flocking, not learned stigmergy.

Our project — PID synergy on a learned shared field in evolutionary multi-agent RL — would be a novel contribution.

---

### DR-2: Causal Emergence (Erik Hoel's Framework)

#### Core Idea

Macroscale descriptions of a system can carry MORE causal information than microscale descriptions. The macro level supervenes on the micro (is fully determined by it), but it can SUPERSEDE it causally.

This happens when coarse-graining averages out noise and collapses many-to-one mappings, producing a cleaner, more deterministic macro-level transition matrix.

#### Effective Information (EI)

EI measures causal strength of a system's transition dynamics. It is the mutual information between current state and next state, computed under a uniform intervention distribution:

```
EI = H(Y_avg) - <H(Y|x)>

where:
  H(Y_avg) = entropy of the average output distribution
  <H(Y|x)> = average entropy of each row of the transition probability matrix (TPM)
```

EI decomposes into:
- **Determinism**: How reliably does a given cause produce a specific effect? (low row entropy = high determinism)
- **Degeneracy**: How many different causes lead to the same effect? (uniform columns = high degeneracy)

```
EI = effectiveness × log2(n)
effectiveness = determinism - degeneracy
```

| Condition | EI |
|-----------|-----|
| Every state maps to same successor (totally degenerate) | 0 |
| Every state transitions uniformly at random | 0 |
| Perfect bijection (each state → unique successor) | log2(n) (maximum) |

#### Computing EI in Python

```python
import numpy as np
from scipy.stats import entropy

def effective_information(tpm):
    """
    Compute EI of a transition probability matrix.
    tpm: (n, n) array where tpm[i,j] = P(next=j | current=i)
    """
    n = tpm.shape[0]
    row_entropies = np.array([entropy(tpm[i], base=2) for i in range(n)])
    avg_row_entropy = np.mean(row_entropies)
    avg_output = np.mean(tpm, axis=0)
    avg_output_entropy = entropy(avg_output, base=2)
    return avg_output_entropy - avg_row_entropy
```

#### Coarse-Graining: Micro → Macro

A coarse-graining is a mapping that groups micro-states into macro-states. Given micro TPM and a partition:

```python
def coarse_grain_tpm(tpm_micro, partition):
    """
    partition: list of lists, e.g. [[0,1], [2,3]]
    """
    k = len(partition)
    tpm_macro = np.zeros((k, k))
    for i, group_i in enumerate(partition):
        for j, group_j in enumerate(partition):
            prob = 0
            for micro_i in group_i:
                prob += sum(tpm_micro[micro_i, micro_j] for micro_j in group_j)
            tpm_macro[i, j] = prob / len(group_i)
    return tpm_macro
```

If EI(tpm_macro) > EI(tpm_micro), the system exhibits causal emergence at that coarse-graining.

#### Applying to Our System

**Micro level**: All 32 agents' positions, energies, alive flags, weights, field state. Astronomically large state space.

**Macro level candidates**:

| Macro Variable | What It Captures |
|----------------|-----------------|
| Population count | Collective survival |
| Mean swarm position (centroid) | Group movement |
| Spatial dispersion (spread) | Exploration vs exploitation |
| Total food collected per step | Collective foraging efficiency |
| Field entropy | Information structure in the medium |
| Field-food mutual information | Whether field encodes food knowledge |
| Cluster count / species count | Specialization level |
| Mean energy | Population fitness |

**The question**: Does EI(field + population stats) > EI(individual agent states)?

**Practical approach**:
1. Discretize: bin continuous variables into 4-8 categories
2. Build TPMs from rollout data (count transitions in trajectory)
3. Compute EI at both levels
4. Use NIS+ to learn optimal coarse-graining automatically from raw data

**The killer experiment**: Compute EI_macro - EI_micro across training. If this gap grows from zero to positive during training, synchronized with the moment agents learn to use the field, that's emergence happening in real time.

#### Challenges

- **State space explosion**: Continuous system needs aggressive discretization or use NIS+ (neural coarse-graining)
- **Combinatorial search**: Number of possible partitions grows super-exponentially. Use greedy/heuristic search (einet library) or domain-informed coarse-graining
- **Non-stationarity**: System changes during training. Use windowed analysis (1000-step windows) to track how CE changes over time
- **Validity**: Not all coarse-grainings are dynamically consistent. CE 2.0 uses KL-divergence checks.

#### Python Libraries

| Library | What It Does | Link |
|---------|-------------|------|
| `einet` (jkbren) | EI for networks, CE search over coarse-grainings. 9 tutorial notebooks. | github.com/jkbren/einet |
| NIS | ML-based EI maximization via invertible neural networks | github.com/jakezj/NIS_for_Causal_Emergence |
| NIS+ | Enhanced — learns coarse-graining from raw time series | github.com/Matthew-ymz/Code-for-Finding-emergence-in-data-5.2 |
| `causalinfo` | Information measures on causal graphs | github.com/brettc/causalinfo |
| ReconcilingEmergences | Psi/Delta/Gamma emergence measures from time series (MATLAB, callable via oct2py) | github.com/pmediano/ReconcilingEmergences |

#### Recent Papers (2023-2025)

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| Causal Emergence 2.0 (Hoel) | 2025 | Axiomatic grounding, emergent complexity measure |
| Engineering Emergence (Jansma & Hoel) | 2025 | Taxonomy: top-heavy, bottom-heavy, scale-free |
| CE via SVD (Zhang et al.) | 2025 | Coarse-graining-independent CE measure |
| Finding Emergence in Data / NIS+ (Yang et al.) | 2024 | Neural network framework to learn CE from raw time series |
| What the Flock Knows (2025) | 2025 | CE applied to flocking — collective dynamics exceed individual components |
| Emergent Behaviors in Multi-Agent Pursuit-Evasion | 2025 | NIS framework for extracting macro-dynamics in MARL |
| Automatic Design of Stigmergy-Based Behaviours | 2024 | Designed stigmergic coordination for robot swarms |

---

### DR-3: Baseline Comparisons (MAPPO, QMIX, Fixed Pheromones)

#### MAPPO — How It Works

Centralized Training with Decentralized Execution (CTDE):
- **Independent actors** (decentralized): Each agent has its own policy mapping local observation → action. During execution, agents act on local info only.
- **Shared centralized critic** (training only): Single value network that receives the global state (all agents' observations concatenated). Used for GAE advantage estimation, then discarded at test time.
- All agents share the same actor and critic weights. Heterogeneity comes from different observations, not different parameters.

Key paper: Yu et al. (2022), "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (NeurIPS 2022).

Implementation notes from the paper:
- Value normalization (PopArt-style running mean/std) — "often helps, never hurts"
- Death masking: zero vector + agent ID for dead agents (analogous to our alive_mask)
- Conservative data reuse: 5-15 PPO epochs (our 4 is in range)

**Differences from our system**:

| Aspect | Our System | MAPPO |
|--------|-----------|-------|
| Critic input | Local observation only (502-dim) | Global state (all agents' obs concatenated) |
| Parameters | Per-agent evolved params | Fully shared params |
| Agent diversity | Mutation + evolution | Same policy, different observations |
| Communication | Indirect via shared field (stigmergy) | None (critic sees global state during training only) |
| Persistent memory | Field retains info across steps | None |

**Key weakness for our comparison**: MAPPO agents all have the same brain and CANNOT specialize. No field, no evolution, no persistent memory. If our system produces specialized species AND outperforms MAPPO, that's a strong result.

**Implementation path**: Modify our existing PPO — add a CentralizedCritic that takes all agents' observations as input, disable evolution, share all weights. ~200 lines of code. Easier than wrapping our env for JaxMARL.

#### QMIX — How It Works

Value-based method (not policy gradient) using value decomposition:
- Per-agent DRQNs output individual Q-values from local observations
- A mixing network combines them into a joint Q_tot
- Mixing network weights are non-negative (enforced by absolute value) — guarantees monotonicity
- Monotonicity means: argmax Q_tot = (argmax Q_1, argmax Q_2, ...) — agents can act independently at execution
- Hypernetworks generate mixing weights conditioned on global state

Key paper: Rashid et al. (2018), ICML.

**Strengths**: Sample efficient (off-policy, replay buffer), explicit credit assignment
**Weaknesses**: Monotonicity constraint fails on non-monotonic coordination, discrete actions only, struggles with sparse rewards, complex to implement

**Compared to MAPPO**:

| | MAPPO | QMIX |
|--|-------|------|
| Type | On-policy, policy gradient | Off-policy, value-based |
| Sample efficiency | Lower | Higher |
| Stability | More stable | Can diverge |
| Credit assignment | Implicit (centralized critic) | Explicit (value decomposition) |
| Sparse rewards | Strong | Weak |
| Scalability | Critic input grows with agents | Mixing network grows modestly |

**Implementation**: JaxMARL has QMIX at `baselines/QLearning/qmix_rnn.py`. 21,500x speedup over PyMARL on MPE.

#### Fixed Pheromone Rules — How to Implement

Standard ACO parameters from Dorigo & Stutzle (2004):

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Pheromone influence | alpha | 1.0 |
| Heuristic influence | beta | 2.0-5.0 |
| Evaporation rate | rho | 0.5 (AS) or 0.02-0.1 (MMAS) |
| Deposit constant | Q | 1.0 |

Hardcoded rules:
```
DEPOSIT: agent found food → deposit pheromone at location
         amount = Q / distance_traveled (or flat rate)
EVAPORATE: field *= (1 - rho) each step
DIFFUSE: use our existing diffusion
FOLLOW: p(move to cell j) = [pheromone_j^alpha * food_heuristic_j^beta] / sum
```

**Two variants for the paper**:
- **ACO-Fixed**: Fully hardcoded, no neural network at all. Deposit on food collection, follow pheromone gradient with softmax. The "classical swarm" baseline.
- **ACO-Hybrid**: Neural network for movement, but field write rules are hardcoded. Isolates the value of learning the field-writing behavior.

**Implementation**: Use our existing field infrastructure (same grid, diffusion, decay). Replace the learned write with hardcoded deposit rules. ~150 lines.

#### Also Add: IPPO (Simplest Baseline)

Independent PPO — our existing PPO with field disabled and shared params. The "no communication at all" lower bound. Costs zero implementation effort — just disable the field.

#### Full Comparison Table for the Paper

| Method | Type | Communication | Learns Field? | Credit Assignment | Key Strength | Key Weakness |
|--------|------|--------------|---------------|-------------------|-------------|-------------|
| **Ours** (Learnable Stigmergy) | PPO + Evolution | Indirect (shared field) | Yes | Evolutionary + field | Emergent specialization, persistent knowledge | Complex system |
| **MAPPO** | On-policy PG | None (centralized critic during training) | No | Centralized critic | Simple, robust, competitive | No specialization, no memory |
| **IPPO** | On-policy PG | None | No | None (fully independent) | Simplest baseline | No coordination |
| **QMIX** | Off-policy value-based | None (mixing network during training) | No | Value decomposition | Sample efficient | Discrete only, monotonicity constraint |
| **ACO-Fixed** | Hardcoded rules | Indirect (fixed pheromone) | No | None | No training needed | Cannot adapt |
| **ACO-Hybrid** | PPO + fixed pheromone | Indirect (fixed deposit) | Partially | PPO critic | Isolates learned writing value | Arbitrary rule design |

#### Implementation Priority

1. **IPPO** (easiest): Disable field, shared params. Zero new code.
2. **ACO-Fixed** (moderate): Hardcode deposit/follow using our field. ~150 lines.
3. **MAPPO** (moderate): Add centralized critic to our PPO. ~200 lines.
4. **QMIX** (hardest): Use JaxMARL or implement from scratch. ~500 lines.

#### JAX Libraries Available

| Library | What | Notes |
|---------|------|-------|
| JaxMARL | MAPPO, IPPO, QMIX, VDN, IQL, TransfQMIX, SHAQ, PQN-VDN | `pip install jaxmarl`. NeurIPS 2024. Up to 12,500x faster. |
| Mava (InstaDeep) | MAPPO with centralized critic, feedforward and recurrent | 10-100x faster than other frameworks |

---

### DR-4: Superlinear Scaling, Hidden Resources, Ablation Standards, and Paper Requirements

#### Superlinear Scaling — The Math

```
Efficiency(N) = F_total(N) / (N × F_solo)

Efficiency > 1.0 → superlinear (collective intelligence)
Efficiency = 1.0 → linear (no interaction effects)
Efficiency < 1.0 → sublinear (interference dominates)
```

The scaling exponent alpha:
```
F_total(N) ∝ N^alpha

alpha > 1 → superlinear
alpha = 1 → linear
alpha < 1 → sublinear
```

Measure by fitting power law to (N, F_total) data across population sizes.

**What causes superlinearity** (Hamann, 2018):
1. Shared information amplification — field amortizes exploration cost across population
2. Collaborative task enablement — some tasks require K agents
3. Cache/memory effects — the field IS the shared cache
4. Bucket brigade — scouts + exploiters create a pipeline where each role amplifies the other

**The killer chart**: X-axis = number of agents (1 to 32), Y-axis = per-agent food collection rate. Three curves: normal field, zeroed field, random field. If the "normal field" curve slopes UP, that's the superlinear signal.

**How to run it**:
1. Experiments at N = 1, 2, 4, 8, 16, 32 agents
2. Hold everything else constant (grid, food, episode length)
3. Plot F_total(N) vs N with reference line y = N × F_solo
4. Fit power law: log(F_total) = alpha × log(N) + c
5. Report alpha with confidence interval

#### Hidden Resources — Experimental Design

Based on Level-Based Foraging (LBF) benchmark (Papoudakis et al., NeurIPS 2021):

```
Hidden food config:
  K: 3                  # agents needed to reveal
  D: 3                  # Chebyshev distance radius
  reveal_duration: 10   # steps food stays revealed after trigger
  hidden_food_value: 5  # 5x more valuable than normal food
  num_hidden: 3         # number of hidden food patches
```

**Why K=3, D=3**:
- K=1 is trivial (solo can do it)
- K=2 can happen by accident (two agents randomly near each other)
- K=3 requires genuine coordination — unlikely by chance on 20x20 grid
- D=3 means 7x7 area — reachable but requires intentional proximity
- D=1 too hard (must be adjacent), D=5 too easy

**Avoiding trivial solutions**:
- Hidden food locations change periodically (no memorization)
- Normal food still exists (balance solo foraging vs cooperative discovery)
- Hidden food dramatically more valuable (5-10x) for selection pressure
- Energy cost for clustering (prevents degenerate "always cluster" strategies)

**Related recent work**:
- The Manitokan Task (2025) — agents must share a key; all tested MARL except MAPPO collapsed
- Threshold-Activated Cooperative Bandits (2025) — tasks require >= K agents to activate
- Composite Task Challenge for Cooperative MARL (Feb 2025) — division of labor is necessary, not just beneficial

#### Stigmergy Ablation — Gold Standard

**Six conditions, priority ordered**:

| Condition | What It Tests | Priority |
|-----------|--------------|----------|
| Normal field (learned read+write) | Full system | Required |
| Zeroed field (always zeros) | Is field encoding useful info? | Required |
| Random field (iid noise each step) | Does learned field beat noise? (Agents that learned to read will be misled) | Required |
| Frozen field (fixed at mid-training snapshot) | Is dynamic updating needed? | Important |
| No-field (input replaced with zeros, write disabled) | Cleanest "without stigmergy" | Important |
| Write-only (write but read zeros) | Reading vs writing value? | Nice-to-have |

The killer comparison: **Normal > Zeroed > Random**. We already have this. Phase 5 makes it rigorous.

#### Statistical Requirements for Publication

**Number of seeds**:
- 5 seeds: bare minimum, statistically fragile (Henderson et al. showed same algorithm can produce "significant" differences between groups of 5)
- **10 seeds: minimum for publication** (Agarwal et al. 2021 standard with IQM + bootstrap CI)
- **20 seeds: gold standard** for robust bootstrap confidence intervals
- Use paired seeds (same seed for baseline and variant) to reduce variance

**Training steps**: Run at 1M, 5M, 10M steps. Show ablation gap at each checkpoint to prove effect is stable or growing.

**Statistical tests (in order of preference)**:
1. Paired BCa bootstrap + permutation test — best for small samples, no distributional assumptions
2. Mann-Whitney U — appropriate when distributions are non-normal (RL rewards usually are)
3. Welch's t-test — acceptable if approximately normal, handles unequal variances
4. **Avoid**: plain Student's t-test (assumes equal variance)

**Reporting standards (Agarwal et al. 2021)**:
- Interquartile Mean (IQM) with 95% stratified bootstrap confidence intervals
- Performance profiles (CDF of normalized scores across seeds)
- Probability of improvement (how often method A beats method B)
- Use `rliable` library (Google) for standardized plotting

```python
from rliable import metrics, plot_utils

# IQM + bootstrap CI
iqm_scores, iqm_cis = metrics.aggregate_func(scores_dict, metrics.aggregate_iqm)

# Pairwise comparison
from scipy import stats
U, p = stats.mannwhitneyu(normal_scores, random_scores, alternative='greater')

# If paired seeds
T, p = stats.wilcoxon(normal_scores - random_scores, alternative='greater')

# Bootstrap CI on difference
deltas = normal_scores - random_scores
boot_means = [np.mean(np.random.choice(deltas, size=len(deltas), replace=True))
              for _ in range(10000)]
ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
```

#### How Top Papers Proved Emergence

**Baker et al. (2019) — OpenAI Hide-and-Seek**:
- Phase transition detection via behavioral statistics (object interactions, tool use frequency)
- Six distinct phases appeared as sharp discontinuities, consistent across seeds
- No explicit incentives for tool use — only reward was hide/seek success
- Transfer evaluation: agents tested on completely different "intelligence test" tasks and outperformed baselines
- Scale dependence: batch sizes >= 32k needed to reach later phases

**Riedl (2025) — Emergent Coordination in Multi-Agent LLMs**:
- Applied PID with time-delayed mutual information
- Found 32% of groups show significant emergence (p < 0.05)
- Fisher's combined probability test across all groups was highly significant
- K=2 quantile binning, Jeffreys smoothing, row/column surrogate tests

**What reviewers will reject**:
1. "Could be explained by simpler mechanisms" — need PID synergy or cooperative-only tasks to rule this out
2. Not enough seeds / no error bars
3. Missing ablations
4. Only tested on one environment — need 2-3 configurations
5. No comparison to communication baselines (CommNet, TarMAC)
6. Emergence is artifact of reward function — must show reward is purely individual
7. Non-stationarity not addressed — show results are stable, not transient

#### What a Strong NeurIPS/AAMAS Paper Needs

```
Minimum experiment matrix:

1. Scaling (N = 1, 2, 4, 8, 16, 32 agents)
   10+ seeds each, report per-agent efficiency + alpha exponent

2. Ablation (4+ conditions: normal, zeroed, frozen, random)
   10+ seeds each, IQM + 95% bootstrap CI + Mann-Whitney p-values

3. Specialization analysis
   Clustering + silhouette + species detection + heredity
   With vs without evolution

4. Phase transition documentation
   Behavioral stats over training (Baker et al. style)
   Consistent across seeds

5. Information-theoretic analysis
   PID synergy > 0 with statistical significance

6. Baselines (3-5 comparisons)
   IPPO, MAPPO, ACO-Fixed, QMIX (if time)

7. At least 2-3 environment configurations
   Vary grid size, food count, or agent count
```

#### Compute Requirements

```
Scaling experiment:     6 agent counts × 20 seeds × 10M steps = 1.2B steps
Ablation:               4 conditions × 20 seeds × 10M steps   = 800M steps
Baselines (MAPPO+ACO):  2 methods × 20 seeds × 10M steps      = 400M steps
Hidden resources:       3 conditions × 20 seeds × 10M steps    = 600M steps
────────────────────────────────────────────────────────────────────────────
Total:                                                          ~3B steps
```

This is why Phase 4B (Kaggle infrastructure) must come first.

---

### DR-5: Libraries We'll Need

| Library | What For | Install | Notes |
|---------|---------|---------|-------|
| `dit` | PID synergy computation (discrete) | `pip install dit` | Gold standard |
| `hoi` | O-information (JAX-native, continuous) | `pip install hoi` | Best fit for our stack |
| `einet` | Causal emergence / Effective Information | `pip install einet` | 9 tutorial notebooks |
| NIS+ | Neural coarse-graining for CE | github clone | PyTorch, interfaces via numpy |
| `rliable` | Statistical reporting (IQM, bootstrap CI, performance profiles) | `pip install rliable` | Google standard |
| `jaxmarl` | MAPPO/QMIX baseline implementations | `pip install jaxmarl` | NeurIPS 2024, 12,500x speedup |

---

### DR-6: The Novel Contribution (Why This Is Publishable)

Three gaps in the literature that our paper fills:

1. **No one has applied PID/synergy to stigmergic multi-agent RL.** Honda (2022) used simple grid-world foraging without RL. Riedl (2025) used LLMs. Rosas (2020) applied to Game of Life and flocking. Nobody has done learned stigmergy + evolution + PID.

2. **No one has measured causal emergence in a learned pheromone field.** Hoel's framework has been applied to Boolean networks, cellular automata, and flocking. Never to a system where the field itself is learned through agent-environment interaction.

3. **No one has combined information-theoretic emergence proof + evolutionary specialization + stigmergy ablation in one system.** Each piece exists separately. The combination is novel.

The paper's core claim: "We present the first system demonstrating mathematically verified emergent collective intelligence through learned stigmergy, with evidence from partial information decomposition, causal emergence analysis, superlinear scaling, and cooperative-only task completion."

---

## Key References (Expanded)

### PID and Synergy
- Williams & Beer (2010) — Original PID paper, I_min
- Riedl (2025) — Emergent Coordination in Multi-Agent Language Models (PID applied to multi-agent)
- Honda Research Institute (2022) — PID for cooperative multi-agent foraging
- Rosas, Mediano et al. (2020) — Reconciling Emergences (Psi, Delta, Gamma via PID)
- Mediano et al. (2022) — Greater than the Parts (review of PID → causal emergence)
- Kolchinsky (2022) — Novel PID approach based on Blackwell sufficiency
- Stevenson et al. (2025) — Information-theoretic analysis of social cohesion emergence

### Causal Emergence
- Hoel, Albantakis, Tononi (2013) — Quantifying Causal Emergence (PNAS)
- Hoel (2025) — Causal Emergence 2.0 (axiomatic grounding)
- Jansma & Hoel (2025) — Engineering Emergence (taxonomy)
- Zhang et al. (2025) — CE via SVD (coarse-graining-independent)
- Yang et al. (2024) — NIS+ (finding emergence in data)

### Baselines
- Yu et al. (2022) — MAPPO (NeurIPS)
- Rashid et al. (2018) — QMIX (ICML)
- Dorigo & Stutzle (2004) — Ant Colony Optimization (MIT Press)

### Proving Emergence
- Baker et al. (2019) — OpenAI Hide-and-Seek (emergent tool use)
- Agarwal et al. (2021) — Deep RL at the Edge of the Statistical Precipice (NeurIPS Outstanding Paper)
- Colas et al. (2018) — How Many Random Seeds?
- Hamann (2018) — Superlinear Scalability in Multi-robot Systems
- Papoudakis et al. (2021) — Level-Based Foraging benchmark (NeurIPS)

### Multi-Agent RL Evaluation
- Gorsane et al. (2022) — Standardised Evaluation Protocol for Cooperative MARL (NeurIPS)
- JaxMARL (2024) — NeurIPS Datasets and Benchmarks
- BenchMARL (2024) — Facebook Research
- rliable library — Google Research
