# ChatGPT Diagnosis Log - Emergence Lab Stigmergy Problem

**Date**: 2026-02-06
**Problem**: Field OFF consistently outperforms Field ON in multi-agent RL stigmergy system

## Context
- Project: emergence-lab (multi-agent RL with pheromone-based communication)
- Goal: Prove collective intelligence emerges from simple agents + shared field + evolution
- Result: Field hurts performance instead of helping

## Experiments Run
1. Field ON vs OFF (30 seeds, 10M steps) → Field OFF won (814 vs 636 reward, p=0.016)
2. Adaptive gate v1 (3 seeds, 2M steps, mutation=0.02) → Gates stuck at 0
3. Adaptive gate v2 (3 seeds, 2M steps, mutation=0.1) → Gates spread but population crashed, Field OFF still won

---

## Q&A Log

### Q1-3 (Architecture - Field Representation)
**Q1**: How is the pheromone field represented (grid resolution, channels, continuous vs discrete), and how do agents observe it?
**Q2**: Is the pheromone field part of the environment state only, or can agents directly write to it via actions?
**Q3**: Is the field dynamics (diffusion, evaporation, clipping) differentiable, learned, or hand-coded?

**A1-3**:
- Grid: (20, 20, 4) float32 — 20x20 spatial, 4 channels, continuous values in [0, 1.0]
- Channels: Ch0=recruitment (volatile), Ch1=territory (persistent), Ch2-3=reserved
- Observation: 5 values per channel (center + N/S/E/W) + temporal dC/dt = 24 dims total
- NO explicit write action — 5 actions (stay/up/down/left/right)
- Implicit writes: Ch0 by laden agents only, Ch1 by all agents (+0.01/step)
- Hand-coded dynamics, not learned. Diffusion 3x3 kernel, exponential decay, per-channel rates

### Q4-6 (Architecture - Agents)
**Q4**: Are agents homogeneous with shared policy weights, or heterogeneous?
**Q5**: Does the policy network have spatial memory (recurrence, attention, convolution)?
**Q6**: Is the pheromone field reset every episode?

**A4-6**:
- Shared policy weights (PPO), with per-agent evolved copies when evolution enabled
- Pure MLP (64→64), no recurrence, no attention, no convolution — field fed as flat features
- Field reset every episode, Ch1 seeded at nest with 1.0

### Q7-9 (Architecture - Interface)
**Q7**: How many agents active simultaneously?
**Q8**: Individual or shared reward?
**Q9**: Any cost for following pheromones vs exploration?

**A7-9**:
- 8 starting, grows to max 32 via reproduction
- Individual rewards (10% on pickup, 90% on nest delivery)
- No explicit cost for field usage. Write steps are energy-FREE. Laden agents travel at half speed (move/write alternation)

### Q10-12 (Training Dynamics)
**Q10**: PPO rollout horizon and batch size?
**Q11**: Field ON vs OFF trained separately or toggled at eval?
**Q12**: Same entropy bonus for both conditions?

**A10-12**:
- Rollout: 128 steps, 32 envs, 32 agents = 131K transitions/update
- Separate training runs, identical hyperparameters, paired seeds
- Entropy coef = 0.01, identical for both conditions

### Q13-15 (Learning Signal)
**Q13**: Advantages computed per-agent or population-level?
**Q14**: Value function sees field observations?
**Q15**: Policy collapse modes in Field ON?

**A13-15**:
- Per-agent independent GAE, minibatch normalization with alive mask
- Yes, value function sees same 45-dim obs including field
- Population crashed to 2.2 agents in adaptive gate v2; behavioral modes not catalogued

### Q16-18 (Signal Stability)
**Q16**: Field observation numerics?
**Q17**: When does Field ON diverge in learning curves?
**Q18**: Non-stationarity from population growth + Ch1?

**A16-18**:
- Field obs NOT normalized before network (fed raw [0,1])
- Learning curve comparison not done (gap)
- Yes, Ch1 accumulates as population grows 8→32, saturates in frequented areas

### Q19-21 (Incentives & Interference)
**Q19**: Do laden agents benefit from own trails?
**Q20**: Write observability timing?
**Q21**: Read-only ablation run?

**A19-21**:
- Mostly for others; trail gradient points toward NEST not food (design flaw)
- One-step delay for observability
- Read-only ablation NOT run (gap)

### Q22-24 (Critical - Food Visibility)
**Q22**: Direct food cues besides pheromones?
**Q23**: Food static or respawned?
**Q24**: Agents anti-follow pheromones?

**A22-24**:
- **YES — agents see K=5 nearest food with exact directions (15 dims)**
- Food respawns randomly after collection
- Policy introspection not done (gap)

### Q25-27 (Behavioral Evidence)
**Q25**: Path overlap/herding in Field ON?
**Q26**: Time-to-food compared?
**Q27**: Failed agents correlate with Ch1 saturation?

**A25-27**: All NOT measured — analysis gaps

### Q28-30 (Failure Modes)
**Q28**: Time near nest compared?
**Q29**: Exploration coverage compared?
**Q30**: Birth rates compared?

**A28-30**: Not explicitly measured; data exists in logs but not analyzed

### Q31-33 (Channel-Specific)
**Q31**: Ch1 influence early vs late?
**Q32**: Ch0-only or Ch1-only ablation?
**Q33**: Nest observable directly?

**A31-33**:
- Not analyzed
- Channel-specific ablations NOT run (gap)
- Yes, nest has direct compass (2 dims) — Ch1 seeding is redundant

### Q34-36 (Critical - Information Scarcity)
**Q34**: Tried removing direct food observations?
**Q35**: Misleading trails from food respawn?
**Q36**: Entropy collapsing faster in Field ON?

**A34-36**:
- **NOT tried — critical missing experiment**
- Yes, trails point to depleted/moved food (design flaw)
- Not compared (gap)

### Q37-39 (Dynamics Variants)
**Q37**: dC/dt usage by policy?
**Q38**: No-diffusion or no-decay variants tested?
**Q39**: Move+write same step tested?

**A37-39**: All NOT tested — gaps

### Q40-50 (Final Batch)
**Q40-42**: Gradient noise, value error, policy variance across seeds? — Not compared
**Q43**: Field from step 0? — Yes, always
**Q44**: Overfitting to early field stats? — Not measured
**Q45-46**: Cross-condition eval (train ON/eval OFF, train OFF/eval ON)? — Not tested
**Q47**: Field resolution/channel variations? — Not tested
**Q48**: MAPPO with field? — Implementation exists, not tested with field
**Q49**: Any seeds where Field ON wins? — Aggregate universally worse, individual seeds unchecked
**Q50**: Is stigmergy desired? — **YES, it's the entire project hypothesis**

---

## ChatGPT Diagnosis

### Executive Summary
**Field ON hurts because the environment systematically teaches agents that the pheromone field is misleading, redundant, and non-stationary. The agents are not "failing to learn stigmergy" — they are correctly learning to ignore or distrust it. This is not a tuning issue. It is a task–signal mismatch.**

### Root Causes (ordered by severity)

#### 1. Stigmergy is strictly redundant
- Agents have exact food locations (K-nearest, global, noiseless)
- Direct nest compass
- No occlusion, no uncertainty
- **Pheromones solve zero unmet information needs**
- Optimal policy is reactive greedy navigation

#### 2. Recruitment trails point in the WRONG direction
- Ch0 laid while returning to nest → gradient points toward nest, not food
- Food respawns randomly → trails point to dead locations
- **Agents trained by experience that pheromones lie**

#### 3. Ch1 territory becomes saturated noise
- +0.01 per agent per step, ~0 decay, population grows 4×
- Early: sparse, meaningful; Late: saturated everywhere
- No normalization, no recurrence, no distribution shift handling
- **Destabilizes both actor and critic**

#### 4. No incentive alignment for writing
- Writing never benefits the writer
- Benefits are delayed, indirect, usually invalid
- Writing has implicit cost (half-speed return)
- **Optimal gradient: suppress field reliance**

#### 5. Observation design destroys spatial meaning
- 5-point sampling flattened into MLP
- No memory, no convolution, no attention
- **Agent cannot represent trail-following behavior**

#### 6. Temporal derivative is noise
- Given diffusion, respawn, multi-agent writes, one-step delay
- **dC/dt ≈ stochastic jitter, not causal signal**

### Why Field OFF Wins
- Lower observation dimensionality
- Stable input distribution
- Direct cues only
- Cleaner value function learning
- **Removed distraction, not coordination**

### Recommended Fixes

#### Phase 1: Make stigmergy NECESSARY (non-negotiable)
1. Remove direct food location observations (no K-nearest food, or severely range-limit)
2. Keep nest compass (or weaken later)

#### Phase 2: Fix pheromone semantics
3. Reverse recruitment trail direction (lay from food outward, or vector pheromones)
4. Make food sources persistent (static locations, no random respawn)

#### Phase 3: Remove pathological dynamics
5. Disable or cap Ch1 saturation
6. Normalize field inputs (per-channel running mean/std)

#### Phase 4: Align incentives
7. Give writers self-benefit (small reward for following own trail)
8. Remove half-speed penalty (allow move+write same step)

#### Phase 5: Make policy capable of stigmergy
9. Add memory (GRU/LSTM)
10. Add spatial inductive bias (conv or polar encoding)

#### Phase 6: Immediate diagnostics
- Field ON → Field OFF at eval (test if field ignored)
- Ch0-only vs Ch1-only (Ch1 will be harmful)
- Remove food obs + Field ON (gap should reverse)
- Visualize trajectories + trails

### Final Verdict
**The system is correctly learning that stigmergy is redundant, misleading, non-stationary, poorly incentivized, and architecturally unsupported. Once information scarcity is forced and trail semantics fixed, stigmergy will emerge naturally.**

---

## Minimal Environment Redesign (from ChatGPT)

### Smallest Change Set (validate diagnosis fast)
1. **REMOVE** direct food localization (K-nearest food vectors)
2. **KEEP** only binary local food detection (on current cell)
3. **KEEP** nest compass unchanged
4. **DISABLE** Ch1 entirely (no writes, zero obs)
5. **FIX** recruitment trail direction (see below)

### Recruitment Trail Mechanics (corrected)
- Trail laid when **LEAVING FOOD**, not while returning
- On food pickup: `agent.last_food_pos = agent.pos`
- While returning: `pheromone_strength = exp(-alpha * dist_to_last_food)` where alpha ~ 0.3
- **Highest intensity at food**, monotonic decay toward nest
- **Gradient ascent = move toward food**
- Move + write in same step (no alternation penalty)

### Food Source Design
- 3 food **patches** (not points), radius=2
- Large/infinite food per patch
- **NO random respawn** — patches persist entire episode
- Trails stay valid, following pheromones pays off

### New Observation Layout (11 dims total)
- Ch0 spatial: 5 dims (center + N/S/E/W)
- Nest compass: 2 dims
- Position: 2 dims
- Energy: 1 dim
- Local food present: 1 dim (binary)

### Field Dynamics
- diffusion_rate = 0.2 (lower)
- decay_rate = 0.01 (slower)
- No temporal derivative, no Ch1, no normalization yet

---

## Adaptive Gate Analysis (from ChatGPT)

### Results Summary
| Condition | Reward | Variance | Population |
|-----------|--------|----------|------------|
| field_on | 1128.0 | 520.7 | 4.7 |
| field_off | 1242.7 | 379.5 | 6.1 |
| adaptive | 1088.7 | 157.6 | **2.2** |

Gate biases: All channels ~0 (neutral), low variance, no bimodal specialization

### ChatGPT Interpretation

**Core insight**: Adaptive gate became a "conflict dampener," not a selector — reducing variance but destroying coordination pressure, collapsing population viability.

#### Why gates stayed neutral (~0)?
- Gate ≈ 0 minimizes gradient damage without destabilizing policy
- **Risk-dominant equilibrium**, not reward-maximizing
- No fitness gradient pushing to extremes because field has no positive causal effect

#### Why population crashed in adaptive?
- Gates decoupled learning from survival
- Agents reduce field exposure individually → collective coordination collapses
- Less food throughput → fewer births → population spiral
- **Tragedy-of-the-commons dynamic**

#### Why lowest reward variance?
- Gate acts as learned noise filter
- Field influence ≈ 0 → policy sees stable input
- Learning smoother but predictably bad
- **Low variance ≠ good learning** — safe but suboptimal basin

#### Is field harmful?
- **Yes** — evolution figured this out faster than PPO
- Adaptive gating = biological ablation study
- Universal damping = no reliable causal advantage from field

#### Are adaptive gates viable?
- **Not in this environment**
- Requires: some agents benefit from field, others from ignoring, both coexist stably
- Current environment violates all three assumptions
- Correct diagnosis by the system, not a gating failure

### Recommendation
Do NOT use adaptive gates until stigmergy is necessary. After environment fixes, expect:
- Bimodal gate distributions
- Field specialists vs explorers
- Higher population + reward

---

## Data Analysis (62GB Archive) - 2026-02-06

### Experiment Results Summary

#### 1. Field ON vs OFF (Original, 30 seeds each, 10M steps)
- **Field OFF**: 5.52 ± 0.19 reward, 19/30 seeds at max population (64)
- **Field ON**: 4.55 ± 1.14 reward, 8 seeds collapsed to 0
- **Statistics**: p < 0.0001, Cohen's d = 1.18 (large effect)
- **Winner**: Field OFF by 21%

#### 2. Field ON vs OFF v2 (Biological Pheromone, 30 seeds)
- **Field ON**: 585.15 ± 57.28 reward
- **Field OFF**: 560.64 ± 89.63 reward
- **Statistics**: p = 0.231 (NOT significant)
- **Trail strength validated**: p < 0.001, d = 2.57
- **Result**: Marginal Field ON advantage, not statistically significant

#### 3. Hidden Food Coordination (30 seeds each)
- **Hidden food revealed**: Field ON 3.63 vs Field OFF 1.97 (+84%)
- **Coordination works**: Field ON enables multi-agent clustering
- **BUT total reward**: Field OFF 39,080 vs Field ON 37,350 (+4.6% for OFF)
- **Conclusion**: Coordination works but costs more than it's worth

#### 4. Pheromone Sweep (24 configs, 3 seeds each, short runs)
- **ALL Field ON configs beat Field OFF**
- **Best config (nest_radius=4)**: 496.7 reward
- **Field OFF baseline**: 263.3 reward
- **Confirmation (best combined)**: 986.7 reward (+274.7% vs OFF)
- **Critical finding**: Short runs show Field ON winning!

#### 5. Config Sweep (Specialization Parameters)
- **Field OFF baseline**: 18,700 reward
- **Field ON baseline**: 4,400 reward (-76%)
- **Freeze-Evolve (95:5)**: 23,300 reward (BEST overall)
- **Key insight**: Evolution-dominated training dramatically outperforms pure gradient

#### 6. Adaptive Gate v2 (High Mutation, 3 seeds)
- **Field OFF**: 1242.67 ± 309.84, population 6.07
- **Field ON**: 1128.00 ± 425.17, population 4.67
- **Adaptive**: 1088.67 ± 128.69, population 2.20 (crashed)
- **Gate biases**: All near zero, no bimodal specialization
- **Conclusion**: Gates became "conflict dampener" not selector

### Key Discrepancy Discovered
- **Short runs (sweep)**: Field ON wins (+88-274%)
- **Long runs (10M steps)**: Field OFF wins (+21%)
- **Hypothesis**: "Coordination collapse" over extended training?

### Critical Insight: Freeze-Evolve Training
Freeze-Evolve (95% gradient freeze, 5% evolution) achieves 23,300 reward — dramatically beats:
- Pure gradient Field ON: 4,400
- Pure gradient Field OFF: 18,700
- Suggests: Evolution is critical, but gradient training kills field usage

### Data Locations
- Main archive: /Users/ashish/Downloads/emergence-labs-data (62GB)
- Field ON vs OFF: emergence-lab 2/analysis_results/field_on_vs_off_results.json
- Hidden food: emergence-lab 2/hidden_food_analysis_results/hidden_food_results.json
- Pheromone sweep: emergence-lab 2/pheromone_sweep/sweep_results.pkl
- Config sweep: emergence-lab 2/config_sweep/sweep_summary.json
- Adaptive gate: emergence-lab 2/adaptive_gate_v2_highmut/all_results.pkl

### Open Questions for ChatGPT
1. Why does Field ON win short runs but lose long runs?
2. Why does freeze-evolve dramatically outperform pure gradient?
3. What's minimum change to make coordination economically viable?

---

## ChatGPT Response: Coordination Collapse Theory (2026-02-06)

### 1. Why Field ON Wins Short Runs But Loses Long Runs

**Short runs measure early learning phase:**
- Sparse policies, high exploration, low population, unspecialized roles
- Pheromones = high signal / low congestion
- Any coordination is net win → Field ON > OFF

**Long training introduces 4 killers:**
1. **Population saturates** → trails congest → over-coordination
2. **Gradient pressure favors short-horizon** → PPO optimizes immediate return, stigmergy payoff is delayed/externalized
3. **Policy homogenization** → shared policy + individual rewards → everyone learns same compromise, specialists wiped out
4. **Catastrophic coordination overshoot** → too many agents exploit same patch → marginal returns negative → starvation spiral

**Core insight**: Field ON collapses because coordination is not self-limiting, and PPO cannot represent "stop coordinating now."

### 2. Why Freeze-Evolve Wins (Critical Insight)

**What gradient descent does wrong:**
- PPO rewards individuals, updates shared weights, penalizes variance, favors smooth myopic policies
- Averages away specialization
- Suppresses behaviors with delayed, indirect, other-dependent payoffs
- **Stigmergy has all three** → gradient learning slowly poisons pheromone channel

**What evolution does right:**
- Selects on lifetime return
- Preserves diversity
- Allows role differentiation
- Doesn't require differentiability

**Why freeze-evolve works:**
- 95% frozen → gradients can't erase coordination
- 5% evolving → exploration + adaptation continue
- Field usage becomes structural, not fragile gradient artifact

**The 23,300 result is not a fluke. It means:**
> "Stigmergy is an evolutionary solution that gradient descent actively destabilizes."

### 3. Minimum Change to Make Stigmergy Economically Viable

**The fix**: Add diminishing returns to solo exploitation.

**Option A: Patch throughput scaling (RECOMMENDED)**
- Food patch yields ∝ √(number of agents exploiting it)
- Solo agent: low yield
- Group: superlinear early, then saturates
- Makes coordination economically dominant

**Option B: Exploration tax**
- Each step without pheromone alignment costs ε energy
- Following trails cheaper than wandering
- Penalizes non-coordination without explicit reward shaping

**Option C: Trail capacity limit**
- Each patch has max throughput
- Without coordination, agents overload and waste effort
- Makes "too many followers" self-correcting

### Unifying Insight

We have demonstrated all three regimes:

| Regime | Evidence |
|--------|----------|
| Mechanism works | Short sweeps, hidden food (+84%) |
| Economics unfavorable | Long PPO runs (Field OFF wins) |
| Evolution rescues it | Freeze-evolve (23,300 vs 4,400) |

**Conclusion**: Stigmergy is a slow, evolutionary coordination strategy that PPO treats as noise unless the environment forces it to dominate economically.

### Falsifiable Prediction (Minimal Fix)

If we:
- Keep freeze-evolve
- Add patch throughput scaling (Option A)
- Do nothing else

Then:
- Long runs will no longer collapse
- Field ON > OFF permanently
- Population will stabilize near cap
- PPO-only will still underperform evolution

---

## ChatGPT Response: Implementation Details (2026-02-06)

### 1. Mechanics: Patch Throughput Scaling Formula

**Use per-agent reward scaling (Option A):**

```python
n = count_agents_within_radius(food_pos, PATCH_RADIUS)
reward_i = E * (sqrt(n) / n)  # per-agent
# Total colony reward at patch = E * sqrt(n)
```

**Why this works:**
- Solo agent (n=1): gets full reward (√1/1 = 1)
- Small group: net gain > solo
- Large group: diminishing returns
- No new resources, no new state

### 2. Patch Definition

**Use radius-based counting (Option C):**
```python
PATCH_RADIUS = 2  # Manhattan or Chebyshev
n = count_agents_within_radius(food_pos, PATCH_RADIUS)
```

- Turns point food into implicit patch
- Preserves existing respawn logic
- Makes coordination spatial, not symbolic

### 3. Edge Cases

**Single agent (n=1):** No penalty, reward = E (avoids punishing exploration)

**Large groups:** Cap the benefit to prevent runaway herding:
```python
N_CAP = 6  # or 8
n_eff = min(n, N_CAP)
reward_i = E * sqrt(n_eff) / n_eff
```

This cap is crucial — makes over-coordination self-limiting.

### 4. Ablation Design (2×2×2 Factorial)

| Factor | Levels |
|--------|--------|
| Field | ON / OFF |
| Patch scaling | ON / OFF |
| Learning | PPO / Freeze-Evolve |

**Key interaction to look for:** Field ON × Patch Scaling ON × Freeze-Evolve (predicted winner)

### 5. Metrics for Paper

**Core metrics (must include):**
1. Population over time (mean, variance, time-to-collapse, survival probability)
2. Total reward (learning curves, early vs late divergence)
3. Coordination index: `CI = mean over t [ max_patch_agents(t) ]`

**Critical diagnostics:**
4. Marginal coordination return: `Δ reward / Δ agents at patch` (should flip sign without scaling)
5. Specialization: gate histogram bimodality, KL divergence between policies
6. Collapse signature: sudden population drop + rising coordination (smoking gun)

**Optional but strong:**
7. Counterfactual eval: disable field at eval, reward should drop sharply only in Patch-Scaling-ON

### 6. Sample Size

- **20-30 seeds per condition** (we can afford 30)
- **8 conditions × 30 seeds = 240 runs at 10M steps**
- Sufficient to detect main effects, two-way interactions, and critical three-way interaction

### Locked-In Falsifiable Predictions (Post Patch Scaling)

1. Field ON > OFF permanently (long runs)
2. No more Field ON collapses
3. Freeze-Evolve dominates PPO
4. Coordination stabilizes instead of runaway
5. Gate distributions become bimodal

**If any fail, coordination-collapse theory is wrong.**

---

## ChatGPT Response: Pass/Fail Criteria & Diagnostics (2026-02-06)

### Binary Pass/Fail Checklist

**P1. Long-horizon dominance (CRITICAL)**
- Metric: Mean episodic reward over last 20% of training
- Threshold: `MeanReward(Field ON + PatchScaling) ≥ 1.15 × MeanReward(Field OFF)`, p < 0.05

**P2. Collapse elimination (CRITICAL)**
- Metric: Population trajectory
- Threshold: ≤ 5% of Field ON seeds collapse to ≤ 2 agents, mean population ≥ 0.75 × cap

**P3. Marginal coordination return sign flip (SMOKING GUN)**
- Metric: ΔReward / ΔAgentsAtPatch
- Threshold: Without scaling = negative for n > 4; With scaling = ≥ 0 for all n ≤ N_CAP

**P4. Coordination without over-concentration (IMPORTANT)**
- Metrics: Max agents per patch, spatial entropy
- Threshold: Mean(max_agents_per_patch) ∈ [3, N_CAP], entropy plateaus (not → 0)

**P5. Learning-rule separation (IMPORTANT)**
- Metric: Reward comparison
- Threshold: Freeze-Evolve ≥ PPO, AND PPO no longer catastrophically collapses

**Confirmation Rule:** Must pass P1, P2, P3 + at least one of {P4, P5}

### Failure Diagnostic Tree

**Case A: Field ON still loses after patch scaling**
1. Check marginal return curve → if still negative, increase N_CAP (6→8) or reward E
2. Check patch occupancy → if agents don't cluster, reduce decay or increase diffusion
3. Check solo efficiency → if solo outperforms, add coordination tax

**Case B: Population still collapses (but early Field ON > OFF)**
1. Check max agents per patch → if > N_CAP regularly, cap not enforced
2. Check patch yield per agent → if decreasing sharply, scaling bug
3. Check trail persistence → if never decay, herding lock-in → increase decay

**Case C: Freeze-evolve no longer dominates PPO**
- This is GOOD news - means gradients now aligned with economics
- Check PPO entropy decay and policy variance

### Implementation Sanity Checks

**Patch scaling unit test:**
- 1 agent at food → reward = E
- 2 agents → each gets ~0.71E
- 4 agents → each gets ~0.5E
- 6 agents → each gets ~0.41E

**Incentive unit test:**
- Two patches, equal distance, one with pheromone trail
- With scaling: agent prefers pheromone patch
- Without scaling: no consistent preference

### Paper Framing

**One-sentence contribution:**
"We show that stigmergic coordination can emerge and function reliably in multi-agent reinforcement learning only when the environment enforces positive marginal returns to coordination; otherwise, gradient-based learning induces coordination collapse despite functional communication."

**Literature connections:**
- Social dilemmas: Hardin (1968), Ostrom (1990)
- Mechanism design in MARL: Perolat et al., Leibo et al.
- Gradient pathologies: Mahajan et al., Lowe et al., Bengio et al.

---

## ChatGPT Response: Engine/Platform Framing (2026-02-06)

### Core Primitive

"Distributed agents optimize global outcomes when coordination is made economically dominant via a shared writable medium."

Investor version: "We can make coordination emerge on demand—without central control—by shaping incentives, not intelligence."

### What Makes This an Engine

An engine is a reusable causal mechanism with control knobs:
- **Incentive structure** (patch scaling, costs)
- **Medium dynamics** (pheromone diffusion/decay)
- **Agent mutation/freeze ratio**
- **Information visibility**

The engine: "A controllable emergence layer that turns local rules + incentives into global coordination."

### "Prompting" and "Fine-tuning" in This Context

**Prompting ≈ Environment/incentive configuration**
- "Optimize throughput" → higher patch scaling
- "Optimize resilience" → stronger decay, higher exploration tax
- "Optimize fairness" → stricter N_CAP

**Fine-tuning ≈ Adapting emergence to a domain**
- Same engine, different reward topology, different medium semantics
- Logistics pheromone ≠ robot pheromone ≠ compute pheromone

### Target Applications

High-value domains where centralized control is brittle/expensive/slow:
- Logistics & supply chains (dynamic routing, load balancing)
- Swarm robotics / drones (search & rescue, exploration)
- Distributed compute / scheduling (task placement, fault tolerance)
- Traffic & mobility (adaptive rerouting)

### Defensibility

1. Understanding the failure mode (coordination collapse under gradients)
2. A design framework (can predict when emergence will fail/succeed)
3. System-level IP (incentive shaping + medium dynamics + learning interplay)

"Big labs can copy code. They can't easily copy understanding plus a working engine."

---

## ChatGPT Response: Freeze-Evolve & Evolution Mechanics (2026-02-06)

### Freeze-Evolve Mechanics

**During 95% frozen periods:**
- Actor parameters: frozen (no gradient application)
- Critic: still trained
- Gradients computed but actor updates skipped
- Coordination behaviors not overwritten

**Why 95:5 works:**
- Gradient poisoning timescale: fast (10³–10⁴ steps)
- Evolutionary discovery timescale: slow (10⁵–10⁶ steps)
- 95% freeze protects coordination gains before they compound
- 90:10 often fails because 10% gradient exposure suppresses weak coordination signals

**During 5% evolve phase:**
- Steady-state evolution (not generational GA)
- Agents accumulate lifetime fitness
- Low-fitness agents die, high-fitness reproduce with mutation
- No explicit crossover
- Selection is asynchronous

### Mutation Dynamics

**Optimal mutation_std:** 0.01–0.03
- < 0.005 → stagnation
- > 0.05 → coordination breaks
- 0.1 (our test) was too high

**Mutation should be smaller than coordination basin, larger than gradient noise.**

### Adaptive Gate Dynamics

**Why gates converged to ~0:**
- Field signal was net-negative
- Partial suppression minimized variance
- Evolution correctly diagnosed bad signal
- This was risk-minimizing attenuation, not failure

**With patch scaling, expect:**
- Gate distribution becomes bimodal
- One peak near 0 (explorers), one peak near 1 (exploiters)
- Variance increases, population stabilizes

**Gates should be evolved, not gradient-learned** (same credit assignment problem)

### Specialization Emergence

- No niche bonus required
- Mechanism: random variation → fitness advantage in context → frequency-dependent fitness → coexistence
- Population size ~32 is sufficient
- NEAT helps but not required

### Field Dynamics Assessment

Current values (Ch0 diffusion=0.5, decay=0.05) are plausible, not bottleneck.

**Symptoms to watch:**
- Agents overshoot → decay too fast
- Herding never disperses → decay too slow
- Trails blur everywhere → diffusion too high
- Trails too brittle → diffusion too low

**Rule:** diffusion_radius ≈ PATCH_RADIUS, decay_half_life ≈ traverse time

---

## ChatGPT Response: Edge Cases & Blind Spots (2026-02-06)

### Edge Cases Status

| Issue | Status | Notes |
|-------|--------|-------|
| Food respawn vs patch | OK | Rewards coordination that happened |
| Timing mismatch (trail vs n) | OK | Healthy noise, mirrors real stigmergy |
| Population oscillations | Monitor | Ecological dynamic, don't preemptively dampen |
| Free riders | OK | Following is valid strategy |
| food_source_pos overlaps nest | **FIX** | Food must not spawn within nest_radius + PATCH_RADIUS |
| Agent dies while carrying | **FIX** | Clear food_source_pos on death |
| Pick up but never deliver | OK | Correct behavior |
| Minimum population | OK | Emergence is the point |
| Mono-strategy trap | Watch | √n and N_CAP prevent, watch entropy |

### Must-Fix Before Running

1. Prevent food-nest overlap for patch counting
2. Clear food_source_pos on death
3. Audit hidden solo incentives

### Codebase Audit Items

Check for:
- Implicit solo rewards (per-step reward unrelated to delivery)
- Movement costs that scale with crowding
- Normalization bugs washing out high-n deliveries
- Critic bootstrap leakage

---

## MASTER REFERENCE: Pre-Run Audit Checklist

### A. Patch Scaling Mechanics
- [ ] Delivery reward scaled by `sqrt(n_eff) / n_eff`
- [ ] Scaling applied only at delivery, not pickup
- [ ] `n_eff = min(n, N_CAP)`
- [ ] N_CAP = 6
- [ ] PATCH_RADIUS = 2 (Manhattan or Chebyshev)
- [ ] Scaling code unit-tested with n = 1, 2, 4, 6

### B. Patch Identity & Counting
- [ ] food_source_pos stored on pickup
- [ ] food_source_pos cleared on delivery
- [ ] food_source_pos cleared on agent death
- [ ] Patch counting uses agent_alive mask
- [ ] Patch counting excludes nest area

### C. Environment Guardrails
- [ ] Food never spawns within nest_radius + PATCH_RADIUS
- [ ] No hidden solo rewards (energy regen, exploration bonus, etc.)
- [ ] No collision penalties that scale with crowding
- [ ] Birth/death logic unchanged

### D. Field Configuration
- [ ] Ch0 ON (recruitment)
- [ ] Ch1 OFF (territory disabled)
- [ ] Diffusion = 0.5
- [ ] Decay = 0.05

### E. Learning Setup
- [ ] Freeze-evolve enabled (95:5)
- [ ] Actor frozen during freeze periods
- [ ] Critic still trains during freeze
- [ ] Critic reinitialized after adding patch scaling

### F. Logging (Must-Have)
- [ ] Population over time
- [ ] Max agents per patch over time
- [ ] Mean reward (rolling)
- [ ] Value estimates, returns, advantages
- [ ] Field intensity at delivery states
- [ ] Marginal reward vs agents-at-patch

---

## MASTER REFERENCE: Complete Diagnosis Summary

### 1. THE PROBLEM (Observed Failures)

**Empirical observations:**
- Field OFF consistently outperformed Field ON in long PPO runs (5.52 vs 4.55, p<0.0001)
- Adaptive gating collapsed population to 2.2 and suppressed field usage (gates → 0)
- Short runs & hidden food showed coordination works (+84% food revealed)
- Long runs showed coordination collapse (8 seeds collapsed to 0)
- Freeze-evolve (95:5) achieved 23,300 vs Field ON 4,400 vs Field OFF 18,700

**Paradox:** Coordination exists and works, yet learning suppresses it over time.

### 2. ROOT CAUSE (Full Mechanism)

**Core failure:** Gradient learning suppresses stigmergy because coordination payoff is externalized and delayed.

**Mechanism chain:**
1. Agent A lays trail → Agent B benefits
2. Agent A receives no immediate reward
3. Critic underestimates value of field-rich states
4. Advantages for field-dependent actions become ≤ 0
5. PPO gradients suppress field sensitivity
6. Policy homogenizes toward field-ignorant behavior
7. Coordination collapses as population grows

**Why critic fails first:**
- Value depends on other agents' future actions
- Local observation is non-Markovian
- TD targets are irreducibly noisy
- Bias propagates into actor updates

**Why evolution survives:**
- Selection on lifetime return
- No need for marginal credit
- Preserves structural behaviors
- Allows specialization

### 3. THE FIX (Patch Throughput Scaling)

**Economic principle:** Make coordination locally rational by aligning private and social return.

**Exact formula at delivery:**
```python
n = count_agents_within_radius(food_source_pos, PATCH_RADIUS)
n_eff = min(n, N_CAP)
reward_i = base_reward * sqrt(n_eff) / n_eff
```

**Parameters (locked):**
- N_CAP = 6
- PATCH_RADIUS = 2
- Apply at delivery only
- Pickup reward unchanged

**Why this works:**
- Converts externalized payoff → individual payoff
- Restores critic accuracy
- Aligns gradients with coordination

### 4. THE EXPERIMENT (2×2×2 Factorial)

| Factor | Levels |
|--------|--------|
| Field | ON / OFF |
| Patch scaling | ON / OFF |
| Learning | PPO / Freeze-Evolve |

**Sample size:** 20-30 seeds per condition, 10M steps per run

### 5. PASS/FAIL CRITERIA (Binary)

**Mandatory (ALL must pass):**

**P1. Long-run dominance**
- `MeanReward(Field ON + Scaling) ≥ 1.15 × MeanReward(Field OFF)`
- p < 0.05

**P2. Collapse elimination**
- ≤ 5% seeds collapse to ≤ 2 agents
- Mean population ≥ 0.75 × cap

**P3. Marginal return sign flip (SMOKING GUN)**
- Without scaling: negative beyond n≈4
- With scaling: ≥ 0 for n ≤ N_CAP

**Supportive (≥1 must pass):**
- P4: Bounded coordination (no herding runaway)
- P5: Freeze-evolve ≥ PPO

### 6. PREDICTIONS (If Theory Correct)

1. Field ON > OFF permanently in long runs
2. No more Field ON population collapses
3. Freeze-Evolve dominates PPO
4. Coordination stabilizes instead of runaway
5. Gate distributions become bimodal (if reintroduced)
6. Marginal coordination return flattens near zero
7. Stable but non-zero spatial entropy

### 7. FAILURE DIAGNOSTICS

**If Field ON still loses:**
1. Check marginal return curve → if still negative, increase N_CAP or reward
2. Check patch occupancy → if no clustering, trail too weak
3. Check solo efficiency → if too high, reduce base reward

**If population still collapses:**
1. Check max agents per patch → cap not enforced?
2. Check patch yield per agent → scaling bug?
3. Increase decay slightly

**If PPO still fails but evolution succeeds:**
- Confirms gradient homogenization
- Engine still valid

### 8. ENGINE IMPLICATIONS

**What is proven:**
- Emergent coordination is economically contingent
- Gradient collapse is predictable and preventable
- Evolution + incentives form a controllable engine

**What the engine is:**
"A coordination layer that turns local rules into global order — without central control."

**Core primitive:**
"Distributed agents optimize global outcomes when coordination is made economically dominant via a shared writable medium."

**Control knobs:**
- Incentive structure (patch scaling, costs)
- Medium dynamics (diffusion/decay)
- Agent mutation/freeze ratio
- Information visibility

**Defensibility:**
- Understanding the failure mode (most teams never diagnose correctly)
- Design framework (can predict when emergence fails/succeeds)
- System-level IP (incentive + dynamics + learning interplay)

---

## Key Insight

**"Use gradients to learn skills. Use evolution to preserve structure. Use economics to align them."**

This is not a hyperparameter fix. It is an economic alignment that turns emergence from a fragile curiosity into an engineered capability.

---

## Codex Codebase Analysis (2026-02-06)

### Implementation Map from Codex

Codex (with codebase access) identified exact locations for all implementation changes:

| Component | File | Location | Description |
|-----------|------|----------|-------------|
| Delivery reward | `env.py` | ~405-432 | Modify `delivery_reward` calculation to apply √n scaling |
| Food pickup | `env.py` | ~238-241 | Store `food_source_pos` when `pickup_mask` is True |
| Agent counting pattern | `env.py` | ~300-314 | Adapt hidden-food reveal pattern for patch counting |
| EnvState field | `state.py` | ~42 | Add `food_source_pos: jnp.ndarray \| None = None` |
| Food respawn | `env.py` | ~259 | Add nest distance check to `new_food_positions` |
| Food init | `env.py` reset() | ~62 | Add nest distance check to initial food positions |
| in_nest pattern | `env.py` | ~396-403 | Use same Chebyshev pattern for "in patch" check |
| Base reward value | `configs.py` | ~248 | `config.evolution.food_energy` = 50 |
| food_energy_val | `env.py` | ~229 | `jnp.float32(config.evolution.food_energy)` |

### Key Code Snippets from Codex

**1. Delivery reward calculation (env.py:405-432):**
```python
delivering = has_food & in_nest & state.agent_alive
delivery_reward = jnp.where(
    delivering, food_energy_val * config.nest.delivery_reward_fraction, 0.0
)
rewards = jnp.where(
    state.agent_alive, pickup_reward + delivery_reward + hidden_food_energy_gained, 0.0
)
```

**2. Food pickup - agent→food mapping (env.py:224-241):**
```python
agent_food_mask = (
    jax.nn.one_hot(closest_agent, max_agents) * newly_collected[:, None]
)
# To get food position per agent:
agent_food = agent_food_mask.T  # (max_agents, F)
picked_food_idx = jnp.argmax(agent_food, axis=1)  # (max_agents,)
picked_food_pos = state.food_positions[picked_food_idx]  # (max_agents, 2)
# Guard with pickup_mask since argmax returns 0 even when no food picked
```

**3. Agent counting pattern (env.py:300-314):**
```python
hf_row_dist = jnp.abs(agent_rows_hf - hf_rows[None, :])
hf_col_dist = jnp.abs(agent_cols_hf - hf_cols[None, :])
hf_chebyshev = jnp.maximum(hf_row_dist, hf_col_dist)
within_reveal = (hf_chebyshev <= reveal_distance) & state.agent_alive[:, None]
agents_near_hf = jnp.sum(within_reveal.astype(jnp.int32), axis=0)
```

**4. in_nest computation (env.py:396-403):**
```python
nest_center_r = config.env.grid_size // 2
nest_center_c = config.env.grid_size // 2
nest_r = config.nest.radius
agent_nest_dr = jnp.abs(new_positions[:, 0] - nest_center_r)
agent_nest_dc = jnp.abs(new_positions[:, 1] - nest_center_c)
in_nest = (agent_nest_dr <= nest_r) & (agent_nest_dc <= nest_r)
```

**5. Current EnvState fields (state.py:42):**
```python
agent_positions: jnp.ndarray   # (max_agents, 2)
food_positions: jnp.ndarray    # (num_food, 2)
food_collected: jnp.ndarray    # (num_food,) bool
has_food: jnp.ndarray | None = None
# ... other fields ...
# ADD: food_source_pos: jnp.ndarray | None = None  # (max_agents, 2)
```

**6. Food spawn (no nest distance check currently):**
- `reset()`: `food_positions = jax.random.randint(k2, shape=(num_food, 2), minval=0, maxval=grid_size)`
- `step()` respawn: `new_food_positions = jax.random.randint(pos_key, shape=(num_food, 2), minval=0, maxval=grid_size)`
- **NEED TO ADD**: rejection/resampling for `distance_to_nest > nest_radius + PATCH_RADIUS`

### Implementation Checklist (Code-Specific)

**1. Add to `state.py` EnvState:**
```python
food_source_pos: jnp.ndarray | None = None  # (max_agents, 2) - where agent picked up food
```

**2. Add to `configs.py`:**
```python
# In NestConfig or new PatchConfig:
patch_radius: int = 2  # Radius for counting agents at patch
patch_n_cap: int = 6   # Max agents for scaling benefit
patch_scaling_enabled: bool = False  # Toggle for experiments
```

**3. Modify `env.py` reset():**
- Initialize `food_source_pos = jnp.zeros((max_agents, 2), dtype=jnp.int32)`
- Add nest distance check for food spawn

**4. Modify `env.py` step():**
- At food pickup: store `food_source_pos` for agents with `pickup_mask`
- At delivery: count agents near `food_source_pos`, compute scaling
- Clear `food_source_pos` on delivery and death
- Add nest distance check for food respawn

---

## Implementation Complete (2026-02-06)

### Changes Made via Codex

All patch scaling implementation steps completed successfully:

| Step | Description | Status | Files Modified |
|------|-------------|--------|----------------|
| 1 | Add `food_source_pos` to EnvState | ✅ Done | state.py, env.py |
| 2 | Store `food_source_pos` at pickup | ✅ Done | env.py |
| 3 | Clear `food_source_pos` on delivery/death | ✅ Done | env.py |
| 4 | Add patch counting + scaling to delivery | ✅ Done | env.py, configs.py |
| 5 | Add food-nest spawn distance check | ✅ Done | env.py |
| 6 | Sanity check - tests pass | ✅ Done | 28 tests passed |
| 7 | Unit test scaling formula | ✅ Done | Formula verified |

### Config Parameters Added

```python
# In NestConfig (configs.py)
patch_radius: int = 2          # Radius for counting agents at food source
patch_n_cap: int = 6           # Max agents for scaling benefit
patch_scaling_enabled: bool = False  # Toggle for A/B experiments
```

### Scaling Formula Verified

| Agents at Patch | Individual Reward | Total Colony Reward |
|-----------------|-------------------|---------------------|
| n=1 | 50.00 (100%) | 50.00 |
| n=2 | 35.36 (71%) | 70.71 (+41%) |
| n=4 | 25.00 (50%) | 100.00 (+100%) |
| n=6 | 20.41 (41%) | 122.47 (+145%) |
| n=8 | 20.41 (capped) | 163.30 |

### Test Results

- **Import check**: ✅ Passed
- **Unit tests (test_env.py)**: ✅ 28 passed in 17.61s
- **Functional test**: ✅ Step completed, shapes correct
- **Formula verification**: ✅ Matches expected values

### Next Steps

1. Run quick validation (3 seeds, 1-2M steps) to confirm training works
2. Run full 2×2×2 experiment (8 conditions × 30 seeds × 10M steps)
3. Analyze results against pass/fail criteria (P1-P5)

---

*Document generated: 2026-02-06*
*Next step: Implement patch scaling, run 3-seed validation, then full 8×30 experiment*
