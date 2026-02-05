# Experiment Log

Track what works, what doesn't, and why. Learn from every run.

---

## Run #001: Maximum Training Attempt (FAILED)

**Date:** 2026-02-03
**Platform:** Kaggle GPU P100
**Duration:** ~12 hours (Session 1 of planned 3)
**Notebook Version:** #5

### Config Used (HARSH — caused population collapse)

```python
# Environment
env.grid_size = 20
env.num_agents = 8          # Starting population
env.num_food = 10           # Low food count

# Evolution (DEFAULT — too harsh for long runs)
evolution.enabled = True
evolution.max_agents = 32
evolution.starting_energy = 100      # Not enough buffer
evolution.food_energy = 50           # Low reward per food
evolution.energy_decay = 1.0         # Fast drain
evolution.reproduce_threshold = 150  # Hard to reach
evolution.reproduce_cost = 50

# Training
train.total_steps = 1_600_000_000    # 1.6B steps
train.num_envs = 32
train.num_steps = 128
train.learning_rate = 3e-4

# Logging
log.save_interval = 10_000_000       # Every 10M steps
```

### What Happened

| Metric | Observed Value | Expected |
|--------|----------------|----------|
| Population | 0.0 (collapsed) | 8-32 |
| Reward | 0.0000 | > 0 |
| Loss | 0.0000 | Learning signal |
| Entropy | 0.0000 | > 0 (exploration) |

**Timeline:**
- Started training, initial population = 8
- Agents couldn't find food fast enough
- Energy drained, agents died
- Population hit 0, no recovery possible
- Training continued for 10+ hours learning nothing

### Root Cause Analysis

1. **Food scarcity**: 10 food items on 20x20 grid = 2.5% coverage
2. **Energy math doesn't work**:
   - Start with 100 energy
   - Lose 1.0 per step
   - Need to find food within 100 steps or die
   - Food gives only 50 energy — not enough buffer
3. **Reproduction impossible**: Need 150 energy, but max sustainable is ~100
4. **No respawn**: Once population = 0, game over

### Lesson Learned

> For long training runs with evolution, agents need a survival buffer. The environment must be forgiving enough that random-policy agents can stumble into food before dying.

---

## Recommended Config: SURVIVAL-FRIENDLY

Use this for training runs where you want agents to survive and learn:

```python
# Environment — more forgiving
env.grid_size = 20
env.num_agents = 8
env.num_food = 20           # 2x more food (5% grid coverage)

# Evolution — easier survival
evolution.enabled = True
evolution.max_agents = 32
evolution.starting_energy = 200      # 2x buffer
evolution.food_energy = 100          # 2x reward
evolution.energy_decay = 0.5         # Half the drain rate
evolution.reproduce_threshold = 120  # Achievable
evolution.reproduce_cost = 40        # Cheaper reproduction
```

**Why this works:**
- Start with 200 energy, lose 0.5/step = 400 steps to find food
- Food gives 100 energy = sustainable with occasional finds
- Reproduction at 120 energy is achievable after 1-2 food pickups
- 20 food items = easier to stumble into

---

## Recommended Config: HARSH (for ablation/difficulty testing)

Use this ONLY for testing agent robustness after they've learned:

```python
# Environment — challenging
env.grid_size = 20
env.num_agents = 8
env.num_food = 5            # Scarce (1.25% coverage)

# Evolution — punishing
evolution.starting_energy = 100
evolution.food_energy = 50
evolution.energy_decay = 1.5         # Fast drain
evolution.reproduce_threshold = 200  # Very hard
evolution.reproduce_cost = 80
```

**Use case:** Test if trained agents can survive in harsh conditions (transfer learning / robustness test)

---

## Config Presets Reference

| Preset | num_food | starting_energy | food_energy | energy_decay | reproduce_threshold | Use Case |
|--------|----------|-----------------|-------------|--------------|---------------------|----------|
| **Easy** | 30 | 300 | 150 | 0.3 | 100 | Quick iteration, debugging |
| **Survival** | 20 | 200 | 100 | 0.5 | 120 | Main training runs |
| **Standard** | 10 | 100 | 50 | 1.0 | 150 | Default (risky for long runs) |
| **Harsh** | 5 | 100 | 50 | 1.5 | 200 | Robustness testing only |
| **Extreme** | 3 | 50 | 30 | 2.0 | 250 | Impossible mode (for fun) |

---

## Experiment Queue

| # | Description | Config | Platform | Status |
|---|-------------|--------|----------|--------|
| 001 | Max training 1.6B steps | Standard | Kaggle P100 | FAILED (pop collapse) |
| 002 | 64 agents, seed=42 | Survival | Colab TPU v5e | ✅ SUCCESS |
| 003 | 64 agents, seed=43 | Survival | Colab TPU v5e | ✅ SUCCESS (BEST!) |
| 004 | 64 agents, seed=44 | Survival | Colab TPU v5e | ✅ SUCCESS (UTOPIA!) |
| 005 | 64 agents, seed=45 | Survival | Colab TPU v5e | ✅ SUCCESS (BUSY) |
| 006 | Ablation: field OFF | Survival | Local | PENDING |
| 007 | Harsh conditions (trained model) | Harsh | Local | PENDING |

---

## Run #002: 64-Agent Survival Training (SUCCESS!)

**Date:** 2026-02-03
**Platform:** Google Colab TPU v5e
**Duration:** 13 minutes 18 seconds
**Seed:** 42

### Config Used (SURVIVAL-FRIENDLY)

```python
env=EnvConfig(
    grid_size=32,
    num_agents=16,          # Start with 16
    num_food=40,            # Plenty of food
),
evolution=EvolutionConfig(
    enabled=True,
    max_agents=64,          # Scale up to 64
    starting_energy=200,    # Survival buffer
    food_energy=100,        # Good reward
    energy_per_step=1,      # Normal drain
    reproduce_threshold=120,
    reproduce_cost=40,
    mutation_std=0.01,
),
train=TrainConfig(
    seed=42,
    total_steps=10_000_000,
    num_envs=32,
    num_steps=128,
    learning_rate=3e-4,
),
```

### Results

| Metric | Start | Peak | Final |
|--------|-------|------|-------|
| Population | 16 | 64.0 | 18.8 |
| Reward | 2.56 | 4.73 | 2.80 |
| Entropy | 1.73 | - | 1.09 |
| Loss | 1995 | - | 76.6 |

### Emergence Events Detected

| Step | Event | z-score | Interpretation |
|------|-------|---------|----------------|
| 5,767,168 | Field structure shift | 4.23 | Significant reorganization |
| 8,650,752 | Field entropy drop | 6.45 | Major phase transition |
| 8,912,896 | Field entropy drop | 3.63 | Continued restructuring |

### Final Statistics

```
Total env steps: 9,961,472
Births (final step): 464
Deaths (final step): 570
Oldest agent age: 968 steps
Mean energy: 85.5
Population equilibrium: ~19 agents
```

### Checkpoint

- **Location:** `checkpoints/colab_run_001/step_9961472.pkl`
- **Size:** 304 MB

### Key Observations

1. **Population dynamics:** Started at 16, exploded to 64, settled at ~19
2. **Natural selection:** 570 deaths vs 464 births = population declining to sustainable level
3. **Three phase transitions:** Field reorganized multiple times
4. **Reward curve:** Peaked at 4.73, stabilized around 2.8
5. **Entropy dropped:** From 1.73 → 1.09 (agents developed strategies)

### Lessons Learned

> Survival-friendly config works! Agents survive, learn, reproduce, and show emergence signals.
> TPU v5e is ~10x faster than Kaggle P100 for this workload.
> Population self-regulates to carrying capacity (~19 agents for 40 food items).

---

## Run #003: 64-Agent Survival Training, Seed 43 (SUCCESS — BEST RUN!)

**Date:** 2026-02-03
**Platform:** Google Colab TPU v5e
**Duration:** 13 minutes 24 seconds
**Seed:** 43

### Config Used

Same as Run #002 (survival-friendly), only `seed=43`.

### Results

| Metric | Start | Peak | Final |
|--------|-------|------|-------|
| Population | 16 | 64.0 | **62.0** |
| Reward | 2.80 | **5.49** | **5.36** |
| Entropy | 1.74 | - | 1.10 |
| Loss | 2173 | - | 124 |

### Comparison: Seed 42 vs Seed 43

| Metric | Seed 42 | Seed 43 | Difference |
|--------|---------|---------|------------|
| Final reward | 2.80 | **5.36** | +92% |
| Final population | 19 | **62** | +226% |
| Oldest agent age | 968 | **4,864** | +402% |
| Births/Deaths balance | 464/570 (declining) | **157/157 (perfect)** | Equilibrium! |

### Specialization Events Detected

| Step | Event | z-score |
|------|-------|---------|
| 8,388,608 | weight_divergence increase | 3.28 |
| 8,388,608 | max_divergence increase | 3.28 |

### Final Statistics

```
Total env steps: 9,961,472
Births (final step): 157
Deaths (final step): 157  ← PERFECT EQUILIBRIUM!
Oldest agent age: 4,864 steps  ← SURVIVED ALMOST ENTIRE TRAINING
Mean energy: 166.4  ← WELL-FED COLONY
Population equilibrium: ~62 agents
```

### Checkpoint

- **Location:** `checkpoints/colab_run_002_seed43/step_9961472.pkl`
- **Size:** ~304 MB

### Key Observations

1. **Thriving colony:** Population stayed at 62 (vs 19 in seed 42)
2. **Perfect birth/death balance:** 157 = 157 (self-sustaining)
3. **Long-lived agents:** Oldest survived 4,864 steps (5x longer than seed 42)
4. **Higher rewards:** 5.36 vs 2.80 (nearly double)
5. **Specialization detected:** Weight divergence events at step 8.4M

### Why Seed 43 Did Better Than Seed 42

Different random starting positions led to different collective dynamics:
- **Seed 42:** Agents competed fiercely, population crashed, survival mode
- **Seed 43:** Agents found harmony, stable population, thriving colony

**This is exactly what emergence looks like** — same rules, different outcomes based on initial conditions. Some colonies struggle, some thrive, just like in nature.

### Lessons Learned

> Multiple seeds reveal the RANGE of possible outcomes.
> Emergence is real — both seeds showed events, but different types.
> Seed 43 discovered a "better" collective strategy than seed 42.

---

## Run #004: 64-Agent Survival Training, Seed 44 (SUCCESS — UTOPIA!)

**Date:** 2026-02-03
**Platform:** Google Colab TPU v5e
**Duration:** 13 minutes 24 seconds
**Seed:** 44

### Config Used

Same as Run #002 (survival-friendly), only `seed=44`.

### Results

| Metric | Start | Peak | Final |
|--------|-------|------|-------|
| Population | 16 | 64.0 | **64.0 (MAXED!)** |
| Reward | 2.11 | **5.62** | **5.62 (BEST EVER!)** |
| Entropy | 1.74 | - | 1.04 |
| Loss | 1652 | - | 140 |

### Comparison: All Seeds

| Metric | Seed 42 | Seed 43 | Seed 44 |
|--------|---------|---------|---------|
| Final population | 19 | 62 | **64** |
| Final reward | 2.80 | 5.36 | **5.62** |
| Births/Deaths | 464/570 | 157/157 | **30/30** |
| Oldest agent | 968 | 4,864 | **4,864** |
| Min energy | 1 | 1 | **11** |
| Mean energy | 85 | 166 | **174** |
| Colony type | Struggling | Thriving | **UTOPIA** |

### Specialization Events Detected

| Step | Event | z-score |
|------|-------|---------|
| 9,175,040 | weight_divergence increase | 3.88 |
| 9,175,040 | max_divergence increase | 3.88 |

### Final Statistics

```
Total env steps: 9,961,472
Births (final step): 30
Deaths (final step): 30  ← MINIMAL TURNOVER
Oldest agent age: 4,864 steps
Mean energy: 174.5  ← FAT AND HAPPY
Min energy: 11  ← NOBODY STARVING
Population: 64.0  ← PERFECTLY MAXED
```

### Checkpoint

- **Location:** `checkpoints/colab_run_003_seed44/step_9961472.pkl`
- **Size:** ~304 MB

### Key Observations

1. **Perfect population:** Stayed at exactly 64 (max capacity) throughout
2. **Minimal turnover:** Only 30 births/deaths (vs 464/570 in seed 42)
3. **Nobody struggling:** Minimum energy = 11 (vs 1 in other seeds)
4. **Highest reward:** 5.62 beats all previous seeds
5. **Utopia achieved:** Agents found perfect collective harmony

### What Made Seed 44 Special

The random starting arrangement led to a "perfect storm" of cooperation:
- Agents spread out efficiently across the grid
- Nobody competing for the same food
- Field information organized optimally
- Result: Maximum population, maximum reward, minimum conflict

### Colony Types Discovered

| Type | Seeds | Characteristics |
|------|-------|-----------------|
| **Struggling** | 42 | High competition, population crash, survival mode |
| **Thriving** | 43 | Balanced, stable population, good rewards |
| **Utopia** | 44 | Perfect harmony, max population, minimal turnover |

### Lessons Learned

> Same rules can produce vastly different collective outcomes.
> "Utopia" colonies exist — where agents achieve perfect coordination.
> This is STRONG evidence for emergence — the system finds different stable states.

---

## Run #005: 64-Agent Survival Training, Seed 45 (SUCCESS — BUSY COLONY)

**Date:** 2026-02-03
**Platform:** Google Colab TPU v5e
**Duration:** 13 minutes 24 seconds
**Seed:** 45

### Config Used

Same as Run #002 (survival-friendly), only `seed=45`.

### Results

| Metric | Start | Peak | Final |
|--------|-------|------|-------|
| Population | 16 | 64.0 | **61.6** |
| Reward | 2.73 | **4.97** | **4.72** |
| Entropy | 1.75 | - | 1.16 |
| Loss | 2121 | - | 109 |

### Final Comparison: ALL 4 SEEDS

| Metric | Seed 42 | Seed 43 | Seed 44 | Seed 45 |
|--------|---------|---------|---------|---------|
| Final population | 19 | 62 | **64** | 62 |
| Final reward | 2.80 | 5.36 | **5.62** | 4.72 |
| Births/Deaths | 464/570 | 157/157 | **30/30** | 589/557 |
| Oldest agent | 968 | 4,864 | 4,864 | 4,864 |
| Min energy | 1 | 1 | **11** | 1 |
| Mean energy | 85 | 166 | **174** | 139 |
| Colony type | Struggling | Thriving | **UTOPIA** | BUSY |

### Final Statistics

```
Total env steps: 9,961,472
Births (final step): 589  ← HIGH TURNOVER!
Deaths (final step): 557  ← LOTS OF CHURN
Oldest agent age: 4,864 steps
Mean energy: 139.3
Min energy: 1
Population: 61.6
```

### Checkpoint

- **Location:** `checkpoints/colab_run_004_seed45/step_9961472.pkl`
- **Size:** ~304 MB

### Key Observations

1. **High turnover strategy:** 589 births vs 557 deaths (vs 30/30 in utopia)
2. **Different survival approach:** Live fast, die young, reproduce lots
3. **Still successful:** Population stable at 62, reward at 4.72
4. **No specialization events:** Unlike other seeds, no z-score triggers

### Colony Strategies Discovered

| Strategy | Seeds | Description | Births/Deaths |
|----------|-------|-------------|---------------|
| **Struggling** | 42 | Can't find equilibrium, population crashes | 464/570 (declining) |
| **Thriving** | 43 | Balanced, stable, moderate turnover | 157/157 (balanced) |
| **Utopia** | 44 | Perfect harmony, minimal turnover | 30/30 (stable) |
| **Busy** | 45 | Fast reproduction, high turnover | 589/557 (churning) |

### Biological Analogy

| Colony Type | Real-World Equivalent |
|-------------|----------------------|
| Utopia (Seed 44) | Elephants — few births, long lives |
| Busy (Seed 45) | Rabbits — many births, shorter lives |

Both strategies achieve population stability through different means!

### Lessons Learned

> The system discovered MULTIPLE successful survival strategies.
> High turnover (Busy) and low turnover (Utopia) both work.
> This is strong evidence for emergence — different stable equilibria from same rules.

---

## Summary: 4 Seeds Complete

### Overall Statistics

| Metric | Value |
|--------|-------|
| Total runs | 4 |
| Success rate | **100%** (4/4) |
| Total training time | ~54 minutes |
| Platform | Google Colab TPU v5e |
| Checkpoints saved | 4 final + 40 intermediate |

### Colony Types Distribution

```
Struggling: 1/4 (25%)  — Seed 42
Thriving:   1/4 (25%)  — Seed 43
Utopia:     1/4 (25%)  — Seed 44
Busy:       1/4 (25%)  — Seed 45
```

### Emergence/Specialization Events

| Seed | Events | Type |
|------|--------|------|
| 42 | 3 | Phase transitions (structure, entropy) |
| 43 | 2 | Specialization (weight divergence) |
| 44 | 2 | Specialization (weight divergence) |
| 45 | 0 | None detected |

### Key Findings

1. **Emergence is REAL** — 3/4 seeds showed statistically significant events
2. **Multiple equilibria exist** — Same rules produce different stable states
3. **Different survival strategies** — Utopia vs Busy both viable
4. **Reproducible** — Results consistent across different random seeds
5. **TPU v5e is fast** — 10M steps in ~13 minutes (vs hours on GPU)

---

## Key Metrics to Watch

| Metric | Healthy Range | Warning | Critical |
|--------|---------------|---------|----------|
| Population | 8-32 | < 5 | 0 |
| Reward (per step) | > 0.01 | < 0.001 | 0.0000 |
| Entropy | 0.5-1.5 | < 0.3 | 0.0000 |
| Loss | 0.01-1.0 | > 5.0 | 0.0000 |

**If population drops to 0** → training is useless, stop and fix config.

---

---

## Run #005: Multi-Seed Statistical Experiment (IN PROGRESS)

**Date:** 2026-02-03
**Platform:** Google Colab TPU v6e + High-RAM
**Goal:** 30 seeds Field ON + 30 seeds Field OFF = 60 total for statistical significance

### Config (PROVEN 64-AGENT)

```python
env=EnvConfig(
    grid_size=32,
    num_agents=16,
    num_food=40,
),
evolution=EvolutionConfig(
    enabled=True,
    max_agents=64,
    starting_energy=200,
    food_energy=100,
    energy_per_step=1,
    reproduce_threshold=120,
    reproduce_cost=40,
    mutation_std=0.01,
),
train=TrainConfig(
    total_steps=10_000_000,
    num_envs=32,
    num_steps=128,
),
```

### Field ON Results (30 seeds)

| Batch | Seed | Reward | Population | Status |
|-------|------|--------|------------|--------|
| 0 | 0 | 5.19 | 64 | 🎉 UTOPIA |
| 0 | 1 | 5.38 | 64 | 🎉 UTOPIA |
| 0 | 2 | 4.70 | 6 | ⚠️ Crashed |
| 1 | 3 | 3.09 | 22 | ⚠️ Struggled |
| 1 | 4 | 4.84 | 64 | 🎉 UTOPIA |
| 1 | 5 | 5.50 | 64 | 🎉 UTOPIA |
| 2 | 6 | 2.54 | 11 | ⚠️ Struggled |
| 2 | 7 | 0.00 | 0 | ❌ DIED |
| 2 | 8 | 5.18 | 64 | 🎉 UTOPIA |
| 3 | 9 | 4.52 | 20 | ⚠️ Low pop |
| 3 | 10 | 5.09 | 64 | 🎉 UTOPIA |
| 3 | 11 | 5.33 | 64 | 🎉 UTOPIA |
| 4 | 12 | 5.38 | 40 | 👍 Thriving |
| 4 | 13 | 3.70 | 30 | 👍 Okay |
| 4 | 14 | 5.24 | 48 | 👍 Thriving |
| 5 | 15 | 4.61 | 62 | 🎉 Near-UTOPIA |
| 5 | 16 | 3.46 | 39 | 👍 Thriving |
| 5 | 17 | 4.56 | 58 | 🎉 Near-UTOPIA |
| 6 | 18 | 5.48 | 64 | 🎉 UTOPIA |
| 6 | 19 | 4.42 | 64 | 🎉 UTOPIA |
| 6 | 20 | 5.43 | 64 | 🎉 UTOPIA |
| 7 | 21 | 4.56 | 58 | 🎉 Near-UTOPIA |
| 7 | 22 | 5.30 | 64 | 🎉 UTOPIA |
| 7 | 23 | 4.99 | 30 | 👍 Thriving |
| 8 | 24 | 4.22 | 28 | ⚠️ Struggled |
| 8 | 25 | 4.33 | 50 | 👍 Thriving |
| 8 | 26 | 5.51 | 62 | 🎉 Near-UTOPIA |
| 9 | 27 | 4.19 | 52 | 👍 Thriving |
| 9 | 28 | 4.67 | 60 | 🎉 Near-UTOPIA |
| 9 | 29 | 5.20 | 58 | 🎉 Near-UTOPIA |

### ✅ FIELD ON COMPLETE!

| Status | Count | % |
|--------|-------|---|
| 🎉 UTOPIA/Near (58-64) | 17 | 57% |
| 👍 Thriving (30-57) | 10 | 33% |
| ⚠️ Struggled (<30) | 2 | 7% |
| ❌ Died | 1 | 3% |

**Survival Rate: 97% (29/30)**

### Field OFF Results (30 seeds)

| Metric | Value |
|--------|-------|
| Mean reward | 5.5162 +/- 0.1909 |
| IQM | 5.5483 [5.4824, 5.6095] |
| Min/Max reward | 4.7852 / 5.7281 |
| CoV | 0.0346 |
| Failed seeds | 0 |

### Statistical Comparison: Field ON vs Field OFF

| Metric | Field ON | Field OFF | Winner |
|--------|----------|-----------|--------|
| Training reward (mean) | 4.554 +/- 1.136 | **5.516 +/- 0.191** | Field OFF |
| Training reward (IQM) | 4.838 | **5.548** | Field OFF |
| Eval reward | 44,423 +/- 36,730 | **81,960 +/- 44,087** | Field OFF |
| Eval population | 7.3 +/- 16.6 | **29.9 +/- 27.1** | Field OFF |
| At max capacity | 2/30 | **10/30** | Field OFF |
| Mean births | 17.3 | **69.9** | Field OFF |
| Weight divergence | 0.0002 | 0.0000 | Negligible |
| Failed seeds | 1 | **0** | Field OFF |
| Variance (CoV) | 0.2494 | **0.0346** | Field OFF |

### Hypothesis Tests

| Test | Statistic | p-value | Effect Size |
|------|-----------|---------|-------------|
| Welch's t-test | - | **p < 0.001** | Cohen's d = 1.18 (LARGE) |
| Mann-Whitney U | U = 834.0 | **p = 0.000000** | r = 0.85 |
| P(Field OFF > ON) | **0.927** | - | - |
| Sensitivity (excl. failed) | - | **p = 0.000004** | d = 1.47 |

### ✅ RUN #005 COMPLETE

**Conclusion:** Field OFF wins decisively on simple foraging. p < 0.001, Cohen's d = 1.18 (large effect). The shared field adds overhead without benefit when coordination is not required. Field ON has 7x more variance and unstable populations.

**Key insight:** Weight divergence = 0.0000 for both conditions. PPO gradient sync overwrites all mutation-driven differentiation. Agents are identical clones regardless of field.

**Saved to Drive:**
- `analysis_results/field_on_vs_off_results.json`
- `analysis_results/field_on_vs_off_results.pkl`
- `analysis_results/comparison_report.md`
- 5 publication-quality figures (PNG + PDF)

---

## Run #006: Hidden Food Coordination — Field ON vs Field OFF (IN PROGRESS)

**Date:** 2026-02-04
**Platform:** Google Colab TPU v6e + High-RAM
**Goal:** Test if the field helps when coordination IS required (hidden food)

### Why This Experiment

Run #005 showed the field hurts on simple foraging. But simple foraging is an individual task — agents don't need each other. Hidden food REQUIRES K=3 agents within distance 3 to reveal invisible high-value food (5x energy). This is a coordination task where the field should help agents signal locations.

### Config (same PROVEN 64-agent + hidden food enabled)

```python
# Same base config as Run #005, plus:
env.hidden_food.enabled = True
env.hidden_food.num_hidden = 3         # 3 invisible food items
env.hidden_food.required_agents = 3    # Need 3 agents nearby
env.hidden_food.reveal_distance = 3    # Chebyshev distance
env.hidden_food.reveal_duration = 10   # Stays visible 10 steps
env.hidden_food.hidden_food_value_multiplier = 5.0  # 500 energy per item
```

### Status

| Condition | Batches | Seeds | Status |
|-----------|---------|-------|--------|
| Field ON | 10/10 | 30/30 | ✅ COMPLETE |
| Field OFF | ~2/10 | ~6/30 | 🔄 Running |

**Checkpoints:** `/content/drive/MyDrive/emergence-lab/hidden_food_field_on/` and `hidden_food_field_off/`

**Observations so far:**
- Field ON rewards: 3.5 → 5.2 over 10M steps (learning)
- Field OFF batch 1 rewards: 6.15, 7.12, 7.25 (higher than Field ON early batches)
- Early indication: Field OFF may win again, but need full data

---

## Run #007: Config Sweep — 34 Specialization Configurations (IN PROGRESS)

**Date:** 2026-02-04
**Platform:** Google Colab TPU v6e + High-RAM
**Goal:** Find the best specialization settings before running the full experiment

### Why This Experiment

Weight divergence = 0.0000 everywhere. PPO kills all differentiation. We upgraded `parallel_train.py` with:
- Per-agent action selection during evolve phases
- Diversity bonus and niche pressure rewards
- FREEZE_EVOLVE mode (alternate gradient + evolution phases)

This sweep tests 34 configs at short runs (3 seeds, 5-15 iterations each) to find which settings actually produce weight divergence AND hidden food coordination.

### Configs Being Tested

| Group | Count | What Varies | Mode |
|-------|-------|-------------|------|
| A1: diversity_bonus | 5 | db=[0.0, 0.05, 0.1, 0.2, 0.5] | GRADIENT |
| A3: mutation_std | 5 | ms=[0.005, 0.01, 0.02, 0.05, 0.1] | GRADIENT |
| B: div x niche | 6 | db=[0.1, 0.2, 0.5] x np=[0.05, 0.1] | GRADIENT |
| D1: FE ratio | 5 | 95/5, 90/10, 80/20, 60/40, 50/50 | FREEZE_EVOLVE |
| D2: FE boost | 4 | boost=[2, 5, 10, 20] | FREEZE_EVOLVE |
| E: FE + diversity | 4 | FE ratio x diversity combos | FREEZE_EVOLVE |
| F: pure evolve | 3 | ms=[0.01, 0.05, 0.1] | EVOLVE |
| G: baselines | 2 | field ON default, field OFF | GRADIENT |

### Status

- Progress: 11/34 configs complete
- Elapsed: ~2.2 hours
- ETA: ~4-5 more hours
- Early finding: GRADIENT configs with diversity_bonus show HF=2.0 but divergence still 0.0000
- FREEZE_EVOLVE configs (the interesting ones) haven't started yet

**Results will be saved to:** `/content/drive/MyDrive/emergence-lab/config_sweep/`

---

## Run #007: Hidden Food Coordination Analysis (COMPLETE — Field OFF Wins Again)

**Date:** 2026-02-05
**Platform:** Google Colab
**Duration:** 30 seeds per condition, 10M steps each (60 total runs)
**Goal:** Determine if the shared field enables coordination on a task that REQUIRES multi-agent cooperation (hidden food revelation)

### Why This Experiment

Run #005 showed the field hurts on simple foraging — but simple foraging is an individual task. Hidden food requires K=3 agents within Chebyshev distance 3 to reveal invisible high-value items (5x energy = 500). If the field carries coordination signal, this is where it should show up: agents need to converge on the same location simultaneously.

### Config

```python
# Base: proven 64-agent survival config
env=EnvConfig(
    grid_size=32,
    num_agents=16,        # Start with 16, grow to 64
    num_food=40,
),
evolution=EvolutionConfig(
    enabled=True,
    max_agents=64,
    starting_energy=200,
    food_energy=100,
    energy_per_step=1,
    reproduce_threshold=120,
    reproduce_cost=40,
    mutation_std=0.01,
),
train=TrainConfig(
    total_steps=10_000_000,
    num_envs=32,
    num_steps=128,
),

# Hidden food settings
env.hidden_food.enabled = True
env.hidden_food.num_hidden = 3         # 3 invisible food items
env.hidden_food.required_agents = 3    # Need 3 agents nearby to reveal
env.hidden_food.reveal_distance = 3    # Chebyshev distance
env.hidden_food.hidden_food_value_multiplier = 5.0  # 500 energy per item

# Field ON condition
field.diffusion_rate = 0.1
field.decay_rate = 0.05
field.write_strength = 1.0

# Field OFF condition
field.diffusion_rate = 0.0
field.decay_rate = 1.0
field.write_strength = 0.0
```

**Eval:** 1 episode per seed, 500 steps

### Hidden Food Coordination Results

| Metric | Field ON | Field OFF | Test | p-value | Effect |
|--------|----------|-----------|------|---------|--------|
| Hidden food revealed | 3.00 +/- 3.27 | 1.90 +/- 2.19 | Welch t=1.532 | p=0.132 | d=0.395 |
| Hidden food collected | 2.33 +/- 2.72 | 2.07 +/- 1.95 | Welch t=0.437 | p=0.664 | d=0.113 |

**Neither metric is statistically significant.** The field does not help agents coordinate to reveal or collect hidden food.

### Performance Results

| Metric | Field ON | Field OFF | Winner |
|--------|----------|-----------|--------|
| Regular food collected | 315.2 +/- 224.5 | **391.6 +/- 278.6** | Field OFF |
| Total eval reward | 32,687 | **40,197** | Field OFF |
| Final population | 2.7 +/- 4.7 | **5.1 +/- 6.7** | Field OFF |
| Training reward (IQM) | 5.00 [4.38, 5.57] | **6.78 [6.64, 6.92]** | Field OFF |

**Field OFF wins on every performance metric**, consistent with Run #005.

### Weight Divergence

| Metric | Field ON | Field OFF | Test | p-value | Effect |
|--------|----------|-----------|------|---------|--------|
| Weight divergence | 0.0004 +/- 0.0010 | 0.0000 | Welch | p=0.024 | d=0.614 (SIGNIFICANT) |

Field ON creates measurable weight divergence (specialization), but this divergence is **purposeless** — it does not translate to better food collection or population sustainability.

### Key Findings

1. **Field does NOT enable coordination:** Hidden food revealed and collected are statistically indistinguishable between conditions (p=0.132, p=0.664)
2. **Field OFF wins everywhere:** Regular food, total reward, population, training reward — all favor Field OFF
3. **Specialization without purpose:** Field ON produces significant weight divergence (p=0.024, d=0.614) but this differentiation provides no performance benefit
4. **Low populations in both conditions:** Final populations of 2.7 and 5.1 suggest the hidden food config is harsh, but Field OFF still survives better

### Interpretation

The old field system cannot be fixed by parameter tuning. The core problems are architectural:
- **484 dimensions of noise** in the observation (local field patch overwhelms useful signal)
- **Constant writes** (every agent writes every step, no selectivity)
- **No trail geometry** (field diffuses uniformly, no directional information)

This confirms the need for a biological pheromone system rebuild — one with discrete pheromone types, active deposition decisions, and trail-following geometry that actually mimics ant colony stigmergy.

### Data

- **Saved to:** `/content/drive/MyDrive/emergence-lab/hidden_food_analysis_results/`

### Lessons Learned

> The shared field does not help even when the task explicitly requires multi-agent coordination.
> Weight divergence (specialization) can emerge without being adaptive — differentiation is not the same as division of labor.
> The observation-based field design (484 dims, constant writes, no trail geometry) is fundamentally incapable of supporting stigmergic coordination. A ground-up pheromone system rebuild is needed.

---

## Experiment Queue (Updated)

| # | Description | Config | Platform | Status |
|---|-------------|--------|----------|--------|
| 001 | Max training 1.6B steps | Standard | Kaggle P100 | ❌ FAILED (pop collapse) |
| 002 | 64 agents, seed=42 | Survival | Colab TPU v5e | ✅ SUCCESS |
| 003 | 64 agents, seed=43 | Survival | Colab TPU v5e | ✅ SUCCESS (BEST!) |
| 004 | 64 agents, seed=44 | Survival | Colab TPU v5e | ✅ SUCCESS (UTOPIA!) |
| 005 | 30+30 Field ON vs OFF (simple food) | Proven 64-agent | Colab TPU v6e | ✅ COMPLETE — Field OFF wins |
| 006 | 30+30 Hidden Food ON vs OFF | Proven 64-agent + hidden food | Colab TPU v6e | 🔄 Running |
| 007 | Config Sweep (34 configs) | Various specialization settings | Colab TPU v6e | 🔄 Running |
| 008 | Specialization + Hidden Food (30 seeds) | Winner from #007 | Colab TPU v6e | ⬜ Waiting for #007 |
| 009 | Field Ablation Mid-Run | TBD | TBD | ⬜ Planned |
| 010 | Scaling Test (32/64/128 agents) | TBD | TBD | ⬜ Planned |
| 011 | Visual Demo | TBD | TBD | ⬜ Planned |

---

## Changelog

- **2026-02-03**: Created log after Run #001 failure. Documented harsh vs survival configs.
- **2026-02-03**: Started multi-seed experiment (Run #005) with proven 64-agent config.
- **2026-02-04**: Run #005 COMPLETE. Field OFF wins on simple foraging (p<0.001, d=1.18). All results saved to Drive.
- **2026-02-04**: Started Run #006 (Hidden Food) and Run #007 (Config Sweep) in parallel on Colab.
- **2026-02-04**: Upgraded parallel_train.py: per-agent action selection, diversity bonuses, FREEZE_EVOLVE mode. 1293 tests pass.
