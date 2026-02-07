# Experiment Log

## 2026-02-07 Session: Field ON vs Field OFF -- Proving Pheromone Stigmergy

### Session Context

Three-way workflow: Claude Code (code changes), Ashish (runs on Colab), ChatGPT (diagnosis/math). Goal: determine whether pheromone stigmergy helps or hurts agent foraging performance under the biological pheromone system (Phase 6).

---

### 1. Delivery Rate Instrumentation

**Problem:** No way to measure actual foraging success (food pickups and nest deliveries).

**Changes:**
- Added `num_pickups` and `num_deliveries` counters to `env.py`
- Fixed metric plumbing through 3 layers: `env.py` -> `rollout.py` -> `train.py`/`parallel_train.py` -> `all_metrics` dict
- Fixed `(0,0)` food stacking bug where food positions initialized at origin caused false pickups

---

### 2. Energy Cap Fix

**Problem:** `max_energy=200` silently destroyed reproduction economics. Delivery energy (food_energy=100 added to agent's current energy) overflowed the cap, meaning agents could never accumulate enough energy to reproduce.

**Fix:** Set `max_energy=300`.

**Result:** Population grew from 8 to 13.97, delivery_rate=0.99.

**Verdict:** GREEN LIGHT -- carry/deliver cycle proven working.

---

### 3. Proven Survivable Config

The following config produces healthy population dynamics with active foraging:

```
grid_size=20, num_agents=8, num_food=10, max_agents=32
starting_energy=200, max_energy=300, food_energy=100
reproduce_threshold=120, reproduce_cost=80, mutation_std=0.01
```

---

### 4. Field ON vs Field OFF Experiments

The central question: does the pheromone field improve foraging performance?

| # | Experiment | Field OFF | Field ON | Winner | Key Insight |
|---|-----------|-----------|----------|--------|-------------|
| 1 | Full food obs (K=5), 20x20, diff=0.5 | 132 del, pop 10 | 93 del, pop 5 | OFF | Field redundant when agents see food directly |
| 2 | No food obs, 20x20 | 0 del, pop 0 | 0 del, pop 0 | Both dead | Can't bootstrap without ANY food info |
| 3 | Food odor (lambda=4), 20x20, diff=0.5 | 0 del, pop 0 | 1.3 del, pop 0.15 | Both dead (ON slightly better) | Odor too weak on large grid, agents die before learning |
| 4 | Food odor, 10x10, 20 food, 400 energy, diff=0.5 | 5875 del, pop 32 | 5212 del, pop 32 | OFF (+12%) | Too easy -- odor solves everything, diffusion washes out trails |
| 5 | Food odor, 14x14, 10 food, 400 energy, diff=0.15 | 2187 del, pop 32 | 1548 del, pop 32 | OFF (+41%) | Scale-matched but field still hurts -- PPO suppresses weak signals |
| 6 | Food patch marking | TBD | TBD | TBD | Simplest possible field: mark where food was found, no diffusion |

**Summary so far:** Field ON consistently underperforms Field OFF across all completed experiments. The field's marginal value is negative, not zero -- it actively degrades performance.

---

### 5. New Features Implemented

#### food_obs_enabled (configs.py, obs.py)
Toggle to hide K=5 nearest-food observations from agents. When disabled, agents lose direct food sensing and must rely on other signals (odor, pheromone trails).

#### food_odor_enabled (configs.py, obs.py)
Passive food odor channel. Each food item emits `exp(-dist/lambda)` scent. Agents sense odor at center + N/S/E/W (5 observation dimensions). Provides partial food information without exact positions.

#### food_patch_marking (configs.py, env.py)
Write Ch0 pheromone ONLY at the exact food pickup location (one-shot write), not along the return path. This is the simplest possible stigmergy: "food was here." No diffusion, no trail following, just location marking.

---

### 6. Key Theoretical Insights

These insights emerged from systematic diagnosis of why Field ON consistently hurts performance.

#### Insight 1: Stigmergy is an amplifier, not a generator
The field can only help when agents have some base success rate to amplify. It cannot bootstrap foraging from zero.

#### Insight 2: Emergent coordination requires partial observability
Too much individual info (K=5 food obs) makes the field redundant. Too little info (no food obs) prevents bootstrapping. The sweet spot is partial observability where individual sensing is insufficient but non-zero.

#### Insight 3: Three coupled length scales must align
- Exploration scale (how far agents roam)
- Coordination benefit scale (distance at which social info helps)
- Field coherence scale (how far pheromone spreads before decaying)

When these are mismatched, the field adds noise rather than signal.

#### Insight 4: Diffusion must scale with grid size
Diffusion=0.5 on a 10x10 grid produces instant global mixing -- the pheromone becomes a uniform background, destroying spatial information. Rule of thumb: diffusion ~ grid_size / 100.

#### Insight 5: PPO suppresses weak coordination channels
Adding noisy social signals to a sufficient individual sensor degrades performance under shared-gradient learning (negative transfer). PPO optimizes the dominant signal path and treats the weaker channel as noise.

#### Insight 6: The field's marginal value is negative, not zero
If the field were simply ignored, we would expect ON ~ OFF. The consistent degradation (12-41%) means agents actively read the field and it misleads them.

#### Insight 7: Write semantics are backwards
Laden agents write pheromone on the return path (food -> nest). This creates trails leading AWAY from food and TOWARD the nest. Other agents following these trails walk toward the nest, not toward food. The write semantics don't clearly encode "go here for food."

---

### 7. Current Status (as of 2026-02-07)

**Food patch marking test running on Colab** -- this is the "unit test for stigmergy." It tests the simplest possible field usage: mark the exact location where food was found, with no diffusion, no trail, just a spatial marker.

**Decision tree:**
- If patch marking helps: adaptive engine is viable (proceed to Path D)
- If patch marking fails: this stigmergy formulation is a valid negative result

---

### 8. Forward Paths

Four paths identified depending on patch marking results:

**Path A: Publish negative result**
"Stigmergy degrades performance under PPO when individual sensing suffices." This is a valid scientific contribution -- it shows the conditions under which stigmergy fails and why.

**Path B: Change task economics**
Make coordination necessary by construction. Design environments where individual sensing is provably insufficient but collective sensing can solve the task.

**Path C: Change learning paradigm**
Evolutionary methods (e.g., MAP-Elites, NEAT) may preserve weak coordination signals that PPO's shared gradient discards.

**Path D: Adaptive stigmergy engine**
Build a system where the field activates only when economically justified. Two-task demo: easy task (gate closes, field off) + coordination task (gate opens, field on). Same model, same hyperparams, different environments. This proves the engine ADAPTS.

---

### 9. Recommended Next Steps (if patch marking works)

1. Build two-task demo: Easy task (gate closes) + Coordination task (gate opens)
2. Same model, same hyperparams, different environments
3. This proves the engine adapts -- not just "field ON" or "field OFF"
4. The demo narrative: "the system learns WHEN to use stigmergy, not just HOW"
