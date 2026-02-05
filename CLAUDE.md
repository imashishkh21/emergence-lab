# CLAUDE.md

## Mission

**Prove that collective intelligence emerges from simple agents + shared medium + evolution — and demonstrate it visually in a way that makes investors say "holy shit."**

This is a research project. Every code change should serve one question: *does the shared field encode collective knowledge that no individual agent possesses?* The evidence must be rigorous (ablation-tested, statistically significant) AND visceral (visualizations that make emergence undeniable at a glance).

## What We've Proven So Far

| Phase | Status | Key Result |
|-------|--------|------------|
| **Phase 1: Digital Petri Dish** | COMPLETE | Agents learn to forage; field develops spatial structure; Normal > Zeroed > Random field in ablation |
| **Phase 2: Evolutionary Pressure** | COMPLETE | Birth/death/reproduction working; population reaches equilibrium (32/32 maxed); 108 births + 108 deaths = perfect turnover; weight inheritance + mutation |
| **Phase 3: Specialization Detection** | COMPLETE | Weight divergence tracking, behavioral clustering (K-means + silhouette), species detection, lineage-strategy correlation, specialization ablation (divergent > uniform > random) |
| **Phase 4/4B: Infrastructure** | COMPLETE | Transfer entropy, division of labor, phase transition detection, dashboard, checkpointing, Kaggle training (9.8M steps) |
| **Phase 5: Prove Emergence** | COMPLETE | Information-theoretic metrics (O-info, PID, Causal Emergence), baselines (IPPO, ACO, MAPPO), statistical reporting (rliable), publication figures, multi-seed experiments |
| **Phase 6: Biological Pheromone System** | IN PROGRESS | Redesigned field based on 103 research sources; success-gated writes, gradient sensing, nest mechanics, food carry-back |

**Key empirical finding**: Random field HURTS agents (585 < 600 food) — they learned to READ the field for information. The field is not noise; it carries signal. This motivated a complete redesign based on ant biology research (see Phase 6).

## The Science

### Core Hypothesis
A shared learnable medium (stigmergy) between simple agents, combined with evolutionary pressure, produces:
1. **Collective knowledge** encoded in field spatial structures
2. **Behavioral specialization** — scouts, exploiters, balanced agents emerge
3. **Species formation** — stable, hereditary behavioral clusters

### Biological Inspiration
- **Ant colony stigmergy**: pheromone trails encode collective knowledge about food sources
- **Response threshold model**: individual insects have variable thresholds for responding to stimuli — some become field-readers, others field-writers
- **NEAT speciation**: genomic distance + fitness sharing drives population diversity
- **Central-place foraging**: nest → food → nest cycle (all reward comes from delivering food home)
- **Success-gated pheromone deposition**: only laden ants lay recruitment trail (Ch0) — prevents noise from unladen agents
- **Tropotaxis**: bilateral gradient sensing for trail following (4-directional field reads per channel)
- **Path integration with noise**: compass degrades linearly with distance from nest, forcing reliance on pheromone trails far from home
- **Dual-channel pheromone**: recruitment = volatile (high diffusion, fast decay), territory = persistent (low diffusion, near-permanent)

### Literature References
- Mouret & Clune (2015) — MAP-Elites, Quality-Diversity
- Lehman & Stanley (2011) — Novelty Search, k-NN behavioral novelty
- Stanley & Miikkulainen (2002) — NEAT speciation, genomic distance
- Baker et al. (2019) — OpenAI Hide-and-Seek, emergent phase detection
- Eysenbach et al. (2018) — DIAYN, mutual information diversity
- Goldberg & Richardson (1987) — Fitness sharing

## Commands

```bash
# Setup
./scripts/setup.sh                    # Create venv and install deps
pip install -e ".[dev]"               # Manual install with dev deps

# Training
python -m src.training.train --train.total-steps 100000   # Quick run
./scripts/train.sh --train.total-steps 10000000            # Full run (sets JAX flags)
# All CLI args map to Config dataclass fields via tyro (use --help for full list)
# CLI uses dashes: --env.grid-size 32 --train.learning-rate 0.001

# Train with specialization pressure
python -m src.training.train --specialization.diversity-bonus 0.1 --specialization.niche-pressure 0.05

# Resume from checkpoint
python -m src.training.train --train.resume-from checkpoints/params.pkl

# Tests
pytest tests/ -v                      # All tests (324 tests, ~60s timeout)
# NOTE: pytest lives in the venv. Use `.venv/bin/pytest` or `source .venv/bin/activate` first.
# Do NOT use bare `python -m pytest` or `python3 -m pytest` — they won't find the installed packages.
pytest tests/test_specialization.py -v # Specialization tests (largest: ~2275 lines, 13 classes)
pytest tests/test_integration.py -v    # End-to-end pipeline tests (needs more RAM)

# Type checking & linting
python -m mypy src/ --ignore-missing-imports
ruff check src/                       # Rules: E, F, I, N, W; ignores E501
black src/ tests/                     # Line-length 88

# Analysis
python -m src.analysis.ablation --checkpoint checkpoints/params.pkl
python scripts/run_ablation.py --iterations 100 --evolution     # Field x evolution 2x2
python scripts/run_specialization_ablation.py --iterations 100  # Divergent vs uniform vs random weights
python scripts/generate_specialization_report.py                # Full markdown report + PNGs

# Phase 5 Analysis
python scripts/compute_emergence_metrics.py --checkpoint <path> --output metrics.json
python scripts/run_stigmergy_ablation.py --checkpoint <path> --dry-run
python scripts/run_scaling_experiment.py --checkpoint <path> --dry-run
python scripts/run_baselines_comparison.py --dry-run
python scripts/generate_paper_figures.py --results-dir results/ --output-dir figures/
python scripts/verify_phase5.py
```

## Architecture

### Core Simulation Loop (each timestep)

```
1. Field diffuses (per-channel rates) + decays (per-channel rates)                    → dynamics.py
2. Agents observe: pos + energy + has_food + nest compass + field gradients            → obs.py
   (4-dir per channel) + temporal derivative + K=5 food = 45 dims
3. ActorCritic → action logits + value estimate                                       → network.py, policy.py
4. Agents move (5 actions: stay/up/down/left/right).                                  → env.py
   Laden agents alternate move/write (skip movement on write steps)
5. Food pickup: adjacent uncollected food → has_food=True, 5% scout sip energy        → env.py
6. Nest delivery: agent with has_food in nest area → has_food=False,                  → env.py
   95% food_energy as reward
7. Field writes: Ch0 recruitment written ONLY by laden agents on write steps.          → ops.py
   Ch1 territory written passively by ALL agents (+territory_write_strength).
   Cap to [0, field_value_cap]
8. Energy drains; agents at 0 energy die (has_food cleared on death)                  → env.py
9. Reproduction: automatic at nest when energy > threshold (no action needed)          → env.py, reproduction.py
```

*Legacy note: The pre-Phase 6 loop used 6 actions (including explicit reproduce), 502-dim observations with a raw 11x11 field patch, and unconditional field writes by all agents. That system is superseded by the biological pheromone design above.*

### Module Map

```
src/
├── configs.py                  # 9 nested dataclasses: Config > Env/Field/Agent/Train/Log/Analysis/Evolution/Specialization/Nest
├── environment/
│   ├── state.py                # EnvState (flax.struct.dataclass): positions, food, field, energy, alive, lineage, agent_params,
│   │                           #   has_food, prev_field_at_pos, laden_cooldown
│   ├── env.py                  # reset() + step() — 10-stage pipeline: movement, food pickup, nest delivery,
│   │                           #   field writes (success-gated), energy drain, death, auto-reproduction at nest
│   ├── obs.py                  # get_observations() → (max_agents, 45) = 2+1+1+2+20+4+15
│   ├── vec_env.py              # VecEnv: jax.vmap over num_envs
│   └── render.py               # RGB frame rendering (heatmap + agents + food)
├── field/
│   ├── field.py                # FieldState: values (H, W, C) float32
│   ├── dynamics.py             # per-channel diffuse() + per-channel decay() via array broadcasting
│   └── ops.py                  # read_local() + write_local() at agent positions, field value capping
├── agents/
│   ├── network.py              # ActorCritic: shared MLP (LayerNorm + tanh) → 5-action actor + scalar critic
│   ├── policy.py               # sample_actions() vmapped over (num_envs, num_agents)
│   └── reproduction.py         # mutate_params(), copy_agent_params(), per-layer mutation rates
├── training/
│   ├── train.py                # Entry point. create_train_state() + train_step() + train() loop
│   ├── rollout.py              # RunnerState + collect_rollout() via lax.scan
│   ├── gae.py                  # compute_gae() via reverse lax.scan
│   └── ppo.py                  # ppo_loss() with alive_mask, clipped surrogate
├── analysis/
│   ├── specialization.py       # THE BIG ONE (~1121 lines): weight divergence, behavioral features, clustering,
│   │                           #   specialization score, novelty score, field usage, lineage correlation,
│   │                           #   species detection, SpecializationTracker
│   ├── trajectory.py           # TrajectoryRecorder + record_episode()
│   ├── visualization.py        # 4 matplotlib plots: PCA/t-SNE clusters, divergence, field usage, score over time
│   ├── field_metrics.py        # field_entropy(), field_structure(), field_food_mi()
│   ├── emergence.py            # EmergenceTracker: rolling z-score phase transition detection
│   ├── ablation.py             # 6 field conditions: normal, zeroed, random, frozen, no_field, write_only
│   ├── lineage.py              # LineageTracker: birth/death records, family trees, dominant lineages
│   ├── o_information.py        # O-information via hoi (Omega < 0 = synergy) [Phase 5]
│   ├── pid_synergy.py          # PID synergy via dit (action + field → outcome) [Phase 5]
│   ├── causal_emergence.py     # Hoel's EI + Rosas Psi (macro vs micro) [Phase 5]
│   ├── surrogates.py           # Surrogate testing framework (row/col/block shuffle) [Phase 5]
│   ├── information.py          # Transfer entropy + division of labor [Phase 4]
│   ├── statistics.py           # rliable integration (IQM, bootstrap CI) [Phase 5]
│   ├── scaling.py              # Superlinear scaling analysis [Phase 5]
│   ├── paper_figures.py        # Publication figures (300 DPI, PDF+PNG) [Phase 5]
│   └── emergence_report.py     # Unified metrics computation + JSON output [Phase 5]
├── baselines/                  # [Phase 5]
│   ├── ippo.py                 # Independent PPO (no field, shared params)
│   ├── aco_fixed.py            # ACO-Fixed + ACO-Hybrid (hardcoded pheromone rules)
│   └── mappo.py                # MAPPO (centralized critic, CTDE)
├── experiments/                # [Phase 5]
│   ├── runner.py               # Multi-seed harness + paired experiments
│   ├── configs.py              # Environment configs (standard, hidden_resources, food_scarcity)
│   └── baselines.py            # Baselines comparison runner
└── utils/
    ├── logging.py              # W&B integration
    └── video.py                # Episode recording to MP4
```

### Key Data Structures

**`EnvState`** (flax.struct.dataclass — JAX pytree):
```python
agent_positions: (max_agents, 2) int32       # grid coordinates
food_positions: (num_food, 2) int32
food_collected: (num_food,) bool
field_state: FieldState                       # (H, W, C) float32
agent_energy: (max_agents,) float32
agent_alive: (max_agents,) bool
agent_ids: (max_agents,) int32               # unique per-agent
agent_parent_ids: (max_agents,) int32        # -1 if original
next_agent_id: scalar int32
agent_birth_step: (max_agents,) int32
agent_params: pytree with (max_agents, ...) leaves  # per-agent network weights
# Phase 6 (Biological Pheromone System) additions:
has_food: (max_agents,) bool                 # carrying food back to nest
prev_field_at_pos: (max_agents, num_channels) float32  # for temporal derivative in obs
laden_cooldown: (max_agents,) bool           # move/write alternation for laden agents
```

**`RunnerState`** (flax.struct.dataclass):
```python
params: Any                  # shared policy (Flax variables dict)
opt_state: Any               # optax optimizer state
env_state: EnvState          # batched (num_envs, ...)
last_obs: (num_envs, max_agents, obs_dim)
key: PRNG key
```

### Tensor Shape Conventions

| Context | Shape | Example |
|---------|-------|---------|
| Single env, per-agent | `(max_agents, ...)` | positions `(32, 2)` |
| Batched across envs | `(num_envs, max_agents, ...)` | obs `(32, 32, 45)` |
| Rollout batch | `(num_steps, num_envs, max_agents, ...)` | rewards `(128, 32, 32)` |
| Flattened for PPO | `(T * E * A, ...)` | `(131072, 45)` |
| Minibatch | `(minibatch_size, ...)` | `(256, 45)` |
| Field | `(H, W, C)` | `(20, 20, 4)` |
| Observation | `(max_agents, obs_dim)` | `(32, 45)` where 45 = 2+1+1+2+20+4+15 |

obs_dim breakdown: 2 (normalized x,y position) + 1 (normalized energy) + 1 (has_food flag) + 2 (nest compass with distance-dependent noise) + 20 (field spatial: center+N+S+E+W per 4 channels) + 4 (field temporal: dC/dt per channel) + 15 (K=5 nearest food: rel_x, rel_y, available)

*Legacy note: Pre-Phase 6 obs_dim was 502 = 3 (pos+energy) + 484 (11x11 field patch x 4ch) + 15 (food).*

## JAX Patterns (CRITICAL)

These patterns are non-negotiable. Violating them causes silent JIT failures or shape errors.

1. **Pure functional**: Every function takes state in, returns new state. No mutation. Use `flax.struct.dataclass` for all stateful objects.

2. **Static config**: Config values are captured in closures before JIT. The JIT boundary is `train_step(runner_state)` — config must NOT be a traced argument. Any new config fields must stay Python-level, not JAX-traced.

3. **Fixed array shapes**: JAX requires static shapes for JIT. Variable population is handled via `(max_agents,)` arrays + `agent_alive` boolean mask. Never dynamically resize arrays.

4. **vmap for parallelism**: `VecEnv` vmaps over `num_envs`. `sample_actions` vmaps over `(num_envs * num_agents)`. When adding new per-agent computation, vmap it.

5. **lax.scan for loops**: Rollout collection, GAE, PPO epochs/minibatches all use `jax.lax.scan`. Reproduction uses sequential `lax.scan` because spawning modifies shared state (next_agent_id, available slots).

6. **PRNG discipline**: Always split keys before use. Never reuse a key. Pattern: `key, subkey = jax.random.split(key)`.

7. **Alive masking in loss**: PPO loss uses `_masked_mean(x, mask)` to average only over alive agents. Any new loss terms must also respect the alive mask. Dead agents get zero observations, zero rewards, zero gradients.

8. **Per-agent params**: When evolution is enabled, `agent_params` is a pytree where every leaf has an extra leading `(max_agents,)` dimension. The shared PPO-updated params are periodically synced to alive agents' slots; dead slots retain their previous weights (preserving genetic diversity).

## Specialization System (Phase 3)

### How Specialization Works

1. **Weight divergence**: Pairwise cosine distance between agents' flattened weight vectors. Tracked over training by `SpecializationTracker`.

2. **Behavioral features** (7 per agent, extracted from trajectory):
   - movement_entropy, food_collection_rate, distance_per_step, reproduction_rate, mean_energy, exploration_ratio, action_stay_fraction

3. **Clustering**: K-means with StandardScaler normalization. `find_optimal_clusters()` tests k=2..5, picks best silhouette.

4. **Specialization score** (composite 0-1):
   - 50% silhouette component (cluster separation quality)
   - 25% weight divergence component (genetic differentiation)
   - 25% behavioral variance component (diverse strategies)

5. **Species detection**: Requires silhouette >= threshold (0.7) AND hereditary membership (>= 70% parent-child pairs in same cluster).

6. **Field usage analysis**: Classifies clusters as writers (high movement, spread deposits), readers (exploit known areas), or balanced.

### Specialization Training Levers

- `specialization.diversity_bonus`: Reward bonus proportional to cosine distance from population centroid (gradient pressure for unique weights)
- `specialization.niche_pressure`: Penalty for nearest-neighbor weight similarity (discourages convergence)
- `specialization.layer_mutation_rates`: Per-layer mutation std overrides (e.g., `{"Dense_0": 0.02}` for higher mutation in early layers)

## What Needs to Happen Next

### Phase 6 Immediate Priorities

1. **Complete pheromone system implementation**: Steps 7-12 of the build plan (food carrying, nest delivery, nest reproduction, success-gated field writes, new observations, integration test). See `docs/plans/2026-02-05-biological-pheromone-system.md` for full spec.

2. **Hyperparameter sweep**: 69-run sweep notebook is ready. Covers diffusion/decay rates, nest radius, food sip fraction, compass noise, territory write strength. Goal: find the config where field ON beats field OFF.

3. **Full 10M step experiment**: Take the best config from the sweep and run a proper long training. This is where trail formation and foraging loops should emerge.

4. **Pheromone system vs field-off baseline**: The moment of truth. If the biological pheromone system reverses the old result (field ON > field OFF), we have proof that the mechanism matters, not just the medium.

### Evidence Gaps (for the paper / investor demo)

1. **Visual "holy shit" moment**: Current rendering is basic (heatmap + dots). We need:
   - Timelapse of recruitment trails forming during a foraging episode
   - Side-by-side: pheromone system vs no-field agent behavior
   - Species visualization: color agents by cluster, show different movement patterns
   - Channel-separated visualization (recruitment trails vs territory map)

2. **Stronger specialization signal**: Need experiments with `diversity_bonus` and `niche_pressure` tuned to find the sweet spot under the new pheromone system.

3. **Phase transition documentation**: The `EmergenceTracker` and `SpecializationTracker` detect sudden metric changes — we need to capture and visualize these "moments of emergence."

4. **Scaling experiments**: Does emergence get STRONGER with more agents? More field channels? Larger grid? This is the key demo narrative: "look what happens when you scale."

### Future Ideas (Phase 7+)

- Channels 2-3: learned neural network writes (DIAL-style communication)
- Food quality variation (higher quality = stronger pheromone)
- Nest congestion mechanics (negative feedback when crowded)
- Lane formation visualization
- Multi-species interaction dynamics
- Competitive/cooperative species relationships
- Predator-prey dynamics
- Sexual reproduction (crossover)

## Development Conventions

### Code Style
- **Formatter**: black (line-length 88)
- **Linter**: ruff (rules: E, F, I, N, W; ignores E501)
- **Types**: mypy with `--ignore-missing-imports`
- **Tests**: pytest with 60s timeout per test

### File Conventions
- Every module function is stateless and JIT-compatible unless explicitly noted
- Config fields use underscores; CLI uses dashes (`--env.grid-size` → `env.grid_size`)
- All analysis functions operate on numpy arrays (not JAX); conversion happens at boundaries
- Visualization uses matplotlib with `Agg` backend (headless)

### Testing
- 324 tests across 11 test files
- `test_specialization.py` is the largest (~2275 lines, 13 test classes)
- `test_integration.py` runs end-to-end pipeline tests (may OOM on low-memory machines)
- Every new feature needs tests that pass before merge

### Checkpoint Format
```python
{
    "params": flax_params_dict,           # shared policy weights
    "agent_params": per_agent_pytree,     # (max_agents, ...) per-agent weights (if evolution enabled)
    "config": config_dict                 # serialized config
}
```
Saved via pickle to `checkpoints/params.pkl` every `save_interval` steps.

## Config Reference

### Critical Parameters for Emergence Experiments

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `env.grid_size` | 20 | Arena size — larger = harder coordination |
| `env.num_agents` | 8 | Starting population (grows to `max_agents` with evolution) |
| `env.num_food` | 10 | Resource scarcity — fewer food = more competition |
| `field.num_channels` | 4 | Field capacity — 4 channels: recruitment, territory, 2 reserved |
| `field.channel_diffusion_rates` | (0.5, 0.01, 0.0, 0.0) | Per-channel diffusion — Ch0 spreads wide, Ch1 stays local |
| `field.channel_decay_rates` | (0.05, 0.0001, 0.0, 0.0) | Per-channel decay — Ch0 fades fast, Ch1 near-permanent |
| `field.territory_write_strength` | 0.01 | Passive territory deposit per agent per step (Ch1) |
| `field.field_value_cap` | 1.0 | Maximum field value per cell (clamped after writes) |
| `field.diffusion_rate` | 0.1 | Legacy fallback — used if channel_diffusion_rates not set |
| `field.decay_rate` | 0.05 | Legacy fallback — used if channel_decay_rates not set |
| `nest.radius` | 2 | Nest half-width (radius=2 gives 5x5 nest area) |
| `nest.food_sip_fraction` | 0.05 | Immediate energy on food pickup (5% scout sip) |
| `nest.compass_noise_rate` | 0.10 | Path integration error — 10% noise per grid_size distance |
| `evolution.mutation_std` | 0.01 | Mutation intensity — higher = faster divergence, less stability |
| `evolution.reproduce_threshold` | 150 | How hard it is to reproduce — higher = more selective |
| `evolution.max_agents` | 32 | Population cap (also array dimension for JAX) |
| `specialization.diversity_bonus` | 0.0 | Reward for unique weights (0 = disabled) |
| `specialization.niche_pressure` | 0.0 | Penalty for similar weights (0 = disabled) |
| `train.total_steps` | 10M | Training duration |
| `train.num_envs` | 32 | Parallel environments (more = better gradient estimates) |

### Survival-Friendly Params (from overnight runs)

For runs where you want robust population dynamics:
```bash
--env.num-food 20 --evolution.starting-energy 200 --evolution.food-energy 100 \
--evolution.reproduce-threshold 120 --evolution.reproduce-cost 50
```
