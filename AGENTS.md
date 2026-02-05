# AGENTS.md — Claude Code Context

## What This Project Is

**Emergence Lab** tests whether collective intelligence can emerge in a shared medium between agents.

Unlike typical multi-agent RL (agents coordinate), we add a **learnable field** that:
- Agents read/write locally
- Has its own dynamics (diffusion, decay)
- Is trained end-to-end with the agents

The hypothesis: The field could develop structures that encode collective knowledge.

## Phase 5: Prove Emergence

We're building **rigorous evidence** that emergence is real:
1. Information-theoretic metrics (PID synergy, O-information, causal emergence)
2. Baselines to beat (IPPO, ACO-Fixed, ACO-Hybrid, MAPPO)
3. Statistical significance (N=20 seeds, IQM + bootstrap CI via rliable)
4. Publication-ready figures and experiment scripts

### Key Decisions (FOLLOW THESE)
- **`dit` for PID** — Gold standard discrete information theory library
- **`hoi` for O-information** — JAX-native, O(n) scaling
- **`rliable` for statistics** — Google standard (Agarwal 2021)
- **K=2 quantile bins + Jeffreys smoothing (alpha=0.5)** for discretization
- **20 seeds minimum** for all experiments
- **Hidden food: K=3 agents, D=3 distance, 5x value, 10 step duration**
- **MAPPO: custom ~200 lines, NOT JaxMARL wrapper**
- **ACO params: alpha=1.0, beta=2.0, rho=0.5, Q=1.0** (Dorigo & Stutzle 2004)
- **Windowed analysis: 1M-step windows, 50% overlap** for non-stationarity
- **Phase 5 deps go in `[project.optional-dependencies]` under `phase5` group** — don't break existing installs

### What Already Exists (from Phases 1-4B)
- `src/analysis/information.py` — Transfer Entropy + KSG estimator (working)
- `src/analysis/specialization.py` — Weight divergence, clustering, species detection (working)
- `src/analysis/emergence.py` — Phase transition detection, EmergenceTracker (working)
- `src/analysis/ablation.py` — 3 field conditions (normal/zeroed/random) + 2x2 evolution grid + specialization ablation (working)
- `src/analysis/field_metrics.py` — Field entropy, structure, mutual information (working)
- `src/analysis/trajectory.py` — TrajectoryRecorder (working)
- `src/analysis/visualization.py` — Matplotlib plots (working)
- `src/training/checkpointing.py` — Full state save/load (working)
- `src/server/` + `dashboard/` — Live visualization (working)
- `src/agents/network.py` — ActorCritic + AgentSpecificActorCritic (working)
- `src/environment/env.py` — reset() + step() with reproduction (working)
- `src/environment/state.py` — EnvState flax dataclass (working)
- `src/configs.py` — 10 nested config dataclasses (working)

### Important: Backward Compatibility
- Hidden food config must be **disabled by default** — all existing tests must still pass
- Phase 5 dependencies are **optional** — use `@pytest.mark.skipif` guards
- New ablation conditions extend existing `ablation.py`, don't rewrite it
- New baselines reuse existing `ActorCritic` and `step()` — minimal new code

## Tech Stack

### Training (Python)
- **JAX/Flax** — ML framework (pure functional, JIT, vmap)
- **Optax** — Optimizers
- **FastAPI** — WebSocket server
- **msgpack** — Binary serialization

### Phase 5 Additions
- **dit** — Discrete information theory (PID, mutual information)
- **hoi** — Higher-order interactions (O-information, JAX-native)
- **rliable** — Statistical reporting (IQM, bootstrap CI, performance profiles)
- **jaxmarl** — Reference only (we write custom MAPPO)

## Build Commands

```bash
# Setup Python
./scripts/setup.sh
source .venv/bin/activate

# Install Phase 5 deps
pip install -e ".[phase5]"

# Run tests
pytest tests/ -v

# Type check
python -m mypy src/ --ignore-missing-imports

# Lint
ruff check src/

# Start training
python -m src.training.train --train.total-steps 100000
```

## Architecture

```
src/
├── configs.py           # Dataclass configs (add HiddenFoodConfig)
├── environment/
│   ├── state.py         # EnvState (add hidden food fields)
│   ├── env.py           # reset() + step() (add hidden food logic)
│   ├── obs.py           # Observations (502-dim)
│   └── vec_env.py       # Vectorized environment
├── field/               # Shared medium (THE KEY INNOVATION)
│   ├── field.py         # FieldState
│   ├── dynamics.py      # diffuse() + decay()
│   └── ops.py           # read_local() + write_local()
├── agents/
│   ├── network.py       # ActorCritic + AgentSpecificActorCritic
│   ├── policy.py        # sample_actions(), get_deterministic_actions()
│   └── reproduction.py  # mutate_agent_params()
├── training/
│   ├── train.py         # PPO + freeze-evolve loop
│   ├── rollout.py       # collect_rollout()
│   ├── gae.py           # compute_gae()
│   ├── ppo.py           # ppo_loss() with alive masking
│   └── checkpointing.py # Full state save/load
├── analysis/
│   ├── specialization.py  # Clustering, species detection
│   ├── emergence.py       # EmergenceTracker, phase transitions
│   ├── information.py     # Transfer entropy (existing)
│   ├── ablation.py        # Field/evolution/specialization ablation (extend)
│   ├── field_metrics.py   # Entropy, structure, MI
│   ├── trajectory.py      # TrajectoryRecorder
│   ├── visualization.py   # Matplotlib plots
│   ├── archive.py         # MAP-Elites
│   ├── o_information.py   # NEW Phase 5: O-information via hoi
│   ├── pid_synergy.py     # NEW Phase 5: PID synergy via dit
│   ├── causal_emergence.py # NEW Phase 5: EI + Rosas Psi
│   ├── surrogates.py      # NEW Phase 5: Surrogate testing framework
│   ├── statistics.py      # NEW Phase 5: rliable integration
│   ├── scaling.py         # NEW Phase 5: Superlinear scaling analysis
│   ├── paper_figures.py   # NEW Phase 5: Publication figures
│   └── emergence_report.py # NEW Phase 5: Unified metrics report
├── baselines/             # NEW Phase 5
│   ├── __init__.py
│   ├── ippo.py            # Independent PPO (no field, no evolution)
│   ├── aco_fixed.py       # ACO-Fixed + ACO-Hybrid
│   └── mappo.py           # MAPPO (centralized critic)
├── experiments/           # NEW Phase 5
│   ├── __init__.py
│   ├── runner.py          # Multi-seed experiment harness
│   ├── configs.py         # 3 env configs (standard, hidden, scarce)
│   └── baselines.py       # Baselines comparison runner
├── server/
│   ├── main.py            # FastAPI app
│   ├── streaming.py       # TrainingBridge
│   └── replay.py          # Session recorder/player
└── utils/
    ├── logging.py         # W&B integration
    └── video.py           # MP4 recording

scripts/
├── ralph.sh                      # Ralph loop runner
├── run_stigmergy_ablation.py     # NEW Phase 5: 6-condition ablation
├── run_scaling_experiment.py     # NEW Phase 5: Superlinear scaling
├── run_baselines_comparison.py   # NEW Phase 5: Full baselines comparison
├── compute_emergence_metrics.py  # NEW Phase 5: All metrics from checkpoint
├── generate_paper_figures.py     # NEW Phase 5: Publication figures
└── verify_phase5.py              # NEW Phase 5: End-to-end verification
```

## Key Patterns

### JAX Patterns
1. **Pure functions** — no side effects, return new state
2. **Explicit PRNG** — split keys before use, never reuse
3. **vmap over agents** — batched operations
4. **JIT everything** — config must be static (captured in closures)
5. **Fixed array shapes** — `(max_agents,)` + `agent_alive` mask
6. **Alive masking in loss** — `_masked_mean(x, mask)` excludes dead agents

### Tracker Pattern (for new metrics)
All trackers follow the same structure (see TransferEntropyTracker):
```python
class MyTracker:
    def __init__(self, window_size=20, z_threshold=3.0):
        self.history = {"metric_name": []}
        self.steps = []
        self.events = []

    def update(self, data, step) -> list[Event]:
        # Compute metric, detect phase transitions via z-score
        ...

    def get_metrics(self) -> dict[str, float]:
        # Return current values for logging
        ...

    def get_summary(self) -> dict[str, Any]:
        # Return overall statistics
        ...
```

### Standardized Result Dict (for baselines)
All baseline evaluation functions return the same format:
```python
{
    "total_reward": float,
    "food_collected": float,
    "final_population": int,
    "per_agent_rewards": list[float],
}
```

### EnvState Modification Pattern (for hidden food)
When adding fields to EnvState (flax.struct.dataclass), you must update:
1. The dataclass definition in `state.py`
2. `create_env_state()` in `state.py`
3. `reset()` in `env.py`
4. Every `EnvState(...)` constructor call in `ablation.py` (_replace_field, _reset_energy, _replace_agent_params)
5. The `step()` function if the new field is modified each timestep

## Testing

Every story must pass its acceptance criteria. Run:
```bash
# Specific test
pytest tests/test_<module>.py -v

# All tests (must pass before committing)
pytest tests/ -v

# Type check
python -m mypy src/ --ignore-missing-imports

# Lint
ruff check src/
```

## Glossary Reference

See PRD.md for the full technical reference (formulas, parameters, etc.).
