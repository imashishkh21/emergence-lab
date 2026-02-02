# AGENTS.md — Claude Code Context

## What This Project Is

**Emergence Lab** tests whether collective intelligence can emerge in a shared medium between agents.

Unlike typical multi-agent RL (agents coordinate), we add a **learnable field** that:
- Agents read/write locally
- Has its own dynamics (diffusion, decay)
- Is trained end-to-end with the agents

The hypothesis: The field could develop structures that encode collective knowledge.

## Phase 4B: Kaggle Infrastructure

We're building **autonomous training infrastructure** to:
1. Wire training to dashboard (real data, not mock)
2. Fix checkpointing (save full state: optimizer, PRNG, step, trackers)
3. Kaggle CLI automation (Titan can run training 24/7)

### Key Decisions (FOLLOW THESE)
- **Enhanced pickle, NOT Orbax** — Orbax has cross-platform GPU→CPU bugs
- **Kaggle only** — Don't mention Colab
- **Save every 100k steps** — Keep last 5 checkpoints with rotation
- **Signal handlers** — Save emergency checkpoint on SIGTERM/SIGINT

### What Already Exists (from Phase 4)
- `src/server/streaming.py` — `TrainingBridge` with `publish_frame()` method
- `src/analysis/emergence.py` — `EmergenceTracker` (needs serialization)
- `src/analysis/specialization.py` — `SpecializationTracker` (needs serialization)
- `config.log.save_interval` — Exists but unused
- `config.log.checkpoint_dir` — Exists
- Dashboard works with mock data via `python -m src.server.main --mock`

## Tech Stack

### Training (Python)
- **JAX/Flax** — ML framework
- **Optax** — Optimizers
- **FastAPI** — WebSocket server
- **msgpack** — Binary serialization

### Dashboard (Web)
- **Svelte 5** — Reactive UI framework
- **Pixi.js v8** — WebGL/WebGPU rendering
- **Plotly.js** — Charts
- **msgpack-lite** — Decode binary frames

## Build Commands

```bash
# Setup Python
./scripts/setup.sh
source .venv/bin/activate

# Run tests
pytest tests/ -v

# Type check
python -m mypy src/ --ignore-missing-imports

# Start training with visualization server
python -m src.server.main --mock  # Mock data (works now)
python -m src.training.train --server  # Real data (after US-001)

# Dashboard
cd dashboard && npm run dev
# Opens http://localhost:5173
```

## Architecture

```
src/
├── configs.py           # Dataclass configs
├── environment/         # Grid world + food
├── field/               # Shared medium (THE KEY INNOVATION)
├── agents/
│   └── network.py       # Neural networks with agent-specific heads
├── training/
│   ├── train.py         # PPO + freeze-evolve
│   └── checkpointing.py # NEW in Phase 4B: full state save/load
├── analysis/
│   ├── specialization.py  # Clustering, metrics
│   ├── emergence.py       # EmergenceTracker (add serialization)
│   ├── information.py     # Transfer entropy
│   └── archive.py         # MAP-Elites
└── server/
    ├── main.py            # FastAPI app
    └── streaming.py       # TrainingBridge

dashboard/                 # Svelte web app
├── src/
│   ├── App.svelte
│   └── lib/
│       ├── AgentCanvas.svelte
│       ├── MetricsPanel.svelte
│       ├── ControlPanel.svelte
│       └── ...
└── package.json

notebooks/
└── kaggle_training.ipynb  # NEW in Phase 4B: Kaggle notebook

scripts/
├── ralph.sh               # Ralph loop runner
├── kaggle_setup.sh        # NEW: Kaggle CLI setup
├── kaggle_push.sh         # NEW: Push to Kaggle
├── kaggle_run.sh          # NEW: Start Kaggle run
├── kaggle_status.sh       # NEW: Check run status
└── kaggle_download.sh     # NEW: Download results
```

## Key Patterns

### JAX Patterns
1. **Pure functions** — no side effects
2. **Explicit PRNG** — split keys before use
3. **vmap over agents** — batched operations
4. **JIT everything** — wrap training step

### Checkpointing Pattern (Phase 4B)
```python
# Save full state
checkpoint = {
    'params': params,
    'opt_state': opt_state,
    'agent_params': agent_params,
    'prng_key': prng_key,
    'step': step,
    'config': config,
    'tracker_state': tracker.to_dict()
}
# Convert JAX arrays to numpy for cross-platform
save_checkpoint(path, checkpoint)
```

## Testing

Every story must pass its acceptance test. Run:
```bash
pytest tests/test_<module>.py::test_<name> -v
```

## Glossary Reference

See PRD.md for the full glossary.
