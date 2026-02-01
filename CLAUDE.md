# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Emergence Lab is a JAX-based multi-agent reinforcement learning environment where agents interact through a shared learnable field (stigmergy). Agents forage for food on a 2D grid while reading/writing to a shared multi-channel field that has its own physics (diffusion + decay). The field is trained end-to-end with agents via PPO. The hypothesis: the field develops spatial structures encoding collective knowledge.

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

# Tests
pytest tests/ -v                      # All tests
pytest tests/test_env.py -v           # Single test file
pytest tests/test_env.py::TestEnvStep::test_step -v  # Single test

# Type checking
python -m mypy src/ --ignore-missing-imports

# Linting / formatting
ruff check src/                       # Lint (rules: E, F, I, N, W; ignores E501)
black src/ tests/                     # Format (line-length 88)

# Analysis
python -m src.analysis.ablation --checkpoint checkpoints/params.pkl
```

## Architecture

### Core Data Flow

Each timestep follows this sequence:
1. Field diffuses (3x3 Gaussian blur) and decays (`dynamics.py`)
2. Agents observe local field patch + K nearest food + own position (`obs.py`)
3. ActorCritic network produces action logits + value (`network.py`, `policy.py`)
4. Agents move on grid (5 actions: stay/up/down/left/right) (`env.py`)
5. Agents write presence to field at their location (`ops.py`)
6. Food collected when agent within 1 cell Chebyshev distance; shared team reward

### Module Responsibilities

- **`src/configs.py`** — All configuration as nested dataclasses (`Config` > `EnvConfig`, `FieldConfig`, `AgentConfig`, `TrainConfig`, `LogConfig`, `AnalysisConfig`). Tyro parses CLI args directly from these. Also supports YAML via `Config.from_yaml()`.
- **`src/environment/`** — Pure-functional env: `reset()` and `step()` take and return `EnvState` (a `flax.struct.dataclass`). `VecEnv` wraps these with `jax.vmap` for parallel simulation. `obs.py` constructs per-agent observations (position + field patch + K=5 nearest food).
- **`src/field/`** — `FieldState` holds a `(H, W, C)` tensor. `dynamics.py` applies diffusion (manual 3x3 conv via shifted sums) then decay. `ops.py` handles local read/write at agent positions.
- **`src/agents/`** — `ActorCritic` is a Flax `nn.Module` with shared MLP backbone (LayerNorm + tanh), orthogonal init. Actor head (scale 0.01) outputs 5-action logits; critic head (scale 1.0) outputs scalar value.
- **`src/training/`** — `train.py` is the entry point. `rollout.py` collects trajectories via `jax.lax.scan`. `gae.py` computes advantages with reverse `lax.scan`. `ppo.py` implements clipped surrogate loss. The entire `train_step` is JIT-compiled.
- **`src/analysis/`** — `EmergenceTracker` monitors field entropy/structure over training, detects phase transitions via z-score. `ablation.py` compares normal vs zeroed vs random field conditions. `field_metrics.py` computes entropy and spatial structure metrics.

### Key Patterns

- **Everything is JAX-functional**: state is passed through and returned, no mutation. All env/field state uses `flax.struct.dataclass` for JAX pytree compatibility.
- **Parallelization via vmap**: `VecEnv` vmaps over `num_envs` independent environments. `sample_actions` vmaps the network over `(num_envs, num_agents)`.
- **Trajectory collection via `lax.scan`**: both rollout collection and GAE computation use `jax.lax.scan` for JIT-compatible loops.
- **Config is static** (not traced by JAX): config values are captured in closures before JIT. The JIT boundary is `train_step(runner_state)` with `config` in the closure.
- **`RunnerState`** carries all mutable state (params, opt_state, env_state, last_obs, PRNG key) across training iterations as a `flax.struct.dataclass`.
- **Tensor shape conventions**: `(num_envs, num_agents, ...)` for batched data; `(num_steps, num_envs, num_agents, ...)` for rollout batches; field is `(H, W, C)`.
- **Actions**: integer 0-4 mapping to stay/up/down/left/right. Movement uses a delta lookup table clipped to grid boundaries.
- **Reward**: shared team reward (+1 per food collected this step, broadcast to all agents).
