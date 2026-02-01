# AGENTS.md — Claude Code Context

## What This Project Is

**Emergence Lab** tests whether collective intelligence can emerge in a shared medium between agents.

Unlike typical multi-agent RL (agents coordinate), we add a **learnable field** that:
- Agents read/write locally
- Has its own dynamics (diffusion, decay)
- Is trained end-to-end with the agents

The hypothesis: The field could develop structures that encode collective knowledge.

## Tech Stack

- **JAX/Flax** — ML framework
- **Optax** — Optimizers
- **Python 3.10+** — Runtime

## Build Commands

```bash
# Setup
./scripts/setup.sh

# Activate
source .venv/bin/activate

# Run tests
pytest tests/ -v

# Type check
python -m mypy src/

# Train
python -m src.training.train --config configs/default.yaml
```

## Architecture

```
src/
├── configs.py       # Dataclass configs
├── environment/     # Grid world + food
├── field/           # Shared medium (THE KEY INNOVATION)
├── agents/          # Neural network policies
├── training/        # PPO implementation
└── analysis/        # Emergence detection
```

## JAX Patterns Required

1. **Pure functions** — no side effects
2. **Explicit PRNG** — split keys before use
3. **vmap over agents** — shared weights
4. **JIT everything** — wrap training step
5. **Flax struct.dataclass** — for pytree states

## The Field

The field is a 2D array (H, W, C) where C = num_channels.

Each timestep:
1. Field diffuses (3x3 blur)
2. Field decays (multiply by 1-rate)
3. Agents read local values (observation)
4. Agents write local values (action effect)

This is what makes us different from OpenAI's hide-and-seek.

## Current PRD Task

Check `PRD.md` for the current story. Check `progress.txt` for status.

## Testing

Every story must pass its acceptance test. Run:
```bash
pytest tests/test_<module>.py::test_<name> -v
```
