# AGENTS.md — Claude Code Context

## What This Project Is

**Emergence Lab** tests whether collective intelligence can emerge in a shared medium between agents.

Unlike typical multi-agent RL (agents coordinate), we add a **learnable field** that:
- Agents read/write locally
- Has its own dynamics (diffusion, decay)
- Is trained end-to-end with the agents

The hypothesis: The field could develop structures that encode collective knowledge.

## Phase 4: Research Microscope

We're building a **live visualization dashboard** to:
1. Watch agents in real-time (Pixi.js)
2. Fix gradient homogenization (agent-specific heads)
3. Measure emergence (transfer entropy, division of labor)

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
python -m src.server.main

# Setup Dashboard (one-time)
cd dashboard
npm install

# Run Dashboard
cd dashboard
npm run dev
# Opens http://localhost:5173
```

## Architecture

```
src/
├── configs.py           # Dataclass configs
├── environment/         # Grid world + food
├── field/               # Shared medium (THE KEY INNOVATION)
├── agents/
│   └── network.py       # Neural networks (ADD agent-specific heads)
├── training/
│   └── train.py         # PPO + freeze-evolve (MODIFY)
├── analysis/
│   ├── specialization.py  # Clustering, metrics (ADD DOL)
│   ├── information.py     # NEW: transfer entropy
│   └── archive.py         # NEW: MAP-Elites
└── server/                # NEW: WebSocket streaming
    ├── main.py            # FastAPI app
    └── streaming.py       # Training bridge

dashboard/                 # NEW: Svelte web app
├── src/
│   ├── App.svelte
│   └── lib/
│       ├── AgentCanvas.svelte   # Pixi.js renderer
│       ├── MetricsPanel.svelte  # Plotly charts
│       ├── ControlPanel.svelte  # Play/pause, sliders
│       ├── GlossaryPanel.svelte # Help for laypeople
│       └── ...
└── package.json
```

## Key Patterns

### JAX Patterns
1. **Pure functions** — no side effects
2. **Explicit PRNG** — split keys before use
3. **vmap over agents** — batched operations
4. **JIT everything** — wrap training step

### Dashboard Patterns
1. **Reactive stores** — Svelte runes ($state, $derived)
2. **Binary streaming** — MessagePack for efficiency
3. **Object pooling** — Pre-allocate Pixi sprites
4. **Tooltips everywhere** — Help for laypeople

## The Gradient Homogenization Problem

**Problem:** PPO shared gradients push all agent weights together.

**Solution 1: Agent-Specific Heads**
```python
class AgentSpecificNetwork(nn.Module):
    shared_encoder: nn.Module  # Same for all agents
    agent_heads: List[nn.Module]  # Different for each agent
    
    def __call__(self, obs, agent_id):
        features = self.shared_encoder(obs)
        return self.agent_heads[agent_id](features)
```

**Solution 2: Freeze-Evolve Cycles**
- GRADIENT phase: Normal PPO training
- EVOLVE phase: Freeze gradients, only evolution (reproduction + mutation)
- Alternate every N steps

## Testing

Every story must pass its acceptance test. Run:
```bash
pytest tests/test_<module>.py::test_<name> -v
```

For dashboard tests (requires Playwright):
```bash
npx playwright test
```

## Glossary Reference

See PRD.md for the full glossary. Key terms:
- **Emergence** — Complex behavior from simple rules
- **Specialization** — Agents developing different roles
- **Transfer Entropy** — Information flow between agents
- **Division of Labor** — How well tasks are split among agents
