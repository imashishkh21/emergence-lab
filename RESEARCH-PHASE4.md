# Phase 4 Research Summary

Compiled from 4 research agents. Full reports available in session transcripts.

---

## 1. Visualization Stack (research-visualization + research-web-stack)

### Recommended Stack
| Layer | Choice | Why |
|-------|--------|-----|
| Backend | FastAPI + WebSocket | Native async, typed |
| Protocol | MessagePack (binary) | 3-10x smaller than JSON |
| Frontend | Svelte 5 | Compiled reactivity, tiny runtime |
| Rendering | Pixi.js v8 | WebGPU-ready, 60fps with 1000+ sprites |
| Charts | Plotly.js | WebGL, real-time trace updates |

### Key Patterns

**Binary WebSocket streaming:**
```python
# Server
await websocket.send_bytes(msgpack.packb({
    'step': step,
    'positions': positions.tobytes(),
    'metrics': {'reward': float(reward)}
}))
```

```javascript
// Client
ws.onmessage = (e) => {
    const frame = msgpack.decode(new Uint8Array(e.data));
    updateCanvas(frame.positions);
};
```

**Pixi.js sprite pooling:**
```javascript
const container = new PIXI.ParticleContainer(100);
const sprites = Array(64).fill().map(() => {
    const s = new PIXI.Sprite(texture);
    container.addChild(s);
    return s;
});
```

### Why NOT Streamlit/Panel/Gradio
- They're for demos, not 60fps real-time
- No native WebSocket support
- Can't handle high-frequency updates

---

## 2. Gradient Isolation (research-gradient-isolation)

### The Problem
PPO shared gradients homogenize all agent weights.
```
37 gradient updates >> subtle mutations → gradients win → no divergence
```

### Solutions (Priority Order)

**1. Agent-Specific Heads (Easiest)**
```python
class SharedBackboneNetwork:
    def __init__(self, n_agents):
        self.encoder = MLP(...)  # Shared
        self.agent_heads = [MLP(...) for _ in range(n_agents)]  # Separate
    
    def forward(self, obs, agent_id):
        features = self.encoder(obs)
        return self.agent_heads[agent_id](features)
```

**2. Agent ID Embedding**
```python
self.agent_embedding = nn.Embedding(n_agents, embed_dim)
obs_with_id = concat([obs, self.agent_embedding(agent_id)])
```

**3. Freeze-Evolve Cycles**
- GRADIENT phase (10K steps): Normal PPO training
- EVOLVE phase (1K steps): Freeze gradients, pure evolution
- Alternate repeatedly

**4. PGA-ME (Advanced)**
- Policy Gradient Assisted MAP-Elites
- Combines gradient updates with behavioral archive
- Already implemented in QDax library

### QDax Integration
```python
from qdax.core.map_elites import MAPElites
from qdax.core.emitters.pga_me_emitter import PGAMEEmitter

emitter = PGAMEEmitter(...)  # Uses policy gradients + archive
map_elites = MAPElites(emitter=emitter, ...)
```

---

## 3. Emergence Metrics (research-emergence-metrics)

### Priority 1: Core Metrics

**Transfer Entropy** — Directed information flow
```python
# If knowing Agent A's past helps predict Agent B's future
# → information is flowing from A to B
TE(A→B) = H(B_future | B_past) - H(B_future | B_past, A_past)
```

**Division of Labor Index**
```python
# 0 = everyone does everything
# 1 = perfect specialization
DOL = 1 - (1/K) * Σ|p_i - 1/K|
```

**Behavioral Diversity (Shannon)**
```python
H = -Σ p_i * log(p_i)
# Where p_i = proportion of agents with behavior type i
```

### Priority 2: Phase Transition Detection

Signs of emergence:
- **Susceptibility spike** — system becomes hypersensitive
- **Critical slowing down** — takes longer to recover from perturbations
- **Power-law distributions** — scale-free behavior

```python
if susceptibility > 3*std AND autocorrelation_increasing:
    flag_phase_transition()
```

### Priority 3: Integrated Information (Φ)

Measures "more than sum of parts" — if cutting the system in half loses information, it's truly integrated.

### Lessons from ALife History

| Project | Key Finding |
|---------|-------------|
| Tierra | Complexity didn't grow over time (open problem) |
| Avida | Complex features evolved incrementally |
| Polyworld | Real-time viz was key to its impact |
| Lenia | Continuous space enables resilient patterns |

**Critical insight:** True open-ended emergence is RARE. Most systems plateau.

### What Makes a Compelling Demo

1. **Show before/after** — random → organized
2. **Perturbation response** — disturb, watch recovery
3. **The surprise factor** — behaviors that weren't programmed
4. **Real-time** — watching it happen, not just results

---

## 4. Implementation Recommendations

### Phase 4 Priority Order

1. **Agent-specific heads** — Simplest fix for homogenization
2. **Basic dashboard** — See what's happening
3. **Freeze-evolve** — Alternative training mode
4. **Transfer entropy** — Quantify coordination
5. **DOL index** — Quantify specialization
6. **Phase detection** — Catch emergence moments
7. **MAP-Elites archive** — Maintain diversity (if needed)

### Tech Stack Summary

```
Training: JAX + Flax + FastAPI + msgpack
Dashboard: Svelte 5 + Pixi.js 8 + Plotly.js + msgpack-lite
Communication: Binary WebSocket @ 30Hz
```

### Expected Timeline

- Week 1: Architecture fix (agent heads, freeze-evolve)
- Week 2: Dashboard core (canvas, charts)
- Week 3: Metrics (TE, DOL, phase detection)
- Week 4: Integration, polish, help system

---

## References

### Key Papers
- Population Based Training (DeepMind, 2017)
- MAP-Elites (Mouret & Clune, 2015)
- PGA-ME (2021) — Policy gradients + QD
- QDPG (2022) — Diversity policy gradient
- QDax (2024) — JAX-accelerated QD library

### Libraries
- QDax: github.com/adaptive-intelligent-robotics/QDax
- Pixi.js v8: pixijs.com
- Svelte 5: svelte.dev
- FastAPI: fastapi.tiangolo.com
