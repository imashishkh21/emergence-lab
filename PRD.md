# Phase 4 PRD: Research Microscope

## Vision

Build a live visualization dashboard that lets us **see what's actually happening** in the simulation, and fix the gradient homogenization problem so specialization can emerge.

**End Goal:** Watch agents differentiate into distinct roles (Scouts/Followers/Hoarders) in real-time, with metrics proving collective intelligence emerges.

---

## 📚 Glossary (Explain Like I'm 5)

Every technical term used in this project, explained simply:

### Core Concepts

| Term | Simple Explanation | Analogy |
|------|-------------------|---------|
| **Agent** | A little creature in our simulation that moves around, eats food, and tries to survive | Like a fish in a fish tank |
| **Neural Network** | The "brain" inside each agent that decides what to do | Like a recipe book that tells the agent "if you see food, go towards it" |
| **Weights** | The numbers inside the brain that determine behavior | Like the ingredient amounts in a recipe — different amounts = different results |
| **Training** | Teaching agents to be better at surviving by adjusting their brain numbers | Like practicing a sport — you get better over time |
| **Evolution** | Agents that do well have babies, babies inherit (slightly modified) brains | Like how puppies look like their parents but aren't identical |
| **Mutation** | Small random changes to a baby's brain | Like a typo when copying a recipe — sometimes makes it better! |
| **Specialization** | Different agents learning to do different jobs | Like in a soccer team: goalie, defender, striker — each has a role |
| **Emergence** | When simple rules create complex, unexpected behavior | Like how ants following simple rules build amazing colonies |

### The Problem We're Solving

| Term | Simple Explanation | Why It Matters |
|------|-------------------|----------------|
| **Gradient Homogenization** | All agents becoming identical copies because training pushes them all the same way | It's like if every soccer player became a striker — no one to defend! |
| **Shared Weights** | All agents using the same brain | Efficient, but means they all act the same |
| **Agent-Specific Heads** | Each agent gets their own "decision-making" part on top of a shared "perception" part | Like siblings with the same eyes but different personalities |
| **Freeze-Evolve** | Sometimes we stop training and just let evolution (births/deaths/mutations) run | Like alternating between practice (training) and natural selection (evolution) |

### Metrics (How We Measure Success)

| Term | Simple Explanation | What Good Looks Like |
|------|-------------------|---------------------|
| **Weight Divergence** | How different are the agents' brains from each other? | High = agents have different "personalities" ✅ |
| **Specialization Score** | How clearly can we group agents by behavior? | High = clear roles (scouts, followers, etc.) ✅ |
| **Transfer Entropy** | Is one agent's behavior influencing another's? | High = agents are communicating/coordinating ✅ |
| **Division of Labor** | Are different agents doing different jobs? | High = like a well-organized team ✅ |
| **Phase Transition** | A sudden shift from "no organization" to "organized" | Like water freezing — sudden change! |

### Dashboard Terms

| Term | Simple Explanation |
|------|-------------------|
| **Real-time** | You see it happening as it happens (no delay) |
| **Heatmap** | Colors showing intensity — red = lots happening, blue = not much |
| **Lineage Tree** | Family tree showing who is whose parent/child |
| **Cluster** | A group of agents that behave similarly |

---

## 🎯 Success Criteria

1. **Weight divergence > 0.1** — Agents have genuinely different brains
2. **Specialization score > 0.7** — Clear behavioral groups emerge
3. **All agents alive** — Population doesn't collapse
4. **30+ fps dashboard** — Smooth, real-time visualization
5. **Layman can understand** — Non-technical person can use dashboard and understand what's happening

---

## 🛠️ Key Problems to Solve

### Problem 1: Gradient Homogenization
**What's happening:** All agents become identical because training pushes them all the same direction.
**Solution:** Agent-specific heads + Freeze-Evolve cycles

### Problem 2: Flying Blind
**What's happening:** We can only see log files, not actual agent behavior.
**Solution:** Live visualization dashboard with Pixi.js

### Problem 3: No Emergence Metrics
**What's happening:** We can't tell if/when emergence is actually happening.
**Solution:** Implement Transfer Entropy, Division of Labor, Phase Transition detection

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING (Python/JAX)                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Agents learn, evolve, specialize                    │   │
│  │  → Positions, behaviors, metrics generated           │   │
│  └───────────────────────┬─────────────────────────────┘   │
│                          │                                   │
│  ┌───────────────────────▼─────────────────────────────┐   │
│  │  FastAPI WebSocket Server                            │   │
│  │  → Streams data to dashboard at 30fps                │   │
│  └───────────────────────┬─────────────────────────────┘   │
└──────────────────────────┼──────────────────────────────────┘
                           │
                     WebSocket
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    DASHBOARD (Web Browser)                   │
│  ┌─────────────┐ ┌─────────────┐ ┌───────────────────────┐  │
│  │  Agent      │ │  Metrics    │ │  Controls & Help      │  │
│  │  Canvas     │ │  Charts     │ │  (with tooltips!)     │  │
│  │  (Pixi.js)  │ │  (Plotly)   │ │                       │  │
│  └─────────────┘ └─────────────┘ └───────────────────────┘  │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Glossary Panel — Click any term to learn more!      │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

---

## 📝 User Stories

### US-001: Agent-Specific Policy Heads
[x] **Task:** Replace shared policy with shared backbone + per-agent output heads.

**Why (ELI5):** Right now all agents share the exact same brain. We want them to share the "seeing" part but have their own "deciding" part — like siblings with same eyes but different personalities.

**Files:** `src/agents/network.py`, `src/configs.py`

**Changes:**
- Add `AgentSpecificActorCritic` class with shared encoder + per-agent heads
- Add `agent_architecture` config option: "shared" | "agent_heads" | "hypernetwork"
- Forward pass takes `agent_id` parameter
- Gradients only flow through relevant agent head

**Verification:** `pytest tests/test_agent.py::test_agent_specific_heads -v` passes

---

### US-002: Agent ID Embedding
[x] **Task:** Add agent identity embedding to observations.

**Why (ELI5):** Give each agent a name tag that the brain can see. Agent #1 knows it's Agent #1, so it can learn to behave differently from Agent #2.

**Files:** `src/agents/network.py`, `src/environment/obs.py`

**Changes:**
- Add learnable `agent_embedding` (n_agents, embed_dim)
- Concatenate embedding to observation before encoding
- Config option `agent_embed_dim` (default 8)

**Verification:** `pytest tests/test_agent.py::test_agent_embedding -v` passes

---

### US-003: Freeze-Evolve Training Mode
[x] **Task:** Implement alternating freeze-gradient and evolve-only phases.

**Why (ELI5):** Sometimes we practice (training), sometimes we just play games and see who wins (evolution). Alternating helps agents develop their own styles instead of all copying the same strategy.

**Files:** `src/training/train.py`, `src/configs.py`

**Changes:**
- Add `TrainingMode` enum: GRADIENT | EVOLVE | HYBRID
- Add `FreezeEvolveConfig`:
  - `gradient_steps`: steps between evolve phases (default 10000)
  - `evolve_steps`: steps of pure evolution (default 1000)
  - `evolve_mutation_std`: mutation during evolve phase (default 0.05)
- During EVOLVE phase: no gradient updates, only reproduction + mutation
- Track phase transitions in metrics

**Verification:** `pytest tests/test_training.py::test_freeze_evolve -v` passes

---

### US-004: MAP-Elites Behavioral Archive
[ ] **Task:** Maintain archive of behaviorally diverse agents.

**Why (ELI5):** Keep a "hall of fame" of different types of agents — the best explorer, the best follower, etc. When making babies, pick parents from different categories to maintain variety.

**Files:** `src/analysis/archive.py` (new), `src/configs.py`

**Changes:**
- Add `BehavioralArchive` class:
  - Grid cells indexed by 2D behavioral descriptor (exploration vs exploitation)
  - Each cell stores best-fitness agent params
  - `add(params, fitness, descriptor)` method
  - `sample(n)` for reproduction parents
- Extract behavioral descriptors from trajectory:
  - Axis 1: movement_entropy (0-1) — how random is movement?
  - Axis 2: field_write_frequency (0-1) — how much do they mark territory?
- Archive size configurable (default 100x100 cells)

**Verification:** `pytest tests/test_archive.py -v` passes

---

### US-005: FastAPI WebSocket Server
[ ] **Task:** Create streaming server for real-time visualization.

**Why (ELI5):** Build a pipe that sends what's happening in the simulation to your web browser, 30 times per second, so you can watch live.

**Files:** `src/server/` (new directory), `src/server/main.py`, `src/server/streaming.py`

**Changes:**
- FastAPI app with WebSocket endpoint `/ws/training`
- Binary MessagePack protocol for efficiency
- `TrainingBridge` class to connect training loop to server
- Frame format:
  ```python
  {
    'step': int,
    'positions': bytes,  # float32 array
    'alive': bytes,      # bool array  
    'clusters': bytes,   # int8 cluster labels
    'metrics': {
      'reward': float,
      'divergence': float,
      'specialization': float,
      'population': int
    }
  }
  ```
- Send at 30Hz (configurable)

**Verification:** Server starts, accepts connection, streams mock data

---

### US-006: Svelte Dashboard Scaffold
[ ] **Task:** Create basic Svelte 5 dashboard with WebSocket client.

**Why (ELI5):** Build the main screen where you'll watch everything happen.

**Files:** `dashboard/` (new directory), standard Svelte structure

**Changes:**
- Svelte 5 with runes ($state, $derived)
- WebSocket client with auto-reconnect
- MessagePack decoder (msgpack-lite)
- Basic layout: header, canvas area, metrics panel, controls
- Reactive state store for training data

**Verification:** `npm run dev` starts, connects to server, displays mock data

---

### US-007: Pixi.js Agent Renderer
[ ] **Task:** Render agents as colored circles with smooth interpolation.

**Why (ELI5):** Show the little creatures moving around on screen as colored dots. Each color = different team/role.

**Files:** `dashboard/src/lib/AgentCanvas.svelte`, `dashboard/src/lib/renderer.js`

**Changes:**
- Pixi.js v8 Application with WebGPU preference
- Pre-allocated sprite pool (max 64 agents)
- Position interpolation (lerp factor 0.3) for smooth movement
- Color by cluster ID (different behavioral groups)
- Trail effect (optional, configurable) — shows where agents have been
- Field heatmap background layer — shows the "pheromone" field

**Verification:** 32 agents render at 60fps, smooth movement

---

### US-008: Real-Time Metrics Charts
[ ] **Task:** Add live-updating charts for key metrics.

**Why (ELI5):** Show graphs that update live, so you can see if things are getting better or worse over time.

**Files:** `dashboard/src/lib/MetricsPanel.svelte`

**Changes:**
- Plotly.js with WebGL (scattergl)
- Charts with helper tooltips explaining each metric:
  - **Reward over time** — "Are agents getting better at collecting food?"
  - **Weight divergence** — "How different are the agents' brains?"
  - **Specialization score** — "Are clear roles emerging?"
  - **Population size** — "How many agents are alive?"
- Rolling window (last 1000 points)
- `Plotly.extendTraces` for efficient updates
- Hover for detailed explanation

**Verification:** Charts update smoothly without memory leaks

---

### US-009: Training Controls
[ ] **Task:** Add pause/resume and hyperparameter controls.

**Why (ELI5):** Let users pause, play, and adjust settings while watching — like a video player with extra knobs.

**Files:** `dashboard/src/lib/ControlPanel.svelte`, `src/server/main.py`

**Changes:**
- WebSocket command channel (JSON)
- Controls with tooltips:
  - **Play/Pause button** — "Stop/resume the simulation"
  - **Speed slider** — "How fast to run (1x, 2x, 4x...)"
  - **Mutation rate slider** — "How much do babies differ from parents?"
  - **Diversity bonus slider** — "Extra reward for being different"
- Server handles commands, updates training config
- Visual feedback for current mode (GRADIENT/EVOLVE)
- Mode indicator with explanation: "🧠 Learning" vs "🧬 Evolving"

**Verification:** Pause stops updates, sliders affect training

---

### US-010: Transfer Entropy Metric
[ ] **Task:** Implement transfer entropy between agent pairs.

**Why (ELI5):** Measure if agents are "talking" to each other through their behavior. If Agent A moves and then Agent B follows, that's information transfer!

**Helper text for UI:** "Transfer Entropy measures how much one agent's behavior predicts another's. High = agents are coordinating. Low = agents ignore each other."

**Files:** `src/analysis/information.py` (new)

**Changes:**
- `compute_transfer_entropy(agent_i_history, agent_j_history, lag=1)`
- Use k-nearest neighbors estimator (tractable)
- Compute TE matrix for all agent pairs
- Aggregate: mean TE, max TE, TE network density
- Add to metrics streamed to dashboard

**Verification:** `pytest tests/test_information.py::test_transfer_entropy -v` passes

---

### US-011: Division of Labor Index
[ ] **Task:** Quantify role specialization.

**Why (ELI5):** Are agents doing different jobs? Like measuring if a team has a goalie, defenders, and strikers vs everyone just running around randomly.

**Helper text for UI:** "Division of Labor shows how well agents have split into different roles. 0 = everyone does everything. 1 = perfect specialization (scouts scout, followers follow)."

**Files:** `src/analysis/specialization.py` (add to existing)

**Changes:**
- `compute_division_of_labor(behavior_features)`:
  - Discretize behaviors into K task types
  - DOL = 1 - (1/K) * Σ|p_i - 1/K|
  - 0 = uniform (no specialization), 1 = perfect division
- Add to specialization report
- Stream to dashboard

**Verification:** `pytest tests/test_specialization.py::test_dol -v` passes

---

### US-012: Phase Transition Detection
[ ] **Task:** Detect emergence events via susceptibility spikes.

**Why (ELI5):** Catch the moment when chaos turns into order — like water freezing. One second it's liquid, next second it's ice. We want to detect that "aha!" moment when organization suddenly appears.

**Helper text for UI:** "Phase Transition Alert! The system just shifted from disorganized to organized (or vice versa). This is emergence happening!"

**Files:** `src/analysis/emergence.py` (add to existing)

**Changes:**
- `PhaseTransitionDetector` class:
  - Track order parameter (specialization score)
  - Compute susceptibility (variance of order parameter)
  - Compute autocorrelation time
  - Flag if susceptibility > 3σ AND autocorrelation increasing
- Log emergence events with timestamp
- Visual marker on dashboard timeline (⚡ icon)

**Verification:** `pytest tests/test_emergence.py::test_phase_detection -v` passes

---

### US-013: Help System & Tooltips
[ ] **Task:** Add contextual help throughout the dashboard.

**Why (ELI5):** Make sure anyone — even a 5-year-old — can understand what's happening.

**Files:** `dashboard/src/lib/HelpSystem.svelte`, `dashboard/src/lib/Tooltip.svelte`, `dashboard/src/lib/GlossaryPanel.svelte`

**Changes:**
- **Tooltip component** — hover over any metric/term to see explanation
- **Glossary button** — opens panel with all terms explained (from this PRD's glossary)
- **"What's happening?" panel** — AI-generated plain-English summary of current state:
  - "32 agents are alive. They're starting to form 2 groups — some explore, some follow."
  - "Specialization is increasing! Looks like roles are emerging."
- **Onboarding tour** — first-time users get guided walkthrough
- **Info icons** (ℹ️) next to every metric with hover explanations
- **Color legend** — what each color means

**Verification:** Every metric has hover help, glossary contains all terms

---

### US-014: Dashboard Integration Test
[ ] **Task:** End-to-end test of training → server → dashboard.

**Why (ELI5):** Make sure everything actually works together before we say "done."

**Files:** `tests/test_dashboard_integration.py`

**Changes:**
- Start training in background thread
- Start server
- Connect headless browser (Playwright)
- Verify:
  - Agents render on canvas
  - Metrics update
  - Controls affect training
  - Help tooltips appear
- Cleanup: stop training, server

**Verification:** `pytest tests/test_dashboard_integration.py -v` passes

---

### US-015: Agent Lineage Visualization
[ ] **Task:** Show parent-child relationships in dashboard.

**Why (ELI5):** Show the family tree — who is whose parent, which families are thriving.

**Helper text for UI:** "This shows the family tree. Lines connect parents to children. Thick branches = successful families. Dead ends = lineages that went extinct."

**Files:** `dashboard/src/lib/LineagePanel.svelte`

**Changes:**
- Tree/graph visualization of agent lineages
- Node = agent, edge = parent-child
- Color by fitness (green = doing well) or cluster (behavioral group)
- Animate births (new nodes appear) and deaths (nodes fade)
- Click agent to highlight in main canvas
- Hover shows: "Agent #7, child of #3, alive for 234 steps, collected 12 food"

**Verification:** Lineage tree updates on reproduction events

---

### US-016: Export and Replay
[ ] **Task:** Save training snapshots for later replay.

**Why (ELI5):** Record what happened so you can rewatch later, show others, or analyze interesting moments.

**Files:** `src/server/replay.py`, `dashboard/src/lib/ReplayControls.svelte`

**Changes:**
- Record frames to disk (compressed MessagePack)
- Replay endpoint `/ws/replay`
- Scrubber to navigate timeline (like a video progress bar)
- Speed controls (0.5x, 1x, 2x, 4x)
- Bookmark interesting moments
- Export to video (stretch goal)

**Verification:** Can replay saved session at variable speed

---

### US-017: Status Indicators & Alerts
[ ] **Task:** Clear visual feedback for system state and important events.

**Why (ELI5):** Make it obvious what's happening — like traffic lights.

**Files:** `dashboard/src/lib/StatusBar.svelte`, `dashboard/src/lib/AlertSystem.svelte`

**Changes:**
- **Connection status:** 🟢 Connected / 🔴 Disconnected
- **Training mode:** 🧠 Learning / 🧬 Evolving / ⏸️ Paused
- **Health indicators:**
  - Population: 🟢 Stable / 🟡 Declining / 🔴 Collapsing
  - Divergence: 🟢 Increasing / 🟡 Flat / 🔴 Homogenizing
- **Alert toasts** for important events:
  - "⚡ Phase transition detected at step 50,000!"
  - "🎉 Specialization score crossed 0.7!"
  - "⚠️ Population dropped below 10"

**Verification:** Alerts appear for simulated events

---

### US-018: Final Review & Documentation
[ ] **Task:** Code review, documentation, and polish.

**Files:** `README.md`, `dashboard/README.md`, various

**Changes:**
- Update README with Phase 4 usage:
  - How to start training with visualization
  - How to open dashboard
  - What each metric means
- Add screenshots/gifs showing emergence
- Add troubleshooting guide
- Performance optimization pass
- Type check clean: `mypy src/`
- All tests pass: `pytest tests/ -v`
- Dashboard has no console errors

**Verification:** Fresh clone → setup → train → dashboard works for new user

---

## 📋 Implementation Order

### Week 1: Foundation (Fix Architecture)
1. US-001: Agent-Specific Policy Heads
2. US-002: Agent ID Embedding
3. US-003: Freeze-Evolve Training Mode
4. US-004: MAP-Elites Behavioral Archive

### Week 2: Dashboard Core
5. US-005: FastAPI WebSocket Server
6. US-006: Svelte Dashboard Scaffold
7. US-007: Pixi.js Agent Renderer
8. US-008: Real-Time Metrics Charts

### Week 3: Metrics & Intelligence
9. US-010: Transfer Entropy Metric
10. US-011: Division of Labor Index
11. US-012: Phase Transition Detection
12. US-013: Help System & Tooltips

### Week 4: Integration & Polish
13. US-009: Training Controls
14. US-014: Dashboard Integration Test
15. US-015: Agent Lineage Visualization
16. US-016: Export and Replay
17. US-017: Status Indicators & Alerts
18. US-018: Final Review & Documentation

---

## 📦 Dependencies

### Python
```
fastapi>=0.100.0
uvicorn>=0.23.0
websockets>=11.0
msgpack>=1.0.0
```

### Node (dashboard)
```json
{
  "dependencies": {
    "svelte": "^5.0.0",
    "pixi.js": "^8.0.0",
    "plotly.js-dist-min": "^2.30.0",
    "msgpack-lite": "^0.1.26"
  },
  "devDependencies": {
    "vite": "^5.0.0",
    "@sveltejs/vite-plugin-svelte": "^3.0.0"
  }
}
```

### Optional (for tests)
```
playwright>=1.40.0
```

---

## 📁 New File Structure

```
emergence-lab/
├── src/
│   ├── server/                    # NEW
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI app
│   │   ├── streaming.py          # WebSocket handlers
│   │   └── replay.py             # Recording/playback
│   ├── analysis/
│   │   ├── archive.py            # NEW: MAP-Elites
│   │   ├── information.py        # NEW: Transfer entropy
│   │   └── ... (existing)
│   └── ... (existing)
├── dashboard/                     # NEW
│   ├── src/
│   │   ├── App.svelte
│   │   ├── lib/
│   │   │   ├── AgentCanvas.svelte
│   │   │   ├── MetricsPanel.svelte
│   │   │   ├── ControlPanel.svelte
│   │   │   ├── LineagePanel.svelte
│   │   │   ├── GlossaryPanel.svelte
│   │   │   ├── HelpSystem.svelte
│   │   │   ├── Tooltip.svelte
│   │   │   ├── StatusBar.svelte
│   │   │   ├── AlertSystem.svelte
│   │   │   ├── ReplayControls.svelte
│   │   │   └── renderer.js
│   │   └── stores/
│   │       └── training.js
│   ├── static/
│   │   └── glossary.json         # All terms + explanations
│   ├── package.json
│   ├── vite.config.js
│   └── README.md
└── ... (existing)
```

---

## 🎯 Definition of Done

Phase 4 is complete when:

1. ✅ Weight divergence > 0.1 in training runs
2. ✅ Specialization score > 0.7 achievable
3. ✅ Dashboard runs at 30+ fps
4. ✅ All 18 user stories marked [x]
5. ✅ A non-technical person can open dashboard and understand what's happening
6. ✅ `mypy src/` clean
7. ✅ `pytest tests/ -v` all pass
8. ✅ `npm run build` succeeds for dashboard
9. ✅ README updated with usage instructions

---

## 📝 Notes

- US-004 (MAP-Elites Archive) can be deferred if Freeze-Evolve alone produces divergence
- QDax full integration is stretch goal — first get basic freeze-evolve working
- Dashboard should work even if training is on different machine (CORS enabled)
- Help system is CRITICAL — the goal is that anyone can understand the research
- Record compelling emergence demos for investor presentations
