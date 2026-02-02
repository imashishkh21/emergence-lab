# Swarm Code Scanner — Product Document

## Executive Summary

**Swarm Code Scanner** is a developer tool that uses emergent collective intelligence to find bugs, vulnerabilities, and code quality issues that no single-pass scanner can detect. It is powered by the **Emergence Engine** — a general-purpose swarm intelligence platform developed through our ongoing research. The product consumes the engine; it does not contain or modify it.

**Key architectural principle:** The research (Emergence Engine) and the product (Swarm Code Scanner) are separate systems. The engine is a powerful, domain-agnostic emergence platform that keeps evolving independently. The product is a thin, domain-specific application layer that translates code scanning into terms the engine understands, calls the engine's API, and translates the results back into developer-friendly output.

**Key technical breakthrough:** The engine supports two field modes. The **numerical field** (used in research) is a simple array of float values — pure stigmergy, proven in the lab. The **LLM-powered field** (used in the product) replaces the dumb numerical medium with a local open-source language model (Ollama/Llama/Mistral). This means the shared medium that agents read and write is itself intelligent — it understands code, knows vulnerability patterns, and can reason about relationships. Agents don't need to be trained on vulnerability datasets. The LLM medium already knows what SQL injection looks like. Agents just need to evolve to explore efficiently and ask the right questions. **Zero pre-training required.**

This separation means:
- Research continues with the pure numerical field (controlled experiments, proving emergence)
- The product uses the LLM-powered field (practical power, immediate usefulness, no pre-training)
- The engine can power multiple products (code scanning today, security monitoring tomorrow, market research next)
- Product iteration is fast (changing CLI output doesn't touch the engine)
- Engine improvements automatically benefit all products

---

## Two-System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  EMERGENCE ENGINE (research — separate repo, separate system)           │
│  ════════════════════════════════════════════════════════════            │
│                                                                         │
│  The engine. Domain-agnostic. Knows nothing about code.                 │
│  Knows everything about: agents, fields, evolution, emergence.          │
│                                                                         │
│  Accepts:                                                               │
│    - A graph (nodes + edges)                                            │
│    - Node feature vectors                                               │
│    - Reward signals                                                     │
│    - Configuration (agent count, field channels, generations, etc.)     │
│                                                                         │
│  Returns:                                                               │
│    - Agent convergence map (which nodes agents clustered on)            │
│    - Field state (the evolved knowledge heatmap)                        │
│    - Species data (what specializations emerged)                        │
│    - Agent lineage data (evolutionary history)                          │
│    - Trained swarm state (saveable, resumable)                          │
│                                                                         │
│  Exposed via:                                                           │
│    - Python SDK: `from emergence_engine import SwarmEngine`             │
│    - REST API: `POST /api/v1/simulate`                                  │
│    - gRPC: for high-performance local use                               │
│                                                                         │
└───────────────┬─────────────────────────────────────────────────────────┘
                │
                │  API calls (graph in, convergence map out)
                │
┌───────────────▼─────────────────────────────────────────────────────────┐
│                                                                         │
│  SWARM CODE SCANNER (product — separate repo, separate system)          │
│  ════════════════════════════════════════════════════════════            │
│                                                                         │
│  The product. Domain-specific. Knows everything about code.             │
│  Knows nothing about: agents, fields, evolution internals.              │
│                                                                         │
│  Responsibilities:                                                      │
│    1. Parse codebase into a graph (tree-sitter AST)                    │
│    2. Extract feature vectors for each node (complexity, patterns...)   │
│    3. Send graph + features to Emergence Engine API                     │
│    4. Receive convergence map + species data back                       │
│    5. Translate convergence hotspots into human-readable findings       │
│    6. Present findings via CLI / TUI / SARIF / JSON                    │
│    7. Collect developer feedback (confirm/dismiss) → send as reward    │
│                                                                         │
│  The product is a TRANSLATION LAYER between code and the engine.       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why This Separation Matters

| Concern | Without Separation | With Separation |
|---------|-------------------|-----------------|
| Research pace | Slowed by product requirements, backwards compatibility | Research moves freely; engine evolves independently |
| Product iteration | Every UI change risks breaking the simulation | Product changes are cosmetic; engine is untouched |
| Multiple products | Can only build one thing | Same engine powers code scanner, security monitor, research assistant, logistics optimizer |
| Engine upgrades | Must coordinate with product release schedule | Engine improves → all products automatically benefit via API |
| Team structure | Everyone works in one codebase | Research team owns the engine; product team owns the scanner |
| Testing | Entangled tests, hard to isolate failures | Engine has its own test suite; product has its own test suite |
| Licensing | Must open-source everything or nothing | Engine can be proprietary; product can be open-source (or vice versa) |

---

## The Hybrid Breakthrough: LLM as the Shared Medium

### The Core Insight

Our research proves that emergence comes from: **simple agents + shared medium + evolution**.

In the lab, the shared medium is a dumb numerical field. This proves the principle.

In the product, the shared medium is a **locally-running open-source LLM**. This makes it practical.

The agents stay simple. The evolution stays the same. Only the medium changes — from a float array to an intelligent language model.

### How It Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  AGENT SWARM                              LLM-POWERED MEDIUM            │
│  (64 simple agents)                       (Ollama / Llama / Mistral)    │
│                                                                         │
│  ┌─────────┐                              ┌─────────────────────────┐  │
│  │ Agent 1  │──── ASK: "What does this ──▶│                         │  │
│  │ (tiny NN)│       function do?"         │  The LLM reads the      │  │
│  │          │◀── "It concatenates user ───│  actual source code at   │  │
│  │          │     input into a SQL query" │  this node and responds  │  │
│  │          │                              │  with semantic analysis  │  │
│  │          │──── ANNOTATE: "Possible ───▶│                         │  │
│  │          │     SQL injection here"     │  The annotation is       │  │
│  │          │                              │  stored in per-node     │  │
│  └─────────┘                              │  memory. Other agents    │  │
│                                            │  can read it later.     │  │
│  ┌─────────┐                              │                         │  │
│  │ Agent 2  │──── ASK: "What have other ─▶│                         │  │
│  │ (tiny NN)│      agents noted here?"    │  The LLM retrieves      │  │
│  │          │◀── "Agent 1 flagged this ───│  Agent 1's annotation   │  │
│  │          │     as possible SQL inject" │  and synthesizes it      │  │
│  │          │                              │  with the code context  │  │
│  │          │──── ASK: "Does the data ───▶│                         │  │
│  │          │     flow to other files?"   │  The LLM checks the     │  │
│  │          │◀── "Yes, via connection.py──│  graph edges + code at   │  │
│  │          │     to db.execute()"        │  neighboring nodes       │  │
│  │          │                              │                         │  │
│  │          │──── FLAG: convergence ──────▶│  Two independent agents │  │
│  │          │     (agrees with Agent 1)   │  now agree. Confidence   │  │
│  └─────────┘                              │  rises.                  │  │
│                                            │                         │  │
│  ┌─────────┐                              │                         │  │
│  │ Agent 47 │──── ASK: "Is this function─▶│                         │  │
│  │ (tiny NN)│     slow or inefficient?"   │  Different agent, same   │  │
│  │          │◀── "The loop at line 23 ────│  medium. This agent      │  │
│  │ (evolved │     iterates over all       │  evolved to ask about    │  │
│  │  to ask  │     users without limit"    │  PERFORMANCE, not        │  │
│  │  about   │                              │  security. It            │  │
│  │  perf)   │──── ANNOTATE: "Unbounded ──▶│  specialized through     │  │
│  │          │     loop, potential DoS"    │  evolution.               │  │
│  └─────────┘                              └─────────────────────────┘  │
│                                                                         │
│  EVOLUTION: Agents that find real issues → get energy → reproduce       │
│             Agents that ask bad questions → waste energy → die           │
│             Over generations, better "question-askers" survive           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### What the Agents Evolve (When the LLM Knows Everything)

The LLM already knows what SQL injection is. So what's left for agents to evolve?

**Everything except domain knowledge:**

| What evolves | Example |
|---|---|
| **Where to go** | Agent A learned to prioritize functions that handle user input. Agent B learned to follow data flow edges. Agent C explores high-complexity nodes. These strategies emerged, not designed. |
| **What to ask** | Agent A asks "Is this input sanitized?" (security-focused). Agent B asks "Is this loop bounded?" (performance-focused). Agent C asks "Does this match any previous annotations?" (pattern-focused). Different evolved weights → different query instincts. |
| **When to flag** | Cautious agents flag only when the LLM response contains words like "vulnerability" or "injection." Aggressive agents flag anything the LLM calls "unusual." Evolution balances this — too cautious = miss bugs = no energy. Too aggressive = false positives = energy penalty. |
| **What to annotate** | Good annotators leave notes that help other agents. Bad annotators leave noise. Notes that lead to multi-agent convergence help the annotator's "descendants" find bugs faster → natural selection for useful communication. |
| **How to specialize** | After several generations, some agent lineages consistently ask security questions. Others consistently ask performance questions. Others trace data flow. Species form — just like in our research petri dish. |

### Why This Is Different from Just Prompting an LLM

"Why not just ask the LLM to review the code directly? Why the agents?"

| Just an LLM (CodeRabbit approach) | LLM + Swarm (Our approach) |
|---|---|
| One perspective | 64 independent perspectives |
| Reviews every line equally | Agents converge on what matters, skip what doesn't |
| No confidence signal | Confidence = how many agents independently converged |
| Same quality forever | Gets better over time through evolution |
| All findings shown (noisy) | Only convergence-filtered findings shown |
| No specialization | Security/performance/pattern species emerge |
| Full LLM cost per file | LLM queried only where agents explore (cheaper) |
| If LLM hallucinates → false positive | If LLM hallucinates → only 1 agent fooled, others disagree → filtered out |

The last point is critical: **convergence-based filtering is a natural defense against LLM hallucinations.** If the LLM gives a bad answer to one agent, that agent might flag a false positive. But the other 63 agents query the LLM independently with different questions, and if they don't converge on the same finding, the false positive is suppressed. This is like a jury — one juror can be fooled, but all 12 agreeing is a strong signal.

### LLM Provider Options

The product supports any local or cloud LLM as the medium:

| Provider | Model | Runs Locally | Cost per Scan | Best For |
|---|---|---|---|---|
| **Ollama** | Llama 3 8B | Yes (MacBook) | $0 | Individual developers, privacy-sensitive |
| **Ollama** | Llama 3 70B | Yes (needs GPU) | $0 | Deeper analysis, larger codebases |
| **Ollama** | CodeLlama 34B | Yes (needs GPU) | $0 | Code-specialized analysis |
| **Ollama** | Mistral 7B | Yes (MacBook) | $0 | Fast, lightweight scanning |
| **Ollama** | Qwen 2.5 Coder | Yes (MacBook) | $0 | Code-specialized, multilingual |
| **OpenAI** | GPT-4o-mini | No (cloud) | ~$0.50/scan | Budget cloud option |
| **OpenAI** | GPT-4o | No (cloud) | ~$2.00/scan | Maximum intelligence |
| **Anthropic** | Claude Sonnet | No (cloud) | ~$1.50/scan | Strong code reasoning |
| **Google** | Gemini Flash | No (cloud) | ~$0.25/scan | Cheapest cloud option |

**Default recommendation:** Ollama with Llama 3 8B. Runs on any modern laptop. Free. Private (code never leaves your machine). Good enough for most code analysis. Upgrade to 70B or cloud models for deeper analysis.

```bash
# Setup (one time)
ollama pull llama3:8b                    # Download the model
swarm config --engine local --llm ollama --model llama3:8b

# Scan (uses local LLM, no internet needed, no API key needed)
swarm scan .
```

---

## System 1: Emergence Engine (The Research Platform)

### What It Is

The Emergence Engine is the core research output — a general-purpose swarm intelligence simulation platform. It is the digital equivalent of a petri dish: you give it an environment (any graph), agents, and evolutionary pressure, and it produces emergent collective intelligence.

The engine does NOT know what "code" is. It does NOT know what a "vulnerability" is. It operates on abstract graphs with features. This abstraction is what makes it reusable across domains.

### The Two Field Modes

The engine's most important design decision is that the shared medium (field) is pluggable. Two modes are supported:

**Mode 1: Numerical Field (Research)**
```
Field = array of float values at each node
Agent writes: deposits a number (e.g., 0.87) at current node
Agent reads: sees float values at current node and neighbors
Intelligence: ZERO — the field is a dumb array
Emergence: agents must EVOLVE all meaning from scratch

Used for: controlled research experiments, ablation studies,
          proving that emergence works with minimal assumptions
```

**Mode 2: LLM-Powered Field (Product)**
```
Field = a local language model (Ollama / Llama / Mistral / etc.)
        maintaining semantic memory per node
Agent writes: leaves a semantic annotation
              ("this function passes unsanitized input to SQL")
Agent reads: queries the medium
              ("what have other agents observed about this node?")
Intelligence: HIGH — the LLM understands code, patterns, context
Emergence: agents evolve HOW TO USE the intelligent medium

Used for: production products where immediate usefulness matters,
          no pre-training on vulnerability datasets needed
```

**Why this matters:**

With the numerical field, agents need extensive pre-training on labeled vulnerability datasets to learn what a "bug" looks like. The numbers have no inherent meaning — agents must discover meaning through millions of training steps.

With the LLM-powered field, agents need ZERO pre-training on vulnerability data. The LLM already knows what SQL injection is. It already knows what buffer overflows look like. It already understands code semantics. The agents don't need to learn about code — they need to learn to explore efficiently, ask the right questions, and leave useful notes for other agents. That's what evolves.

Think of it this way:
- **Numerical field** = ants leaving pheromone chemicals. The chemicals have no meaning. Ants evolved over millions of years to interpret them.
- **LLM field** = humans leaving written notes on a shared whiteboard. The notes are immediately meaningful. Humans just need to learn which notes to write and where.

Both are stigmergy. Both produce emergence. But the LLM field gives you a massive head start because the medium itself is pre-trained.

**The research proves the principle (emergence works). The LLM field makes it practical (no training needed).**

### Current State (emergence-lab research)

The engine currently lives in our `emergence-lab` repository as a research codebase:

```
emergence-lab/
├── src/
│   ├── environment/     # Grid-based environment (will generalize to graph)
│   │   ├── state.py     # EnvState dataclass
│   │   ├── env.py       # reset() + step() pipeline
│   │   ├── obs.py       # Agent observations
│   │   └── vec_env.py   # Vectorized parallel environments
│   ├── field/           # Shared stigmergic field
│   │   ├── field.py     # FieldState
│   │   ├── dynamics.py  # Diffusion + decay
│   │   └── ops.py       # Read/write operations
│   ├── agents/          # Agent neural networks + behavior
│   │   ├── network.py   # ActorCritic architecture
│   │   ├── policy.py    # Action sampling
│   │   └── reproduction.py  # Mutation + inheritance
│   ├── training/        # PPO training loop
│   │   ├── train.py     # Entry point
│   │   ├── rollout.py   # Trajectory collection
│   │   ├── gae.py       # Advantage estimation
│   │   └── ppo.py       # Loss computation
│   └── analysis/        # Emergence detection + metrics
│       ├── specialization.py  # Species detection, clustering
│       ├── emergence.py       # Phase transition detection
│       ├── ablation.py        # Controlled experiments
│       └── lineage.py         # Evolutionary history
```

### Engine Evolution Path

The research codebase will evolve into a general-purpose engine through these stages:

**Stage 1 (Current): Grid-Based Research**
- 2D spatial grid environment with numerical field
- Proving core hypotheses (field carries signal, specialization emerges)
- All experiments run locally

**Stage 2: Graph Abstraction**
- Generalize grid environment to arbitrary graph environments
- Nodes can be anything (grid cells, code functions, network endpoints)
- Edges can be anything (spatial adjacency, function calls, network connections)
- Field diffusion works along graph edges instead of spatial neighbors
- This is the KEY engineering milestone that enables productization

**Stage 3: LLM Field Integration**
- Add pluggable field backend: numerical (existing) or LLM-powered (new)
- LLM field uses Ollama for local inference (no cloud dependency)
- Agent actions expand: ASK (query medium), ANNOTATE (write semantic note)
- Agent observations include LLM responses instead of just float arrays
- Research continues on numerical field; product uses LLM field
- This is the KEY product milestone — eliminates need for pre-training

**Stage 4: Engine API**
- Package the graph-based engine as an installable Python SDK
- Expose a clean API surface (see below)
- Engine can be run locally (SDK) or remotely (REST API / gRPC)
- Engine becomes a standalone service that any product can consume
- API supports both field modes (numerical and LLM)

**Stage 5: Engine as a Service**
- Cloud-hosted engine for products that need scale
- GPU-accelerated simulation + LLM inference
- Multi-tenant with isolated swarm states per customer
- Usage-based pricing for API calls

### Engine API Surface

```python
from emergence_engine import SwarmEngine, EngineConfig, GraphEnvironment

# ─── CONFIGURATION ───────────────────────────────────────────

# RESEARCH MODE: Pure numerical field (controlled experiments)
research_config = EngineConfig(
    # Agent parameters
    num_agents=64,
    max_agents=128,
    agent_hidden_dims=[64, 64],

    # Field parameters — NUMERICAL MODE
    field_type="numerical",           # Simple float array field
    field_channels=4,                 # Number of float channels per node
    diffusion_rate=0.1,               # How far signals spread along edges
    decay_rate=0.05,                  # How fast signals fade

    # Evolution parameters
    evolution_enabled=True,
    mutation_std=0.01,
    reproduce_threshold=150,
    starting_energy=200,

    # Simulation parameters
    num_steps_per_generation=500,
    num_generations=20,
    num_parallel_envs=32,

    # Tracking
    track_specialization=True,
    track_lineage=True,
    track_field_metrics=True,
)

# PRODUCT MODE: LLM-powered field (practical applications)
product_config = EngineConfig(
    # Agent parameters
    num_agents=64,
    max_agents=128,
    agent_hidden_dims=[64, 64],

    # Field parameters — LLM MODE
    field_type="llm",                 # LLM-powered semantic field
    llm_provider="ollama",            # Local LLM via Ollama (no cloud needed)
    llm_model="llama3:8b",            # Any Ollama-supported model
    # llm_provider="openai",          # Or use OpenAI API
    # llm_model="gpt-4o-mini",        # For cloud-based inference
    field_memory_per_node=10,         # Max annotations stored per node
    field_context_window=4096,        # Context window for LLM queries

    # Evolution parameters (same as research)
    evolution_enabled=True,
    mutation_std=0.01,
    reproduce_threshold=150,
    starting_energy=200,

    # Simulation parameters
    num_steps_per_generation=500,
    num_generations=20,
    num_parallel_envs=1,              # LLM mode: sequential (LLM is the bottleneck)

    # Tracking
    track_specialization=True,
    track_lineage=True,
    track_field_metrics=True,
)

# ─── DEFINE THE ENVIRONMENT (product-specific) ──────────────

# The product constructs this graph — the engine doesn't care what the
# nodes represent. Nodes are just feature vectors. Edges are just connections.

# For NUMERICAL mode: nodes are float feature vectors
graph = GraphEnvironment(
    num_nodes=3291,
    node_features=node_feature_matrix,        # (num_nodes, feature_dim) float32
    adjacency=adjacency_matrix,               # (num_nodes, num_nodes) sparse bool
    food_locations=known_bug_node_ids,        # Nodes where "food" exists (for training)
    food_values=bug_severity_scores,          # Reward value per food item
)

# For LLM mode: nodes carry source code and context (the LLM reads the code directly)
llm_graph = GraphEnvironment(
    num_nodes=3291,
    node_content={                            # Raw content the LLM can read
        0: "def login(request):\n    query = 'SELECT * FROM users WHERE name=' + request.POST['user']",
        1: "def connect_db(query):\n    return db.execute(query)",
        # ... one entry per node with actual source code
    },
    node_metadata={                           # Structural info per node
        0: {"file": "src/auth/login.py", "line": 47, "type": "function", "language": "python"},
        1: {"file": "src/db/connection.py", "line": 12, "type": "function", "language": "python"},
        # ...
    },
    adjacency=adjacency_matrix,               # Same graph structure
    # No food_locations needed — the LLM field can assess "suspiciousness" directly
    # No pre-labeled training data needed
)

# ─── RUN THE ENGINE ──────────────────────────────────────────

# Research mode (numerical field)
research_engine = SwarmEngine(research_config)
result = research_engine.simulate(graph)

# Product mode (LLM field) — NO PRE-TRAINING NEEDED
product_engine = SwarmEngine(product_config)
result = product_engine.simulate(llm_graph)

# ─── READ THE RESULTS ────────────────────────────────────────

# Where did agents converge? (the core output)
result.convergence_map
# → Dict[node_id, ConvergenceInfo]
# → ConvergenceInfo: { agent_count: int, confidence: float, field_intensity: float }

# What species emerged?
result.species
# → List[Species]
# → Species: { id: int, size: int, behavioral_profile: Dict, member_agent_ids: List }

# The evolved field state (persistent knowledge map)
result.field_state
# → (num_nodes, field_channels) float32 — the collective knowledge

# Agent lineage (who descended from whom)
result.lineage
# → LineageTree with birth/death records, family trees, dominant lineages

# Specialization metrics
result.specialization_score    # 0-1 composite score
result.species_detected        # bool — did stable species form?
result.weight_divergence       # float — genetic diversity

# The trained swarm state (save and resume later)
swarm_state = engine.get_state()
engine.save_state("swarm_state.pkl")
engine.load_state("swarm_state.pkl")

# ─── INCREMENTAL UPDATE (for subsequent scans) ──────────────

# Only update changed nodes; keep field state for unchanged nodes
engine.load_state("previous_state.pkl")
result = engine.simulate_incremental(
    graph=updated_graph,
    changed_nodes=[12, 45, 67, 234],    # Only these nodes changed
    freeze_unchanged=True,                # Don't re-analyze stable nodes
)

# ─── FEEDBACK LOOP (for self-improvement) ────────────────────

# Developer confirmed finding at node 47 was a real bug
engine.apply_reward(node_id=47, reward=+10.0)

# Developer dismissed finding at node 23 as false positive
engine.apply_reward(node_id=23, reward=-5.0)

# Re-evolve with feedback incorporated
result = engine.evolve(num_generations=5)

# ─── ABLATION TESTING (for research validation) ─────────────

# Run the same graph with field disabled (control experiment)
ablation_result = engine.simulate(graph, ablation="zero_field")

# Compare: does the field add value?
field_value = result.total_food_collected - ablation_result.total_food_collected
```

### Engine REST API (for remote / cloud deployment)

```
POST /api/v1/engine/create
  Body: { config: EngineConfig }
  Returns: { engine_id: string }

POST /api/v1/engine/{id}/simulate
  Body: { graph: GraphEnvironment }
  Returns: { result: SimulationResult }

POST /api/v1/engine/{id}/simulate-incremental
  Body: { graph: GraphEnvironment, changed_nodes: int[], freeze_unchanged: bool }
  Returns: { result: SimulationResult }

POST /api/v1/engine/{id}/feedback
  Body: { node_id: int, reward: float }
  Returns: { status: "ok" }

POST /api/v1/engine/{id}/evolve
  Body: { num_generations: int }
  Returns: { result: SimulationResult }

GET  /api/v1/engine/{id}/state
  Returns: { swarm_state: binary }

POST /api/v1/engine/{id}/state
  Body: { swarm_state: binary }
  Returns: { status: "ok" }

GET  /api/v1/engine/{id}/species
  Returns: { species: Species[] }

GET  /api/v1/engine/{id}/convergence
  Returns: { convergence_map: Dict[int, ConvergenceInfo] }

GET  /api/v1/engine/{id}/field
  Returns: { field_state: float[][] }

DELETE /api/v1/engine/{id}
  Returns: { status: "ok" }
```

### Engine API Key Model

```
# Products authenticate to the engine via API keys

EMERGENCE_ENGINE_API_KEY=em_live_abc123...

# Usage tiers:
#
# Free:       1,000 simulation steps/month   (for testing)
# Starter:    100,000 steps/month             ($49/mo)
# Growth:     1,000,000 steps/month           ($199/mo)
# Scale:      10,000,000 steps/month          ($499/mo)
# Enterprise: Unlimited + dedicated instance  (custom)
#
# A typical code scan (64 agents, 20 generations, 500 steps/gen) = 640,000 steps
# So Starter tier = ~15 scans/month, Growth = ~150 scans/month
```

---

## System 2: Swarm Code Scanner (The Product)

### What It Is

The Swarm Code Scanner is a **thin, domain-specific translation layer** between the code scanning domain and the Emergence Engine. It does three things:

1. **Translate code → graph**: Parse a codebase into a graph that the engine understands
2. **Call the engine**: Send the graph to the Emergence Engine and get results back
3. **Translate results → findings**: Convert convergence data into developer-friendly output

The product contains ZERO simulation logic, ZERO agent logic, ZERO field logic, ZERO evolution logic. All of that lives in the engine. The product is purely: parsing, API calls, and presentation.

### Product Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       SWARM CODE SCANNER                             │
│                                                                      │
│  ┌──────────────────┐                                                │
│  │  1. CODE PARSER   │  Responsibilities:                            │
│  │                    │  - tree-sitter AST parsing                   │
│  │  Input: codebase   │  - Build file/function dependency graph      │
│  │  Output: graph     │  - Extract feature vectors per node          │
│  │    + features      │  - Detect language-specific patterns         │
│  └────────┬───────────┘                                              │
│           │                                                          │
│           │  graph + node_features + adjacency_matrix                │
│           ▼                                                          │
│  ┌──────────────────┐     ┌──────────────────────────────────┐      │
│  │  2. ENGINE CLIENT │────▶│  EMERGENCE ENGINE (external)      │      │
│  │                    │    │                                    │      │
│  │  Sends graph to    │◀───│  Returns convergence map,         │      │
│  │  engine API        │    │  species data, field state         │      │
│  │                    │    │                                    │      │
│  │  Handles:          │    │  (This is the research system.     │      │
│  │  - API key auth    │    │   It runs the swarm simulation.    │      │
│  │  - State caching   │    │   The product never touches its    │      │
│  │  - Incremental     │    │   internals.)                      │      │
│  │    updates         │    │                                    │      │
│  └────────┬───────────┘    └──────────────────────────────────┘      │
│           │                                                          │
│           │  convergence_map + species + field_state                 │
│           ▼                                                          │
│  ┌──────────────────┐                                                │
│  │  3. RESULT        │  Responsibilities:                            │
│  │     TRANSLATOR    │  - Map node IDs back to file:line locations   │
│  │                    │  - Compute confidence from agent convergence │
│  │  Input: engine     │  - Classify severity (critical/warning/note) │
│  │    results         │  - Detect cross-file vulnerability chains    │
│  │  Output: findings  │  - Generate human-readable descriptions      │
│  └────────┬───────────┘                                              │
│           │                                                          │
│           │  structured findings                                     │
│           ▼                                                          │
│  ┌──────────────────┐                                                │
│  │  4. PRESENTER     │  Responsibilities:                            │
│  │                    │  - CLI output (rich terminal formatting)     │
│  │  Formats:          │  - TUI heatmap visualization                │
│  │  - Terminal        │  - SARIF export (GitHub/GitLab integration) │
│  │  - TUI             │  - JSON export (machine-readable)           │
│  │  - SARIF           │  - Feedback collection (confirm/dismiss)    │
│  │  - JSON            │                                              │
│  └──────────────────┘                                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Product Code Structure

```
swarm-code-scanner/                   # SEPARATE REPO from emergence-lab
├── swarm_scan/
│   ├── __init__.py
│   ├── cli.py                        # CLI entry point (tyro-based)
│   ├── parser/                       # Layer 1: Code → Graph
│   │   ├── __init__.py
│   │   ├── tree_sitter_parser.py     # AST parsing via tree-sitter
│   │   ├── graph_builder.py          # Build dependency graph from AST
│   │   ├── feature_extractor.py      # Extract feature vectors per node
│   │   ├── languages/
│   │   │   ├── python.py             # Python-specific patterns
│   │   │   ├── javascript.py         # JS/TS-specific patterns
│   │   │   ├── go.py                 # Go-specific patterns
│   │   │   └── ...
│   │   └── patterns/
│   │       ├── security.py           # Known security anti-patterns
│   │       ├── performance.py        # Known performance anti-patterns
│   │       └── quality.py            # Code smell patterns
│   ├── engine_client/                # Layer 2: Talk to Engine
│   │   ├── __init__.py
│   │   ├── client.py                 # API client for Emergence Engine
│   │   ├── local_client.py           # Direct SDK call (engine installed locally)
│   │   ├── remote_client.py          # REST API call (engine running remotely)
│   │   ├── state_cache.py            # Cache swarm state in .swarm/ directory
│   │   └── config.py                 # Engine configuration presets for code scanning
│   ├── translator/                   # Layer 3: Engine Results → Findings
│   │   ├── __init__.py
│   │   ├── finding.py                # Finding dataclass
│   │   ├── convergence_analyzer.py   # Convert convergence map to findings
│   │   ├── chain_detector.py         # Detect cross-file vulnerability chains
│   │   ├── severity_classifier.py    # Classify critical/warning/note
│   │   └── description_generator.py  # Generate human-readable descriptions
│   ├── presenter/                    # Layer 4: Findings → User Output
│   │   ├── __init__.py
│   │   ├── terminal.py               # Rich terminal output
│   │   ├── tui.py                    # Textual-based TUI heatmap
│   │   ├── sarif_export.py           # SARIF format for GitHub/GitLab
│   │   ├── json_export.py            # JSON export
│   │   └── feedback.py               # Collect confirm/dismiss from user
│   └── config.py                     # Product-level config (paths, thresholds, etc.)
├── tests/
│   ├── test_parser/
│   ├── test_engine_client/
│   ├── test_translator/
│   └── test_presenter/
├── pyproject.toml
└── README.md
```

### Data Flow: A Complete Scan

```
Step 1: PARSE
───────────────────────────────────────────────────────────

User runs: swarm scan ./my-project

Parser reads all source files via tree-sitter.
Builds a graph:
  - 847 files → 3,291 function nodes
  - Import/call analysis → 12,408 edges
  - Feature extraction → (3291, 20) feature matrix

Each node gets a feature vector:
  [complexity, loc, params, nesting, branches, loops,
   external_calls, string_concats, user_inputs, eval_usage,
   raw_sql, fs_ops, net_ops, crypto_ops, error_handling,
   input_validation, is_test, imports, exports, comment_density]


Step 2: CALL ENGINE (with LLM-powered field)
───────────────────────────────────────────────────────────

Product sends graph + source code to Emergence Engine:

  engine.simulate(
    graph=GraphEnvironment(
      num_nodes=3291,
      node_content={ ... source code per node ... },
      node_metadata={ ... file/line/type per node ... },
      adjacency=adjacency_matrix,
    )
  )

The engine (with LLM field):
  - Releases 64 agents onto the graph
  - Agent arrives at a node → ASKs the LLM: "What is this code doing?"
  - LLM reads the actual source code and responds semantically
  - Agent decides to FLAG, ANNOTATE, or MOVE based on LLM response
  - Other agents arrive at same node → read previous annotations via LLM
  - LLM synthesizes: "Agent 1 noted SQL injection. Agent 7 confirmed data flow."
  - Agents independently converge on suspicious nodes
  - Evolution: agents that find real issues reproduce, false-positive agents die
  - Species emerge: security questioners, performance questioners, flow tracers
  - Returns convergence map + species data + field state

The product does NOT know or care what happens inside this call.
It's a black box. Graph in, convergence map out.
The LLM interaction is entirely internal to the engine.


Step 3: TRANSLATE
───────────────────────────────────────────────────────────

Engine returns:

  convergence_map = {
    node_47:  { agent_count: 31, confidence: 0.94, field_intensity: 0.87 },
    node_892: { agent_count: 8,  confidence: 0.71, field_intensity: 0.45 },
    node_23:  { agent_count: 3,  confidence: 0.45, field_intensity: 0.22 },
  }

  species = [
    { id: 0, name: "cluster_0", size: 18, profile: { flags_security: 0.82, ... } },
    { id: 1, name: "cluster_1", size: 12, profile: { flags_performance: 0.76, ... } },
    { id: 2, name: "cluster_2", size: 34, profile: { flags_patterns: 0.68, ... } },
  ]

Translator maps node IDs back to source locations:
  node_47  → src/auth/login.py:47 (function: executeQuery)
  node_892 → src/api/handlers.js:156 (function: processData)
  node_23  → lib/cache.py:23 (function: hash_key)

Translator checks graph edges between converged nodes:
  node_47 → node_112 → node_234 (connected by data flow)
  Maps to: login.py → connection.py → user.py
  CROSS-FILE CHAIN DETECTED.

Translator classifies severity:
  31/64 agents + security species + cross-file chain → CRITICAL
  8/64 agents + performance species → WARNING
  3/64 agents + pattern species → NOTE


Step 4: PRESENT
───────────────────────────────────────────────────────────

Presenter formats findings for terminal:

  🔴 CRITICAL [confidence: 94%] (31/64 agents converged)
  ├── src/auth/login.py:47 — SQL injection via unsanitized user input
  ├── Related: src/db/connection.py:12 (data flows here)
  ├── Related: src/models/user.py:88 (reaches database sink here)
  └── Cross-file chain: input → login.py → connection.py → user.py

Or exports as SARIF for GitHub integration.
Or opens TUI for interactive exploration.


Step 5: FEEDBACK (optional)
───────────────────────────────────────────────────────────

Developer reviews findings:
  - Confirms the SQL injection → swarm scan confirm 1
  - Dismisses the MD5 note as intentional → swarm scan dismiss 3

Product sends feedback to engine:
  engine.apply_reward(node_id=47, reward=+10.0)   # confirmed
  engine.apply_reward(node_id=23, reward=-5.0)     # dismissed

Engine re-evolves briefly:
  engine.evolve(num_generations=5)

Saves improved state:
  engine.save_state(".swarm/state.pkl")

Next scan loads this improved state → fewer false positives.
```

---

## Product Interface

### Primary Interface: CLI

```bash
# Install the product (installs engine SDK as a dependency)
pip install swarm-scan

# Set your engine API key (one-time setup)
export EMERGENCE_ENGINE_API_KEY=em_live_abc123...
# Or for local engine (no API key needed):
pip install emergence-engine
swarm config --engine local

# Basic scan
swarm scan .

# Scan with custom engine settings
swarm scan . --agents 128 --generations 30

# Scan specific directory
swarm scan src/

# Output formats
swarm scan . --format json          # Machine-readable
swarm scan . --format sarif         # GitHub/GitLab compatible
swarm scan . --format table         # Terminal table (default)

# Resume from previous swarm state (incremental scanning)
swarm scan . --resume

# Exclude paths
swarm scan . --exclude "tests/,node_modules/,vendor/"

# Set minimum convergence threshold
swarm scan . --min-convergence 5    # Only report if 5+ agents converge

# Verbose: show agent behavior in real-time
swarm scan . --verbose

# Feedback commands
swarm confirm 1                     # Confirm finding #1 was real
swarm dismiss 3                     # Dismiss finding #3 as false positive

# View interactive heatmap
swarm view

# Engine status
swarm engine status                 # Check engine connection + API usage
swarm engine usage                  # See steps consumed this month
```

### Terminal Output

```
$ swarm scan .

  ╔══════════════════════════════════════════════════════════╗
  ║  SWARM CODE SCANNER v0.1.0                               ║
  ║  Engine: Emergence Engine v2.0 (local)                    ║
  ║  64 agents · 4 field channels · 20 generations           ║
  ╚══════════════════════════════════════════════════════════╝

  [Parser]  Scanning codebase... 847 files, 3,291 functions, 12,408 edges
  [Engine]  Releasing 64 agents onto code graph...

  Gen  1/20  ████░░░░░░░░░░░░░░░░  agents exploring...
  Gen  5/20  ██████████░░░░░░░░░░  3 clusters forming
  Gen 10/20  ██████████████░░░░░░  species emerging: security(18) perf(12) pattern(34)
  Gen 20/20  ████████████████████  converged

  ┌─────────────────────────────────────────────────────────┐
  │  SPECIES EMERGED                                         │
  │  🔴 Security Hunters:   18 agents (28%)                  │
  │  🟡 Performance Scouts: 12 agents (19%)                  │
  │  🔵 Pattern Detectors:  34 agents (53%)                  │
  └─────────────────────────────────────────────────────────┘

  FINDINGS (convergence-filtered):

  #1 🔴 CRITICAL [confidence: 94%] (31/64 agents converged)
  ├── src/auth/login.py:47 — SQL injection via unsanitized user input
  ├── Related: src/db/connection.py:12 (data flows here)
  ├── Related: src/models/user.py:88 (reaches database sink here)
  └── Cross-file chain detected: input → login.py → connection.py → user.py
      The vulnerability spans 3 files. No single-file scanner would see this.

  #2 🟠 WARNING [confidence: 71%] (8/64 agents converged)
  ├── src/api/handlers.js:156 — Unbounded iteration over user-supplied array
  └── Potential DoS vector under high load

  #3 🟡 NOTE [confidence: 45%] (3/64 agents converged)
  ├── lib/cache.py:23 — MD5 hashing (deprecated, weak collision resistance)
  └── Consider upgrading to SHA-256

  ─────────────────────────────────────────────────────────
  Summary: 1 critical · 1 warning · 1 note
  False positive rate: estimated 8% (based on evolutionary filtering)
  Swarm state saved to .swarm/state.pkl

  Provide feedback to improve future scans:
    swarm confirm 1      # Yes, this is a real bug
    swarm dismiss 3      # No, this is intentional

  Run 'swarm view' for interactive heatmap
```

### TUI Visualization

```bash
$ swarm view

┌─ FILE TREE ──────────────┬─ CODE VIEW ─────────────────────────────┐
│                          │                                          │
│  src/                    │  47│ def login(request):                  │
│  ├── auth/               │  48│     username = request.POST["user"]  │
│  │   ├── 🔴 login.py     │  49│     password = request.POST["pass"]  │
│  │   └──    session.py   │  50│     ████████████████████████████████  │
│  ├── api/                │  51│     query = "SELECT * FROM users"    │
│  │   ├── 🟠 handlers.js  │  52│          + " WHERE name='" + user   │
│  │   └──    routes.js    │  53│     ████████████████████████████████  │
│  ├── db/                 │  54│     result = db.execute(query)       │
│  │   ├── 🟡 connection.  │  55│     return result                    │
│  │   └──    models.py    │                                          │
│  ├── lib/                │  ─── FIELD HEATMAP ──────────────────── │
│  │   └── 🟡 cache.py     │  ██░░██████░░░░░░██████░░░░░░░░░░░░░░  │
│  └── tests/              │  ░░░░████████░░░░████░░░░░░░░░░░░░░░░  │
│       └──    ...         │  ░░░░░░██████████████░░░░░░░░░░░░░░░░  │
│                          │                                          │
│  AGENTS: ··●●●●●●●··    │  [f] toggle field  [a] toggle agents    │
│  (dots near login.py)    │  [g] replay evolution  [q] quit          │
│                          │                                          │
└──────────────────────────┴──────────────────────────────────────────┘
```

### CI/CD Integration

```yaml
# GitHub Actions
name: Swarm Scan
on: [pull_request]
jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: swarm-scan/action@v1
        with:
          engine-api-key: ${{ secrets.EMERGENCE_ENGINE_API_KEY }}
          agents: 64
          generations: 10
          min-convergence: 3
          format: sarif
      - uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: swarm-results.sarif
```

```yaml
# GitLab CI
swarm-scan:
  image: swarm-scan:latest
  variables:
    EMERGENCE_ENGINE_API_KEY: $EMERGENCE_ENGINE_API_KEY
  script:
    - swarm scan . --format json --output swarm-results.json
  artifacts:
    reports:
      sast: swarm-results.json
```

---

## Scaling Architecture

### What Happens as Codebases Grow

| Codebase Size | Files | Functions (Nodes) | Edges | Field Size (naive) |
|---|---|---|---|---|
| Small project | 50 | 200 | 800 | ~3 KB |
| Medium startup | 500 | 5,000 | 20,000 | ~80 KB |
| Large company | 5,000 | 50,000 | 200,000 | ~800 KB |
| Monorepo | 50,000+ | 500,000+ | 2,000,000+ | ~8 MB+ |

The product handles scaling through hierarchical scanning. The engine doesn't need to change — it just receives smaller graphs.

### Hierarchical Scanning (Product-Side Logic)

```
The PRODUCT, not the engine, handles scale.

Step 1: Product parses codebase into a full graph
Step 2: Product partitions the graph into clusters (modules/packages)
Step 3: Product creates a COARSE graph (one node per module)
Step 4: Product sends coarse graph to engine → engine returns convergence
Step 5: Product identifies hot modules (where agents converged)
Step 6: Product sends DETAILED graphs of only hot modules to engine
Step 7: Product merges coarse + detailed results

The engine always receives manageable-sized graphs.
The product handles the zoom-in/zoom-out logic.
```

```
100K function codebase:

  Level 1 (coarse):
    50 modules → 50 nodes sent to engine
    Engine returns: 5 modules are hot

  Level 2 (detailed):
    5 modules × ~200 functions each → 1,000 nodes per engine call
    5 engine calls, each with ~1,000 nodes

  Total engine work: 50 + 5,000 = 5,050 nodes
  Instead of: 100,000 nodes (naive)
  Reduction: 20x
```

### Incremental Scanning (Product-Side Logic)

```bash
# First scan: full analysis
swarm scan .                    # Sends full graph to engine
                                # Engine saves state to .swarm/state.pkl

# Subsequent scans: only changed files
swarm scan . --resume           # Product computes git diff
                                # Product identifies changed nodes + neighbors
                                # Product sends only changed subgraph to engine
                                # Engine loads previous state, updates only changed region
                                # 95% of the field stays frozen from last scan
```

### Scaling Comparison

| Codebase Size | SonarQube | CodeRabbit | Swarm Scanner |
|---|---|---|---|
| 10K LOC | ~30s | ~5 min (LLM calls) | ~30s |
| 100K LOC | ~5 min | ~20 min | ~2 min (hierarchical) |
| 1M LOC | ~30 min | ~2 hrs (rate limited) | ~10 min (partitioned) |
| 10M LOC | ~3 hrs | Impractical (cost) | ~30 min (multi-level hierarchy) |

Our cost advantage: engine runs tiny local neural networks, not cloud LLM API calls. Cost scales sub-linearly because hierarchy means most of the codebase is analyzed at coarse resolution.

### Hierarchical Agent Tiers (Agent-Level Scaling)

Beyond partitioning the graph (product-side scaling), the engine itself can scale agents hierarchically. Not all agents need to be equally expensive.

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  TIER 1: SCOUTS                                                     │
│  ─────────────────                                                  │
│  Count:    1,000 - 10,000                                           │
│  NN size:  16-dim hidden (tiny)                                     │
│  Field:    Numerical only (no LLM calls)                            │
│  Input:    Structural features: complexity, nesting, call depth,    │
│            parameter count, string operations, external calls       │
│  Job:      Fast graph traversal. Cover the entire codebase.         │
│            Flag "structurally interesting" nodes based on features.  │
│  Output:   Heatmap of suspicious neighborhoods                      │
│  Cost:     ~0 per scout (pure NN inference, no LLM)                 │
│  Time:     <5 seconds for 10K nodes (parallelized on CPU/GPU)       │
│                                                                     │
│  Scouts are the research agents. Same as our current emergence-lab  │
│  agents running on a numerical field. Proven technology.            │
│                                                                     │
│            │                                                        │
│            │  "These 50 nodes (out of 3,000) look interesting"      │
│            ▼                                                        │
│                                                                     │
│  TIER 2: INVESTIGATORS                                              │
│  ─────────────────────                                              │
│  Count:    64 - 256                                                 │
│  NN size:  64-dim hidden (standard)                                 │
│  Field:    LLM-powered (Ollama queries)                             │
│  Input:    Scout heatmap + actual source code via LLM               │
│  Job:      Visit scout-flagged nodes. ASK the LLM targeted          │
│            questions. Leave annotations. Build convergence signal.   │
│            Follow graph edges to trace cross-file data flow.         │
│  Output:   Convergence map with confidence scores                   │
│  Cost:     ~100ms LLM inference per ASK                             │
│  Time:     ~60 seconds for 50 hot nodes × 500 steps                 │
│                                                                     │
│  Investigators are where emergence matters most. They specialize    │
│  (security, performance, data-flow), leave notes for each other,   │
│  and build the convergence signal through independent agreement.    │
│                                                                     │
│            │                                                        │
│            │  "12 nodes have high convergence (10+ agents agree)"   │
│            ▼                                                        │
│                                                                     │
│  TIER 3: VERIFIERS                                                  │
│  ──────────────────                                                 │
│  Count:    8 - 16                                                   │
│  NN size:  128-dim hidden (larger)                                  │
│  Field:    LLM-powered (detailed reasoning mode)                    │
│  Input:    High-convergence nodes + all investigator annotations    │
│  Job:      Deep analysis of confirmed hotspots only. Trace full     │
│            data flow paths. Cross-reference with known CVE          │
│            patterns. Generate human-readable explanations.           │
│  Output:   Verified findings with severity, description, chain      │
│  Cost:     ~500ms LLM inference per detailed query                  │
│  Time:     ~10 seconds for 12 verified nodes                        │
│                                                                     │
│  Verifiers are few but thorough. They only look at what the         │
│  swarm already agreed on. This is where false positives die.        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Cost comparison for a 3,000-node codebase:

  Flat approach (all agents equal, all use LLM):
    64 agents × 500 steps × 1 LLM call = 32,000 LLM calls
    At 100ms each = ~53 minutes

  Tiered approach:
    Scouts:        1,000 agents × 200 steps × 0 LLM calls = 0 LLM calls (~3s)
    Investigators: 128 agents × 500 steps on 50 nodes = 64,000 LLM calls (~107 min)
                   BUT: batched to 8 parallel + only hot nodes = ~8 minutes
    Verifiers:     16 agents × 50 steps on 12 nodes = 800 LLM calls (~7s)
    Total: ~9 minutes with BETTER coverage (scouts saw everything)

  Result: 5.9x faster, better coverage, same or better quality
```

### Engine Deployment Options for Scale

| Deployment | Use Case | Performance |
|---|---|---|
| **Local SDK** | Individual developer, small projects | 64 agents on CPU, ~30s for 10K LOC |
| **Local SDK + GPU** | Developer with GPU, medium projects | 128 agents on GPU, ~10s for 10K LOC |
| **Remote API (shared)** | Teams, CI/CD pipelines | Multi-tenant cloud, GPU-accelerated, ~15s for 10K LOC |
| **Remote API (dedicated)** | Enterprise, large codebases | Dedicated GPU instance, ~5s for 10K LOC |
| **Self-hosted engine** | Air-gapped / regulated environments | Customer runs engine on own infrastructure |

---

## The Problem (Market Context)

### What Every Existing Tool Gets Wrong

Based on deep competitive research across 13 products (CodeRabbit, SonarQube, Snyk, Semgrep, CodeQL, Checkmarx, Veracode, Sourcery, DeepSource, Codacy, Qodana, GitHub Copilot Code Review, Amazon CodeGuru), the industry shares these structural problems:

| Problem | Impact | Evidence |
|---------|--------|----------|
| **Alert fatigue / noise** | ~60% of implementations suffer. Teams start ignoring ALL alerts. | CodeRabbit: "50-50 useful vs useless comments." Checkmarx: 500+ issues, most false positives. |
| **No cross-file intelligence** | Tools analyze diffs in isolation. No persistent codebase model. | "None of these tools maintains a persisted model of the entire codebase" — Qodo analysis |
| **False positive / negative tradeoff** | Best precision: 65% (1 in 3 wrong). Best detection: 42-48%. | Augment Code benchmark: 7 tools, 50 real PRs. >50% of real bugs go undetected. |
| **Configuration overhead** | 35% of implementations need significant tuning before useful. | SonarQube: "steep learning curve." Checkmarx: "requires dedicated administrators." |
| **Business logic blindness** | No tool understands architectural intent or business rules. | BlueDot benchmark: no tool caught a severe S3 region configuration issue. |
| **AI-generated code amplifies issues** | AI code produces 1.7x more issues. 41% of new code is AI-assisted. | CodeRabbit study: 470 real-world open-source PRs. |

### Competitive Landscape Summary

**Tier 1 — Established SAST/SCA:**

| Tool | Approach | Price | Key Weakness |
|------|----------|-------|-------------|
| **SonarQube** | Rule-based AST, 6500+ rules, 35 languages | Free–$35,700/yr | No SCA. Very noisy. Declining market share (26.5% → 19.2%). |
| **Snyk** | Hybrid AI (DeepCode) + SCA + Container + IaC | Free–$57/dev/mo | Expensive ($96K/yr avg enterprise). SAST called "very weak." |
| **Checkmarx** | Graph-based SAST, customizable queries | Custom (expensive) | Extremely high false positive rate. Needs 32GB+ RAM. |
| **Veracode** | Binary SAST (unique). Claims <1.1% FP rate. | $15K–$500K+/yr | SaaS-only. Very expensive. Weak on modern architectures. |
| **Semgrep** | Lightweight AST pattern matching, YAML rules | Free–$40/dev/mo | OSS version is single-file only. Pattern-based, not model-based. |
| **CodeQL** | Code-as-database, QL query language | Free (OSS)–$49/mo | 44% FP rate. Requires build. Closed-source. Steep QL learning curve. |

**Tier 2 — AI-Native Code Review:**

| Tool | Approach | Price | Key Weakness |
|------|----------|-------|-------------|
| **CodeRabbit** | Multi-model LLM + 40 linters + vector DB | Free–$24/dev/mo | 44% bug catch rate. Noise problem. 10-20 min per review. |
| **GitHub Copilot Review** | LLM + CodeQL + ESLint | $10–$39/mo | "Too noisy, low value." Surface-level. Declining quality reports. |
| **Sourcery** | LLM (OpenAI) + rules | Free–$24/user/mo | Slow (>1 min/file). No cross-file. Python-centric. |
| **DeepSource** | Static analysis + Gemini agents | Free–$24/user/mo | Core is traditional SAST. LLM dependency. |
| **Codacy** | Rules (49 languages) + AI reviewer | Free–$18/user/mo | Legacy perception. AI features less mature. |
| **Qodana** | Deterministic (IntelliJ engine) | Free–EUR 1/contributor/mo | Not LLM-powered. No auto-fix. |

**No existing tool uses true swarm intelligence** — emergent specialization, stigmergic knowledge sharing, evolutionary self-improvement. This is the gap.

---

## The Three Innovations

### 1. Convergence-Based Alerting (Kills Alert Fatigue)

Traditional: one engine flags an issue → alert fires → probably false positive.

Ours: a finding surfaces only when N independent agents converge on the same node. Each agent has mutated weights (different "instincts"). If 31/64 agents independently flag the same function, that's high confidence. If 1 agent flags something, it's noise — suppressed.

Confidence = independent convergence, not single-model probability.

### 2. Shared Knowledge Field (Enables Cross-File Intelligence)

Traditional: analyze each file independently.

Ours: agents deposit marks on a shared field as they crawl the code graph. Suspicious marks at `auth.py` diffuse along edges to `connection.py` and `user.py`. Other agents read these marks and follow the trail. The field builds a persistent, evolving map of cross-file vulnerability chains — without anyone programming the connections.

### 3. Evolutionary Self-Improvement (Adapts to Your Codebase)

Traditional: same rules for every codebase. Manual tuning required.

Ours: agents that find real bugs reproduce. False-positive generators die. Over time, the swarm evolves specialists for YOUR codebase's patterns. SQL-heavy code → more injection hunters. Async-heavy code → more concurrency detectors. Nobody configures this. It emerges.

---

## Training Strategy: Why the LLM Field Eliminates Pre-Training

### The Old Approach (Numerical Field — Research Only)

With a pure numerical field, agents need extensive pre-training:

| What agents must learn from scratch | How long it takes |
|-------------------------------------|-------------------|
| What a SQL injection looks like | Millions of training steps |
| What unsafe data flow means | Millions of training steps |
| What a buffer overflow is | Millions of training steps |
| How to interpret field signals | Millions of training steps |

This requires labeled vulnerability datasets (OWASP, Juliet, CVE Fix, etc.) and extensive compute time. The agents must evolve ALL knowledge from zero because the field is a dumb array of floats that carries no inherent meaning.

**This is the correct approach for research** — it proves emergence from first principles with minimal assumptions.

### The New Approach (LLM Field — Product)

With an LLM-powered field, agents need ZERO pre-training on vulnerability data:

```
Agent arrives at a code node containing:
  def login(request):
      query = "SELECT * FROM users WHERE name='" + request.POST["user"]
      result = db.execute(query)

Agent action: ASK the medium "What is suspicious about this code?"

LLM field responds: "This function concatenates user input directly
  into a SQL query string without parameterization. This is a classic
  SQL injection vulnerability (CWE-89). User-controlled input from
  request.POST['user'] flows directly into the query variable which
  is then executed via db.execute()."

Agent action: ANNOTATE this node with "SQL injection — unsanitized input"
Agent action: FLAG this node for convergence tracking

The agent didn't need to be trained on what SQL injection is.
The medium already knew.
```

**What agents DO need to evolve (through natural selection):**

| What evolves | Why it matters |
|---|---|
| Exploration strategy | Which nodes to visit, in what order, how to navigate the graph efficiently |
| Question quality | Agents that ask vague questions waste LLM compute and find less. Agents that ask targeted questions find more and survive. |
| Annotation quality | Agents that leave useful notes help other agents find related issues. Agents that leave useless notes don't contribute to the collective. |
| When to flag vs. move on | Agents that flag everything are noisy (false positives → energy penalty). Agents that never flag miss bugs (no energy gain). The balance evolves. |
| Specialization | Some agents evolve to ask security questions. Others evolve to ask performance questions. Others evolve to trace data flow. Nobody assigns these roles. |
| LLM prompt engineering | This is the most interesting emergence: agents evolve their "instincts" for how to query the LLM effectively. Different evolved weights lead to different query strategies. |

### How the Feedback Loop Works (Per User)

```
Scan 1: Fresh swarm + LLM field → scan codebase → findings
        User confirms finding #1 (real bug) → +10 energy to agents that converged there
        User dismisses finding #3 (intentional) → -5 energy to agents that converged there
        Swarm evolves for 5 generations with this feedback
        State saved to .swarm/state.pkl

Scan 2: Load evolved swarm → scan again → better results
        Agents that wasted time on intentional patterns are dead
        Agents that found real bugs have reproduced
        New generation asks better questions, explores more efficiently

Scan 3: Even better...

The swarm doesn't learn what a vulnerability IS (the LLM knows that).
The swarm learns what matters in YOUR SPECIFIC codebase.
```

### Comparison: Training Requirements

| Approach | Pre-training data needed | Time to first useful scan | Improves over time |
|---|---|---|---|
| SonarQube | 6,500+ manually written rules | Immediate (rule-based) | Only via manual rule updates |
| CodeRabbit | GPT-4/Claude training (done by OpenAI/Anthropic) | Immediate (LLM-based) | Thumbs up/down, limited |
| Swarm + Numerical Field | OWASP, Juliet, CVE Fix datasets + millions of training steps | Days to weeks of training | Yes, via evolution |
| **Swarm + LLM Field** | **NONE** | **Immediate** (LLM already knows code) | **Yes, via evolution** (strongest improvement) |

The LLM field gives us immediate usefulness (like CodeRabbit) PLUS evolutionary self-improvement (unique to us) PLUS convergence-based filtering (unique to us). Best of all worlds.

---

## Go-to-Market Strategy

### Launch Plan

**Phase 1: Open-Source CLI (awareness)**
- Release `swarm-scan` as open-source (MIT license)
- Ships with local engine SDK (no API key needed for basic use)
- Focus on the visual demo: agents swarming code in the terminal
- Channel: Hacker News, Reddit r/programming, Twitter/X

**Phase 2: Pro Features via Engine API (revenue)**
- Pro tier unlocks the cloud engine:
  - More agents (128 vs 64)
  - More generations (50 vs 10)
  - Persistent state across CI/CD
  - SARIF export
  - Priority engine access
- Price: $15/developer/month

**Phase 3: Team/Enterprise (scale)**
- Shared swarm state across organization
- Self-hosted engine option
- Custom training on internal vulnerability patterns
- Compliance reporting
- Price: $30/developer/month

### Revenue Model

```
Revenue comes from ENGINE API USAGE, not from the product itself.

The product (CLI) is free and open-source.
The engine (local SDK) is free for basic use.
The engine (cloud API) is paid:

  Free:       1,000 steps/month    (test it out)
  Starter:    100,000 steps/month  ($49/mo)   ~15 full scans
  Growth:     1,000,000 steps/month ($199/mo)  ~150 full scans
  Scale:      10,000,000 steps/month ($499/mo) ~1,500 full scans
  Enterprise: Unlimited             (custom)

This model means:
  - Any product built on the engine generates revenue
  - Code scanner is product #1; security monitor is product #2; etc.
  - The engine is the business. Products are distribution channels.
```

### Enterprise Platform: Companies Train Their Own Models

The most powerful business model isn't selling scans — it's selling the engine as a platform for companies to build their own swarm-powered systems. Companies bring their own graph, their own LLM, their own reward signals. The engine provides the swarm intelligence layer.

**How it works:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  EMERGENCE ENGINE PLATFORM                                          │
│  ═════════════════════════                                          │
│                                                                     │
│  What WE provide:                                                   │
│    - The swarm simulation engine (agents, evolution, field, species)│
│    - The API / SDK                                                  │
│    - Agent architectures (scout, investigator, verifier tiers)      │
│    - Convergence algorithms                                         │
│    - Specialization tracking                                        │
│    - Scaling infrastructure                                         │
│                                                                     │
│  What the CUSTOMER provides:                                        │
│    - Their graph (their domain: code, network, market, supply chain)│
│    - Their LLM (their Ollama instance, their fine-tuned model,     │
│      or their cloud LLM API key)                                   │
│    - Their reward signals (what counts as a "good find" in their   │
│      domain — bug reports, incident data, expert ratings)          │
│    - Their data (never leaves their infrastructure)                 │
│                                                                     │
│  What they GET:                                                     │
│    - A swarm that evolves to be an expert at THEIR specific domain │
│    - Emergent specialization tuned to THEIR patterns                │
│    - Self-improving system that gets better with every feedback loop│
│    - Cross-node intelligence via field that no single-pass tool has │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Enterprise use cases:**

| Customer | Their Graph | Their LLM | Their Rewards | What Emerges |
|----------|-------------|-----------|---------------|--------------|
| **Bank** | Transaction flow graph (accounts, transfers, merchants) | Fine-tuned fraud detection LLM | Confirmed fraud cases | Fraud pattern hunters, money-laundering chain tracers, anomaly detectors — specialized species per fraud type |
| **Hospital** | Patient record graph (symptoms, diagnoses, treatments, outcomes) | Medical LLM (Med-PaLM, BioMistral) | Expert-validated diagnoses | Diagnostic specialists, drug interaction checkers, rare condition detectors |
| **Logistics company** | Supply chain graph (warehouses, routes, suppliers, demand centers) | Operations LLM | Known bottleneck resolutions | Route optimizers, demand forecasters, supplier risk assessors |
| **Cybersecurity firm** | Network topology graph (endpoints, connections, traffic patterns) | Security LLM (SecLM, fine-tuned Llama) | Confirmed intrusion incidents | Lateral movement trackers, exfiltration detectors, C2 communication finders |
| **Pharma company** | Molecular graph (compounds, reactions, targets, pathways) | Chemistry LLM (ChemLlama) | Successful drug candidates | Binding site predictors, toxicity screeners, novel pathway explorers |

**Enterprise pricing tiers:**

```
Engine Platform Pricing:

  SDK License (self-hosted):
    Annual license fee: $50,000 - $200,000/year
    Customer runs everything on their infrastructure
    Includes: SDK, documentation, support SLA, quarterly updates
    Data: never leaves customer's network

  Engine-as-a-Service (EaaS):
    Managed cloud deployment in customer's cloud (VPC)
    Pay per simulation step: $0.001 per 1,000 steps
    Typical enterprise usage: 10M-100M steps/month = $10K-$100K/month
    Includes: managed infrastructure, monitoring, auto-scaling
    Data: stays in customer's VPC

  Consulting + Custom Integration:
    Custom graph adapters for customer's domain
    Custom reward signal design
    Custom LLM fine-tuning guidance
    Custom agent architecture tuning
    $200-$400/hour or fixed-price engagements

  Engine Marketplace (future):
    Community-contributed "domain packs":
      - Graph adapters (parse X domain into engine graph format)
      - Reward templates (common reward signals for a domain)
      - Pre-evolved swarm states (kickstart with agents already specialized)
    Revenue share: 70% creator / 30% platform
```

**Why companies would pay for this instead of building their own swarm:**

1. **The engine is the hard part.** Building a swarm simulation with evolution, field dynamics, species detection, convergence tracking, and scaling — that's years of research. The graph adapter is the easy part.
2. **Research backing.** Our published research proves emergence works. Companies don't want to gamble on unproven approaches.
3. **Continuous improvement.** As we advance the engine (better evolution algorithms, better scaling, new agent architectures), all customers automatically benefit.
4. **Cross-domain insights.** We learn patterns from running the engine across many domains — a convergence pattern that works for fraud detection might also work for code scanning. Customers benefit from this shared learning (while their data stays private).

---

## Other Products the Engine Can Power

The same Emergence Engine, unchanged, can power multiple products by changing what the graph represents:

| Product | Graph Nodes | Graph Edges | Food/Reward | Field Meaning |
|---------|-------------|-------------|-------------|---------------|
| **Swarm Code Scanner** | Functions/files | Imports, calls, data flow | Bugs found | Where the swarm thinks issues are |
| **Swarm Security Monitor** | Network endpoints/services | Network connections, API calls | Anomalies detected | Shared threat intelligence map |
| **Swarm Research Assistant** | Document chunks | Citation links, semantic similarity | Relevant passages found | Shared relevance/importance map |
| **Swarm Market Intelligence** | Web pages/data sources | Links, topic similarity | Competitive insights found | Shared market knowledge map |
| **Swarm Logistics Optimizer** | Warehouses/routes | Roads, shipping lanes | Efficient routes found | Shared congestion/efficiency map |

Each product is a thin translation layer. The engine does the heavy lifting.

**This is the real business model:** build one powerful engine, sell it through many domain-specific products.

---

## The Central Brain: Collective Learning Across All Swarms

### The Limitation of Local-Only Swarms

Without a central brain, every customer's swarm is isolated:

```
Company A runs swarm on their codebase → swarm evolves for Company A
Company B runs swarm on their codebase → swarm evolves for Company B

Company A's swarm knows nothing about Company B.
Company B's swarm knows nothing about Company A.
They never talk. They're completely independent.
```

Each swarm is like a doctor who only ever sees one patient. They get really good at treating THAT patient, but they never learn from other patients.

### Federated Learning: Every Swarm Makes Every Other Swarm Smarter

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  CENTRAL BRAIN (our servers)                                     │
│                                                                  │
│  Collects from all customer swarms:                              │
│    - Which agent STRATEGIES work across many codebases           │
│      (not the code itself — just the evolved agent weights)      │
│    - Which finding TYPES get confirmed vs dismissed              │
│    - Which species PATTERNS are universally useful               │
│                                                                  │
│  Does NOT collect:                                               │
│    - Customer source code (NEVER)                                │
│    - Customer-specific findings (NEVER)                          │
│    - Any identifiable data (NEVER)                               │
│                                                                  │
│  Produces:                                                       │
│    - "Global swarm" — pre-evolved agents with industry-wide      │
│      instincts, updated monthly                                  │
│    - "Species library" — proven specialist types that work       │
│      across codebases                                            │
│    - "Pattern confidence" — global stats on what's usually       │
│      a real bug vs usually a false positive                      │
│                                                                  │
└──────────────────────┬───────────────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
     Company A    Company B    Company C

     Each company:
     1. Downloads the global swarm as a starting point
     2. Evolves it further on their specific codebase
     3. Sends back anonymized learnings (agent weights +
        confirm/dismiss stats, NOT source code)
```

### What Gets Sent Back (And What Doesn't)

| Sent back (safe) | NOT sent back (private) |
|---|---|
| Agent weight vectors (just numbers, meaningless without context) | Source code |
| "Finding type X was confirmed 80% of the time" | The actual code that was flagged |
| "Species with these behavioral features performed well" | File names, function names |
| "Swarm converged on 12 nodes in this scan" | What those nodes contained |
| Aggregate stats: scan count, confirm rate, species count | Any identifiable information |

Think of it like how a phone keyboard learns. It sends back "people often type 'the' after 'in'" — it doesn't send back your actual messages. We send back "agents that ask about input sanitization near database calls get confirmed 85% of the time" — not the actual code.

### How the Flywheel Compounds

```
Day 1:     10 companies using the tool
           Central brain has basic patterns from 10 codebases

Month 6:   500 companies using the tool
           Central brain has seen patterns across 500 codebases
           New customers get a swarm that's already seen 500 codebases worth of bugs

Year 2:    10,000 companies
           Central brain has the most comprehensive bug pattern database
           in existence — built by evolution, not by human rule-writers

           A NEW competitor starts. Their swarm has seen 0 codebases.
           Ours has seen 10,000. They can never catch up because
           the learning compounds.
```

This is a **data network effect.** Every new customer makes the product better for every other customer. Same reason Google Search got better as more people used it — more queries = better ranking = more users = more queries.

SonarQube's 6,500 rules were written by humans over years. Our "rules" would be evolved across thousands of real codebases automatically.

### Three Privacy Models

| Model | What's shared | Who it's for |
|---|---|---|
| **Fully Local** | Nothing. Ever. Your swarm only learns from YOUR codebase. You don't benefit from the collective. | Air-gapped enterprises, regulated industries, free tier |
| **Federated** | Anonymized agent weights + confirm/dismiss stats. No source code leaves your machine. You benefit from the collective brain. | Standard paid tier — best value |
| **Community** | Full findings shared (with consent) in a public database. Like a community CVE database but built by swarms. | Open-source projects |

### Emergence at Two Levels

This is actually emergence happening at TWO scales:

```
Level 1: Emergence WITHIN a swarm
  64 agents → collective intelligence about ONE codebase
  This is what the research proves (Phase 1-6)

Level 2: Emergence ACROSS swarms
  10,000 swarms → collective intelligence about ALL code everywhere
  This is the product endgame

Level 1 is ants in a colony.
Level 2 is colonies sharing knowledge across an ecosystem.
```

Level 1 proves the science works. Level 2 is where the real business value is — the flywheel that makes this a platform, not just a tool.

---

## Feedback Resilience: What If the Developer Is Wrong?

### The Problem

A developer dismisses a real bug by mistake. The agents that found it lose energy, some die, and the swarm learns the wrong lesson. Bad feedback corrupts the swarm.

### Why It's Not Catastrophic

One dismiss doesn't kill a strategy — it kills a few agents. The strategy itself exists across many agents in the population. Evolution is noisy by design. A few bad signals don't erase a strong instinct.

But **consistent** bad feedback IS dangerous. If a developer keeps dismissing a real vulnerability category (e.g., XSS warnings), agents that hunt XSS will gradually die off. That species goes extinct. The swarm becomes blind to XSS.

### Five Mitigations

**1. Species Protection (Minimum Viable Population)**

No species can drop below 5% of the population, regardless of feedback. Even if every security finding gets dismissed, security-hunter agents survive at minimum levels. Like endangered species protection.

The MAP-Elites archive from Phase 4 already does this — it preserves behavioral diversity even when selection pressure pushes against it.

**2. Confidence Warnings**

If the LLM is highly confident AND many agents converged, but the developer dismisses:

```
You dismissed: SQL injection in login.py:47
LLM confidence: 95% | Agent convergence: 31/64

Are you sure? This has strong signals.
  [Yes, dismiss anyway]  [Let me look again]
```

**3. Graded Dismiss Reasons**

Not all dismissals mean "this isn't a bug":

```
Why are you dismissing this?

  [Not a bug]        → agents lose energy (real negative signal)
  [Known/accepted]   → agents lose NO energy (it IS a bug, we accept the risk)
  [Won't fix]        → agents lose NO energy (real but low priority)
  [Not sure]         → agents lose NO energy (inconclusive)
```

Only "Not a bug" actually penalizes agents. Everything else is neutral. The swarm only learns from confident developer judgment.

**4. Team Consensus**

In a team setting, one developer's feedback doesn't override everything:

```
Developer A dismissed finding #7.
Developer B confirmed finding #7.

Result: conflicting → agents get ZERO feedback.
        Finding stays flagged for team discussion.
```

**5. Fresh Swarm Audits**

Every N scans, run a small population of "fresh" agents with no evolutionary history. They look at the codebase with zero bias from previous feedback.

```
Main swarm (evolved): finds 5 issues
Fresh swarm (reset):  finds 8 issues

3 issues the main swarm stopped finding →
Alert: "Your feedback may have suppressed real findings.
        Review these 3 items."
```

This catches feedback corruption before it becomes permanent.

### With the Central Brain, Bad Feedback Self-Corrects

If Company A's developer dismisses SQL injection, but 847 other companies confirm it, the central brain knows SQL injection is real. When Company A's swarm syncs with the global model, the SQL injection hunting instinct gets reinforced — overriding the local bad feedback.

The larger the network, the more resilient it is to any single developer's mistakes.

---

## Success Metrics

### Research Gates (Must Pass Before Building Product)

#### Gate 1: Core Emergence (Numerical Field)

| Metric | Required Result | Maps to Research |
|--------|----------------|-----------------|
| Swarm > single agent | 64 agents find more bugs than 1 agent with 64x compute | Ablation: multi-agent vs single-agent |
| Field adds value | Agents WITH field > agents WITHOUT | Ablation: normal vs zeroed field |
| Convergence reduces FP | <15% FP rate (convergence-filtered) vs >40% (individual agents) | Convergence threshold analysis |
| Evolution improves | Generation 20 > generation 1 on same codebase | Training curve analysis |
| Specialization emerges | Distinct species form spontaneously | Specialization tracking |

#### Gate 2: LLM Field Intelligence Ablation

The critical question: **does giving agents an intelligent medium kill emergence, or enhance it?**

If the LLM medium is too smart, agents may become irrelevant — the LLM does all the work, and agents add no value beyond a single LLM call. This must be tested before investing in the product.

| Experiment | Setup | What We Measure | Pass Condition |
|------------|-------|-----------------|----------------|
| **LLM-only baseline** | 1 agent, LLM field, visits every node once | Bugs found, false positives | Baseline to beat |
| **Swarm + dumb field** | 64 agents, numerical field, no LLM | Bugs found, false positives | Proves agents add value without LLM |
| **Swarm + smart field** | 64 agents, LLM field | Bugs found, false positives | Must beat BOTH baselines |
| **LLM intelligence throttle** | Vary LLM capability: tiny (1B) → small (7B) → medium (34B) → large (70B) | Agent contribution margin at each level | Sweet spot where agents + LLM > LLM alone |
| **Agent complexity sweep** | Fix LLM at 7B. Vary agent NN size: 16 → 32 → 64 → 128 hidden dims | Whether smarter agents help with smart medium | Identify minimum agent complexity needed |
| **Emergence metrics** | For each setup above: specialization score, species count, weight divergence | Whether emergence HAPPENS with LLM field | Specialization score > 0.5, species detected = True |

**Key hypothesis to test:** There's a sweet spot where the LLM is smart enough to understand code but dumb enough that agent exploration strategy, specialization, and convergence still add significant value. Our bet is that a local 7B-8B model hits this sweet spot — it knows what SQL injection is but can't reliably find cross-file vulnerability chains on its own. The swarm provides the exploration, convergence, and cross-file intelligence that the small LLM lacks.

**Failure mode:** If swarm + LLM field performs no better than a single LLM pass over all nodes, the product reduces to "yet another LLM scanner" and our moat disappears. This is the existential risk to the product thesis.

#### Gate 3: Optimal Agent Count

Does scaling from 32 to 10,000+ agents improve results? Or do diminishing returns kick in early?

| Experiment | Agent Count | What We Measure | Expected Outcome |
|------------|------------|-----------------|------------------|
| Baseline | 32 | Bugs found, time, convergence quality | Current performance |
| 2x | 64 | Same | Meaningful improvement expected |
| 4x | 128 | Same | Still improving, but gains slowing |
| 8x | 256 | Same | Diminishing returns starting |
| 32x | 1,024 | Same | Likely diminishing, but more species may emerge |
| 300x | 10,000 | Same | Test whether new dynamics appear at scale |

**Why 10,000+ agents might matter — or not:**

Arguments FOR scaling:
- More agents = more independent perspectives = higher convergence confidence
- At large scale, rare specializations can emerge (an agent that only checks crypto operations, for example) — with 32 agents, there aren't enough population slots for niche specialists
- More agents explore more of the graph per generation = faster convergence
- Statistical power: 100/10,000 agents converging is a stronger signal than 5/32

Arguments AGAINST scaling:
- LLM is the bottleneck, not agents. With LLM field, each agent ASK takes ~100ms of LLM inference. 10,000 agents × 500 steps × 1 query/step = 5,000,000 LLM calls per scan. At 100ms each = ~139 hours sequential. Even with batching, this is impractical.
- Diminishing returns on convergence: the difference between 50/64 converging and 500/640 converging is not 10x better signal — it's marginally better.
- Communication overhead: more agents writing annotations = more noise in the shared medium.
- Memory: 10,000 agent neural networks × weight arrays = significant RAM.

**Practical scaling architecture (if agent count helps):**

```
Hierarchical Agent Tiers:

Tier 1 — SCOUTS (cheap, many, no LLM)
  Count: 1,000-10,000
  Cost per agent: ~0 (tiny NN, numerical field only)
  Job: Fast graph traversal. Identify suspicious neighborhoods
       using only structural features (complexity, nesting, call depth).
  Output: "These 50 nodes look structurally interesting" (out of 3,000)

Tier 2 — INVESTIGATORS (moderate, fewer, some LLM)
  Count: 64-256
  Cost per agent: Low (small LLM queries)
  Job: Visit nodes flagged by scouts. ASK the LLM targeted questions.
       Leave annotations. Build the convergence signal.
  Output: Convergence map with confidence scores

Tier 3 — VERIFIERS (expensive, few, heavy LLM)
  Count: 8-16
  Cost per agent: High (detailed LLM reasoning)
  Job: Visit only high-convergence nodes (where 10+ investigators agreed).
       Deep LLM analysis: trace full data flow, check across files,
       generate human-readable explanation.
  Output: Verified findings with detailed descriptions

Total LLM calls: 256 × 500 + 16 × 50 = 128,800 (manageable)
vs. flat 10,000 agents: 10,000 × 500 = 5,000,000 (impractical)
```

This tiered approach gets the exploration benefits of 10,000+ agents (scouts are cheap) with the intelligence of LLM-powered analysis (investigators + verifiers) without the cost explosion.

#### Gate 4: Cross-Domain Generalization

Can the same engine, without modification, work on a different domain just by changing the graph?

| Test Domain | Graph | Reward Signal | Pass Condition |
|-------------|-------|---------------|----------------|
| Code scanning (primary) | AST function graph | Developer confirms bug | Baseline |
| Network security | Network topology graph | Known anomalies | Agents converge on known vulnerabilities |
| Research papers | Citation graph | Expert-rated relevance | Agents find related work that experts agree is relevant |
| Logistics | Warehouse/route graph | Known bottlenecks | Agents identify inefficient routes |

This gate validates the "one engine, many products" thesis. If the engine only works for code scanning, the platform business model collapses.

### Product KPIs

| Metric | Target | Rationale |
|--------|--------|-----------|
| Bug catch rate | >50% on OWASP Benchmark | Competitive with existing tools |
| False positive rate | <15% | Must beat industry average (~35-44%) |
| Cross-file unique finds | >10% of findings | Proves swarm value over single-file tools |
| Scan time | <60s for typical project | CI/CD compatible |
| Developer NPS | >40 | Developers like using it |
| OSS weekly active users | 1,000 in first 3 months | Adoption traction |
| Pro conversion | >5% of free users | Revenue viability |
| Engine API revenue | $10K MRR within 6 months of launch | Business viability |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **LLM medium kills emergence** | Medium-High | **Fatal** | Gate 2 ablation tests this explicitly. If a single LLM pass matches swarm+LLM, agents add no value and the product thesis collapses. Mitigation: test with throttled LLM (small models like 1B-7B) where agent exploration still adds value. The sweet spot is likely a model smart enough to understand code but not smart enough to find cross-file chains alone. |
| Swarm doesn't outperform single agent | Medium | Fatal | Research ablation tests determine this BEFORE product investment. |
| Too slow for CI/CD | Medium | High | Hierarchical agent tiers (scouts/investigators/verifiers) + graph partitioning + incremental scanning + GPU acceleration. |
| 10,000+ agents impractical with LLM | High | Medium | Tiered architecture: scouts (no LLM, cheap, many) + investigators (LLM, moderate) + verifiers (LLM, few). Don't send all agents through the LLM. |
| Training data insufficient | Low | High | With LLM field, pre-training data is NOT needed. LLM already knows vulnerability patterns. Agents evolve exploration strategy, not domain knowledge. |
| Competitors copy multi-agent approach | High | Medium | Moat is the engine + published research proving emergence works + years of head start on evolution algorithms. |
| Developers don't trust "swarm" framing | Medium | Medium | Lead with results ("found 3 cross-file vulns"), not mechanism. |
| Engine API latency too high | Medium | Medium | Offer local SDK as fallback. Cloud API for CI/CD where latency is acceptable. |
| Engine dependency creates single point of failure | Low | High | Local SDK mode works without internet. Engine is our own system. |
| Enterprise customers demand custom engines | Medium | Medium | SDK license model lets them self-host. Consulting tier supports custom integration. Platform approach means we don't need to maintain N forks. |
| LLM hallucinations propagate through swarm | Low | Medium | Convergence filtering is a natural defense — hallucinated findings from one agent's LLM query are suppressed when other agents don't independently converge. Multi-agent diversity is the mitigation. |
| Bad developer feedback corrupts the swarm | Medium | Medium | Species protection (minimum 5% population), graded dismiss reasons, confidence warnings, team consensus, fresh swarm audits, and central brain override from global network. |

---

## Development Roadmap

### Milestone 1: Research Validation (Emergence Engine — Numerical Field)
- Complete field ablation at scale (10M+ steps)
- Generalize engine from grid to arbitrary graphs
- Prove emergence on code-graph-like environments
- **GATE 1: proceed only if ablation proves swarm + field > single agent**

### Milestone 2: LLM Field Intelligence Ablation (Critical Research)
- Implement pluggable field backend (numerical + LLM)
- Integrate Ollama for local LLM inference
- Run Gate 2 ablation matrix: LLM-only vs swarm-only vs swarm+LLM
- Test LLM intelligence throttle: 1B → 7B → 34B → 70B models
- Measure agent contribution margin at each LLM intelligence level
- Verify that emergence metrics (specialization, species, divergence) survive LLM field
- **GATE 2: swarm + LLM must outperform LLM-only baseline. If it doesn't, re-evaluate product thesis.**
- **GATE 2b: emergence must still occur (specialization score > 0.5, species detected) with LLM field**

### Milestone 3: Agent Scaling Research
- Run Gate 3 agent count sweep: 32 → 64 → 128 → 256 → 1,024 → 10,000
- Measure diminishing returns curve for each count
- Implement and test hierarchical agent tiers (scouts/investigators/verifiers)
- Benchmark tiered vs flat approach on cost, speed, and quality
- Determine optimal tier ratios (how many scouts per investigator per verifier)
- **GATE 3: identify the cost-effective agent configuration. Document the scaling curve.**

### Milestone 4: Engine API (Emergence Engine)
- Package engine as installable Python SDK
- Define clean API surface (simulate, feedback, state management)
- Support both field modes via config
- Support tiered agent architecture via config
- Engine test suite independent of any product
- Local and remote execution modes
- Run Gate 4: cross-domain generalization test (code, network, research, logistics)
- **GATE 4: engine must work on at least 2 domains without modification**

### Milestone 5: Product Prototype (Swarm Code Scanner)
- Code parser (tree-sitter based)
- Engine client (calls engine SDK with LLM field mode + tiered agents)
- Basic result translator (convergence → findings)
- Basic CLI output
- `ollama pull llama3:8b && swarm scan .` works end-to-end
- No pre-training needed — immediate first scan

### Milestone 6: Product Alpha
- Full CLI with rich terminal output
- SARIF export for GitHub/GitLab
- Python, JavaScript, TypeScript support
- Incremental scanning
- Feedback loop (confirm/dismiss → evolution)
- Publish benchmarks vs SonarQube, Semgrep, and CodeRabbit

### Milestone 7: Product Beta
- TUI visualization
- CI/CD integration (GitHub Actions, GitLab CI)
- Support multiple LLM providers (Ollama, OpenAI, Anthropic, Google)
- Cloud engine API option
- Go, Java, Rust support

### Milestone 8: GA Launch
- Open-source product on GitHub + PyPI
- Engine cloud API with usage-based pricing
- Documentation and tutorials
- Launch: Hacker News, Reddit, Twitter/X
- First 100 paying users

### Milestone 9: Enterprise Platform
- SDK license model: companies self-host the engine
- Engine-as-a-Service (EaaS) in customer VPC
- Custom graph adapter framework (make it easy for companies to plug their domain in)
- Custom reward signal templates
- Enterprise documentation: "How to build a swarm-powered [X] using the Emergence Engine"
- First 3 enterprise customers across different domains
- Consulting pipeline for custom integrations

### Milestone 10: Platform Expansion
- Second first-party product (Swarm Security Monitor or Swarm Research Assistant)
- Engine marketplace for community-contributed domain packs
- Pre-evolved swarm states (share evolved agents across similar domains)
- Specialized fine-tuned LLM for the medium (trained on security + code patterns)
- Cross-domain insights: patterns that work in one domain applied to others

---

## Appendix A: Three Approaches Compared

| Dimension | LLM-Only (CodeRabbit, Copilot) | Swarm-Only (Numerical Field) | **Swarm + LLM Field (Ours)** |
|-----------|---|---|---|
| **Pre-training needed** | Done by OpenAI/Anthropic | Extensive (OWASP, CVE datasets) | **NONE — LLM already knows code** |
| **Time to first useful scan** | Immediate | Days/weeks of training | **Immediate** |
| **Cost per scan** | High ($0.50-$5.00 cloud LLM) | Near-zero (local computation) | **Low ($0 with Ollama locally)** |
| **Latency** | 10-20 minutes | <60 seconds | **1-3 minutes (LLM inference)** |
| **Privacy** | Code sent to cloud LLMs | 100% local | **100% local with Ollama** |
| **Self-improvement** | Limited (thumbs up/down) | Strong (evolution) | **Strongest (evolution + semantic feedback)** |
| **Cross-file reasoning** | Limited by context window | Field propagates across graph | **LLM reasons across graph + field propagates** |
| **False positive handling** | Manual dismissal | Evolution kills FP-prone agents | **Convergence filtering + evolution + LLM hallucination defense** |
| **Specialization** | None (one model does everything) | Emergent (species form) | **Emergent (species form around different question types)** |
| **LLM hallucination risk** | High (single pass, no verification) | N/A | **Low (convergence filters out single-agent errors)** |
| **Offline capability** | Requires internet | Fully offline | **Fully offline with Ollama** |
| **Novel bug detection** | Limited to LLM training data | Emergent discovery | **LLM knowledge + emergent discovery** |
| **Scaling cost** | Linear with codebase size | Sub-linear | **Sub-linear (agents explore selectively)** |

## Appendix B: Taglines and Positioning

**For developers:**
> "Install Ollama. Run swarm scan. No config. No training. No cloud. It just works."

**For security teams:**
> "47 agents found what no single scanner could. Zero pre-training required."

**For engineering leaders:**
> "Stop configuring rules. Stop training models. Drop a swarm on your code and let intelligence emerge."

**For the pitch deck:**
> "CodeRabbit uses one LLM brain. We put 64 evolving agents on a shared LLM medium. Each agent asks different questions, leaves notes for others, and converges independently on what matters. False positives are filtered by consensus, not configuration. The system gets smarter every scan through evolution. No training data needed — the LLM medium already understands code. The agents just learn to use it better than any single prompt ever could."

**For the engine as a platform:**
> "One engine. Any LLM as the medium. Any graph as the environment. Emergence as a service."

**The one-liner:**
> "Swarm Code Scanner: 64 agents, one shared brain, zero training required."

**The technical one-liner:**
> "Stigmergic intelligence on an LLM substrate — convergence-filtered, evolutionarily self-improving code analysis."
