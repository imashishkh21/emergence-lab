# 🧬 Emergence Lab

**Autonomous Multi-Agent Reinforcement Learning for Emergent Intelligence**

This project studies emergent behavior in populations of simple AI agents. Instead of building one sophisticated agent, we create conditions where collective intelligence arises from the interactions of many simple agents.

## The Vision

> "AlphaGo is one genius. We're building a hive mind."

Inspired by:
- OpenAI's Hide-and-Seek (emergent tool use from competition)
- Ant colonies (stigmergy, collective problem-solving)
- Neural assemblies (intelligence from simple neurons)

## What We're Building

A "digital petri dish" where:
- 50+ simple agents compete/cooperate in a grid world
- No agent knows the "big picture" — only local observations
- Emergent strategies arise from multi-agent dynamics
- We measure and detect when emergence happens

## Project Structure

```
emergence-lab/
├── src/
│   ├── environment/     # Grid world, physics, objects
│   ├── agents/          # Neural network policies
│   ├── training/        # PPO/MAPPO implementation
│   └── analysis/        # Emergence detection, metrics
├── configs/             # Hyperparameters, experiment configs
├── tests/               # Unit and integration tests
├── scripts/
│   └── ralph/           # Autonomous build system
└── results/             # Training outputs, videos, logs
```

## Getting Started

### Prerequisites
- Python 3.10+
- Mac with M-series chip (optimized for Apple Silicon)
- Claude Code (for Ralph autonomous building)

### Installation
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### Running Training
```bash
python -m src.training.train --config configs/default.yaml
```

### Running Ralph (Autonomous Building)
```bash
./scripts/ralph/ralph.sh --tool claude 50
```

## The Experiment

### Phase 1: Simple Tag
- Hiders vs Seekers
- Grid world with movable objects
- Can agents learn to build shelters?

### Phase 2: Complexity
- More object types
- Longer episodes
- Larger populations

### Phase 3: Emergence
- Detect novel strategies
- Measure collective intelligence
- Document surprising behaviors

## Research Questions

1. What's the minimum complexity needed for emergence?
2. How does population size affect emergent behavior?
3. Can emergence be reliably reproduced?
4. What metrics best capture "emergence"?

## Built With

- **JAX/Flax** — High-performance ML on Apple Silicon
- **Ralph** — Autonomous AI coding loop
- **Claude Code** — AI pair programmer

## Team

- **Ashish** — Founder, Visionary
- **Titan** — AI Research Partner

---

*"The whole is greater than the sum of its parts." — Aristotle*

🧬 Let's discover what emerges.
