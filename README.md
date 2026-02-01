# Emergence Lab

**Can collective intelligence emerge in the space BETWEEN agents?**

Emergence Lab is a JAX-based multi-agent reinforcement learning environment where agents interact through a **shared learnable field**. Unlike typical multi-agent RL where agents coordinate through actions alone, agents here read and write to a shared 2D medium that has its own physics (diffusion and decay). The field is trained end-to-end with the agents via PPO.

The hypothesis: the field develops spatial structures that encode collective knowledge — information no single agent possesses individually.

## How It Works

```
Each timestep:
1. Field diffuses (3x3 Gaussian blur)
2. Field decays (values * (1 - decay_rate))
3. Agents observe local field + nearby food + own position
4. Agents act (move in 4 directions or stay)
5. Agents write presence to field at their location
6. Food collected when agent is within 1 cell (shared reward)
```

The field acts as a shared external memory — similar to stigmergy in ant colonies, where pheromone trails encode collective knowledge about the environment.

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Install

```bash
# Clone and set up
./scripts/setup.sh

# Or manually:
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Train

```bash
# Quick test run (a few minutes)
source .venv/bin/activate
python -m src.training.train --train.total-steps 100000

# Full training run with W&B logging
./scripts/train.sh --train.total-steps 10000000 --log.wandb true

# Override any config field via CLI
python -m src.training.train --env.grid-size 32 --env.num-agents 16 --train.learning-rate 0.001
```

All CLI arguments map to config fields via [tyro](https://github.com/brentyi/tyro). Run `python -m src.training.train --help` for the full list.

### Evaluate & Visualize

```bash
# Run ablation test (normal vs zeroed vs random field)
python -m src.analysis.ablation --checkpoint checkpoints/params.pkl
```

### Run Tests

```bash
# All tests
pytest tests/ -v

# Type check
python -m mypy src/ --ignore-missing-imports
```

## Project Structure

```
emergence-lab/
├── src/
│   ├── configs.py              # Dataclass configs (EnvConfig, FieldConfig, etc.)
│   ├── environment/
│   │   ├── env.py              # reset() and step() functions
│   │   ├── state.py            # EnvState dataclass
│   │   ├── obs.py              # Observation construction
│   │   ├── vec_env.py          # Vectorized env via jax.vmap
│   │   └── render.py           # RGB frame rendering
│   ├── field/
│   │   ├── field.py            # FieldState dataclass
│   │   ├── dynamics.py         # Diffusion + decay physics
│   │   └── ops.py              # Local read/write operations
│   ├── agents/
│   │   ├── network.py          # ActorCritic (Flax nn.Module)
│   │   └── policy.py           # Action sampling + deterministic policy
│   ├── training/
│   │   ├── train.py            # Full training loop + train_step
│   │   ├── rollout.py          # Trajectory collection via lax.scan
│   │   ├── gae.py              # Generalized Advantage Estimation
│   │   └── ppo.py              # PPO clipped surrogate loss
│   ├── analysis/
│   │   ├── field_metrics.py    # Entropy, structure, mutual information
│   │   ├── ablation.py         # Field ablation experiments
│   │   └── emergence.py        # Phase transition detection
│   └── utils/
│       ├── logging.py          # W&B integration
│       └── video.py            # Episode recording to MP4
├── configs/
│   └── default.yaml            # Default hyperparameters
├── scripts/
│   ├── setup.sh                # Environment setup
│   └── train.sh                # Training launcher with JAX flags
└── tests/                      # Unit tests for all modules
```

## Expected Results

### Training Metrics

During training you should see:

- **Reward** increasing over time as agents learn to forage efficiently
- **Entropy** decreasing as the policy specializes from uniform random
- **Policy loss** and **value loss** stabilizing
- **Approx KL** staying small (~0.01), indicating stable updates

### Field Behavior

The field typically evolves through phases:

1. **Early training**: Field is noisy — agents write randomly, diffusion creates uniform blur
2. **Mid training**: Field begins showing spatial gradients around high-activity areas
3. **Late training**: Field develops stable structures that correlate with food locations and agent paths

### Ablation Results

The ablation test (`src/analysis/ablation`) compares three conditions:

| Condition | What it tests |
|-----------|---------------|
| **Normal** | Field operates as trained (baseline) |
| **Zeroed** | Field values replaced with zeros each step — agents get no field info |
| **Random** | Field replaced with random noise each step — agents get misleading info |

A positive result looks like: **Normal > Zeroed > Random** in mean episode reward, indicating the field carries useful information that agents have learned to exploit.

## Interpreting Field Visualizations

In rendered frames and videos:

- **Background heatmap**: Shows field intensity (sum across all 4 channels). Blue = low values, red = high values.
- **Green dots**: Uncollected food items
- **Colored circles**: Agents (each agent gets a distinct color)
- **Grid lines**: Cell boundaries on the 20x20 grid

What to look for:

- **Hotspots around food**: If the field develops high values near food, agents are using it to mark resources
- **Trail patterns**: Persistent gradients along common paths suggest agents are leaving navigational traces
- **Uniform field**: If the field looks the same everywhere, it hasn't learned useful structure yet
- **Channel specialization**: Different field channels may encode different types of information (though this requires per-channel visualization to observe)

## Configuration

Default parameters (see `configs/default.yaml`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `env.grid_size` | 20 | Grid dimensions (20x20) |
| `env.num_agents` | 8 | Number of foraging agents |
| `env.num_food` | 10 | Food items per episode |
| `env.max_steps` | 500 | Steps per episode |
| `field.num_channels` | 4 | Field channel depth |
| `field.diffusion_rate` | 0.1 | How fast field values spread |
| `field.decay_rate` | 0.05 | How fast field values fade |
| `train.total_steps` | 10M | Total environment steps |
| `train.num_envs` | 32 | Parallel environments |
| `train.learning_rate` | 3e-4 | Adam learning rate |

## Tech Stack

- **JAX** — Accelerated numerical computing
- **Flax** — Neural network library for JAX
- **Optax** — Gradient processing and optimization
- **Weights & Biases** — Experiment tracking (optional)
- **tyro** — CLI argument parsing from dataclasses
