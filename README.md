# Emergence Lab

**Can collective intelligence emerge in the space BETWEEN agents?**

Emergence Lab is a JAX-based multi-agent reinforcement learning environment where agents interact through a **shared learnable field** and evolve under **evolutionary pressure**. Unlike typical multi-agent RL where agents coordinate through actions alone, agents here read and write to a shared 2D medium that has its own physics (diffusion and decay). The field is trained end-to-end with the agents via PPO, while an energy/reproduction system drives population dynamics and behavioral specialization.

The hypothesis: the field develops spatial structures that encode collective knowledge — information no single agent possesses individually — and evolutionary pressure amplifies this emergence.

## How It Works

```
Each timestep:
1. Field diffuses (3x3 Gaussian blur)
2. Field decays (values * (1 - decay_rate))
3. Agents observe local field + nearby food + own position + energy level
4. Agents act (move in 4 directions, stay, or reproduce)
5. Agents write presence to field at their location
6. Food collected when agent is within 1 cell (shared reward, +energy)
7. Energy drains each step; agents with 0 energy die
8. Agents above energy threshold can reproduce (action 5), spawning a mutated child
```

The field acts as a shared external memory — similar to stigmergy in ant colonies, where pheromone trails encode collective knowledge about the environment.

## Phase 2: Evolutionary Pressure

Phase 2 adds an energy system and reproduction mechanics on top of the field-based learning:

- **Energy**: Each agent has an energy level (0 to `max_energy`). Energy drains by `energy_per_step` each timestep. Collecting food restores `food_energy` units. Agents with 0 energy die (are removed from the simulation).
- **Death**: When an agent's energy reaches 0, its slot is freed. A minimum population (`min_agents`) is maintained by respawning random agents if needed.
- **Reproduction**: Agents with energy above `reproduce_threshold` can take action 5 (reproduce). This costs `reproduce_cost` energy and spawns a child agent in an adjacent cell with a copy of the parent's network weights.
- **Mutation**: Child weights are perturbed by Gaussian noise with standard deviation `mutation_std`, enabling behavioral variation and selection.
- **Lineage tracking**: Each agent receives a unique ID. Parent-child relationships are tracked, enabling lineage analysis and weight divergence measurement.

These mechanics create a natural selection loop: agents that forage efficiently accumulate energy, reproduce more, and pass their (mutated) weights to offspring — driving population-level adaptation alongside gradient-based learning.

## Phase 3: Specialization Detection

Phase 3 adds analysis tools to detect when agents evolve into distinct "species" with specialized strategies:

- **Weight divergence**: Measures pairwise cosine distance between agents' neural network weights over time, tracking how genetically different agents become.
- **Behavioral feature extraction**: Extracts 7 features per agent from trajectories — movement entropy, food collection rate, distance per step, reproduction rate, mean energy, exploration ratio, and stay fraction.
- **Behavioral clustering**: K-means clustering with silhouette score evaluation identifies distinct behavioral groups (e.g., scouts vs exploiters).
- **Species detection**: Formally detects "species" — stable, hereditary behavioral clusters where children consistently adopt the same strategy as parents.
- **Field usage analysis**: Classifies clusters as "writers" (high movement, spread field deposits), "readers" (exploit known field areas), or "balanced".
- **Lineage-strategy correlation**: Checks whether agents from the same evolutionary lineage cluster into the same behavioral strategy.
- **Specialization training options**: Optional `diversity_bonus` and `niche_pressure` reward modifiers to encourage population diversity, plus per-layer mutation rates.

### Phase 3 Commands

```bash
# Generate a full specialization analysis report (trains briefly, then analyzes)
python scripts/generate_specialization_report.py

# Generate report from an existing checkpoint
python scripts/generate_specialization_report.py --checkpoint checkpoints/params.pkl --skip-ablation

# Run specialization ablation (divergent vs uniform vs random weights)
python scripts/run_specialization_ablation.py --iterations 100

# Train with diversity bonus to encourage specialization
python -m src.training.train --specialization.diversity-bonus 0.1 --specialization.niche-pressure 0.05
```

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
# Run field ablation test (normal vs zeroed vs random field)
python -m src.analysis.ablation --checkpoint checkpoints/params.pkl

# Run evolution ablation (field x evolution 2x2 comparison)
python scripts/run_ablation.py --iterations 100 --seed 42 --evolution

# Generate specialization analysis report with visualizations
python scripts/generate_specialization_report.py --checkpoint checkpoints/params.pkl

# Run specialization ablation (divergent vs uniform vs random weights)
python scripts/run_specialization_ablation.py --iterations 100
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
│   ├── configs.py              # Dataclass configs (EnvConfig, FieldConfig, EvolutionConfig, etc.)
│   ├── environment/
│   │   ├── env.py              # reset() and step() with energy/reproduction
│   │   ├── state.py            # EnvState dataclass (includes evolution fields)
│   │   ├── obs.py              # Observation construction (includes energy)
│   │   ├── vec_env.py          # Vectorized env via jax.vmap
│   │   └── render.py           # RGB frame rendering
│   ├── field/
│   │   ├── field.py            # FieldState dataclass
│   │   ├── dynamics.py         # Diffusion + decay physics
│   │   └── ops.py              # Local read/write operations
│   ├── agents/
│   │   ├── network.py          # ActorCritic (Flax nn.Module, 6 actions)
│   │   ├── policy.py           # Action sampling + deterministic policy
│   │   └── reproduction.py     # Weight mutation for offspring
│   ├── training/
│   │   ├── train.py            # Full training loop + train_step
│   │   ├── rollout.py          # Trajectory collection via lax.scan
│   │   ├── gae.py              # Generalized Advantage Estimation
│   │   └── ppo.py              # PPO clipped surrogate loss
│   ├── analysis/
│   │   ├── field_metrics.py    # Entropy, structure, mutual information
│   │   ├── ablation.py         # Field + specialization ablation experiments
│   │   ├── emergence.py        # Phase transition detection
│   │   ├── lineage.py          # Lineage tracking and weight divergence
│   │   ├── specialization.py   # Weight divergence, clustering, species detection
│   │   ├── trajectory.py       # Trajectory recording for behavior analysis
│   │   └── visualization.py    # Specialization visualization (PCA, charts)
│   └── utils/
│       ├── logging.py          # W&B integration
│       └── video.py            # Episode recording to MP4
├── configs/
│   ├── default.yaml            # Default hyperparameters
│   └── phase2.yaml             # Phase 2 evolution config
├── scripts/
│   ├── setup.sh                # Environment setup
│   ├── train.sh                # Training launcher with JAX flags
│   ├── run_ablation.py         # Ablation runner (field x evolution)
│   ├── run_specialization_ablation.py  # Specialization ablation (Phase 3)
│   └── generate_specialization_report.py  # Full analysis report (Phase 3)
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

The field ablation test (`src/analysis/ablation`) compares three conditions:

| Condition | What it tests |
|-----------|---------------|
| **Normal** | Field operates as trained (baseline) |
| **Zeroed** | Field values replaced with zeros each step — agents get no field info |
| **Random** | Field replaced with random noise each step — agents get misleading info |

A positive result looks like: **Normal > Zeroed > Random** in mean episode reward, indicating the field carries useful information that agents have learned to exploit.

### Evolution Results

With evolution enabled, you should also observe:

- **Population dynamics**: Population size fluctuates as agents die and reproduce, eventually stabilizing around a carrying capacity determined by available food
- **Lineage diversity**: Multiple lineages coexist initially; over time, fitter lineages dominate
- **Weight divergence**: Child weights drift from parent weights over generations, measurable via the lineage tracker
- **Energy efficiency**: Mean energy levels increase as the population adapts to forage more effectively

The evolution ablation (`scripts/run_ablation.py --evolution`) runs a 2x2 comparison:

| Condition | Field | Evolution | What it tests |
|-----------|-------|-----------|---------------|
| **Both** | Normal | Enabled | Full system (baseline) |
| **Field only** | Normal | Disabled | Field contribution without selection |
| **Evolution only** | Zeroed | Enabled | Selection without field communication |
| **Neither** | Zeroed | Disabled | Agents learning alone |

### Specialization Results

With specialization detection enabled (Phase 3), training also reports:

- **Weight divergence**: Pairwise cosine distance between agents' weight vectors. Increasing divergence indicates agents are differentiating genetically.
- **Specialization score**: Composite 0-1 metric combining cluster silhouette (50%), weight divergence (25%), and behavioral variance (25%). Higher = more specialized population.
- **Specialization events**: Sudden increases in divergence detected via z-score threshold, analogous to phase transitions.

The specialization ablation (`scripts/run_specialization_ablation.py`) compares three weight conditions:

| Condition | Description | What it tests |
|-----------|-------------|---------------|
| **Divergent** | Trained per-agent weights (as-is) | Baseline with evolved specialization |
| **Uniform** | All agents cloned to mean weights | Whether weight diversity matters |
| **Random** | Mean weights + Gaussian noise | Whether learned divergence beats noise |

A positive result looks like: **Divergent > Uniform > Random** in food collected, indicating that evolved weight specialization improves collective foraging.

The specialization report (`scripts/generate_specialization_report.py`) produces a comprehensive markdown report with:
- Specialization score breakdown
- Detected species and their characteristics
- Field usage patterns per behavioral cluster
- PCA/t-SNE scatter plots of agent behaviors
- Weight divergence over training
- Ablation comparison results

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

### Evolution Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `evolution.enabled` | true | Enable energy/death/reproduction system |
| `evolution.starting_energy` | 100 | Initial energy for each agent |
| `evolution.energy_per_step` | 1 | Energy drained per timestep |
| `evolution.food_energy` | 50 | Energy gained from collecting food |
| `evolution.max_energy` | 200 | Energy cap |
| `evolution.reproduce_threshold` | 150 | Minimum energy to reproduce |
| `evolution.reproduce_cost` | 80 | Energy spent on reproduction |
| `evolution.mutation_std` | 0.01 | Gaussian noise std for weight mutation |
| `evolution.max_agents` | 32 | Maximum population size (array dimension) |
| `evolution.min_agents` | 2 | Minimum population (respawn if below) |

### Specialization Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `specialization.diversity_bonus` | 0.0 | Reward bonus for weight diversity (0 = disabled) |
| `specialization.niche_pressure` | 0.0 | Penalty for identical strategies (0 = disabled) |
| `specialization.layer_mutation_rates` | None | Per-layer mutation rate overrides (e.g., `{"Dense_0": 0.02}`) |

## Tech Stack

- **JAX** — Accelerated numerical computing
- **Flax** — Neural network library for JAX
- **Optax** — Gradient processing and optimization
- **scikit-learn** — Clustering and dimensionality reduction (Phase 3)
- **matplotlib** — Visualization and plotting
- **Weights & Biases** — Experiment tracking (optional)
- **tyro** — CLI argument parsing from dataclasses
