# PRD: Emergence Lab — Phase 1: Digital Petri Dish

## Introduction

Build a minimal multi-agent reinforcement learning environment where agents interact through a **shared learnable field**. The goal is to test whether collective intelligence can emerge in the space BETWEEN agents, not just in the agents themselves.

This is Phase 1 — get the infrastructure working and observe what happens.

## Goals

- Create a JAX-based grid environment with N foraging agents
- Implement a shared field that agents read/write locally
- Field has its own dynamics (diffusion, decay)
- Train agents AND field dynamics end-to-end with PPO
- Visualize field patterns during/after training
- Measure whether field carries information agents don't have individually

## Non-Goals (Phase 1)

- Complex game mechanics (hide-and-seek, tag)
- Distributed training across machines
- Production-ready code
- Novel neural architectures
- Publication-ready experiments

## Technical Stack

- **JAX/Flax** — ML framework
- **Optax** — Optimizers
- **Weights & Biases** — Logging
- **Python 3.10+** — Runtime

## Architecture Overview

```
┌─────────────────────────────────────────┐
│              ENVIRONMENT                │
│  ┌─────────────────────────────────┐   │
│  │         FIELD (NxN)             │   │
│  │   - Diffuses over time          │   │
│  │   - Decays over time            │   │
│  │   - Agents read local values    │   │
│  │   - Agents write local values   │   │
│  │   - Dynamics are LEARNED        │   │
│  └─────────────────────────────────┘   │
│                                         │
│  Agent 1    Agent 2    ...    Agent K   │
│  (pos, vel) (pos, vel)       (pos, vel) │
│                                         │
│  Food sources spawn randomly            │
│  Agents get reward for finding food     │
└─────────────────────────────────────────┘
```

## User Stories

### Epic 1: Project Foundation

#### [ ] 1.1 Create project structure
Create the folder structure with `__init__.py` files:
```
src/
├── __init__.py
├── environment/
│   └── __init__.py
├── agents/
│   └── __init__.py
├── training/
│   └── __init__.py
├── field/
│   └── __init__.py
└── analysis/
    └── __init__.py
configs/
tests/
```
**Acceptance:** `python -c "import src"` works without errors.

#### [ ] 1.2 Set up dependencies and virtual environment
Create setup script that:
- Creates venv if not exists
- Installs dependencies from pyproject.toml
- Verifies JAX is working

**Acceptance:** `./scripts/setup.sh && python -c "import jax; print(jax.devices())"` runs successfully.

#### [ ] 1.3 Create base config dataclasses
Create `src/configs.py` with dataclasses for:
- `EnvConfig` (grid_size, num_agents, num_food, max_steps)
- `FieldConfig` (diffusion_rate, decay_rate, num_channels)
- `AgentConfig` (hidden_dims, observation_radius)
- `TrainConfig` (lr, batch_size, num_envs, total_steps, gamma, gae_lambda)
- `Config` (combines all above)

**Acceptance:** `python -c "from src.configs import Config; c = Config(); print(c)"` prints valid config.

---

### Epic 2: The Field

#### [ ] 2.1 Implement Field dataclass
Create `src/field/field.py` with:
- `FieldState` as `@flax.struct.dataclass`
- Contains: `values: jnp.ndarray` shape (H, W, C) where C = num_channels
- Initialize to zeros or small random values

**Acceptance:** Tests pass: `pytest tests/test_field.py::test_field_state_creation -v`

#### [ ] 2.2 Implement field dynamics (diffusion + decay)
In `src/field/dynamics.py`:
- `diffuse(field, rate)` — 3x3 convolution blur
- `decay(field, rate)` — multiply by (1 - decay_rate)
- `step_field(field, diffusion_rate, decay_rate)` — applies both
- All ops must be JIT-compatible

**Acceptance:** Tests pass: `pytest tests/test_field.py::test_diffusion -v`

#### [ ] 2.3 Implement field read/write operations
In `src/field/ops.py`:
- `read_local(field, positions, radius)` — returns values around each position
- `write_local(field, positions, values)` — adds values at positions
- Use `jax.ops.segment_sum` or scatter for batched writes

**Acceptance:** Tests pass: `pytest tests/test_field.py::test_read_write -v`

---

### Epic 3: Environment Core

#### [ ] 3.1 Implement EnvState dataclass
Create `src/environment/state.py` with:
- `EnvState` as `@flax.struct.dataclass`
- Fields: `agent_positions`, `agent_velocities`, `food_positions`, `food_collected`, `field_state`, `step`, `key`

**Acceptance:** Tests pass: `pytest tests/test_env.py::test_env_state -v`

#### [ ] 3.2 Implement environment reset
Create `src/environment/env.py` with:
- `reset(key, config) -> EnvState`
- Random agent positions (no overlap)
- Random food positions
- Fresh field state

**Acceptance:** Tests pass: `pytest tests/test_env.py::test_reset -v`

#### [ ] 3.3 Implement environment step
In `src/environment/env.py`:
- `step(state, actions, config) -> (EnvState, rewards, dones, info)`
- Actions: discrete 5 (stay, up, down, left, right)
- Agents move, collect food if adjacent
- Field updates (diffuse, decay, agents write presence)
- Rewards: +1 per food collected by team (shared reward)

**Acceptance:** Tests pass: `pytest tests/test_env.py::test_step -v`

#### [ ] 3.4 Implement observation function
In `src/environment/obs.py`:
- `get_observations(state, config) -> dict[str, jnp.ndarray]`
- Each agent sees: own position, local field values, relative food positions (if in range)
- Observations normalized to [-1, 1]

**Acceptance:** Tests pass: `pytest tests/test_env.py::test_observations -v`

#### [ ] 3.5 Vectorize environment with vmap
In `src/environment/vec_env.py`:
- `VecEnv` class wrapping reset/step with `jax.vmap`
- Handles `num_envs` parallel environments
- Batch shape: (num_envs, num_agents, ...)

**Acceptance:** Tests pass: `pytest tests/test_env.py::test_vec_env -v`

---

### Epic 4: Agent Neural Network

#### [ ] 4.1 Implement actor-critic network
Create `src/agents/network.py` with:
- `ActorCritic` Flax module
- MLP with LayerNorm and Tanh activations
- Shared backbone → actor head (5 actions) + critic head (1 value)
- Proper orthogonal initialization (sqrt(2) hidden, 0.01 actor, 1.0 critic)

**Acceptance:** Tests pass: `pytest tests/test_agent.py::test_network_forward -v`

#### [ ] 4.2 Implement action sampling
In `src/agents/policy.py`:
- `sample_actions(params, obs, key)` — returns actions, log_probs, values, entropy
- Works with batched observations (num_envs, num_agents, obs_dim)
- Uses `jax.vmap` over agents (shared params)

**Acceptance:** Tests pass: `pytest tests/test_agent.py::test_action_sampling -v`

---

### Epic 5: PPO Training

#### [ ] 5.1 Implement GAE calculation
Create `src/training/gae.py` with:
- `compute_gae(rewards, values, dones, gamma, gae_lambda)` 
- Using `jax.lax.scan` with reverse iteration
- Returns advantages and returns

**Acceptance:** Tests pass: `pytest tests/test_training.py::test_gae -v`

#### [ ] 5.2 Implement PPO loss function
Create `src/training/ppo.py` with:
- `ppo_loss(params, batch, clip_eps, vf_coef, ent_coef)`
- Clipped surrogate objective
- Value loss (MSE)
- Entropy bonus
- Returns total loss and metrics dict

**Acceptance:** Tests pass: `pytest tests/test_training.py::test_ppo_loss -v`

#### [ ] 5.3 Implement rollout collection
Create `src/training/rollout.py` with:
- `collect_rollout(runner_state, num_steps)` using `jax.lax.scan`
- Collects: obs, actions, rewards, dones, values, log_probs
- Returns batch and final runner_state

**Acceptance:** Tests pass: `pytest tests/test_training.py::test_rollout -v`

#### [ ] 5.4 Implement training step
In `src/training/train.py`:
- `train_step(runner_state, config)` 
- Collect rollout → compute GAE → PPO update (multiple epochs)
- Returns updated runner_state and metrics

**Acceptance:** Tests pass: `pytest tests/test_training.py::test_train_step -v`

#### [ ] 5.5 Implement full training loop
In `src/training/train.py`:
- `train(config)` main function
- Initialize env, network, optimizer
- JIT-compile training step
- Loop with logging every N steps
- Save checkpoints every M steps

**Acceptance:** `python -m src.training.train --total_steps=10000` runs without error and logs to console.

---

### Epic 6: Logging & Visualization

#### [ ] 6.1 Implement W&B logging
Create `src/utils/logging.py` with:
- `init_wandb(config)`
- `log_metrics(metrics, step)`
- `log_video(frames, name, step)`

**Acceptance:** Training with `--wandb=true` logs to W&B project.

#### [ ] 6.2 Implement environment rendering
Create `src/environment/render.py` with:
- `render_frame(state, config) -> np.ndarray` (RGB image)
- Show: grid, agents (colored dots), food (green), field (heatmap overlay)

**Acceptance:** `python -c "from src.environment.render import render_frame; ..."` produces valid image.

#### [ ] 6.3 Implement episode video recording
Create `src/utils/video.py` with:
- `record_episode(env, policy, config) -> list[np.ndarray]`
- `save_video(frames, path, fps=30)`

**Acceptance:** Running evaluation produces MP4 video file.

---

### Epic 7: Emergence Analysis

#### [ ] 7.1 Implement field analysis metrics
Create `src/analysis/field_metrics.py` with:
- `field_entropy(field)` — spatial entropy of field patterns
- `field_structure(field)` — measure of non-random structure (autocorrelation)
- `mutual_information(field, food_positions)` — does field encode food locations?

**Acceptance:** Tests pass: `pytest tests/test_analysis.py::test_field_metrics -v`

#### [ ] 7.2 Implement ablation test
Create `src/analysis/ablation.py` with:
- `ablation_test(policy, env, config)` 
- Run episodes with: normal field, zeroed field, random field
- Compare performance across conditions

**Acceptance:** `python -m src.analysis.ablation --checkpoint=...` produces comparison results.

#### [ ] 7.3 Implement emergence detection
Create `src/analysis/emergence.py` with:
- Track field structure over training
- Detect phase transitions (sudden changes in metrics)
- Log emergence events to W&B

**Acceptance:** Training logs include emergence metrics.

---

### Epic 8: Integration & Polish

#### [ ] 8.1 Create default config YAML
Create `configs/default.yaml` with sensible defaults:
- 20x20 grid, 8 agents, 10 food sources
- 4 field channels
- 32 parallel envs, 128 steps per rollout
- 10M total steps

**Acceptance:** `python -m src.training.train --config=configs/default.yaml` works.

#### [ ] 8.2 Create training launch script
Create `scripts/train.sh`:
- Activates venv
- Sets env vars (JAX flags, W&B)
- Runs training with passed args

**Acceptance:** `./scripts/train.sh --total_steps=100000` runs training.

#### [ ] 8.3 Add README with quick start
Update `README.md` with:
- What this project is testing
- Quick start commands
- Expected results
- How to interpret field visualizations

**Acceptance:** Following README instructions starts training successfully.

#### [ ] 8.4 Full integration test
Create `tests/test_integration.py`:
- Test full pipeline: init → train 1000 steps → evaluate → analyze
- Should complete in < 2 minutes

**Acceptance:** `pytest tests/test_integration.py -v` passes.

---

## Definition of Done

Phase 1 is complete when:
1. ✅ Training runs for 1M+ steps without crashing
2. ✅ W&B dashboard shows learning curves
3. ✅ Videos show agents moving and interacting with field
4. ✅ Field develops visible (non-random) patterns
5. ✅ Ablation test shows field contribution to performance

## Success Metrics

| Metric | Target |
|--------|--------|
| Training stability | No NaN/Inf for 10M steps |
| Performance | Agents collect >50% of food per episode |
| Field structure | Autocorrelation > random baseline |
| Ablation gap | Normal field > zeroed field (statistically significant) |

---

*This PRD is designed for autonomous execution via Ralph loop.*
*Each story should complete in one Claude Code context window.*
