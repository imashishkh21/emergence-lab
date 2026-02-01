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

---

## User Stories

### US-001: Create project structure [x]

**Description:** As a developer, I need the folder structure set up so I can organize code properly.

**Acceptance Criteria:**
- [x] Create `src/` with subfolders: `environment/`, `agents/`, `training/`, `field/`, `analysis/`
- [x] Each folder has `__init__.py`
- [x] `python -c "import src"` works without errors
- [x] Typecheck passes: `python -m mypy src/ --ignore-missing-imports`

---

### US-002: Set up dependencies and virtual environment [x]

**Description:** As a developer, I need dependencies installed so I can run the project.

**Acceptance Criteria:**
- [x] `scripts/setup.sh` creates venv if not exists
- [x] Installs package with `pip install -e .`
- [x] `./scripts/setup.sh` completes without error
- [x] `python -c "import jax; print(jax.devices())"` works after setup
- [x] Typecheck passes

---

### US-003: Create base config dataclasses [x]

**Description:** As a developer, I need configuration objects so training is configurable.

**Acceptance Criteria:**
- [x] `src/configs.py` has `EnvConfig`, `FieldConfig`, `AgentConfig`, `TrainConfig`, `LogConfig`, `Config`
- [x] `python -c "from src.configs import Config; c = Config(); print(c)"` prints valid config
- [x] Config can load from YAML: `Config.from_yaml(path)`
- [x] Typecheck passes

---

### US-004: Implement FieldState dataclass [x]

**Description:** As a developer, I need a data structure for the shared field so agents can read/write it.

**Acceptance Criteria:**
- [x] Create `src/field/field.py`
- [x] `FieldState` is a `@flax.struct.dataclass`
- [x] Contains `values: jnp.ndarray` with shape `(H, W, C)`
- [x] `create_field(height, width, channels)` function returns initialized FieldState
- [x] Tests pass: `pytest tests/test_field.py::TestFieldState -v`
- [x] Typecheck passes

---

### US-005: Implement field dynamics (diffusion + decay) [x]

**Description:** As a developer, I need field physics so the field evolves over time.

**Acceptance Criteria:**
- [x] Create `src/field/dynamics.py`
- [x] `diffuse(field, rate)` applies 3x3 Gaussian blur
- [x] `decay(field, rate)` multiplies values by `(1 - rate)`
- [x] `step_field(field, diffusion_rate, decay_rate)` applies both
- [x] All functions are JIT-compatible (no Python control flow)
- [x] Tests pass: `pytest tests/test_field.py::TestFieldDynamics -v`
- [x] Typecheck passes

---

### US-006: Implement field read/write operations [x]

**Description:** As a developer, I need agents to interact with the field locally.

**Acceptance Criteria:**
- [x] Create `src/field/ops.py`
- [x] `read_local(field, positions, radius)` returns local field values for each position
- [x] `write_local(field, positions, values)` adds values at agent positions
- [x] Works with batched positions `(N, 2)`
- [x] Tests pass: `pytest tests/test_field.py::TestFieldOps -v`
- [x] Typecheck passes

---

### US-007: Implement EnvState dataclass [x]

**Description:** As a developer, I need environment state to track simulation.

**Acceptance Criteria:**
- [x] Create `src/environment/state.py`
- [x] `EnvState` is a `@flax.struct.dataclass`
- [x] Fields: `agent_positions`, `food_positions`, `food_collected`, `field_state`, `step`, `key`
- [x] Tests pass: `pytest tests/test_env.py::TestEnvState -v`
- [x] Typecheck passes

---

### US-008: Implement environment reset [x]

**Description:** As a developer, I need to initialize fresh episodes.

**Acceptance Criteria:**
- [x] Create `src/environment/env.py`
- [x] `reset(key, config) -> EnvState` returns fresh state
- [x] Agent positions are random, non-overlapping
- [x] Food positions are random
- [x] Field is initialized fresh
- [x] Tests pass: `pytest tests/test_env.py::TestEnvReset -v`
- [x] Typecheck passes

---

### US-009: Implement environment step [x]

**Description:** As a developer, I need the core simulation loop.

**Acceptance Criteria:**
- [x] `step(state, actions, config) -> (EnvState, rewards, dones, info)`
- [x] Actions: 0=stay, 1=up, 2=down, 3=left, 4=right
- [x] Agents collect food when adjacent (within 1 cell)
- [x] Field updates: diffuse, decay, agents write presence
- [x] Reward: +1 per food collected (shared across team)
- [x] Done when `step >= max_steps`
- [x] Tests pass: `pytest tests/test_env.py::TestEnvStep -v`
- [x] Typecheck passes

---

### US-010: Implement observation function [x]

**Description:** As a developer, I need observations for agent policies.

**Acceptance Criteria:**
- [x] Create `src/environment/obs.py`
- [x] `get_observations(state, config) -> jnp.ndarray` shape `(num_agents, obs_dim)`
- [x] Each agent sees: own position (normalized), local field values, relative food positions (if in range)
- [x] All values normalized to `[-1, 1]`
- [x] Tests pass: `pytest tests/test_env.py::TestObservations -v`
- [x] Typecheck passes

---

### US-011: Vectorize environment with vmap [ ]

**Description:** As a developer, I need parallel environments for efficient training.

**Acceptance Criteria:**
- [ ] Create `src/environment/vec_env.py`
- [ ] `VecEnv` class with `reset(key)` and `step(states, actions)`
- [ ] Uses `jax.vmap` to parallelize across `num_envs`
- [ ] Batch shapes: `(num_envs, num_agents, ...)`
- [ ] Tests pass: `pytest tests/test_env.py::TestVecEnv -v`
- [ ] Typecheck passes

---

### US-REVIEW-01: Review Environment Epic [ ]

**Description:** Review US-004 through US-011 as a cohesive system.

**Acceptance Criteria:**
- [ ] All environment tests pass: `pytest tests/test_env.py tests/test_field.py -v`
- [ ] Manually verify: create env, reset, step 10 times, print state shapes
- [ ] Check: field values change after agent writes
- [ ] Check: food collection works
- [ ] If issues found: create fix tasks (US-XXXa)
- [ ] Typecheck passes

---

### US-012: Implement actor-critic network [ ]

**Description:** As a developer, I need neural network for agent policy.

**Acceptance Criteria:**
- [ ] Create `src/agents/network.py`
- [ ] `ActorCritic` is a Flax `nn.Module`
- [ ] MLP backbone with LayerNorm and Tanh activations
- [ ] Actor head outputs logits for 5 actions
- [ ] Critic head outputs scalar value
- [ ] Orthogonal init: `sqrt(2)` for hidden, `0.01` for actor, `1.0` for critic
- [ ] Tests pass: `pytest tests/test_agent.py::TestNetwork -v`
- [ ] Typecheck passes

---

### US-013: Implement action sampling [ ]

**Description:** As a developer, I need to sample actions from policy.

**Acceptance Criteria:**
- [ ] Create `src/agents/policy.py`
- [ ] `sample_actions(network, params, obs, key)` returns `(actions, log_probs, values, entropy)`
- [ ] Works with batched obs `(num_envs, num_agents, obs_dim)`
- [ ] Uses `jax.vmap` over agents with shared params
- [ ] Tests pass: `pytest tests/test_agent.py::TestActionSampling -v`
- [ ] Typecheck passes

---

### US-014: Implement GAE calculation [ ]

**Description:** As a developer, I need advantage estimation for PPO.

**Acceptance Criteria:**
- [ ] Create `src/training/gae.py`
- [ ] `compute_gae(rewards, values, dones, gamma, gae_lambda)` returns `(advantages, returns)`
- [ ] Uses `jax.lax.scan` with reverse iteration
- [ ] Handles episode boundaries correctly
- [ ] Tests pass: `pytest tests/test_training.py::TestGAE -v`
- [ ] Typecheck passes

---

### US-015: Implement PPO loss function [ ]

**Description:** As a developer, I need the PPO objective.

**Acceptance Criteria:**
- [ ] Create `src/training/ppo.py`
- [ ] `ppo_loss(network, params, batch, clip_eps, vf_coef, ent_coef)` returns `(loss, metrics)`
- [ ] Implements clipped surrogate objective
- [ ] Value loss is MSE
- [ ] Entropy bonus for exploration
- [ ] Metrics include: `policy_loss`, `value_loss`, `entropy`, `approx_kl`, `clip_fraction`
- [ ] Tests pass: `pytest tests/test_training.py::TestPPOLoss -v`
- [ ] Typecheck passes

---

### US-016: Implement rollout collection [ ]

**Description:** As a developer, I need to collect trajectories for training.

**Acceptance Criteria:**
- [ ] Create `src/training/rollout.py`
- [ ] `RunnerState` dataclass holds: `params`, `opt_state`, `env_state`, `last_obs`, `key`
- [ ] `collect_rollout(runner_state, network, vec_env, num_steps)` returns `(new_runner_state, batch)`
- [ ] Uses `jax.lax.scan` for efficiency
- [ ] Batch contains: `obs`, `actions`, `rewards`, `dones`, `values`, `log_probs`
- [ ] Tests pass: `pytest tests/test_training.py::TestRollout -v`
- [ ] Typecheck passes

---

### US-017: Implement training step [ ]

**Description:** As a developer, I need the core training update.

**Acceptance Criteria:**
- [ ] In `src/training/train.py`
- [ ] `train_step(runner_state, network, config)` returns `(new_runner_state, metrics)`
- [ ] Collects rollout → computes GAE → updates policy (multiple epochs)
- [ ] Normalizes advantages per minibatch
- [ ] Uses gradient clipping
- [ ] Tests pass: `pytest tests/test_training.py::TestTrainStep -v`
- [ ] Typecheck passes

---

### US-018: Implement full training loop [ ]

**Description:** As a developer, I need the complete training pipeline.

**Acceptance Criteria:**
- [ ] In `src/training/train.py`
- [ ] `train(config)` function is the main entry point
- [ ] Initializes: env, network, optimizer, runner_state
- [ ] JIT-compiles `train_step`
- [ ] Loops for `total_steps` with progress bar
- [ ] Logs metrics every `log_interval` steps
- [ ] `python -m src.training.train --train.total_steps=10000` runs without error
- [ ] Typecheck passes

---

### US-REVIEW-02: Review Training Epic [ ]

**Description:** Review US-012 through US-018 as a cohesive system.

**Acceptance Criteria:**
- [ ] All training tests pass: `pytest tests/test_agent.py tests/test_training.py -v`
- [ ] Training runs for 10k steps without NaN/Inf
- [ ] Loss decreases over time (or reward increases)
- [ ] If issues found: create fix tasks
- [ ] Typecheck passes

---

### US-019: Implement W&B logging [ ]

**Description:** As a developer, I need experiment tracking.

**Acceptance Criteria:**
- [ ] Create `src/utils/logging.py`
- [ ] `init_wandb(config)` initializes W&B run with config
- [ ] `log_metrics(metrics, step)` logs scalars
- [ ] `log_video(frames, name, step)` logs video
- [ ] `finish_wandb()` closes run
- [ ] Training with `--log.wandb=true` logs to W&B
- [ ] Typecheck passes

---

### US-020: Implement environment rendering [ ]

**Description:** As a developer, I need to visualize the simulation.

**Acceptance Criteria:**
- [ ] Create `src/environment/render.py`
- [ ] `render_frame(state, config) -> np.ndarray` returns RGB image
- [ ] Shows: grid lines, agents as colored circles, food as green dots
- [ ] Field shown as heatmap overlay (sum across channels)
- [ ] Image is at least 400x400 pixels
- [ ] Test: render a frame and save as PNG
- [ ] Typecheck passes

---

### US-021: Implement episode video recording [ ]

**Description:** As a developer, I need to record agent behavior.

**Acceptance Criteria:**
- [ ] Create `src/utils/video.py`
- [ ] `record_episode(network, params, env, config) -> list[np.ndarray]`
- [ ] `save_video(frames, path, fps=30)` saves MP4
- [ ] Running evaluation produces valid MP4 file
- [ ] Typecheck passes

---

### US-022: Implement field analysis metrics [ ]

**Description:** As a developer, I need to measure field properties.

**Acceptance Criteria:**
- [ ] Create `src/analysis/field_metrics.py`
- [ ] `field_entropy(field)` computes spatial entropy
- [ ] `field_structure(field)` measures autocorrelation (structure > random)
- [ ] `field_food_mi(field, food_positions)` estimates mutual information
- [ ] Tests pass: `pytest tests/test_analysis.py -v`
- [ ] Typecheck passes

---

### US-023: Implement ablation test [ ]

**Description:** As a developer, I need to test if the field matters.

**Acceptance Criteria:**
- [ ] Create `src/analysis/ablation.py`
- [ ] `ablation_test(network, params, env, config, num_episodes=20)`
- [ ] Tests 3 conditions: normal field, zeroed field, random field
- [ ] Returns mean rewards per condition with std
- [ ] `python -m src.analysis.ablation --checkpoint=path` works
- [ ] Typecheck passes

---

### US-024: Implement emergence detection [ ]

**Description:** As a developer, I need to detect when emergence happens.

**Acceptance Criteria:**
- [ ] Create `src/analysis/emergence.py`
- [ ] `EmergenceTracker` class tracks field metrics over training
- [ ] Detects phase transitions (sudden metric changes)
- [ ] Integrates with training loop to log emergence events
- [ ] Typecheck passes

---

### US-025: Create training launch script [ ]

**Description:** As a developer, I need easy training launch.

**Acceptance Criteria:**
- [ ] Create `scripts/train.sh`
- [ ] Activates venv
- [ ] Sets JAX flags for performance
- [ ] Runs training with passed arguments
- [ ] `./scripts/train.sh --train.total_steps=100000` works
- [ ] Typecheck passes

---

### US-026: Update README with quick start [ ]

**Description:** As a user, I need documentation to get started.

**Acceptance Criteria:**
- [ ] README explains what this project tests
- [ ] Quick start: install, train, visualize commands
- [ ] Expected results section
- [ ] How to interpret field visualizations
- [ ] Following README starts training successfully

---

### US-027: Full integration test [ ]

**Description:** As a developer, I need end-to-end validation.

**Acceptance Criteria:**
- [ ] Create `tests/test_integration.py`
- [ ] Test: init → train 1000 steps → evaluate → render → analyze
- [ ] Completes in < 2 minutes
- [ ] All assertions pass
- [ ] `pytest tests/test_integration.py -v` passes

---

### US-REVIEW-FINAL: Final Phase 1 Review [ ]

**Description:** Validate Phase 1 is complete and working.

**Acceptance Criteria:**
- [ ] All tests pass: `pytest tests/ -v`
- [ ] Typecheck passes: `python -m mypy src/ --ignore-missing-imports`
- [ ] Training runs for 100k steps without crash
- [ ] W&B shows learning curves
- [ ] Video shows agents moving and field changing
- [ ] Ablation test shows field > zeroed field
- [ ] Mark COMPLETE in progress.txt

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

*PRD designed for Ralph Loop autonomous execution.*
*Each story completes in one context window.*
*Total: 27 stories + 3 reviews = ~30 iterations expected.*
