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

### US-011: Vectorize environment with vmap [x]

**Description:** As a developer, I need parallel environments for efficient training.

**Acceptance Criteria:**
- [x] Create `src/environment/vec_env.py`
- [x] `VecEnv` class with `reset(key)` and `step(states, actions)`
- [x] Uses `jax.vmap` to parallelize across `num_envs`
- [x] Batch shapes: `(num_envs, num_agents, ...)`
- [x] Tests pass: `pytest tests/test_env.py::TestVecEnv -v`
- [x] Typecheck passes

---

### US-REVIEW-01: Review Environment Epic [x]

**Description:** Review US-004 through US-011 as a cohesive system.

**Acceptance Criteria:**
- [x] All environment tests pass: `pytest tests/test_env.py tests/test_field.py -v`
- [x] Manually verify: create env, reset, step 10 times, print state shapes
- [x] Check: field values change after agent writes
- [x] Check: food collection works
- [x] If issues found: create fix tasks (US-XXXa)
- [x] Typecheck passes

---

### US-012: Implement actor-critic network [x]

**Description:** As a developer, I need neural network for agent policy.

**Acceptance Criteria:**
- [x] Create `src/agents/network.py`
- [x] `ActorCritic` is a Flax `nn.Module`
- [x] MLP backbone with LayerNorm and Tanh activations
- [x] Actor head outputs logits for 5 actions
- [x] Critic head outputs scalar value
- [x] Orthogonal init: `sqrt(2)` for hidden, `0.01` for actor, `1.0` for critic
- [x] Tests pass: `pytest tests/test_agent.py::TestNetwork -v`
- [x] Typecheck passes

---

### US-013: Implement action sampling [x]

**Description:** As a developer, I need to sample actions from policy.

**Acceptance Criteria:**
- [x] Create `src/agents/policy.py`
- [x] `sample_actions(network, params, obs, key)` returns `(actions, log_probs, values, entropy)`
- [x] Works with batched obs `(num_envs, num_agents, obs_dim)`
- [x] Uses `jax.vmap` over agents with shared params
- [x] Tests pass: `pytest tests/test_agent.py::TestActionSampling -v`
- [x] Typecheck passes

---

### US-014: Implement GAE calculation [x]

**Description:** As a developer, I need advantage estimation for PPO.

**Acceptance Criteria:**
- [x] Create `src/training/gae.py`
- [x] `compute_gae(rewards, values, dones, gamma, gae_lambda)` returns `(advantages, returns)`
- [x] Uses `jax.lax.scan` with reverse iteration
- [x] Handles episode boundaries correctly
- [x] Tests pass: `pytest tests/test_training.py::TestGAE -v`
- [x] Typecheck passes

---

### US-015: Implement PPO loss function [x]

**Description:** As a developer, I need the PPO objective.

**Acceptance Criteria:**
- [x] Create `src/training/ppo.py`
- [x] `ppo_loss(network, params, batch, clip_eps, vf_coef, ent_coef)` returns `(loss, metrics)`
- [x] Implements clipped surrogate objective
- [x] Value loss is MSE
- [x] Entropy bonus for exploration
- [x] Metrics include: `policy_loss`, `value_loss`, `entropy`, `approx_kl`, `clip_fraction`
- [x] Tests pass: `pytest tests/test_training.py::TestPPOLoss -v`
- [x] Typecheck passes

---

### US-016: Implement rollout collection [x]

**Description:** As a developer, I need to collect trajectories for training.

**Acceptance Criteria:**
- [x] Create `src/training/rollout.py`
- [x] `RunnerState` dataclass holds: `params`, `opt_state`, `env_state`, `last_obs`, `key`
- [x] `collect_rollout(runner_state, network, vec_env, num_steps)` returns `(new_runner_state, batch)`
- [x] Uses `jax.lax.scan` for efficiency
- [x] Batch contains: `obs`, `actions`, `rewards`, `dones`, `values`, `log_probs`
- [x] Tests pass: `pytest tests/test_training.py::TestRollout -v`
- [x] Typecheck passes

---

### US-017: Implement training step [x]

**Description:** As a developer, I need the core training update.

**Acceptance Criteria:**
- [x] In `src/training/train.py`
- [x] `train_step(runner_state, network, config)` returns `(new_runner_state, metrics)`
- [x] Collects rollout → computes GAE → updates policy (multiple epochs)
- [x] Normalizes advantages per minibatch
- [x] Uses gradient clipping
- [x] Tests pass: `pytest tests/test_training.py::TestTrainStep -v`
- [x] Typecheck passes

---

### US-018: Implement full training loop [x]

**Description:** As a developer, I need the complete training pipeline.

**Acceptance Criteria:**
- [x] In `src/training/train.py`
- [x] `train(config)` function is the main entry point
- [x] Initializes: env, network, optimizer, runner_state
- [x] JIT-compiles `train_step`
- [x] Loops for `total_steps` with progress bar
- [x] Logs metrics every `log_interval` steps
- [x] `python -m src.training.train --train.total_steps=10000` runs without error
- [x] Typecheck passes

---

### US-REVIEW-02: Review Training Epic [x]

**Description:** Review US-012 through US-018 as a cohesive system.

**Acceptance Criteria:**
- [x] All training tests pass: `pytest tests/test_agent.py tests/test_training.py -v`
- [x] Training runs for 10k steps without NaN/Inf
- [x] Loss decreases over time (or reward increases)
- [x] If issues found: create fix tasks
- [x] Typecheck passes

---

### US-019: Implement W&B logging [x]

**Description:** As a developer, I need experiment tracking.

**Acceptance Criteria:**
- [x] Create `src/utils/logging.py`
- [x] `init_wandb(config)` initializes W&B run with config
- [x] `log_metrics(metrics, step)` logs scalars
- [x] `log_video(frames, name, step)` logs video
- [x] `finish_wandb()` closes run
- [x] Training with `--log.wandb=true` logs to W&B
- [x] Typecheck passes

---

### US-020: Implement environment rendering [x]

**Description:** As a developer, I need to visualize the simulation.

**Acceptance Criteria:**
- [x] Create `src/environment/render.py`
- [x] `render_frame(state, config) -> np.ndarray` returns RGB image
- [x] Shows: grid lines, agents as colored circles, food as green dots
- [x] Field shown as heatmap overlay (sum across channels)
- [x] Image is at least 400x400 pixels
- [x] Test: render a frame and save as PNG
- [x] Typecheck passes

---

### US-021: Implement episode video recording [x]

**Description:** As a developer, I need to record agent behavior.

**Acceptance Criteria:**
- [x] Create `src/utils/video.py`
- [x] `record_episode(network, params, env, config) -> list[np.ndarray]`
- [x] `save_video(frames, path, fps=30)` saves MP4
- [x] Running evaluation produces valid MP4 file
- [x] Typecheck passes

---

### US-022: Implement field analysis metrics [x]

**Description:** As a developer, I need to measure field properties.

**Acceptance Criteria:**
- [x] Create `src/analysis/field_metrics.py`
- [x] `field_entropy(field)` computes spatial entropy
- [x] `field_structure(field)` measures autocorrelation (structure > random)
- [x] `field_food_mi(field, food_positions)` estimates mutual information
- [x] Tests pass: `pytest tests/test_analysis.py -v`
- [x] Typecheck passes

---

### US-023: Implement ablation test [x]

**Description:** As a developer, I need to test if the field matters.

**Acceptance Criteria:**
- [x] Create `src/analysis/ablation.py`
- [x] `ablation_test(network, params, env, config, num_episodes=20)`
- [x] Tests 3 conditions: normal field, zeroed field, random field
- [x] Returns mean rewards per condition with std
- [x] `python -m src.analysis.ablation --checkpoint=path` works
- [x] Typecheck passes

---

### US-024: Implement emergence detection [x]

**Description:** As a developer, I need to detect when emergence happens.

**Acceptance Criteria:**
- [x] Create `src/analysis/emergence.py`
- [x] `EmergenceTracker` class tracks field metrics over training
- [x] Detects phase transitions (sudden metric changes)
- [x] Integrates with training loop to log emergence events
- [x] Typecheck passes

---

### US-025: Create training launch script [x]

**Description:** As a developer, I need easy training launch.

**Acceptance Criteria:**
- [x] Create `scripts/train.sh`
- [x] Activates venv
- [x] Sets JAX flags for performance
- [x] Runs training with passed arguments
- [x] `./scripts/train.sh --train.total_steps=100000` works
- [x] Typecheck passes

---

### US-026: Update README with quick start [x]

**Description:** As a user, I need documentation to get started.

**Acceptance Criteria:**
- [x] README explains what this project tests
- [x] Quick start: install, train, visualize commands
- [x] Expected results section
- [x] How to interpret field visualizations
- [x] Following README starts training successfully

---

### US-027: Full integration test [x]

**Description:** As a developer, I need end-to-end validation.

**Acceptance Criteria:**
- [x] Create `tests/test_integration.py`
- [x] Test: init → train 1000 steps → evaluate → render → analyze
- [x] Completes in < 2 minutes
- [x] All assertions pass
- [x] `pytest tests/test_integration.py -v` passes

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
