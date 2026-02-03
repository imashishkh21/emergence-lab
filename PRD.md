# Phase 5 PRD: Prove Emergence

## Vision

Prove with **rigorous information-theoretic metrics** and **statistical significance** that collective intelligence emerges from simple agents + shared field + evolution. Beat classical swarm AND modern MARL baselines. Produce publication-ready figures and data that make the case undeniable.

---

## Goals

- Quantify emergence with PID synergy, O-information, and causal emergence metrics
- Statistical significance on all claims (N=20 seeds, IQM + bootstrap CI)
- Beat IPPO, ACO, and MAPPO baselines on collective tasks
- Demonstrate superlinear scaling (field-mediated agents > sum of individuals)
- 6-condition stigmergy ablation proving the field encodes collective knowledge
- Hidden resources task proving swarm solves what individuals cannot
- Publication-ready figures (300 DPI, PDF + PNG)

## Non-Goals

- Writing the paper prose (figures + data only)
- QMIX baseline (MAPPO covers MARL comparison space)
- NIS+ neural coarse-graining (basic causal emergence sufficient)
- CommNet/TarMAC communication baselines
- Multi-grid-size experiments (food scarcity is the third env config)
- Real-time streaming of experiments

---

## Technical Decisions (ALREADY MADE)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| PID library | `dit` (discrete) | Gold standard, handles I_min, I_BROJA |
| O-information | `hoi` (JAX-native) | Fits our JAX stack, scales O(n) |
| Statistical reporting | `rliable` | Google standard (Agarwal 2021) |
| MAPPO implementation | Custom (~200 lines) | Easier than wrapping env for JaxMARL |
| QMIX | Deferred | MAPPO covers MARL comparison |
| Dependencies | `dit`, `hoi`, `rliable`, `jaxmarl` | Per DR-5 analysis |
| Hidden food config | K=3 agents, D=3 distance, value=5x, duration=10 steps | Per DR-4 LBF analysis |
| Seed count | 20 minimum | DR-4 gold standard |
| Discretization | K=2 quantile bins + Jeffreys smoothing (alpha=0.5) | Per DR-1, Riedl 2025 |
| Environment configs | 3: standard, hidden resources, food scarcity | DR-4 reviewer requirement |
| Windowed analysis | 1M-step windows, 50% overlap | Standard for non-stationarity |
| ACO variants | Both Fixed and Hybrid | DR-3: isolates learning-to-write value |
| Causal emergence | Both Hoel EI and Rosas Psi | Complementary measures |

---

## What This Phase Builds On

- **Transfer Entropy** — `src/analysis/information.py` (Phase 4, working)
- **Division of Labor** — `src/analysis/information.py` (Phase 4, working)
- **Phase Transition Detection** — `src/analysis/emergence.py` (Phase 4, working)
- **Specialization Score** — `src/analysis/specialization.py` (Phase 3, working)
- **Field Ablation** — `src/analysis/ablation.py` (Phase 1-3, 3 conditions)
- **Evolution Ablation** — `src/analysis/ablation.py` (Phase 2, 2x2 grid)
- **Dashboard** — `src/server/` + `dashboard/` (Phase 4, working)
- **Checkpointing** — `src/training/checkpointing.py` (Phase 4B, working)
- **Kaggle training** — `notebooks/kaggle_training.ipynb` (Phase 4B, 9.8M steps trained)

---

## User Stories

### US-001: Phase 5 Dependencies and Library Verification [x]

**Task:** Add `dit`, `hoi`, `rliable`, `jaxmarl` to pyproject.toml and verify each library imports and runs a basic operation.

**Acceptance Criteria:**
- `dit`, `hoi`, `rliable`, `jaxmarl` added to `[project.optional-dependencies]` under a new `phase5` group
- New test file with one smoke test per library (import + one function call)
- Each smoke test is `@pytest.mark.skipif` guarded so CI doesn't break if libs aren't installed
- `python -m mypy src/ --ignore-missing-imports` passes
- `pytest tests/test_phase5_deps.py -v` passes
- `ruff check src/` passes

**Verification:**
```bash
pip install -e ".[phase5]"
pytest tests/test_phase5_deps.py -v
```

---

### US-002: O-Information Metric via hoi [x]

**Task:** Implement O-information (Omega = TC - DTC) over agent behavioral features using the `hoi` library. Omega < 0 means synergy dominates, which is our emergence signal.

**Acceptance Criteria:**
- New file `src/analysis/o_information.py`
- `compute_o_information(behavioral_features)` returns scalar Omega value
- `OInformationTracker` class following `TransferEntropyTracker` pattern (history, events, get_metrics, get_summary)
- Handles edge cases: fewer than 3 agents, constant features, NaN inputs
- Computes for normal field vs zeroed vs random to show field-mediated emergence
- `python -m mypy src/ --ignore-missing-imports` passes
- `pytest tests/test_o_information.py -v` passes
- `ruff check src/` passes

**Verification:**
```bash
pytest tests/test_o_information.py -v
python -c "from src.analysis.o_information import compute_o_information; print('OK')"
```

---

### US-003: Pairwise PID Synergy via dit [x]

**Task:** Implement pairwise Partial Information Decomposition over agent pairs using the `dit` library. Variables: agent action (discrete, 6 values), field summary (K=2 quantile bins), future food (K=2 quantile bins).

**Acceptance Criteria:**
- New file `src/analysis/pid_synergy.py`
- `compute_pairwise_pid(actions, field_summary, future_food)` returns synergy, redundancy, unique info per pair
- **Jeffreys smoothing** (alpha=0.5 pseudocount) applied before passing to `dit`
- `compute_interaction_information(X, Y, Z)` as quick screening function (negative II = synergy dominates)
- `compute_median_synergy(agent_data)` returns median synergy across all agent pairs
- Row/column shuffle surrogate significance test
- Handles edge cases: single agent, constant variables, empty trajectories
- `python -m mypy src/ --ignore-missing-imports` passes
- `pytest tests/test_pid_synergy.py -v` passes
- `ruff check src/` passes

**Verification:**
```bash
pytest tests/test_pid_synergy.py -v
python -c "from src.analysis.pid_synergy import compute_interaction_information; print('OK')"
```

---

### US-004: Causal Emergence (Effective Information + Rosas Psi) [x]

**Task:** Implement two complementary causal emergence measures: Hoel's Effective Information (EI) on micro vs macro transition probability matrices, and Rosas' Psi as a single scalar.

**Acceptance Criteria:**
- New file `src/analysis/causal_emergence.py`
- `compute_effective_information(micro_tpm, macro_tpm)` returns EI gap (macro EI - micro EI > 0 = causal emergence)
- `compute_rosas_psi(macro_var_t, macro_var_t1, micro_vars_t)` returns Psi scalar
- Macro variable candidates: population count, mean field intensity, total food collected, spatial dispersion, field entropy
- Discretize continuous vars into 4-8 bins, build TPMs from trajectory windows
- **Windowed analysis**: 1M-step windows, 50% overlap, track CE gap over training
- `CausalEmergenceTracker` class following tracker pattern
- Handles degenerate cases: constant macro variables, too-short windows
- `python -m mypy src/ --ignore-missing-imports` passes
- `pytest tests/test_causal_emergence.py -v` passes
- `ruff check src/` passes

**Verification:**
```bash
pytest tests/test_causal_emergence.py -v
python -c "from src.analysis.causal_emergence import compute_effective_information; print('OK')"
```

---

### US-005: Surrogate Testing Framework [x]

**Task:** Create a reusable surrogate testing framework for statistical significance of all emergence metrics. Row shuffle breaks cross-agent coordination, column/block shuffle breaks temporal dependencies.

**Acceptance Criteria:**
- New file `src/analysis/surrogates.py`
- `row_shuffle(data, rng)` — shuffle across agents (break spatial coordination)
- `column_shuffle(data, rng)` — shuffle across time (break temporal dependencies)
- `block_shuffle(data, block_size, rng)` — shuffle temporal blocks (preserve short-range structure)
- `bootstrap_ci(statistic_fn, data, n_bootstrap=1000, ci=0.95)` — BCa bootstrap confidence intervals
- `mann_whitney_u(x, y)` — wrapper returning U-statistic, p-value, effect size (rank-biserial)
- `wilcoxon_signed_rank(x, y)` — wrapper returning statistic, p-value, effect size
- `surrogate_test(metric_fn, real_data, shuffle_fn, n_surrogates=100)` — returns p-value and significance flag
- Uses only numpy/scipy (no Phase 5 deps needed)
- `python -m mypy src/ --ignore-missing-imports` passes
- `pytest tests/test_surrogates.py -v` passes
- `ruff check src/` passes

**Verification:**
```bash
pytest tests/test_surrogates.py -v
python -c "from src.analysis.surrogates import bootstrap_ci, mann_whitney_u; print('OK')"
```

---

### US-006: IPPO Baseline (No Field, Shared Params) [x]

**Task:** Create the simplest baseline: Independent PPO with no field and no evolution. Shared parameters, individual rewards. This is the "no communication at all" lower bound.

**Acceptance Criteria:**
- New file `src/baselines/__init__.py` (empty)
- New file `src/baselines/ippo.py`
- `ippo_config()` returns a Config with field zeroed (write_strength=0, decay_rate=1.0), evolution disabled
- `run_ippo_episode(network, params, config, key)` returns standardized result dict: `{"total_reward", "food_collected", "final_population", "per_agent_rewards"}`
- `evaluate_ippo(network, params, config, n_episodes, seed)` runs N episodes and returns aggregate results
- Zero new neural network code — reuses existing `ActorCritic` and `step()`
- `python -m mypy src/ --ignore-missing-imports` passes
- `pytest tests/test_baselines.py -v` passes
- `ruff check src/` passes

**Verification:**
```bash
pytest tests/test_baselines.py::TestIPPO -v
python -c "from src.baselines.ippo import ippo_config; print(ippo_config())"
```

---

### US-007: ACO-Fixed Baseline (+ ACO-Hybrid Variant) [x]

**Task:** Implement two Ant Colony Optimization baselines. ACO-Fixed uses hardcoded pheromone rules (no neural network). ACO-Hybrid uses a neural network for movement but hardcoded field write rules. This isolates the value of LEARNING the write behavior.

**Acceptance Criteria:**
- New file `src/baselines/aco_fixed.py`
- `ACOAgent` class with hardcoded rules: deposit pheromone on food collection, follow field gradient with softmax
- ACO parameters: `alpha=1.0, beta=2.0, rho=0.5, Q=1.0` (Dorigo & Stutzle 2004)
- `run_aco_fixed_episode(config, key)` — no neural network at all
- `run_aco_hybrid_episode(network, params, config, key)` — NN for movement, hardcoded field writes
- Both return standardized result dict (same format as US-006)
- Uses existing field infrastructure (same grid, diffusion, decay)
- `python -m mypy src/ --ignore-missing-imports` passes
- `pytest tests/test_baselines.py::TestACO -v` passes
- `ruff check src/` passes

**Verification:**
```bash
pytest tests/test_baselines.py::TestACO -v
python -c "from src.baselines.aco_fixed import run_aco_fixed_episode; print('OK')"
```

---

### US-008: MAPPO Baseline (Centralized Critic) [x]

**Task:** Implement Multi-Agent PPO with a centralized critic (all agents' observations concatenated) and decentralized actors. Modify our existing PPO, not a JaxMARL wrapper. Disable field and evolution, shared weights for all actors.

**Acceptance Criteria:**
- New file `src/baselines/mappo.py`
- `CentralizedCritic` Flax module: takes concatenated observations of all agents, outputs single value per agent
- `mappo_config()` returns Config with field disabled, evolution disabled, shared weights
- `mappo_loss(network, critic, params, critic_params, batch, clip_eps, vf_coef, ent_coef)` — PPO loss using centralized value
- Value normalization (running mean/std of returns)
- Death masking (dead agents excluded from centralized obs via zero-padding)
- `evaluate_mappo(network, critic, params, critic_params, config, n_episodes, seed)` returns standardized result dict
- ~200 lines of code
- `python -m mypy src/ --ignore-missing-imports` passes
- `pytest tests/test_baselines.py::TestMAPPO -v` passes
- `ruff check src/` passes

**Verification:**
```bash
pytest tests/test_baselines.py::TestMAPPO -v
python -c "from src.baselines.mappo import CentralizedCritic; print('OK')"
```

---

### US-009: Experiment Harness and Multi-Seed Runner [x]

**Task:** Create a generic experiment harness that runs any method across N seeds with paired seed support and standardized results.

**Acceptance Criteria:**
- New file `src/experiments/__init__.py` (empty)
- New file `src/experiments/runner.py`
- `ExperimentConfig` dataclass: method name, n_seeds, env_config_name, paired_seeds flag
- `ExperimentResult` dataclass: per-seed results list, aggregate metrics (mean, std, IQM, CI)
- `run_experiment(experiment_config, method_fn, seed_offset)` runs method_fn for each seed, returns ExperimentResult
- Paired seed support: same seed used for baseline and variant (reduces variance)
- Save/load ExperimentResult to pickle
- New file `src/experiments/configs.py`
- Three environment configs: `standard_config()`, `hidden_resources_config()`, `food_scarcity_config()` (num_food=5)
- `python -m mypy src/ --ignore-missing-imports` passes
- `pytest tests/test_experiments.py -v` passes
- `ruff check src/` passes

**Verification:**
```bash
pytest tests/test_experiments.py -v
python -c "from src.experiments.runner import ExperimentConfig, run_experiment; print('OK')"
```

---

### US-010: Hidden Resources Environment Modification [x]

**Task:** Modify the environment so that K=3 agents within Chebyshev distance D=3 of a hidden food location reveals it. Hidden food has 5x value, stays revealed for 10 steps, then re-hides. This creates a task that REQUIRES coordination — individuals cannot solve it alone.

**Acceptance Criteria:**
- New `HiddenFoodConfig` dataclass nested in `EnvConfig` (disabled by default for backward compatibility)
- Fields: `enabled=False`, `num_hidden=3`, `required_agents=3`, `reveal_distance=3`, `reveal_duration=10`, `hidden_food_value_multiplier=5.0`
- New fields in `EnvState`: `hidden_food_positions (num_hidden, 2)`, `hidden_food_revealed (num_hidden,) bool`, `hidden_food_reveal_timer (num_hidden,) int32`
- When `hidden_food.enabled`: each step checks if >= K alive agents within D of each hidden food position
- Revealed hidden food is collectible like normal food but gives `food_energy * value_multiplier`
- Timer counts down; when 0, food re-hides and respawns at new random position
- Normal food still exists simultaneously (balanced foraging)
- All existing tests pass unchanged (hidden food disabled by default)
- `python -m mypy src/ --ignore-missing-imports` passes
- `pytest tests/test_hidden_food.py -v` passes
- `pytest tests/test_env.py -v` passes (no regressions)
- `ruff check src/` passes

**Verification:**
```bash
pytest tests/test_hidden_food.py -v
pytest tests/test_env.py -v
python -c "from src.configs import Config; c = Config(); print('hidden_food enabled:', c.env.hidden_food.enabled)"
```

---

### US-011: Statistical Reporting Module (rliable Integration) [x]

**Task:** Create a statistical reporting module using `rliable` for IQM + bootstrap CI, performance profiles, and probability of improvement. Wrap scipy stats tests for convenience.

**Acceptance Criteria:**
- New file `src/analysis/statistics.py`
- `compute_iqm(scores, n_bootstrap=10000)` — Interquartile Mean with bootstrap CI via rliable
- `performance_profiles(score_dict, tau_range)` — CDFs for each method via rliable
- `probability_of_improvement(scores_x, scores_y)` — P(X > Y) via rliable
- `StatisticalReport` dataclass: method_name, iqm, ci_lower, ci_upper, median, mean, std, n_seeds
- `mann_whitney_test(x, y)` — wrapper returning full report
- `wilcoxon_test(x, y)` — wrapper returning full report
- `welch_t_test(x, y)` — wrapper returning full report
- `compare_methods(results_dict)` — returns pairwise comparison table
- `python -m mypy src/ --ignore-missing-imports` passes
- `pytest tests/test_statistics.py -v` passes
- `ruff check src/` passes

**Verification:**
```bash
pytest tests/test_statistics.py -v
python -c "from src.analysis.statistics import compute_iqm, StatisticalReport; print('OK')"
```

---

### US-012: Stigmergy Ablation at Scale (Launch Script) [x]

**Task:** Extend the existing ablation module with 3 new conditions (frozen field, no-field, write-only) and create a launch script for running all 6 conditions at scale with 20 seeds across 3 environment configs.

**Acceptance Criteria:**
- Extend `src/analysis/ablation.py` with new conditions:
  - `frozen`: field initialized from checkpoint but never updated (preserves structure, no new writes)
  - `no_field`: field completely removed from observations (zero-padded obs)
  - `write_only`: agents write to field but read zeros (can they still benefit?)
- Total: 6 conditions (normal, zeroed, random, frozen, no_field, write_only)
- New script `scripts/run_stigmergy_ablation.py`:
  - 6 conditions x 20 seeds x 3 env configs (standard, hidden, scarce)
  - `--dry-run` flag for verification without running
  - `--checkpoint` flag for model to evaluate
  - `--steps` flag for checkpoint milestones (1M, 5M, 10M)
  - Saves results to pickle per condition
- New test file for extended ablation conditions
- `python -m mypy src/ --ignore-missing-imports` passes
- `pytest tests/test_ablation_extended.py -v` passes
- `ruff check src/` passes

**Verification:**
```bash
pytest tests/test_ablation_extended.py -v
python scripts/run_stigmergy_ablation.py --dry-run --checkpoint checkpoints/params.pkl
```

---

### US-013: Superlinear Scaling Experiment (Launch Script) [x]

**Task:** Create scaling analysis module and launch script. Test N = 1, 2, 4, 8, 16, 32 agents with 3 field conditions and 20 seeds. Compute per-agent efficiency and fit power law.

**Acceptance Criteria:**
- New file `src/analysis/scaling.py`
- `compute_per_agent_efficiency(total_food, n_agents, solo_food)` — F_total(N) / (N * F_solo)
- `fit_power_law(n_agents_list, total_food_list)` — log(F_total) = alpha * log(N) + c, returns alpha with CI
- `ScalingResult` dataclass: n_agents, field_condition, efficiency, total_food, per_agent_food
- New script `scripts/run_scaling_experiment.py`:
  - N = [1, 2, 4, 8, 16, 32] agents x 3 field conditions (normal, zeroed, no_field) x 20 seeds
  - `--dry-run` flag
  - `--checkpoint` flag
  - Saves results to pickle
- The killer chart: X=agents, Y=per-agent food rate, 3 curves (normal > zeroed > no_field proves field helps)
- `python -m mypy src/ --ignore-missing-imports` passes
- `pytest tests/test_scaling.py -v` passes
- `ruff check src/` passes

**Verification:**
```bash
pytest tests/test_scaling.py -v
python scripts/run_scaling_experiment.py --dry-run
```

---

### US-014: Baselines Comparison Experiment (Launch Script) [x]

**Task:** Create the full baselines comparison: all 4 methods (Ours, IPPO, ACO-Fixed/Hybrid, MAPPO) across 3 env configs and 20 seeds with the same compute budget.

**Acceptance Criteria:**
- New file `src/experiments/baselines.py`
- `run_baselines_comparison(config, methods, n_seeds, paired)` — runs all methods, returns dict of ExperimentResults
- Methods: ["ours", "ippo", "aco_fixed", "aco_hybrid", "mappo"]
- Same seeds (paired) across methods for fair comparison
- Collects per method: reward, food_collected, specialization_score, mean_te, o_information, population_dynamics
- New script `scripts/run_baselines_comparison.py`:
  - All methods x 3 env configs x 20 seeds
  - `--dry-run` flag
  - `--methods` flag to select subset
  - Saves results to pickle
- `python -m mypy src/ --ignore-missing-imports` passes
- `pytest tests/test_experiments.py -v` passes (extended)
- `ruff check src/` passes

**Verification:**
```bash
pytest tests/test_experiments.py -v
python scripts/run_baselines_comparison.py --dry-run --methods ours,ippo
```

---

### US-015: Emergence Metrics Integration Script [x]

**Task:** Create a unified script that loads a checkpoint and computes ALL emergence metrics with surrogate tests for statistical significance. Windowed analysis for temporal stability.

**Acceptance Criteria:**
- New file `src/analysis/emergence_report.py`
- `EmergenceReport` dataclass aggregating: o_information, median_pid_synergy, causal_emergence_ei, rosas_psi, mean_transfer_entropy, specialization_score, division_of_labor — each with value, p_value, significant flag
- `compute_all_emergence_metrics(trajectory, config)` — computes everything, runs surrogate tests
- **Windowed analysis**: 1M-step windows with 50% overlap, returns per-window metrics for temporal stability
- JSON output format for downstream consumption
- New script `scripts/compute_emergence_metrics.py`:
  - `--checkpoint` flag
  - `--output` flag for JSON path
  - `--window-size` and `--window-overlap` flags
- `python -m mypy src/ --ignore-missing-imports` passes
- `pytest tests/test_emergence_report.py -v` passes
- `ruff check src/` passes

**Verification:**
```bash
pytest tests/test_emergence_report.py -v
python scripts/compute_emergence_metrics.py --checkpoint checkpoints/params.pkl --output /tmp/emergence.json
```

---

### US-016: Publication Figures — Scaling + Ablation [x]

**Task:** Create publication-quality matplotlib figures for scaling curves and ablation results. Consistent style, 300 DPI, PDF + PNG output.

**Acceptance Criteria:**
- New file `src/analysis/paper_figures.py`
- `plot_scaling_curve(scaling_results, output_path)` — X=agents, Y=per-agent food, 3 curves with power law fit + alpha=1.0 reference line
- `plot_ablation_bars(ablation_results, output_path)` — bar chart with IQM + bootstrap CI error bars, 6 conditions
- `plot_performance_profiles(score_dict, output_path)` — CDFs via rliable
- `plot_probability_of_improvement(scores_dict, output_path)` — heatmap
- Consistent style: font size 12, color palette (colorblind-safe), 300 DPI, PDF + PNG
- New script `scripts/generate_paper_figures.py`:
  - `--results-dir` flag pointing to experiment results
  - `--output-dir` flag for figure output
  - Generates all figures in one run
- `python -m mypy src/ --ignore-missing-imports` passes
- `pytest tests/test_paper_figures.py -v` passes
- `ruff check src/` passes

**Verification:**
```bash
pytest tests/test_paper_figures.py -v
python scripts/generate_paper_figures.py --results-dir results/ --output-dir figures/
```

---

### US-017: Publication Figures — Emergence Metrics [x]

**Task:** Add the "holy shit" emergence visualization functions: PID synergy heatmaps, O-information trajectories, causal emergence gap, and phase transition annotations.

**Acceptance Criteria:**
- Extend `src/analysis/paper_figures.py` with:
  - `plot_pid_synergy_heatmap(synergy_matrix, output_path)` — agent-pair matrix with colorbar
  - `plot_o_information_trajectory(o_info_over_time, surrogate_ci, output_path)` — line plot with shaded surrogate CI band
  - `plot_causal_emergence_gap(macro_ei, micro_ei, steps, output_path)` — two lines showing gap growing over training
  - `plot_phase_transitions(metrics_over_time, transition_events, output_path)` — training curves with vertical annotations at phase transitions (Baker et al. 2019 style)
- Same consistent style as US-016 (font, color, DPI)
- `python -m mypy src/ --ignore-missing-imports` passes
- `pytest tests/test_paper_figures.py -v` passes (extended)
- `ruff check src/` passes

**Verification:**
```bash
pytest tests/test_paper_figures.py -v
```

---

### US-018: Documentation and Final Verification [ ]

**Task:** Update CLAUDE.md with Phase 5 status, new modules, and commands. Create end-to-end verification script. Copy PRD to completed file.

**Acceptance Criteria:**
- Update `CLAUDE.md`:
  - Phase 5 row in status table (COMPLETE)
  - New modules in module map (o_information, pid_synergy, causal_emergence, surrogates, statistics, scaling, paper_figures, emergence_report)
  - New baselines in module map (ippo, aco_fixed, mappo)
  - New experiments in module map (runner, configs, baselines)
  - New commands section for Phase 5 scripts
- New script `scripts/verify_phase5.py`:
  - Runs mini version of every metric (1 seed, 1000 steps)
  - Imports all new modules
  - Verifies output formats
  - Exit 0 on success, exit 1 on failure
- Copy PRD to `PRD-PHASE5-COMPLETE.md` with all stories marked `[x]`
- `python -m mypy src/ --ignore-missing-imports` passes
- `pytest tests/ -v` passes (ALL tests)
- `ruff check src/` passes

**Verification:**
```bash
python scripts/verify_phase5.py
pytest tests/ -v
python -m mypy src/ --ignore-missing-imports
ruff check src/
```

---

## Technical Reference

### O-Information (DR-1)

```
Omega(X) = TC(X) - DTC(X)
         = (n-2) * H(X) + sum_i H(X_i) - sum_i H(X_{-i})

Omega < 0  =>  synergy dominates (emergence!)
Omega > 0  =>  redundancy dominates (no emergence)
Omega = 0  =>  balanced
```

Library: `hoi` (JAX-native, O(n) scaling)

### Partial Information Decomposition (DR-1)

```
I(S1, S2; T) = Synergy + Redundancy + Unique_S1 + Unique_S2

I_min(S1, S2; T) = min over Q: I_Q(S1; T) + I_Q(S2; T)  (Williams & Beer 2010)

Variables:
  S1 = agent_i action (discrete, 6 values)
  S2 = field summary at agent_i position (K=2 quantile bins)
  T  = future food collected by agent_i (K=2 quantile bins)
```

Jeffreys smoothing: add alpha=0.5 pseudocounts to joint distribution before passing to `dit`.

Interaction Information (quick screen):
```
II(X; Y; Z) = I(X; Y | Z) - I(X; Y)
II < 0  =>  synergy dominates
```

### Effective Information / Causal Emergence (DR-2)

Hoel EI:
```
EI(M) = H_max(input) - <H(output | do(input))>
      = log(N) - (1/N) * sum_i H(row_i of TPM)

CE_gap = EI(macro_TPM) - EI(micro_TPM)
CE_gap > 0  =>  causal emergence at macro scale
```

Rosas Psi:
```
Psi(V) = I(V_t; V_{t+1}) - sum_i I(X_{i,t}; V_{t+1})

Psi > 0  =>  macro variable has causal power beyond sum of micro parts
```

Macro variable candidates:

| Variable | Description | Bins |
|----------|-------------|------|
| Population count | sum(alive) | 4-8 |
| Mean field intensity | mean(field_values) | 4-8 |
| Total food collected | cumulative food | 4-8 |
| Spatial dispersion | std(agent_positions) | 4-8 |
| Field entropy | shannon_entropy(field) | 4-8 |

### ACO Parameters (DR-3)

```
p_ij = (tau_ij^alpha * eta_ij^beta) / sum_k (tau_ik^alpha * eta_ik^beta)

alpha = 1.0   (pheromone importance)
beta  = 2.0   (heuristic importance — distance to food)
rho   = 0.5   (evaporation rate)
Q     = 1.0   (deposit quantity)
```

Reference: Dorigo & Stutzle (2004), "Ant Colony Optimization"

### Superlinear Scaling (DR-4)

```
Efficiency(N) = F_total(N) / (N * F_solo)

Power law: log(F_total) = alpha * log(N) + c
  alpha > 1.0  =>  superlinear (field helps more with more agents)
  alpha = 1.0  =>  linear (no coordination benefit)
  alpha < 1.0  =>  sublinear (crowding hurts)
```

Reference: Hamann (2018), "Swarm Robotics: A Formal Approach"

### Statistical Standards (DR-4)

Per Agarwal et al. (2021), "Deep RL at the Edge of the Statistical Precipice":

- **IQM** (Interquartile Mean): robust to outliers, better than mean for RL
- **Bootstrap CI**: BCa bootstrap with 10,000 resamples
- **Performance profiles**: CDF of normalized scores across tasks
- **Probability of improvement**: P(X > Y) with bootstrap CI
- **Minimum 20 seeds** for reliable statistics
- **Paired seeds**: same seed across methods reduces variance

Statistical tests:
- Mann-Whitney U: non-parametric, unpaired comparison
- Wilcoxon signed-rank: non-parametric, paired comparison
- Welch's t-test: parametric with unequal variance assumption

### Hidden Resources (DR-4)

```
Reveal condition: count(alive agents within Chebyshev(D=3) of hidden food) >= K=3
Reward: food_energy * 5.0 (5x normal value)
Duration: 10 steps revealed, then re-hides
Count: 3 hidden food items on grid
```

Reference: Level-Based Foraging (LBF) benchmark, Papoudakis et al. (2021)

---

## Implementation Order

### Group 1 (start together — no dependencies):
- US-001: Phase 5 Dependencies
- US-005: Surrogate Testing Framework
- US-006: IPPO Baseline
- US-010: Hidden Resources Environment

### Group 2 (after US-001):
- US-002: O-Information
- US-003: PID Synergy
- US-004: Causal Emergence
- US-011: Statistical Reporting

### Group 3 (after US-006):
- US-007: ACO Baselines
- US-008: MAPPO Baseline
- US-009: Experiment Harness

### Group 4 (after Groups 2 + 3):
- US-012: Stigmergy Ablation at Scale
- US-013: Superlinear Scaling
- US-014: Baselines Comparison
- US-015: Emergence Metrics Integration

### Group 5 (after Group 4):
- US-016: Publication Figures — Scaling + Ablation
- US-017: Publication Figures — Emergence Metrics

### Group 6 (final):
- US-018: Documentation and Final Verification

---

## Dependency Graph

```
US-001 (deps) ---+---> US-002 (O-info) ----------------+
                 +---> US-003 (PID) --------------------+
                 +---> US-004 (CE) ---------------------+--> US-015 (metrics) --> US-017 (figs) --> US-018
                 +---> US-011 (statistics) ------------>+--> US-016 (figs) ----------------------->
                                                        |
US-005 (surrogates) ---------------------------------->+

US-006 (IPPO) ---+---> US-007 (ACO + Hybrid) ----------+
                 +---> US-008 (MAPPO) ------------------+
                 +---> US-009 (harness) ----------------+--> US-014 (baselines) --> US-018
                            |                           |
                            +---> US-012 (ablation) --> US-013 (scaling) --> US-016
                            |
US-010 (hidden food) -----------------------------------+
```

---

## Files to Create/Modify

| File | Action | Story |
|------|--------|-------|
| `pyproject.toml` | MOD | US-001 |
| `tests/test_phase5_deps.py` | NEW | US-001 |
| `src/analysis/o_information.py` | NEW | US-002 |
| `tests/test_o_information.py` | NEW | US-002 |
| `src/analysis/pid_synergy.py` | NEW | US-003 |
| `tests/test_pid_synergy.py` | NEW | US-003 |
| `src/analysis/causal_emergence.py` | NEW | US-004 |
| `tests/test_causal_emergence.py` | NEW | US-004 |
| `src/analysis/surrogates.py` | NEW | US-005 |
| `tests/test_surrogates.py` | NEW | US-005 |
| `src/baselines/__init__.py` | NEW | US-006 |
| `src/baselines/ippo.py` | NEW | US-006 |
| `tests/test_baselines.py` | NEW | US-006 |
| `src/baselines/aco_fixed.py` | NEW | US-007 |
| `src/baselines/mappo.py` | NEW | US-008 |
| `src/experiments/__init__.py` | NEW | US-009 |
| `src/experiments/runner.py` | NEW | US-009 |
| `src/experiments/configs.py` | NEW | US-009 |
| `tests/test_experiments.py` | NEW | US-009 |
| `src/configs.py` | MOD | US-010 |
| `src/environment/state.py` | MOD | US-010 |
| `src/environment/env.py` | MOD | US-010 |
| `tests/test_hidden_food.py` | NEW | US-010 |
| `src/analysis/statistics.py` | NEW | US-011 |
| `tests/test_statistics.py` | NEW | US-011 |
| `src/analysis/ablation.py` | MOD | US-012 |
| `scripts/run_stigmergy_ablation.py` | NEW | US-012 |
| `tests/test_ablation_extended.py` | NEW | US-012 |
| `src/analysis/scaling.py` | NEW | US-013 |
| `scripts/run_scaling_experiment.py` | NEW | US-013 |
| `tests/test_scaling.py` | NEW | US-013 |
| `src/experiments/baselines.py` | NEW | US-014 |
| `scripts/run_baselines_comparison.py` | NEW | US-014 |
| `src/analysis/emergence_report.py` | NEW | US-015 |
| `scripts/compute_emergence_metrics.py` | NEW | US-015 |
| `tests/test_emergence_report.py` | NEW | US-015 |
| `src/analysis/paper_figures.py` | NEW | US-016, US-017 |
| `scripts/generate_paper_figures.py` | NEW | US-016 |
| `tests/test_paper_figures.py` | NEW | US-016, US-017 |
| `CLAUDE.md` | MOD | US-018 |
| `scripts/verify_phase5.py` | NEW | US-018 |
| `PRD-PHASE5-COMPLETE.md` | NEW | US-018 |

---

## Definition of Done

1. All 18 user stories marked `[x]`
2. O-information, PID synergy, and causal emergence metrics implemented and tested
3. Surrogate tests prove significance (p < 0.05) for at least one metric
4. IPPO, ACO-Fixed, ACO-Hybrid, and MAPPO baselines implemented
5. Hidden resources environment works with backward compatibility
6. Experiment harness runs paired-seed multi-method comparisons
7. 6-condition ablation script ready to launch
8. Superlinear scaling script ready to launch
9. Publication-quality figures generate from experiment results
10. `pytest tests/ -v` — all pass
11. `python -m mypy src/ --ignore-missing-imports` — clean
12. `ruff check src/` — clean
13. `scripts/verify_phase5.py` exits 0

---

## Key References

- Williams & Beer (2010) — Partial Information Decomposition
- Riedl et al. (2025) — PID for multi-agent emergence
- Rosas et al. (2020) — Causal emergence, Psi metric
- Hoel et al. (2013) — Effective Information, macro-scale causation
- Dorigo & Stutzle (2004) — Ant Colony Optimization
- Yu et al. (2022) — MAPPO, centralized training decentralized execution
- Agarwal et al. (2021) — Statistical reporting for deep RL (rliable)
- Hamann (2018) — Swarm robotics formal scaling analysis
- Baker et al. (2019) — OpenAI Hide-and-Seek, emergent phase detection
- Papoudakis et al. (2021) — Level-Based Foraging benchmark
- Lehman & Stanley (2011) — Novelty Search

---

*Total: 18 stories, ~18 Ralph iterations, ~4,500 lines new code.*
*PRD designed for Ralph Loop autonomous execution.*
*Each story should complete in one Claude context window.*
