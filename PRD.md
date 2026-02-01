# Phase 3 PRD: Specialization Detection

## Vision

Detect when agents evolve into different "species" with distinct strategies. The field's value may not be higher average reward — it may be enabling DIVERSITY and SPECIALIZATION.

## Key Hypothesis

With shared field + evolution, agents should differentiate into roles:
- **Scouts**: Explore, find food, write locations to field
- **Followers**: Read field, exploit known food locations
- **Hoarders**: Stay near food, defend territory

## Success Criteria

1. Behavioral clustering shows 2+ distinct strategies
2. Weight divergence increases over training
3. Field usage patterns differ between clusters
4. Specialization improves collective efficiency vs uniform population

---

## User Stories

### US-001: Weight Divergence Metric [ ]
**Task:** Measure how different agents' neural network weights become over time.
**Files:** `src/analysis/specialization.py` (new)
**Changes:**
- `compute_weight_divergence(agent_params)` — pairwise cosine distance between agent weight vectors
- `mean_divergence`, `max_divergence`, `divergence_matrix`
- Track divergence over training iterations
**Verification:** `pytest tests/test_specialization.py::test_weight_divergence -v` passes

### US-002: Behavioral Feature Extraction [ ]
**Task:** Extract behavioral features from agent trajectories for clustering.
**Files:** `src/analysis/specialization.py`
**Changes:**
- `extract_behavior_features(trajectory)` returns feature vector:
  - Movement entropy (how random is movement)
  - Field write frequency
  - Field read reliance (correlation between field values and actions)
  - Food collection rate
  - Distance traveled per episode
  - Reproduction frequency
  - Average energy level
**Verification:** `pytest tests/test_specialization.py::test_behavior_features -v` passes

### US-003: Trajectory Recording [ ]
**Task:** Record agent trajectories during evaluation for analysis.
**Files:** `src/analysis/trajectory.py` (new)
**Changes:**
- `TrajectoryRecorder` class that captures per-step:
  - Agent positions
  - Actions taken
  - Field reads/writes
  - Energy levels
  - Rewards
- `record_episode(network, params, config, key)` returns trajectory data
**Verification:** `pytest tests/test_specialization.py::test_trajectory_recording -v` passes

### US-004: Behavioral Clustering [ ]
**Task:** Cluster agents by behavioral features to detect specialization.
**Files:** `src/analysis/specialization.py`
**Changes:**
- `cluster_agents(behavior_features, n_clusters=3)` using K-means
- `find_optimal_clusters(features, max_k=5)` using silhouette score
- Return cluster labels, centroids, silhouette score
**Verification:** `pytest tests/test_specialization.py::test_clustering -v` passes

### US-005: Specialization Score [ ]
**Task:** Single metric for "how specialized is this population?"
**Files:** `src/analysis/specialization.py`
**Changes:**
- `specialization_score(behavior_features)` returns 0-1 score:
  - 0 = all agents identical
  - 1 = completely distinct clusters
- Based on: silhouette score + weight divergence + behavioral variance
**Verification:** `pytest tests/test_specialization.py::test_specialization_score -v` passes

### US-006: Field Usage Analysis [ ]
**Task:** Analyze how different clusters use the field differently.
**Files:** `src/analysis/specialization.py`
**Changes:**
- `analyze_field_usage(trajectories, cluster_labels)` returns per-cluster:
  - Write frequency
  - Write patterns (where do they write?)
  - Read patterns (how much do actions depend on field?)
- Identify "writers" vs "readers"
**Verification:** `pytest tests/test_specialization.py::test_field_usage -v` passes

### US-007: Specialization Tracker [ ]
**Task:** Track specialization metrics during training.
**Files:** `src/analysis/specialization.py`, `src/training/train.py`
**Changes:**
- `SpecializationTracker` class (like EmergenceTracker)
- Compute metrics every N steps
- Detect "specialization events" (sudden increase in divergence)
- Log to training metrics
**Verification:** Training output shows specialization metrics

### US-008: Lineage-Strategy Correlation [ ]
**Task:** Do dominant lineages have distinct strategies?
**Files:** `src/analysis/specialization.py`
**Changes:**
- `correlate_lineage_strategy(lineage_tracker, cluster_labels, agent_ids)`
- Check if agents from same lineage cluster together
- Identify "specialist lineages"
**Verification:** `pytest tests/test_specialization.py::test_lineage_correlation -v` passes

### US-009: Diversity vs Performance Ablation [ ]
**Task:** Test if specialization helps collective performance.
**Files:** `src/analysis/ablation.py`, `scripts/run_specialization_ablation.py`
**Changes:**
- Compare populations with high vs low specialization scores
- Clone experiment: force uniform weights vs allow divergence
- Report: collective food collected, survival rate, population stability
**Verification:** `python scripts/run_specialization_ablation.py` produces comparison

### US-010: Specialization Visualization [ ]
**Task:** Visualize agent clusters and their behaviors.
**Files:** `src/analysis/visualization.py` (new)
**Changes:**
- `plot_behavior_clusters(features, labels)` — 2D scatter (PCA/t-SNE)
- `plot_weight_divergence_over_time(divergence_history)`
- `plot_field_usage_by_cluster(usage_data)`
- `plot_specialization_score_over_time(scores)`
**Verification:** Running visualization produces PNG files

### US-011: Species Detection [ ]
**Task:** Formally detect when distinct "species" have emerged.
**Files:** `src/analysis/specialization.py`
**Changes:**
- `detect_species(behavior_features, threshold=0.7)`:
  - Species = stable behavioral cluster that persists across generations
  - Check: cluster membership inherited from parent?
  - Check: cluster boundaries are clear (silhouette > threshold)?
- Return: list of species with characteristics
**Verification:** `pytest tests/test_specialization.py::test_species_detection -v` passes

### US-012: Specialization Report [ ]
**Task:** Generate comprehensive specialization analysis report.
**Files:** `scripts/generate_specialization_report.py` (new)
**Changes:**
- Run trained model through analysis
- Generate markdown report with:
  - Specialization score over training
  - Detected species and their characteristics
  - Field usage patterns per species
  - Lineage-strategy correlations
  - Visualizations
**Verification:** `python scripts/generate_specialization_report.py --checkpoint checkpoints/params.pkl` produces report

### US-013: Update Training to Encourage Specialization [ ]
**Task:** Add config options that encourage specialization.
**Files:** `src/configs.py`, `src/training/train.py`
**Changes:**
- `specialization.diversity_bonus: float = 0.0` — reward diversity in population
- `specialization.niche_pressure: float = 0.0` — penalize identical strategies
- Optional: different mutation rates for different weight layers
**Verification:** Training with diversity_bonus > 0 increases specialization score

### US-014: Integration Test [ ]
**Task:** End-to-end specialization detection test.
**Files:** `tests/test_integration.py`
**Changes:**
- Train for N steps
- Extract trajectories
- Compute specialization score
- Assert: specialization > baseline random
**Verification:** `pytest tests/test_integration.py::test_specialization_emerges -v` passes

### US-015: Final Review [ ]
**Task:** Code quality and documentation.
**Files:** All modified files
**Verification:**
- `pytest tests/ -v` — all pass
- `python -m mypy src/ --ignore-missing-imports` — no errors
- README updated with Phase 3 instructions
- Example outputs documented

---

## Technical Notes

### Behavioral Features Vector (per agent)
```python
features = [
    movement_entropy,      # 0=deterministic, 1=random
    field_write_rate,      # writes per step
    field_read_influence,  # correlation(field_values, actions)
    food_collection_rate,  # food per step
    distance_per_episode,  # total movement
    reproduction_rate,     # babies per 100 steps
    mean_energy,           # average energy level
    exploration_vs_exploit # ratio of new vs revisited cells
]
```

### Species Detection Criteria
A "species" requires:
1. Distinct behavioral cluster (silhouette > 0.5)
2. Weight divergence from other clusters
3. Hereditary: children cluster with parents
4. Stable: cluster persists for 100+ steps

### Expected Species Types
- **Explorers**: High movement entropy, high field writes, low food rate
- **Exploiters**: Low entropy, high field reads, high food rate
- **Balanced**: Medium everything, generalist strategy

## Out of Scope (Phase 4+)

- Multi-species interaction dynamics
- Competitive/cooperative species relationships
- Explicit communication between agents
- Environment modification (nest building)
- Predator-prey dynamics

---

*PRD designed for Ralph Loop autonomous execution.*
*Each story should complete in one Claude context window.*
*Total: 15 stories expected.*
