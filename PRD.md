# Phase 4B PRD: Kaggle Infrastructure

## Vision

Enable **autonomous 24/7 training** on Kaggle with proper checkpointing, and wire the training loop to the dashboard so we can visualize **real** emergence.

---

## Goals

- Real training data streams to dashboard (not mock)
- Full checkpoint recovery (optimizer, PRNG, step, trackers)
- Kaggle CLI automation for autonomous training
- Zero data loss on disconnect (lose at most 100k steps)

## Non-Goals

- No Colab support (Kaggle only)
- No distributed training across machines  
- No real-time streaming from Kaggle to local dashboard
- No Orbax (use enhanced pickle for cross-platform)

---

## Technical Decisions (ALREADY MADE)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Checkpoint format | Enhanced pickle | Orbax has GPU→CPU bugs |
| Save frequency | Every 100k steps | ~15-30 min, keep last 5 |
| Cloud platform | Kaggle only | Background exec, 30hr/week |
| Dashboard streaming | Every 10 steps | 30fps feel |

---

## User Stories

### US-001: Wire Training to Dashboard [x]

**Task:** Connect training loop to WebSocket streaming server.

**Acceptance Criteria:**
- `train.py` imports `TrainingBridge` when `--server` flag passed
- Every 10 steps, call `bridge.publish_frame()` with current state
- Frame includes: agent positions, field state, metrics
- Training works normally without `--server` (no regression)
- `python -m mypy src/ --ignore-missing-imports` passes
- `pytest tests/test_training.py -v` passes

**Verification:**
```bash
python -m src.training.train --server --train.total-steps 100
# Dashboard at localhost:5173 shows real agents
```

---

### US-002: Tracker Serialization [x]

**Task:** Add serialize/deserialize methods to EmergenceTracker and SpecializationTracker.

**Acceptance Criteria:**
- `EmergenceTracker.to_dict()` returns serializable dict
- `EmergenceTracker.from_dict(d)` restores state
- Same for `SpecializationTracker`
- Round-trip preserves all history
- `python -m mypy src/ --ignore-missing-imports` passes
- `pytest tests/test_tracker_serialization.py -v` passes

**Verification:**
```bash
pytest tests/test_tracker_serialization.py -v
```

---

### US-003: Checkpointing Module [x]

**Task:** Create checkpointing module that saves/loads full training state.

**Acceptance Criteria:**
- New file: `src/training/checkpointing.py`
- `save_checkpoint(path, state_dict)` saves: params, opt_state, agent_params, prng_key, step, config, tracker_state
- `load_checkpoint(path)` restores all above
- JAX arrays → numpy before pickling (cross-platform)
- Checkpoint rotation: keep last 5, delete oldest
- `python -m mypy src/ --ignore-missing-imports` passes
- `pytest tests/test_checkpointing.py -v` passes

**Verification:**
```bash
pytest tests/test_checkpointing.py -v
```

---

### US-004: Periodic Checkpoint Saves [x]

**Task:** Modify training loop to save checkpoints periodically.

**Acceptance Criteria:**
- Use `config.log.save_interval` (currently unused)
- Save checkpoint every `save_interval` steps
- Print: "Checkpoint saved: step {N} → {path}"
- Save final checkpoint on normal completion
- `python -m mypy src/ --ignore-missing-imports` passes
- `pytest tests/ -v` passes

**Verification:**
```bash
python -m src.training.train --train.total-steps 1000 --log.save-interval 200
ls checkpoints/  # Should show 5 checkpoints
```

---

### US-005: Proper Resume Logic [x]

**Task:** Fix resume to restore full training state.

**Acceptance Criteria:**
- `--train.resume-from` loads full checkpoint
- Optimizer state restored (Adam momentum continues)
- PRNG key restored (same trajectory)
- Step counter restored (correct logging)
- Tracker state restored (metrics history continues)
- `python -m mypy src/ --ignore-missing-imports` passes
- `pytest tests/ -v` passes

**Verification:**
```bash
python -m src.training.train --train.total-steps 500
python -m src.training.train --train.total-steps 1000 --train.resume-from checkpoints/latest.pkl
# Metrics continuous, no reset
```

---

### US-006: Signal Handlers [ ]

**Task:** Add signal handlers to save checkpoint on interrupt.

**Acceptance Criteria:**
- SIGTERM handler saves emergency checkpoint
- SIGINT (Ctrl+C) handler saves emergency checkpoint
- `atexit` handler as fallback
- Emergency checkpoints: `emergency_step_{N}.pkl`
- Works on Mac and Linux
- `python -m mypy src/ --ignore-missing-imports` passes
- `pytest tests/ -v` passes

**Verification:**
```bash
python -m src.training.train --train.total-steps 100000 &
sleep 10 && kill -TERM $!
# Check for emergency checkpoint
```

---

### US-007: Kaggle Notebook [ ]

**Task:** Create ready-to-use Kaggle notebook.

**Acceptance Criteria:**
- New file: `notebooks/kaggle_training.ipynb`
- Cell 1: Install JAX with CUDA, verify GPU
- Cell 2: Clone repo, install dependencies
- Cell 3: Configuration (hyperparameters, paths)
- Cell 4: Resume-or-start logic
- Cell 5: Training with progress display
- All cells syntactically valid
- `python -m mypy src/ --ignore-missing-imports` passes

**Verification:**
- Notebook opens without error in Jupyter
- Manual Kaggle upload test (documented in progress.txt)

---

### US-008: Kaggle CLI Scripts [ ]

**Task:** Create scripts for automated Kaggle interaction.

**Acceptance Criteria:**
- `scripts/kaggle_setup.sh` — setup instructions
- `scripts/kaggle_push.sh` — push notebook
- `scripts/kaggle_run.sh` — start execution
- `scripts/kaggle_status.sh` — check status
- `scripts/kaggle_download.sh` — download results
- All scripts executable and documented
- README section on Kaggle workflow
- `python -m mypy src/ --ignore-missing-imports` passes

**Verification:**
```bash
./scripts/kaggle_push.sh --help  # Shows usage
```

---

### US-009: Documentation and Final Review [ ]

**Task:** Document Kaggle workflow and verify integration.

**Acceptance Criteria:**
- `docs/kaggle-workflow.md` with step-by-step guide
- All previous stories' code works together
- Test: save checkpoint → load → training continues correctly
- Update README with Phase 4B section
- `python -m mypy src/ --ignore-missing-imports` passes
- `pytest tests/ -v` passes (all tests)

**Verification:**
```bash
pytest tests/ -v
# Manual: Document Kaggle test results in progress.txt
```

---

## Definition of Done

1. Dashboard shows real training data
2. Training resumes with identical state
3. Kaggle scripts exist and are documented
4. All 9 stories marked `[x]`
5. All tests pass
6. mypy clean

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/training/checkpointing.py` | NEW |
| `src/training/train.py` | MOD |
| `src/analysis/emergence.py` | MOD |
| `src/analysis/specialization.py` | MOD |
| `notebooks/kaggle_training.ipynb` | NEW |
| `scripts/kaggle_*.sh` | NEW (5 scripts) |
| `tests/test_checkpointing.py` | NEW |
| `tests/test_tracker_serialization.py` | NEW |
| `docs/kaggle-workflow.md` | NEW |

---

*Total: 9 stories, ~9-12 Ralph iterations expected.*
