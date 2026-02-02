# Phase 4B: Kaggle Infrastructure

## Goal

Fix checkpointing and set up Kaggle so we can run long training sessions 24/7 without losing progress.

This is a prerequisite for Phase 5. You can't run 30x 10M-step ablation experiments if a Kaggle disconnect at step 800k wipes everything.

---

## Part 1: Fix Checkpointing

### The Problem

Current checkpointing is broken:

- Optimizer state NOT saved (Adam momentum resets on resume — training quality degrades)
- PRNG key NOT saved (different random trajectory on resume — results not reproducible)
- Step counter NOT saved (can't tell where you left off)
- Tracker state NOT saved (metrics history lost — can't plot training curves across resumes)
- save_interval exists in config but is never actually used in the training loop

If Kaggle disconnects at step 800k, you lose everything since the last manual save.

### What Needs to Be Built

| File | Change |
|------|--------|
| `src/training/checkpointing.py` | NEW — save/load full state with checkpoint rotation |
| `src/training/train.py` | MOD — periodic saves using save_interval, proper resume |
| `src/analysis/emergence.py` | MOD — tracker serialize/deserialize |

### Key Decisions

- **Enhanced pickle, NOT Orbax.** Orbax has cross-platform issues. Pickle is simple and works everywhere.
- **Save every ~100k steps.** Keep last 3-5 checkpoints with rotation (delete oldest when new one is saved).

### What Gets Saved (Full RunnerState)

```
Checkpoint contents:
  - params           (shared policy weights)
  - opt_state        (Adam optimizer momentum + variance)
  - agent_params     (per-agent weights, if evolution enabled)
  - env_state        (full environment snapshot)
  - prng_key         (exact random state for reproducibility)
  - step_counter     (where we are in training)
  - tracker_state    (EmergenceTracker + SpecializationTracker history)
  - config           (full config for reference)
```

### Success Criteria (Part 1)

- Can resume training from checkpoint with no quality degradation
- Optimizer momentum is preserved across resume (loss doesn't spike)
- Step counter is continuous across resume
- Metrics history is continuous across resume (no gaps in plots)
- Last 3-5 checkpoints kept, oldest auto-deleted
- save_interval config actually works

---

## Part 2: Kaggle CLI Setup

### The Problem

Training on a local MacBook is too slow for the long runs Phase 5 needs. Kaggle offers free GPUs (T4/P100, 30hrs/week) but we have no setup for running there.

### What Needs to Be Built

| File | Change |
|------|--------|
| `notebooks/kaggle_training.ipynb` | NEW — ready-to-use Kaggle notebook |

### What the Notebook Does

1. Installs the project dependencies
2. Clones or uploads the codebase
3. Detects available GPU and configures JAX accordingly
4. Runs training with checkpointing enabled
5. Saves checkpoints to Kaggle's persistent `/kaggle/working/` directory
6. On resume: loads latest checkpoint and continues training seamlessly

### Success Criteria (Part 2)

- Kaggle notebook runs end-to-end: install → train → checkpoint → resume → continue
- Can start a run, let Kaggle disconnect, restart notebook, and pick up where we left off
- Training uses Kaggle GPU (not CPU)
- Checkpoints survive between Kaggle sessions
- Can download trained checkpoints for local analysis
