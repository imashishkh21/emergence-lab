# Kaggle Training Workflow

Step-by-step guide to running autonomous training on Kaggle with GPU acceleration.

## Prerequisites

- A [Kaggle account](https://www.kaggle.com/) with phone verification (required for GPU access)
- Kaggle API credentials configured locally
- The `kaggle` CLI installed (`pip install kaggle`)

## Quick Start

```bash
# 1. One-time setup
./scripts/kaggle_setup.sh --username YOUR_KAGGLE_USERNAME

# 2. Push notebook to Kaggle
./scripts/kaggle_push.sh

# 3. Start training
./scripts/kaggle_run.sh

# 4. Monitor progress
./scripts/kaggle_status.sh --watch

# 5. Download results when complete
./scripts/kaggle_download.sh
```

---

## Detailed Setup

### 1. Kaggle API Credentials

Create an API token from your Kaggle account settings:

1. Go to https://www.kaggle.com/settings
2. Scroll to "API" section
3. Click "Create New Token"
4. Save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`
5. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### 2. Initialize Kernel Metadata

Run the setup script to create the kernel configuration:

```bash
./scripts/kaggle_setup.sh --username YOUR_KAGGLE_USERNAME
```

This creates `notebooks/kernel-metadata.json` with:
- Kernel ID: `YOUR_USERNAME/emergence-lab-training`
- GPU accelerator enabled
- Internet access enabled (for repo clone)

### 3. Configure Training Parameters

Edit Cell 3 in `notebooks/kaggle_training.ipynb` to customize:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `total_steps` | 10,000,000 | ~8-12 hours on T4 GPU |
| `num_envs` | 32 | Kaggle T4 has ~16GB VRAM |
| `save_interval` | 100,000 | Checkpoint every ~15-30 min |
| `diversity_bonus` | 0.1 | Encourages weight diversity |
| `niche_pressure` | 0.05 | Penalizes similar agents |

Default evolution params are survival-friendly:
```
num_food=20, starting_energy=200, food_energy=100,
reproduce_threshold=120, reproduce_cost=50
```

---

## Running Training

### First Run

```bash
# Push and start
./scripts/kaggle_push.sh
./scripts/kaggle_run.sh

# Monitor (polls every 60s)
./scripts/kaggle_status.sh --watch
```

### Resuming After Interruption

The notebook auto-detects existing checkpoints. Simply re-run:

```bash
./scripts/kaggle_run.sh
```

The resume logic in Cell 4:
1. Checks for `checkpoints/latest.pkl` in the Kaggle working directory
2. If found, loads the checkpoint and continues from the saved step
3. Optimizer state, PRNG key, per-agent params, and tracker history are all restored

### Downloading Results

```bash
# Download to default directory (./kaggle_output/)
./scripts/kaggle_download.sh

# Or specify a custom directory
./scripts/kaggle_download.sh --output-dir ./my_results
```

Downloaded checkpoints can be used locally:

```bash
# Resume training locally
python -m src.training.train --train.resume-from kaggle_output/checkpoints/latest.pkl

# Run analysis
python -m src.analysis.ablation --checkpoint kaggle_output/checkpoints/latest.pkl
```

---

## Checkpointing System

### How It Works

Training saves full state checkpoints periodically and on interruption:

**Periodic saves** (every `save_interval` steps):
- Named `step_{N}.pkl` in the checkpoint directory
- Last 5 checkpoints are kept; older ones are deleted automatically
- A `latest.pkl` symlink always points to the most recent save

**Emergency saves** (on SIGTERM/SIGINT/Kaggle timeout):
- Named `emergency_step_{N}.pkl`
- Never trigger checkpoint rotation (won't delete regular checkpoints)
- `atexit` handler provides a fallback if signal handlers miss

### What's Saved

Each checkpoint contains the full training state:

| Field | Description |
|-------|-------------|
| `params` | Shared policy weights (Flax pytree) |
| `opt_state` | Adam optimizer momentum/variance accumulators |
| `agent_params` | Per-agent evolved weights (if evolution enabled) |
| `prng_key` | JAX PRNG key for deterministic replay |
| `step` | Training step counter |
| `config` | Full configuration as dict |
| `tracker_state` | Emergence + Specialization tracker history |

All JAX arrays are converted to numpy before saving for cross-platform compatibility (GPU checkpoints load on CPU and vice versa).

### Resume Guarantees

When resuming from a checkpoint:
- **Optimizer continues** — Adam momentum is not reset, so learning rate warmup and gradient statistics carry over
- **Random stream continues** — PRNG key is restored, so the same sequence of random numbers is used
- **Step counter continues** — Logging, checkpointing intervals, and W&B all pick up from the correct step
- **Tracker history continues** — Emergence events and specialization metrics history is preserved
- **Per-agent weights preserved** — Evolved agent weights are restored exactly

---

## Kaggle Resource Budget

| Resource | Limit | Notes |
|----------|-------|-------|
| GPU time | 30 hours/week | T4 GPU; resets weekly |
| Session time | 12 hours max | Single session limit |
| Disk space | ~20 GB | /kaggle/working |
| Internet | Available | Needed for repo clone |

**Typical run**: 10M steps takes ~8-12 hours on T4. One full training run fits in a single session.

**Multi-session strategy**: For longer runs (50M+ steps):
1. Run first session (10M steps) — checkpoints saved
2. Download checkpoints
3. Push checkpoints back as Kaggle dataset (optional)
4. Run next session — auto-resumes from latest checkpoint

---

## Troubleshooting

### "No GPU detected" in Cell 1

Enable GPU accelerator in Kaggle notebook settings (Settings > Accelerator > GPU T4 x2).

### Kernel times out before completion

Kaggle sessions have a 12-hour limit. Checkpoints are saved every 100k steps, so you lose at most ~15-30 minutes of work. Re-run the notebook to resume from the last checkpoint.

If the session is killed by SIGTERM (Kaggle's standard shutdown signal), an emergency checkpoint is saved automatically.

### Out of memory on Kaggle

Reduce batch size:
```python
config = Config(
    train=TrainConfig(num_envs=16),  # Reduce from 32
    # ... rest of config
)
```

### Checkpoint won't load

If you see errors loading a checkpoint saved on GPU to a CPU machine (or vice versa), the checkpoint format handles this automatically — all JAX arrays are stored as numpy. If you still see issues, verify the pickle protocol version matches your Python version.

### "Module not found" errors in Cell 2

The repo may need to be re-cloned. Delete the working directory and re-run Cell 2:
```python
!rm -rf /kaggle/working/emergence-lab
```

---

## Script Reference

| Script | Purpose | Key Options |
|--------|---------|-------------|
| `kaggle_setup.sh` | One-time setup | `--username NAME` |
| `kaggle_push.sh` | Push notebook to Kaggle | `--new` (first push) |
| `kaggle_run.sh` | Start kernel execution | `--push-first` |
| `kaggle_status.sh` | Check execution status | `--watch`, `--interval N` |
| `kaggle_download.sh` | Download results | `--output-dir DIR` |
