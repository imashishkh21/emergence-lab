#!/bin/bash
# Training launch script for Emergence Lab
#
# Usage:
#   ./scripts/train.sh                              # Run with defaults
#   ./scripts/train.sh --train.total-steps 100000   # Override total steps
#   ./scripts/train.sh --env.num-agents 16          # Override any config
#
# All arguments are passed directly to the training module (tyro CLI).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Activate virtual environment
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
else
    echo "ERROR: Virtual environment not found at $PROJECT_DIR/.venv"
    echo "Run ./scripts/setup.sh first."
    exit 1
fi

# JAX performance flags
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"
export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-cpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Run training with all passed arguments
python -m src.training.train "$@"
