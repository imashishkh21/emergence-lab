#!/bin/bash
# Setup script for Emergence Lab

set -e

echo "🧬 Setting up Emergence Lab..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install package in editable mode with dev dependencies
echo "Installing dependencies..."
pip install -e ".[dev]"

# Verify JAX
echo ""
echo "Verifying JAX installation..."
python -c "import jax; print(f'JAX devices: {jax.devices()}')"

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To start training:"
echo "  python -m src.training.train --config configs/default.yaml"
