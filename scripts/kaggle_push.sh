#!/usr/bin/env bash
# kaggle_push.sh — Push the training notebook to Kaggle
#
# Usage:
#   ./scripts/kaggle_push.sh [--new]
#
# Options:
#   --new   Create a new kernel (first push). Omit for updates.
#   --help  Show this help message
#
# Prerequisites:
#   - Run ./scripts/kaggle_setup.sh first
#   - kernel-metadata.json must exist in notebooks/

set -euo pipefail

# --- Config ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
NOTEBOOK_DIR="$PROJECT_DIR/notebooks"
METADATA_FILE="$NOTEBOOK_DIR/kernel-metadata.json"

# --- Parse args ---
NEW_KERNEL=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --new)
            NEW_KERNEL=true
            shift
            ;;
        --help|-h)
            echo "Usage: ./scripts/kaggle_push.sh [--new]"
            echo ""
            echo "Pushes the emergence-lab training notebook to Kaggle."
            echo ""
            echo "Options:"
            echo "  --new   Create a new kernel (first push only)"
            echo "  --help  Show this help message"
            echo ""
            echo "The kernel-metadata.json in notebooks/ controls the kernel ID,"
            echo "title, GPU settings, and privacy. Run kaggle_setup.sh first."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage."
            exit 1
            ;;
    esac
done

# --- Validate ---
if [[ ! -f "$METADATA_FILE" ]]; then
    echo "ERROR: kernel-metadata.json not found at $METADATA_FILE"
    echo "Run ./scripts/kaggle_setup.sh first."
    exit 1
fi

if ! command -v kaggle &>/dev/null; then
    echo "ERROR: kaggle CLI not found. Run ./scripts/kaggle_setup.sh first."
    exit 1
fi

# --- Read kernel info ---
KERNEL_ID=$(python3 -c "import json; print(json.load(open('$METADATA_FILE'))['id'])")
echo "=== Kaggle Push ==="
echo "  Kernel: $KERNEL_ID"
echo "  Notebook: $NOTEBOOK_DIR/kaggle_training.ipynb"
echo ""

# --- Push ---
echo "Pushing notebook to Kaggle..."
cd "$NOTEBOOK_DIR"

if $NEW_KERNEL; then
    echo "  (Creating new kernel)"
fi

kaggle kernels push -p "$NOTEBOOK_DIR"

echo ""
echo "=== Push Complete ==="
echo ""
echo "View at: https://www.kaggle.com/code/$KERNEL_ID"
echo ""
echo "Next steps:"
echo "  - Start execution: ./scripts/kaggle_run.sh"
echo "  - Or run directly from the Kaggle web UI"
