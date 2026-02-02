#!/usr/bin/env bash
# kaggle_run.sh — Start kernel execution on Kaggle
#
# Usage:
#   ./scripts/kaggle_run.sh [--push-first]
#
# Options:
#   --push-first  Push the latest notebook before running
#   --help        Show this help message
#
# This triggers the kernel to execute on Kaggle's servers with GPU.
# The kernel runs in the background — use kaggle_status.sh to monitor.

set -euo pipefail

# --- Config ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
NOTEBOOK_DIR="$PROJECT_DIR/notebooks"
METADATA_FILE="$NOTEBOOK_DIR/kernel-metadata.json"

# --- Parse args ---
PUSH_FIRST=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --push-first)
            PUSH_FIRST=true
            shift
            ;;
        --help|-h)
            echo "Usage: ./scripts/kaggle_run.sh [--push-first]"
            echo ""
            echo "Starts the emergence-lab training kernel on Kaggle."
            echo ""
            echo "Options:"
            echo "  --push-first  Push latest notebook before running"
            echo "  --help        Show this help message"
            echo ""
            echo "The kernel runs in the background on Kaggle's GPU servers."
            echo "Use ./scripts/kaggle_status.sh to monitor progress."
            echo "Use ./scripts/kaggle_download.sh to retrieve checkpoints."
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

KERNEL_ID=$(python3 -c "import json; print(json.load(open('$METADATA_FILE'))['id'])")

echo "=== Kaggle Run ==="
echo "  Kernel: $KERNEL_ID"
echo ""

# --- Optionally push first ---
if $PUSH_FIRST; then
    echo "Pushing latest notebook first..."
    "$SCRIPT_DIR/kaggle_push.sh"
    echo ""
fi

# --- Trigger execution ---
# Kaggle CLI doesn't have a direct "run" command separate from push.
# Pushing a kernel with `kaggle kernels push` automatically queues it for execution.
# If we already pushed, it's already running. Otherwise, push now to trigger.
if ! $PUSH_FIRST; then
    echo "Pushing notebook to trigger execution..."
    kaggle kernels push -p "$NOTEBOOK_DIR"
fi

echo ""
echo "=== Execution Queued ==="
echo ""
echo "The kernel is now queued for execution on Kaggle."
echo "Kaggle will allocate a GPU and start running the notebook."
echo ""
echo "Monitor progress:"
echo "  ./scripts/kaggle_status.sh"
echo "  ./scripts/kaggle_status.sh --watch"
echo ""
echo "View in browser:"
echo "  https://www.kaggle.com/code/$KERNEL_ID"
