#!/usr/bin/env bash
# kaggle_download.sh — Download results and checkpoints from Kaggle
#
# Usage:
#   ./scripts/kaggle_download.sh [--output-dir DIR]
#
# Downloads the kernel output (checkpoints, logs) to a local directory.
# By default, saves to ./kaggle_output/

set -euo pipefail

# --- Config ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
NOTEBOOK_DIR="$PROJECT_DIR/notebooks"
METADATA_FILE="$NOTEBOOK_DIR/kernel-metadata.json"
DEFAULT_OUTPUT_DIR="$PROJECT_DIR/kaggle_output"

# --- Parse args ---
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: ./scripts/kaggle_download.sh [--output-dir DIR]"
            echo ""
            echo "Downloads emergence-lab training results from Kaggle."
            echo ""
            echo "Options:"
            echo "  --output-dir, -o DIR  Where to save files (default: ./kaggle_output/)"
            echo "  --help                Show this help message"
            echo ""
            echo "Downloaded files typically include:"
            echo "  checkpoints/step_*.pkl   - Training checkpoints"
            echo "  checkpoints/latest.pkl   - Symlink to latest checkpoint"
            echo ""
            echo "To resume locally from a downloaded checkpoint:"
            echo "  python -m src.training.train --train.resume-from kaggle_output/checkpoints/latest.pkl"
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

echo "=== Kaggle Download ==="
echo "  Kernel: $KERNEL_ID"
echo "  Output dir: $OUTPUT_DIR"
echo ""

# --- Check status first ---
STATUS_OUTPUT=$(kaggle kernels status "$KERNEL_ID" 2>&1) || true
echo "  Kernel status: $STATUS_OUTPUT"
echo ""

# --- Download ---
mkdir -p "$OUTPUT_DIR"
echo "Downloading kernel output..."
kaggle kernels output "$KERNEL_ID" --path "$OUTPUT_DIR"

echo ""

# --- Summary ---
echo "=== Download Complete ==="
echo ""

# List downloaded files
if [[ -d "$OUTPUT_DIR" ]]; then
    FILE_COUNT=$(find "$OUTPUT_DIR" -type f | wc -l | tr -d ' ')
    TOTAL_SIZE=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)
    echo "Downloaded $FILE_COUNT files ($TOTAL_SIZE) to $OUTPUT_DIR"
    echo ""

    # Show checkpoints specifically
    CHECKPOINTS=$(find "$OUTPUT_DIR" -name "*.pkl" -type f 2>/dev/null)
    if [[ -n "$CHECKPOINTS" ]]; then
        echo "Checkpoints found:"
        echo "$CHECKPOINTS" | while read -r f; do
            SIZE=$(du -h "$f" | cut -f1)
            echo "  $(basename "$f") ($SIZE)"
        done
        echo ""
        echo "Resume locally with:"
        echo "  python -m src.training.train --train.resume-from $OUTPUT_DIR/checkpoints/latest.pkl"
    else
        echo "No checkpoint files (.pkl) found in output."
        echo "The kernel may still be running, or the output path may differ."
    fi
else
    echo "No files downloaded. The kernel may still be running."
fi
