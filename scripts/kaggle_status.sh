#!/usr/bin/env bash
# kaggle_status.sh — Check Kaggle kernel execution status
#
# Usage:
#   ./scripts/kaggle_status.sh
#   ./scripts/kaggle_status.sh --watch          # Poll every 60s
#   ./scripts/kaggle_status.sh --watch --interval 30  # Custom interval
#
# Shows: execution status, runtime, and output file list.

set -euo pipefail

# --- Config ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
NOTEBOOK_DIR="$PROJECT_DIR/notebooks"
METADATA_FILE="$NOTEBOOK_DIR/kernel-metadata.json"

# --- Parse args ---
WATCH=false
INTERVAL=60
while [[ $# -gt 0 ]]; do
    case "$1" in
        --watch|-w)
            WATCH=true
            shift
            ;;
        --interval|-i)
            INTERVAL="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: ./scripts/kaggle_status.sh [--watch] [--interval SECONDS]"
            echo ""
            echo "Checks the status of the emergence-lab Kaggle kernel."
            echo ""
            echo "Options:"
            echo "  --watch, -w       Poll status repeatedly"
            echo "  --interval, -i N  Seconds between polls (default: 60)"
            echo "  --help            Show this help message"
            echo ""
            echo "Status values:"
            echo "  queued     - Waiting for GPU allocation"
            echo "  running    - Currently executing"
            echo "  complete   - Finished successfully"
            echo "  error      - Failed with error"
            echo "  cancelAcknowledged - Cancelled by user"
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

# --- Status function ---
check_status() {
    echo "=== Kaggle Status ==="
    echo "  Kernel: $KERNEL_ID"
    echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # Get kernel status
    STATUS_OUTPUT=$(kaggle kernels status "$KERNEL_ID" 2>&1) || true
    echo "  Status: $STATUS_OUTPUT"
    echo ""

    # List output files if available
    echo "  Output files:"
    kaggle kernels output "$KERNEL_ID" --path /dev/null 2>&1 | head -20 || echo "  (no output yet or kernel still running)"
    echo ""
    echo "  View: https://www.kaggle.com/code/$KERNEL_ID"
    echo "---"
}

# --- Execute ---
if $WATCH; then
    echo "Watching kernel status (every ${INTERVAL}s, Ctrl+C to stop)..."
    echo ""
    while true; do
        check_status
        echo ""
        sleep "$INTERVAL"
    done
else
    check_status
fi
