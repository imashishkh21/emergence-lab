#!/usr/bin/env bash
# kaggle_setup.sh — Setup Kaggle CLI and kernel metadata
#
# Prerequisites:
#   - Python 3.8+ with pip
#   - Kaggle account with API token
#
# Usage:
#   ./scripts/kaggle_setup.sh [--username YOUR_USERNAME]
#
# This script:
#   1. Installs the kaggle CLI if not present
#   2. Verifies API credentials (~/.kaggle/kaggle.json)
#   3. Creates kernel-metadata.json for the training notebook

set -euo pipefail

# --- Config ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
NOTEBOOK_DIR="$PROJECT_DIR/notebooks"
NOTEBOOK_FILE="$NOTEBOOK_DIR/kaggle_training.ipynb"
METADATA_FILE="$NOTEBOOK_DIR/kernel-metadata.json"

# --- Parse args ---
USERNAME=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --username)
            USERNAME="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: ./scripts/kaggle_setup.sh [--username YOUR_KAGGLE_USERNAME]"
            echo ""
            echo "Sets up the Kaggle CLI and creates kernel metadata for the"
            echo "emergence-lab training notebook."
            echo ""
            echo "Options:"
            echo "  --username  Your Kaggle username (auto-detected if kaggle CLI is configured)"
            echo "  --help      Show this help message"
            echo ""
            echo "Prerequisites:"
            echo "  1. Create a Kaggle account at https://www.kaggle.com"
            echo "  2. Go to Settings > API > Create New Token"
            echo "  3. Save kaggle.json to ~/.kaggle/kaggle.json"
            echo "  4. chmod 600 ~/.kaggle/kaggle.json"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage."
            exit 1
            ;;
    esac
done

echo "=== Kaggle Setup ==="
echo ""

# --- Step 1: Install kaggle CLI ---
echo "Step 1: Checking kaggle CLI..."
if command -v kaggle &>/dev/null; then
    echo "  kaggle CLI found: $(kaggle --version 2>/dev/null || echo 'installed')"
else
    echo "  Installing kaggle CLI..."
    pip install --quiet kaggle
    echo "  Installed: $(kaggle --version 2>/dev/null || echo 'installed')"
fi

# --- Step 2: Verify credentials ---
echo ""
echo "Step 2: Verifying API credentials..."
KAGGLE_JSON="$HOME/.kaggle/kaggle.json"
if [[ -f "$KAGGLE_JSON" ]]; then
    echo "  Found: $KAGGLE_JSON"

    # Check permissions
    PERMS=$(stat -f "%Lp" "$KAGGLE_JSON" 2>/dev/null || stat -c "%a" "$KAGGLE_JSON" 2>/dev/null)
    if [[ "$PERMS" != "600" ]]; then
        echo "  Fixing permissions (should be 600, got $PERMS)..."
        chmod 600 "$KAGGLE_JSON"
    fi

    # Auto-detect username if not provided
    if [[ -z "$USERNAME" ]]; then
        USERNAME=$(python3 -c "import json; print(json.load(open('$KAGGLE_JSON'))['username'])" 2>/dev/null || true)
    fi

    if [[ -z "$USERNAME" ]]; then
        echo "  WARNING: Could not detect username from kaggle.json"
        echo "  Re-run with: ./scripts/kaggle_setup.sh --username YOUR_USERNAME"
        exit 1
    fi

    echo "  Username: $USERNAME"

    # Verify by listing kernels (quick API test)
    if kaggle kernels list --mine --page-size 1 &>/dev/null; then
        echo "  API connection verified."
    else
        echo "  WARNING: API test failed. Check your token at:"
        echo "  https://www.kaggle.com/settings > API > Create New Token"
    fi
else
    echo "  ERROR: $KAGGLE_JSON not found!"
    echo ""
    echo "  To set up Kaggle API credentials:"
    echo "  1. Go to https://www.kaggle.com/settings"
    echo "  2. Scroll to 'API' section"
    echo "  3. Click 'Create New Token'"
    echo "  4. Save the downloaded kaggle.json to ~/.kaggle/"
    echo "  5. Run: chmod 600 ~/.kaggle/kaggle.json"
    echo "  6. Re-run this script"
    exit 1
fi

# --- Step 3: Create kernel metadata ---
echo ""
echo "Step 3: Creating kernel metadata..."

if [[ ! -f "$NOTEBOOK_FILE" ]]; then
    echo "  ERROR: Notebook not found at $NOTEBOOK_FILE"
    echo "  Run from the project root directory."
    exit 1
fi

KERNEL_SLUG="emergence-lab-training"

cat > "$METADATA_FILE" << EOF
{
  "id": "${USERNAME}/${KERNEL_SLUG}",
  "title": "Emergence Lab Training",
  "code_file": "kaggle_training.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "keywords": ["emergence", "multi-agent", "stigmergy", "jax"],
  "dataset_sources": [],
  "kernel_sources": [],
  "competition_sources": []
}
EOF

echo "  Created: $METADATA_FILE"
echo "  Kernel ID: ${USERNAME}/${KERNEL_SLUG}"

# --- Done ---
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Push notebook:   ./scripts/kaggle_push.sh"
echo "  2. Start training:  ./scripts/kaggle_run.sh"
echo "  3. Check status:    ./scripts/kaggle_status.sh"
echo "  4. Download results: ./scripts/kaggle_download.sh"
