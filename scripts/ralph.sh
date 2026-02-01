#!/bin/bash
# Ralph Loop - Autonomous coding with Claude Code
# Usage: ./scripts/ralph.sh [iterations]
#
# Based on Geoffrey Huntley's Ralph methodology
# https://www.yourfuturebrain.blog/p/sitting-in-a-loop

set -e

ITERATIONS=${1:-50}
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_DIR"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "🤖 Starting Ralph Loop"
echo "   Project: $PROJECT_DIR"
echo "   Iterations: $ITERATIONS"
echo "   PRD: PRD.md"
echo "   Prompt: PROMPT.md"
echo ""

# Check if Claude Code is installed
if ! command -v claude &> /dev/null; then
    echo "❌ Claude Code CLI not found. Install with:"
    echo "   npm install -g @anthropic-ai/claude-code"
    exit 1
fi

# Ensure PROMPT.md exists
if [ ! -f "PROMPT.md" ]; then
    echo "❌ PROMPT.md not found. Create it first."
    exit 1
fi

# Run Ralph loop
for i in $(seq 1 $ITERATIONS); do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔄 Iteration $i/$ITERATIONS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Run Claude Code with the prompt file
    cat PROMPT.md | claude --print
    
    # Check for completion signal
    if grep -q "COMPLETE:" progress.txt 2>/dev/null; then
        echo ""
        echo "✅ All tasks completed!"
        break
    fi
    
    echo ""
    sleep 2
done

echo ""
echo "🏁 Ralph Loop finished after $i iterations"
echo "   Check progress.txt for final status"
echo "   Check PRD.md for task completion"
