# Ralph Build Prompt

You are completing tasks from PRD.md autonomously.

## Instructions

1. Read `PRD.md` to understand the full project
2. Read `progress.txt` to see what's been done and learned
3. Read `AGENTS.md` for operational commands
4. Find the NEXT unchecked `[ ]` task in PRD.md (scan top to bottom)
5. Complete that ONE task fully
6. Run validation (tests, typecheck) as specified in verification
7. Update `progress.txt` with:
   - What you did
   - Decisions made
   - Files changed
   - Any issues encountered
8. Mark the task `[x]` in PRD.md
9. Git commit with message: `[Ralph] Complete US-XXX: <title>`

## Critical Rules

- Complete ONE task per iteration, then STOP
- Do NOT skip ahead or do multiple tasks
- Before making changes, SEARCH the codebase first
- All verification steps must pass before marking complete
- If stuck, document blocker in progress.txt and move on

## Context for Phase 4B

We're building **Kaggle Infrastructure** to enable:
1. **Real data in dashboard** — Wire `train.py` → WebSocket server
2. **Reliable checkpointing** — Save FULL state (optimizer, PRNG, step, trackers)
3. **Autonomous training** — Kaggle CLI so Titan can run 24/7

### Key Decisions (FOLLOW THESE)
- **Enhanced pickle, NOT Orbax** — Orbax has cross-platform bugs
- **Kaggle only** — Don't mention Colab
- **Save every 100k steps** — Keep last 5 checkpoints
- **Signal handlers** — Save on SIGTERM/SIGINT

### What Already Exists
- `src/server/streaming.py` — `TrainingBridge` with `publish_frame()`
- `src/analysis/emergence.py` — `EmergenceTracker` (needs serialization)
- `src/analysis/specialization.py` — `SpecializationTracker` (needs serialization)
- `config.log.save_interval` — Exists but unused
- `config.log.checkpoint_dir` — Exists

## Backpressure (After EVERY Task)

```bash
python -m mypy src/ --ignore-missing-imports
pytest tests/ -v
```

## Completion Signal

When ALL tasks have `[x]`, add to progress.txt:
```
COMPLETE: All Phase 4B tasks finished.
```

Then STOP.
