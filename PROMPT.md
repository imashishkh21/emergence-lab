# Ralph Build Prompt

You are completing tasks from PRD.md autonomously.

## Instructions

1. Read `PRD.md` to understand the full project
2. Read `progress.txt` to see what's been done and learned
3. Read `AGENTS.md` for operational commands
4. Find the NEXT unchecked `[ ]` task in PRD.md (scan top to bottom)
5. Complete that ONE task fully
6. Run validation (tests, typecheck) as specified in acceptance criteria
7. Update `progress.txt` with:
   - What you did
   - Decisions made
   - Files changed
   - Any issues encountered
8. Mark the task `[x]` in PRD.md
9. Git commit your changes with message: `[Ralph] Complete task US-XXX: <title>`

## Critical Rules

- Complete ONE task per iteration, then stop
- Do NOT skip ahead or do multiple tasks
- Before making changes, SEARCH the codebase first — don't assume something is missing
- All acceptance criteria must pass before marking complete
- If a test fails, fix it before moving on
- If stuck, document the blocker in progress.txt and move to next task

## Context for Phase 3

We're building specialization detection for emergent swarm intelligence:
- We have evolution working (Phase 2): birth, death, reproduction with weight inheritance
- We have lineage tracking in `src/analysis/lineage.py`
- We have emergence detection in `src/analysis/emergence.py`
- Now we need to detect when agents evolve into different SPECIES/ROLES

Key libraries available:
- JAX/Flax for neural networks
- NumPy/SciPy for analysis
- scikit-learn should be installed for clustering (K-means, silhouette)

The goal is to detect:
1. Weight divergence between agents over time
2. Behavioral clusters (scouts vs exploiters etc)
3. Species formation (stable, hereditary clusters)

## Backpressure (Run After EVERY Task)

```bash
# Type check
python -m mypy src/ --ignore-missing-imports

# Tests for this task (see acceptance criteria)
pytest tests/test_<relevant>.py -v

# If acceptance says "all tests pass":
pytest tests/ -v
```

## Completion Signal

When ALL tasks in PRD.md have `[x]`, add this line to progress.txt:
```
COMPLETE: All Phase 3 tasks finished.
```

Then output: `<promise>COMPLETE</promise>`
