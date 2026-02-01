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
9. Git commit your changes with message: `[Ralph] Complete task X.Y: <title>`

## Critical Rules

- Complete ONE task per iteration, then stop
- Do NOT skip ahead or do multiple tasks
- Before making changes, SEARCH the codebase first — don't assume something is missing
- All acceptance criteria must pass before marking complete
- If a test fails, fix it before moving on
- If stuck, document the blocker in progress.txt and move to next task

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
COMPLETE: All Phase 1 tasks finished.
```

Then output: `<promise>COMPLETE</promise>`
