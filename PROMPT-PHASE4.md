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

## Context for Phase 4

We're building the **Research Microscope** — a live visualization dashboard to:
1. **SEE** what agents are actually doing (Pixi.js canvas)
2. **FIX** gradient homogenization (agent-specific heads, freeze-evolve)
3. **MEASURE** emergence (transfer entropy, division of labor, phase transitions)

### Key Technical Decisions
- **Backend:** FastAPI + WebSocket + MessagePack (binary streaming)
- **Frontend:** Svelte 5 + Pixi.js v8 + Plotly.js
- **Architecture Fix:** Agent-specific heads on shared backbone
- **Training Mode:** Freeze-Evolve cycles (alternate gradient training with pure evolution)

### What We Already Have (Phases 1-3)
- Grid world environment with shared field (stigmergy)
- Evolution: energy, death, reproduction, mutation, lineage tracking
- Specialization detection: clustering, species detection, analysis tools
- Problem identified: PPO shared gradients homogenize weights

### Key Files
- `src/agents/network.py` — Neural network definitions
- `src/training/train.py` — Training loop (need to modify for freeze-evolve)
- `src/analysis/specialization.py` — Specialization metrics (add to this)
- `src/environment/render.py` — Existing render logic (reference for dashboard)

### Dependencies to Add
**Python** (add to pyproject.toml):
- fastapi, uvicorn, websockets, msgpack

**Dashboard** (new package.json):
- svelte@5, pixi.js@8, plotly.js-dist-min, msgpack-lite

## Backpressure (Run After EVERY Task)

```bash
# Type check
python -m mypy src/ --ignore-missing-imports

# Tests for this task (see acceptance criteria)
pytest tests/test_<relevant>.py -v

# If acceptance says "all tests pass":
pytest tests/ -v
```

## Dashboard Development

When working on dashboard (US-006 through US-018):
```bash
cd dashboard
npm install
npm run dev
# Dashboard runs at http://localhost:5173
```

To test with training:
```bash
# Terminal 1: Start training with server
python -m src.server.main

# Terminal 2: Start dashboard
cd dashboard && npm run dev
```

## Important: Help System

Every UI element needs to be understandable by a non-technical person. When building dashboard components:
- Add tooltips to every metric
- Include helper text explaining what things mean
- Reference the Glossary section in PRD.md for explanations
- Think "would a 5-year-old understand what this shows?"

## Completion Signal

When ALL tasks in PRD.md have `[x]`, add this line to progress.txt:
```
COMPLETE: All Phase 4 tasks finished.
```

Then output: `<promise>COMPLETE</promise>`
