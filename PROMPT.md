# Ralph Build Prompt

You are completing tasks from PRD.md autonomously.

## Instructions

1. Read `PRD.md` to understand the full project
2. Read `progress.txt` to see what's been done and learned
3. Read `AGENTS.md` for operational commands
4. Find the NEXT unchecked `[ ]` task in PRD.md (scan top to bottom)
5. Complete that ONE task fully
6. Run validation (tests, typecheck, lint) as specified in verification
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

## Context for Phase 5

We're building **rigorous proof of emergence** with:
1. **Information-theoretic metrics** — PID synergy, O-information, causal emergence
2. **Baselines** — IPPO (no field), ACO-Fixed/Hybrid (hardcoded rules), MAPPO (centralized critic)
3. **Statistical significance** — N=20 seeds, IQM + bootstrap CI via rliable
4. **Experiments** — Hidden resources (requires cooperation), superlinear scaling, 6-condition ablation
5. **Publication figures** — 300 DPI, PDF + PNG, colorblind-safe

### Key Decisions (FOLLOW THESE)
- **`dit` for PID** — Gold standard discrete information theory
- **`hoi` for O-information** — JAX-native, O(n) scaling
- **`rliable` for statistics** — Google standard (Agarwal 2021)
- **K=2 quantile bins + Jeffreys smoothing (alpha=0.5)** for discretization
- **Phase 5 deps in `[project.optional-dependencies]` under `phase5` group**
- **`@pytest.mark.skipif` guards** on tests that need Phase 5 deps
- **Hidden food disabled by default** — backward compatible
- **MAPPO: custom code (~200 lines), NOT JaxMARL wrapper**
- **ACO params: alpha=1.0, beta=2.0, rho=0.5, Q=1.0**
- **Standardized result dict** from all baselines: `{"total_reward", "food_collected", "final_population", "per_agent_rewards"}`

### What Already Exists
- `src/analysis/information.py` — Transfer Entropy + TransferEntropyTracker
- `src/analysis/specialization.py` — Weight divergence, clustering, SpecializationTracker
- `src/analysis/emergence.py` — EmergenceTracker, PhaseTransitionDetector
- `src/analysis/ablation.py` — 3 field conditions + 2x2 evolution + specialization ablation
- `src/analysis/field_metrics.py` — field_entropy, field_structure, field_food_mi
- `src/agents/network.py` — ActorCritic (shared) + AgentSpecificActorCritic (per-agent heads)
- `src/environment/env.py` — reset() + step() with 10-stage pipeline
- `src/environment/state.py` — EnvState flax.struct.dataclass
- `src/configs.py` — Config with 10 nested dataclasses
- `src/training/ppo.py` — ppo_loss() with alive masking

### EnvState Modification Warning (US-010)
When adding hidden food fields to EnvState, you MUST update every `EnvState(...)` constructor call:
- `state.py`: create_env_state()
- `env.py`: reset(), step() (the final EnvState construction)
- `ablation.py`: _replace_field(), _reset_energy(), _replace_agent_params()

## Backpressure (After EVERY Task)

```bash
python -m mypy src/ --ignore-missing-imports
ruff check src/
pytest tests/ -v
```

## Completion Signal

When ALL tasks have `[x]`, add to progress.txt:
```
COMPLETE: All Phase 5 tasks finished.
```

Then STOP.
