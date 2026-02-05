# Fix 3 Pre-existing Errors Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 3 failing tests (missing glossary.json), 2 ruff lint errors (import order + unused variable) so all verification commands pass clean.

**Architecture:** Minimal targeted fixes — create one missing directory+file, auto-fix one import ordering, delete one unused variable line. No behavioral changes.

**Tech Stack:** Python, ruff, mypy, pytest

---

### Task 1: Create `dashboard/static/glossary.json`

The 3 failing tests in `tests/test_dashboard_integration.py::TestHelpSystemData` expect a glossary file at `dashboard/static/glossary.json`. The file already exists at `dashboard/public/glossary.json` with correct content (19 terms, all with `term`+`simple` keys, many with `analogy`, includes "agent", "emergence", "specialization", "evolution"). The tests need it at the `static/` path.

**Files:**
- Create: `dashboard/static/glossary.json` (copy content from `dashboard/public/glossary.json`)

**Step 1: Create `dashboard/static/` directory and copy glossary**

```bash
mkdir -p dashboard/static
cp dashboard/public/glossary.json dashboard/static/glossary.json
```

**Step 2: Verify the 3 glossary tests pass**

Run: `.venv/bin/pytest tests/test_dashboard_integration.py::TestHelpSystemData -v`
Expected: 3 PASSED (`test_glossary_json_exists`, `test_glossary_json_valid`, `test_glossary_terms_have_analogies`)

---

### Task 2: Fix ruff import ordering in `src/training/train.py`

Ruff reports `I001` — import block is un-sorted. This is auto-fixable.

**Files:**
- Modify: `src/training/train.py` (lines 3-28, import block)

**Step 1: Auto-fix import ordering**

```bash
.venv/bin/ruff check src/training/train.py --fix
```

This will reorder the import block. The sorted order should keep stdlib → third-party → local grouping but fix the internal sort.

**Step 2: Verify the I001 error is gone**

Run: `.venv/bin/ruff check src/training/train.py`
Expected: Only the F841 error remains (handled in Task 3)

---

### Task 3: Remove unused `cycle_length` variable in `src/training/train.py`

Line 858 assigns `cycle_length` but it's never used. Ruff reports `F841`.

**Files:**
- Modify: `src/training/train.py:858` — delete the line and its comment on line 857

**Step 1: Remove the unused variable**

Delete these two lines (857-858):
```python
        # Cycle length in env steps
        cycle_length = fe_cfg.gradient_steps + fe_cfg.evolve_steps
```

**Step 2: Verify ruff is clean**

Run: `.venv/bin/ruff check src/`
Expected: 0 errors

---

### Task 4: Run all 3 verification commands

**Step 1: mypy**

Run: `.venv/bin/python -m mypy src/ --ignore-missing-imports`
Expected: `Success: no issues found in 39 source files`

**Step 2: ruff**

Run: `.venv/bin/ruff check src/`
Expected: `All checks passed!` (0 errors)

**Step 3: pytest**

Run: `.venv/bin/pytest tests/ -v --timeout=60`
Expected: 0 failures

---

### Task 5: Commit

```bash
git add dashboard/static/glossary.json src/training/train.py
git commit -m "fix: resolve 3 pre-Phase-4B errors (glossary, ruff lint)

- Create dashboard/static/glossary.json for dashboard integration tests
- Fix import ordering in train.py (ruff I001)
- Remove unused cycle_length variable in train.py (ruff F841)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```
