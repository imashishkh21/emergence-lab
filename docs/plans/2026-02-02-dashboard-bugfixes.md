# Dashboard Bugfixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix three bugs found during dashboard testing — frame decode errors (high), glossary not loading (medium), and missing accessibility labels (low).

**Architecture:** Three independent bug fixes targeting the Svelte 5 dashboard frontend. No backend changes needed. Bug 1 fixes a buffer handling issue in `unpackArray()` that silently drops array data on every frame. Bug 2 moves `glossary.json` into Vite's default `public/` directory so it gets served correctly. Bug 3 adds `aria-label` attributes to all interactive buttons.

**Tech Stack:** Svelte 5, Vite 6, JavaScript (no TypeScript), msgpack-lite

---

### Task 1: Fix frame decode error in `unpackArray()`

**Files:**
- Modify: `dashboard/src/stores/training.svelte.js:17-44`

**Context:**
The server packs numpy arrays as `{ shape: [32, 2], dtype: "<f4", data: <raw bytes> }` via msgpack. On the client, `msgpack-lite` decodes the raw bytes into a `Uint8Array`. The current code does `packed.data.buffer` to get the underlying `ArrayBuffer`, then passes it directly to a typed array constructor like `new Float32Array(buffer, offset, length)`.

The problem: msgpack-lite may decode multiple fields into a single shared `ArrayBuffer` with different `byteOffset` values. The typed array constructor `new Float32Array(buffer, offset, length)` requires that:
1. `offset` is aligned to the typed array's `BYTES_PER_ELEMENT` (4 for Float32Array/Int32Array)
2. `buffer.byteLength - offset >= length * BYTES_PER_ELEMENT`

When msgpack uses a shared buffer, both conditions can fail — the offset may not be 4-byte aligned, and the buffer may not be large enough from that offset. This causes `RangeError: Invalid typed array length` on every frame (~30/sec).

**Fix:** Copy the raw bytes into a fresh, correctly-sized `ArrayBuffer` before creating the typed array view. This guarantees alignment and correct sizing.

**Step 1: Implement the fix**

Replace the `unpackArray` function at lines 17-44 of `dashboard/src/stores/training.svelte.js` with:

```javascript
function unpackArray(packed) {
  if (!packed || !packed.data || !packed.shape) {
    return { data: new Float32Array(0), shape: [0] };
  }

  const dtype = packed.dtype || "<f4";
  const shape = packed.shape;
  const totalElements = shape.reduce((a, b) => a * b, 1);

  // Copy raw bytes into a fresh ArrayBuffer to guarantee correct size and alignment.
  // msgpack-lite may decode into a shared buffer with arbitrary offsets that break
  // typed array constructors (e.g., Float32Array needs 4-byte alignment).
  const raw = packed.data instanceof Uint8Array
    ? packed.data
    : new Uint8Array(packed.data);
  const buffer = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);

  let data;
  if (dtype === "<f4" || dtype === "float32") {
    data = new Float32Array(buffer, 0, totalElements);
  } else if (dtype === "<i4" || dtype === "int32") {
    data = new Int32Array(buffer, 0, totalElements);
  } else if (dtype === "<i1" || dtype === "int8") {
    data = new Int8Array(buffer, 0, totalElements);
  } else if (dtype === "|u1" || dtype === "uint8" || dtype === "<u1") {
    data = new Uint8Array(buffer, 0, totalElements);
  } else {
    data = new Float32Array(buffer, 0, totalElements);
  }

  return { data, shape };
}
```

**Step 2: Manual verification in browser**

Run: Start the dashboard with `cd dashboard && npm run dev` and server with `python3 -m src.server.main --mock`

Expected:
- Open Chrome DevTools console at http://localhost:5173
- Zero `RangeError` or `Frame decode error` messages in the console
- Lineage data (agent_ids, parent_ids, birth_steps) should now populate in the LineagePanel

**Step 3: Commit**

```bash
git add dashboard/src/stores/training.svelte.js
git commit -m "fix: copy msgpack buffer in unpackArray to fix typed array RangeError

The msgpack-lite decoder uses shared ArrayBuffers with arbitrary byte
offsets. Typed array constructors (Float32Array, Int32Array) require
aligned offsets and exact buffer sizes. Copy to a fresh buffer first."
```

---

### Task 2: Fix glossary not loading

**Files:**
- Move: `dashboard/static/glossary.json` → `dashboard/public/glossary.json`
- Delete: `dashboard/static/` (will be empty after move)

**Context:**
Vite's default `publicDir` is `public/`. The glossary file is in `static/`, so `fetch("/glossary.json")` gets Vite's SPA fallback (index.html). The `.json()` parse fails silently and the glossary shows "No glossary terms available."

We move the file to `public/` rather than changing `publicDir` in vite.config.js, because `public/` already has `favicon.svg` and changing `publicDir` to `static/` would break the favicon.

**Step 1: Move the file**

```bash
mv dashboard/static/glossary.json dashboard/public/glossary.json
rmdir dashboard/static
```

**Step 2: Manual verification in browser**

Run: Refresh http://localhost:5173, click the glossary button (book icon).

Expected:
- Glossary panel opens with 19 terms visible (Agent, Neural Network, Weights, Training, Evolution, etc.)
- Search box filters terms correctly
- Network tab shows `GET /glossary.json` returning 200 with JSON content (not HTML)

**Step 3: Commit**

```bash
git add dashboard/public/glossary.json
git rm dashboard/static/glossary.json
git commit -m "fix: move glossary.json to public/ so Vite serves it correctly

Vite's default publicDir is public/, not static/. The fetch was getting
the SPA fallback HTML, causing silent parse failure."
```

---

### Task 3: Add aria-labels to buttons

**Files:**
- Modify: `dashboard/src/lib/ControlPanel.svelte`
- Modify: `dashboard/src/lib/AgentCanvas.svelte`
- Modify: `dashboard/src/lib/HelpSystem.svelte`
- Modify: `dashboard/src/lib/ReplayControls.svelte`
- Modify: `dashboard/src/lib/LineagePanel.svelte`

**Context:**
25+ buttons across 5 components have no `aria-label`. For buttons that have visible text (like "Resume", "Back", "Skip"), `aria-label` is technically optional since the text itself is the accessible name. But icon-only buttons and emoji-only buttons definitely need labels. We'll add labels to icon/emoji-only buttons that lack visible descriptive text.

Focus on buttons where the visible content is an icon, emoji, or ambiguous:
- `?` (help tour trigger)
- book emoji (glossary trigger)
- `↻` (refresh sessions)
- `&times;` (close buttons)
- Speed buttons showing just `0.5x`, `1x`, etc.

Buttons with clear visible text like "Resume", "Pause", "Back", "Skip", "Next", "Got it!", "Close Replay", "Cluster", "Fitness", "Show Trails", "Hide Trails" are already accessible — the visible text IS the accessible name. We skip those.

**Step 1: Add aria-labels to icon-only buttons**

**ControlPanel.svelte** — Speed buttons in the `{#each}` loop:
```svelte
<button
  class="speed-btn"
  class:active={store.speedMultiplier === speed}
  onclick={() => store.setSpeed(speed)}
  disabled={!store.connected}
  aria-label="Set speed to {speed}x"
>
```

**HelpSystem.svelte** — Tour trigger button:
```svelte
<button
  class="wh-tour-btn"
  onclick={restartTour}
  title="Restart the guided tour"
  aria-label="Restart guided tour"
>
```

**HelpSystem.svelte** — Glossary trigger button:
```svelte
<button
  class="wh-glossary-btn"
  onclick={() => showGlossary?.()}
  title="Open glossary — look up any term"
  aria-label="Open glossary"
>
```

**ReplayControls.svelte** — Refresh button:
```svelte
<button class="btn-refresh" onclick={fetchSessions} title="Refresh session list" aria-label="Refresh session list">
```

**ReplayControls.svelte** — Speed buttons in the `{#each}` loop:
```svelte
<button
  class="speed-btn"
  class:active={store.replayStatus.speed === speed}
  onclick={() => store.sendReplayCommand({ type: "set_speed", value: speed })}
  aria-label="Set replay speed to {speed}x"
>
```

**ReplayControls.svelte** — Bookmark buttons:
```svelte
<button
  class="bookmark-item"
  onclick={() => store.sendReplayCommand({ type: "seek_bookmark", index: idx })}
  title="Step {bookmark.step}"
  aria-label="Jump to bookmark: {bookmark.label} at step {bookmark.step}"
>
```

**Step 2: Verify no Svelte accessibility warnings**

Run: `cd dashboard && npm run dev` — Svelte 5's compiler emits `a11y` warnings for buttons without accessible names. After this fix, those warnings should be gone.

Expected: No `a11y` warnings in the terminal output related to buttons.

**Step 3: Commit**

```bash
git add dashboard/src/lib/ControlPanel.svelte dashboard/src/lib/HelpSystem.svelte dashboard/src/lib/ReplayControls.svelte
git commit -m "fix: add aria-labels to icon-only buttons for accessibility"
```

---

## Verification

After all three tasks, verify together:

1. Open http://localhost:5173 with mock server running
2. Open Chrome DevTools console — should have zero `RangeError` or `Frame decode error` messages
3. Click glossary button — should show all 19 terms
4. Check terminal for Svelte — no `a11y` accessibility warnings
5. Lineage panel should show agent data (was broken because int32 arrays failed to decode)
