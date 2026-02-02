<script>
  /**
   * Replay controls for playing back recorded training sessions.
   *
   * Provides:
   * - Session selector (load a recorded session)
   * - Play/Pause controls
   * - Timeline scrubber (like a video progress bar)
   * - Speed controls (0.5x, 1x, 2x, 4x)
   * - Bookmark navigation
   */
  import Tooltip from "./Tooltip.svelte";

  let { store } = $props();

  const SPEED_OPTIONS = [0.5, 1, 2, 4];

  // Replay state
  let sessions = $state([]);
  let loadingSession = $state(false);
  let replayError = $state(null);

  // Fetch available sessions from the server
  async function fetchSessions() {
    try {
      const response = await fetch("http://localhost:8765/sessions");
      if (response.ok) {
        sessions = await response.json();
      }
    } catch (e) {
      // Server not available — expected when not connected
    }
  }

  // Load a session for replay
  function loadSession(sessionPath) {
    if (!store.replayConnected) return;
    loadingSession = true;
    replayError = null;
    store.sendReplayCommand({ type: "load", session: sessionPath });
    // Status update will come back via WebSocket
    setTimeout(() => { loadingSession = false; }, 2000);
  }

  // Scrubber: seek to position on input
  function handleScrub(e) {
    const position = parseInt(e.target.value);
    store.sendReplayCommand({ type: "seek", position });
  }

  // Format step number for display
  function formatStep(step) {
    if (step >= 1000000) return (step / 1000000).toFixed(1) + "M";
    if (step >= 1000) return (step / 1000).toFixed(1) + "K";
    return step.toString();
  }

  // Format time for display (seconds)
  function formatDuration(seconds) {
    if (!seconds) return "—";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
  }

  // Fetch sessions when the component appears and server is available
  $effect(() => {
    if (store.connected) {
      fetchSessions();
    }
  });
</script>

<div class="replay-panel">
  <h2 class="panel-title">
    Replay
    <Tooltip text="Load and replay recorded training sessions. Watch past experiments at any speed, scrub through the timeline, and jump to bookmarked moments." />
  </h2>

  {#if !store.replayConnected && !store.replayStatus}
    <!-- Session selector -->
    <div class="session-selector">
      <div class="control-header">
        <span class="control-label">Recorded Sessions</span>
        <button class="btn-refresh" onclick={fetchSessions} title="Refresh session list">
          &#8635;
        </button>
      </div>

      {#if sessions.length === 0}
        <div class="empty-state">
          No recorded sessions found. Start recording from the Controls panel to save a session.
        </div>
      {:else}
        <div class="session-list">
          {#each sessions as session}
            <button
              class="session-item"
              onclick={() => {
                store.connectReplay();
                // Small delay for WebSocket to connect
                setTimeout(() => loadSession(session.path), 300);
              }}
            >
              <div class="session-name">{session.name}</div>
              <div class="session-meta">
                {#if session.metadata.frame_count}
                  {session.metadata.frame_count} frames
                {/if}
                {#if session.metadata.duration_seconds}
                  &middot; {formatDuration(session.metadata.duration_seconds)}
                {/if}
              </div>
            </button>
          {/each}
        </div>
      {/if}
    </div>
  {:else}
    <!-- Replay controls (after loading a session) -->
    <div class="replay-controls">
      {#if store.replayStatus}
        <!-- Timeline scrubber -->
        <div class="scrubber-section">
          <div class="scrubber-info">
            <span class="scrubber-step">Step {formatStep(store.replayStatus.step)}</span>
            <span class="scrubber-progress">
              {store.replayStatus.position + 1} / {store.replayStatus.total}
            </span>
          </div>
          <input
            type="range"
            class="scrubber"
            min="0"
            max={Math.max(0, store.replayStatus.total - 1)}
            value={store.replayStatus.position}
            oninput={handleScrub}
          />
          <div class="scrubber-labels">
            <span>Start</span>
            <span>End</span>
          </div>
        </div>

        <!-- Play/Pause + Speed -->
        <div class="playback-row">
          <button
            class="btn-play"
            onclick={() => store.sendReplayCommand({
              type: store.replayStatus.playing ? "pause" : "play"
            })}
          >
            {store.replayStatus.playing ? "\u23F8 Pause" : "\u25B6 Play"}
          </button>

          <div class="speed-buttons">
            {#each SPEED_OPTIONS as speed}
              <button
                class="speed-btn"
                class:active={store.replayStatus.speed === speed}
                onclick={() => store.sendReplayCommand({ type: "set_speed", value: speed })}
              >
                {speed}x
              </button>
            {/each}
          </div>
        </div>

        <!-- Bookmarks -->
        {#if store.replayStatus.bookmarks && store.replayStatus.bookmarks.length > 0}
          <div class="bookmarks-section">
            <div class="control-header">
              <span class="control-label">Bookmarks</span>
              <Tooltip text="Jump to interesting moments that were saved during recording." />
            </div>
            <div class="bookmark-list">
              {#each store.replayStatus.bookmarks as bookmark, idx}
                <button
                  class="bookmark-item"
                  onclick={() => store.sendReplayCommand({ type: "seek_bookmark", index: idx })}
                  title="Step {bookmark.step}"
                >
                  <span class="bookmark-icon">&#128278;</span>
                  <span class="bookmark-label">{bookmark.label}</span>
                  <span class="bookmark-step">Step {formatStep(bookmark.step)}</span>
                </button>
              {/each}
            </div>
          </div>
        {/if}

        <!-- Session info -->
        {#if store.replayStatus.metadata}
          <div class="session-info">
            {#if store.replayStatus.metadata.frame_count}
              <span>{store.replayStatus.metadata.frame_count} frames</span>
            {/if}
            {#if store.replayStatus.metadata.duration_seconds}
              <span>{formatDuration(store.replayStatus.metadata.duration_seconds)}</span>
            {/if}
          </div>
        {/if}

        <!-- Close replay -->
        <button
          class="btn-close"
          onclick={() => store.disconnectReplay()}
        >
          Close Replay
        </button>
      {:else if loadingSession}
        <div class="loading-state">Loading session...</div>
      {:else if replayError}
        <div class="error-state">{replayError}</div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .replay-panel {
    padding: 12px;
    border-bottom: 1px solid #1a1a3e;
  }

  .panel-title {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #666;
    margin: 0 0 10px 0;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  /* Session selector */
  .session-selector {
    margin-bottom: 8px;
  }

  .control-header {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 6px;
  }

  .control-label {
    font-size: 12px;
    color: #aaa;
    flex: 1;
  }

  .btn-refresh {
    background: none;
    border: 1px solid #2a2a4e;
    color: #888;
    font-size: 14px;
    cursor: pointer;
    padding: 2px 6px;
    border-radius: 4px;
    transition: all 0.15s;
  }

  .btn-refresh:hover {
    background: rgba(255, 255, 255, 0.05);
    color: #ccc;
  }

  .empty-state {
    font-size: 11px;
    color: #555;
    line-height: 1.5;
    padding: 8px;
    text-align: center;
  }

  .session-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
    max-height: 150px;
    overflow-y: auto;
  }

  .session-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 8px 10px;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid #2a2a4e;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.15s;
    text-align: left;
    color: inherit;
    font-family: inherit;
  }

  .session-item:hover {
    background: rgba(78, 205, 196, 0.08);
    border-color: rgba(78, 205, 196, 0.3);
  }

  .session-name {
    font-size: 12px;
    font-weight: 600;
    color: #e0e0e0;
  }

  .session-meta {
    font-size: 10px;
    color: #666;
  }

  /* Replay controls */
  .replay-controls {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  /* Scrubber */
  .scrubber-section {
    margin-bottom: 4px;
  }

  .scrubber-info {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    margin-bottom: 4px;
  }

  .scrubber-step {
    color: #4ecdc4;
    font-weight: 600;
    font-family: "JetBrains Mono", "Fira Code", monospace;
  }

  .scrubber-progress {
    color: #888;
    font-family: "JetBrains Mono", "Fira Code", monospace;
  }

  .scrubber {
    width: 100%;
    height: 6px;
    -webkit-appearance: none;
    appearance: none;
    background: #2a2a4e;
    border-radius: 3px;
    outline: none;
    margin: 0;
  }

  .scrubber::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #4ecdc4;
    cursor: pointer;
    border: 2px solid #111128;
  }

  .scrubber::-moz-range-thumb {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #4ecdc4;
    cursor: pointer;
    border: 2px solid #111128;
  }

  .scrubber-labels {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: #555;
    margin-top: 2px;
  }

  /* Playback row */
  .playback-row {
    display: flex;
    gap: 6px;
    align-items: center;
  }

  .btn-play {
    padding: 6px 14px;
    border: 1px solid #4ecdc4;
    background: rgba(78, 205, 196, 0.1);
    color: #4ecdc4;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s;
    font-family: inherit;
    white-space: nowrap;
  }

  .btn-play:hover {
    background: rgba(78, 205, 196, 0.2);
  }

  .speed-buttons {
    display: flex;
    gap: 3px;
    flex: 1;
  }

  .speed-btn {
    flex: 1;
    padding: 5px 0;
    border: 1px solid #2a2a4e;
    background: rgba(255, 255, 255, 0.03);
    color: #888;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
    font-family: inherit;
  }

  .speed-btn:hover {
    background: rgba(255, 255, 255, 0.08);
    color: #ccc;
  }

  .speed-btn.active {
    background: rgba(78, 205, 196, 0.15);
    border-color: rgba(78, 205, 196, 0.4);
    color: #4ecdc4;
  }

  /* Bookmarks */
  .bookmarks-section {
    margin-top: 2px;
  }

  .bookmark-list {
    display: flex;
    flex-direction: column;
    gap: 3px;
    max-height: 100px;
    overflow-y: auto;
  }

  .bookmark-item {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 5px 8px;
    background: rgba(255, 217, 61, 0.04);
    border: 1px solid rgba(255, 217, 61, 0.15);
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.15s;
    color: inherit;
    font-family: inherit;
    font-size: 11px;
  }

  .bookmark-item:hover {
    background: rgba(255, 217, 61, 0.1);
    border-color: rgba(255, 217, 61, 0.3);
  }

  .bookmark-icon {
    font-size: 12px;
  }

  .bookmark-label {
    flex: 1;
    color: #e0e0e0;
  }

  .bookmark-step {
    color: #888;
    font-family: "JetBrains Mono", "Fira Code", monospace;
    font-size: 10px;
  }

  /* Session info */
  .session-info {
    display: flex;
    gap: 12px;
    font-size: 10px;
    color: #666;
    justify-content: center;
  }

  /* Close button */
  .btn-close {
    width: 100%;
    padding: 6px 12px;
    border: 1px solid #555;
    background: rgba(255, 255, 255, 0.03);
    color: #888;
    border-radius: 6px;
    font-size: 11px;
    cursor: pointer;
    transition: all 0.15s;
    font-family: inherit;
  }

  .btn-close:hover {
    background: rgba(255, 255, 255, 0.08);
    color: #ccc;
  }

  /* Loading/Error states */
  .loading-state {
    font-size: 12px;
    color: #888;
    text-align: center;
    padding: 16px 0;
  }

  .error-state {
    font-size: 12px;
    color: #e94560;
    text-align: center;
    padding: 8px;
  }
</style>
