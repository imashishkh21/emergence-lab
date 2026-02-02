<script>
  /**
   * Training control panel with pause/resume, speed, and hyperparameter controls.
   *
   * Provides:
   * - Play/Pause button
   * - Speed slider (1x, 2x, 4x, 8x)
   * - Mutation rate slider
   * - Diversity bonus slider
   * - Training mode indicator (Gradient / Evolve / Paused)
   */
  import Tooltip from "./Tooltip.svelte";

  let { store } = $props();

  const SPEED_OPTIONS = [0.5, 1, 2, 4, 8];

  let mutationRate = $state(0.01);
  let diversityBonus = $state(0.0);

  function handleMutationChange(e) {
    mutationRate = parseFloat(e.target.value);
    store.setParam("mutation_std", mutationRate);
  }

  function handleDiversityChange(e) {
    diversityBonus = parseFloat(e.target.value);
    store.setParam("diversity_bonus", diversityBonus);
  }

  let modeLabel = $derived(
    store.trainingMode === "gradient"
      ? "Learning"
      : store.trainingMode === "evolve"
        ? "Evolving"
        : "Paused"
  );

  let modeIcon = $derived(
    store.trainingMode === "gradient"
      ? "\u{1F9E0}"
      : store.trainingMode === "evolve"
        ? "\u{1F9EC}"
        : "\u23F8\uFE0F"
  );

  let modeClass = $derived(
    store.trainingMode === "gradient"
      ? "mode-gradient"
      : store.trainingMode === "evolve"
        ? "mode-evolve"
        : "mode-paused"
  );
</script>

<div class="control-panel">
  <h2 class="panel-title">Controls</h2>

  <!-- Training Mode Indicator -->
  <div class="mode-indicator {modeClass}">
    <span class="mode-icon">{modeIcon}</span>
    <span class="mode-label">{modeLabel}</span>
    <Tooltip text="Current training phase. Gradient = adjusting brain weights via math. Evolve = letting natural selection drive changes." />
  </div>

  <!-- Play/Pause -->
  <div class="control-group">
    <button
      class="btn-primary"
      onclick={() => store.paused ? store.resume() : store.pause()}
      disabled={!store.connected}
      title="Stop or resume the simulation"
    >
      {store.paused ? "\u25B6 Resume" : "\u23F8 Pause"}
    </button>
  </div>

  <!-- Speed Slider -->
  <div class="control-group">
    <div class="control-header">
      <label class="control-label">Speed</label>
      <span class="control-value">{store.speedMultiplier}x</span>
      <Tooltip text="How fast to run the simulation. Higher = more steps per second, but may reduce frame smoothness." />
    </div>
    <div class="speed-buttons">
      {#each SPEED_OPTIONS as speed}
        <button
          class="speed-btn"
          class:active={store.speedMultiplier === speed}
          onclick={() => store.setSpeed(speed)}
          disabled={!store.connected}
        >
          {speed}x
        </button>
      {/each}
    </div>
  </div>

  <!-- Mutation Rate Slider -->
  <div class="control-group">
    <div class="control-header">
      <label class="control-label">Mutation Rate</label>
      <span class="control-value">{mutationRate.toFixed(3)}</span>
      <Tooltip text="How much do babies differ from parents? Higher = more variation between generations. Too high = chaotic. Too low = everyone stays the same." />
    </div>
    <input
      type="range"
      min="0.001"
      max="0.1"
      step="0.001"
      value={mutationRate}
      oninput={handleMutationChange}
      disabled={!store.connected}
      class="slider"
    />
    <div class="slider-labels">
      <span>Stable</span>
      <span>Chaotic</span>
    </div>
  </div>

  <!-- Diversity Bonus Slider -->
  <div class="control-group">
    <div class="control-header">
      <label class="control-label">Diversity Bonus</label>
      <span class="control-value">{diversityBonus.toFixed(2)}</span>
      <Tooltip text="Extra reward for being different from other agents. Higher = stronger push for agents to develop unique strategies. 0 = no bonus." />
    </div>
    <input
      type="range"
      min="0"
      max="1.0"
      step="0.01"
      value={diversityBonus}
      oninput={handleDiversityChange}
      disabled={!store.connected}
      class="slider"
    />
    <div class="slider-labels">
      <span>None</span>
      <span>Strong</span>
    </div>
  </div>

  <!-- Status hint -->
  <div class="control-hint">
    {#if !store.connected}
      Connect to server to enable controls
    {:else if store.paused}
      Training is paused. Click Resume to continue.
    {:else}
      Training is running. Agents are learning and evolving.
    {/if}
  </div>
</div>

<style>
  .control-panel {
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
  }

  /* Mode indicator */
  .mode-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-radius: 6px;
    margin-bottom: 10px;
    font-size: 13px;
    font-weight: 600;
    transition: background 0.3s, border-color 0.3s;
    cursor: default;
  }

  .mode-gradient {
    background: rgba(78, 205, 196, 0.1);
    border: 1px solid rgba(78, 205, 196, 0.3);
    color: #4ecdc4;
  }

  .mode-evolve {
    background: rgba(255, 217, 61, 0.1);
    border: 1px solid rgba(255, 217, 61, 0.3);
    color: #ffd93d;
  }

  .mode-paused {
    background: rgba(102, 102, 102, 0.1);
    border: 1px solid rgba(102, 102, 102, 0.3);
    color: #999;
  }

  .mode-icon {
    font-size: 16px;
  }

  .mode-label {
    letter-spacing: 0.02em;
  }

  /* Control groups */
  .control-group {
    margin-bottom: 12px;
  }

  .control-header {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 4px;
  }

  .control-label {
    font-size: 12px;
    color: #aaa;
    flex: 1;
  }

  .control-value {
    font-size: 12px;
    color: #e0e0e0;
    font-weight: 600;
    font-family: "JetBrains Mono", "Fira Code", monospace;
  }

  /* Play/Pause button */
  .btn-primary {
    width: 100%;
    padding: 8px 16px;
    border: 1px solid #e94560;
    background: rgba(233, 69, 96, 0.1);
    color: #e94560;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s;
    font-family: inherit;
  }

  .btn-primary:hover:not(:disabled) {
    background: rgba(233, 69, 96, 0.2);
  }

  .btn-primary:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  /* Speed buttons */
  .speed-buttons {
    display: flex;
    gap: 4px;
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

  .speed-btn:hover:not(:disabled) {
    background: rgba(255, 255, 255, 0.08);
    color: #ccc;
  }

  .speed-btn.active {
    background: rgba(78, 205, 196, 0.15);
    border-color: rgba(78, 205, 196, 0.4);
    color: #4ecdc4;
  }

  .speed-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  /* Sliders */
  .slider {
    width: 100%;
    height: 4px;
    -webkit-appearance: none;
    appearance: none;
    background: #2a2a4e;
    border-radius: 2px;
    outline: none;
    margin: 4px 0;
  }

  .slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #4ecdc4;
    cursor: pointer;
    border: 2px solid #111128;
  }

  .slider::-moz-range-thumb {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #4ecdc4;
    cursor: pointer;
    border: 2px solid #111128;
  }

  .slider:disabled {
    opacity: 0.4;
  }

  .slider:disabled::-webkit-slider-thumb {
    cursor: not-allowed;
  }

  .slider-labels {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: #555;
  }

  /* Hint */
  .control-hint {
    font-size: 11px;
    color: #666;
    line-height: 1.4;
    margin-top: 4px;
  }
</style>
