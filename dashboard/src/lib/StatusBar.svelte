<!--
  StatusBar.svelte

  Clear visual feedback for system state:
  - Connection status (Connected / Disconnected / Reconnecting)
  - Training mode (Learning / Evolving / Paused)
  - Health indicators for population and divergence
-->
<script>
  import Tooltip from "./Tooltip.svelte";

  let { store } = $props();
</script>

<div class="status-bar">
  <!-- Connection Status -->
  <div class="status-item">
    <span
      class="status-dot"
      class:connected={store.connected}
      class:reconnecting={store.reconnecting}
      class:disconnected={!store.connected && !store.reconnecting}
    ></span>
    <span class="status-label">
      {#if store.connected}
        Connected
      {:else if store.reconnecting}
        Reconnecting...
      {:else}
        Disconnected
      {/if}
    </span>
  </div>

  <!-- Training Mode -->
  <div class="status-item">
    <span
      class="mode-badge"
      class:mode-gradient={store.trainingMode === "gradient"}
      class:mode-evolve={store.trainingMode === "evolve"}
      class:mode-paused={store.paused || store.trainingMode === "paused"}
    >
      {#if store.paused || store.trainingMode === "paused"}
        <span class="mode-icon">&#x23F8;&#xFE0E;</span> Paused
      {:else if store.trainingMode === "evolve"}
        <span class="mode-icon">&#x1F9EC;</span> Evolving
      {:else}
        <span class="mode-icon">&#x1F9E0;</span> Learning
      {/if}
    </span>
    <Tooltip text="Training mode: Learning = adjusting brain weights via gradients. Evolving = only reproduction and mutation. Paused = simulation stopped." />
  </div>

  <!-- Separator -->
  <span class="separator"></span>

  <!-- Population Health -->
  <div class="status-item health-item">
    <span class="health-label">Population</span>
    <span
      class="health-indicator"
      class:health-stable={store.populationHealth === "stable"}
      class:health-declining={store.populationHealth === "declining"}
      class:health-collapsing={store.populationHealth === "collapsing"}
      class:health-unknown={store.populationHealth === "unknown"}
    >
      {#if store.populationHealth === "stable"}
        Stable
      {:else if store.populationHealth === "declining"}
        Declining
      {:else if store.populationHealth === "collapsing"}
        Collapsing
      {:else}
        --
      {/if}
    </span>
    <Tooltip text="Population health: Stable = 30%+ alive. Declining = 15-30%. Collapsing = below 15%. Watch out for crashes!" />
  </div>

  <!-- Divergence Health -->
  <div class="status-item health-item">
    <span class="health-label">Divergence</span>
    <span
      class="health-indicator"
      class:health-stable={store.divergenceHealth === "increasing"}
      class:health-declining={store.divergenceHealth === "flat"}
      class:health-collapsing={store.divergenceHealth === "homogenizing"}
      class:health-unknown={store.divergenceHealth === "unknown"}
    >
      {#if store.divergenceHealth === "increasing"}
        Increasing
      {:else if store.divergenceHealth === "flat"}
        Flat
      {:else if store.divergenceHealth === "homogenizing"}
        Homogenizing
      {:else}
        --
      {/if}
    </span>
    <Tooltip text="Weight divergence: Increasing = agents becoming different (good!). Flat = not changing. Homogenizing = agents converging to same brain (bad)." />
  </div>
</div>

<style>
  .status-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 6px 12px;
    background: #0e0e24;
    border-bottom: 1px solid #1a1a3e;
    font-size: 12px;
    flex-wrap: wrap;
    min-height: 32px;
  }

  .status-item {
    display: flex;
    align-items: center;
    gap: 5px;
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .status-dot.connected {
    background: #4caf50;
    box-shadow: 0 0 6px rgba(76, 175, 80, 0.6);
  }

  .status-dot.reconnecting {
    background: #ff9800;
    animation: pulse-dot 1.2s ease-in-out infinite;
  }

  .status-dot.disconnected {
    background: #e94560;
  }

  .status-label {
    color: #ccc;
    white-space: nowrap;
  }

  .mode-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 600;
    white-space: nowrap;
  }

  .mode-icon {
    font-size: 12px;
  }

  .mode-gradient {
    background: rgba(78, 205, 196, 0.15);
    color: #4ecdc4;
    border: 1px solid rgba(78, 205, 196, 0.3);
  }

  .mode-evolve {
    background: rgba(255, 217, 61, 0.15);
    color: #ffd93d;
    border: 1px solid rgba(255, 217, 61, 0.3);
  }

  .mode-paused {
    background: rgba(150, 150, 150, 0.15);
    color: #999;
    border: 1px solid rgba(150, 150, 150, 0.3);
  }

  .separator {
    width: 1px;
    height: 16px;
    background: #2a2a4e;
  }

  .health-item {
    gap: 4px;
  }

  .health-label {
    color: #888;
    font-size: 11px;
  }

  .health-indicator {
    padding: 1px 6px;
    border-radius: 8px;
    font-size: 11px;
    font-weight: 500;
    white-space: nowrap;
  }

  .health-stable {
    background: rgba(76, 175, 80, 0.15);
    color: #4caf50;
    border: 1px solid rgba(76, 175, 80, 0.25);
  }

  .health-declining {
    background: rgba(255, 152, 0, 0.15);
    color: #ff9800;
    border: 1px solid rgba(255, 152, 0, 0.25);
  }

  .health-collapsing {
    background: rgba(233, 69, 96, 0.15);
    color: #e94560;
    border: 1px solid rgba(233, 69, 96, 0.25);
  }

  .health-unknown {
    background: rgba(100, 100, 100, 0.1);
    color: #666;
    border: 1px solid rgba(100, 100, 100, 0.2);
  }

  @keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }
</style>
