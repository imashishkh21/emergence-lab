<script>
  let { store } = $props();
</script>

<header class="header">
  <div class="title-group">
    <h1 class="title">Emergence Lab</h1>
    <span class="subtitle">Research Microscope</span>
  </div>

  <div class="status-group">
    <div class="step-counter">
      Step <span class="step-value">{store.step.toLocaleString()}</span>
    </div>

    <div class="connection-status" class:connected={store.connected} class:reconnecting={store.reconnecting}>
      <span class="status-dot"></span>
      {#if store.connected}
        Connected
      {:else if store.reconnecting}
        Reconnecting...
      {:else}
        Disconnected
      {/if}
    </div>
  </div>
</header>

<style>
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 16px;
    background: #111128;
    border-bottom: 1px solid #1a1a3e;
    height: 48px;
    flex-shrink: 0;
  }

  .title-group {
    display: flex;
    align-items: baseline;
    gap: 12px;
  }

  .title {
    font-size: 16px;
    font-weight: 700;
    margin: 0;
    color: #e94560;
    letter-spacing: -0.02em;
  }

  .subtitle {
    font-size: 12px;
    color: #666;
    font-weight: 400;
  }

  .status-group {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  .step-counter {
    font-size: 12px;
    color: #888;
  }

  .step-value {
    font-variant-numeric: tabular-nums;
    color: #ccc;
    font-weight: 600;
  }

  .connection-status {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: #888;
  }

  .status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #666;
  }

  .connected .status-dot {
    background: #4caf50;
    box-shadow: 0 0 4px #4caf50;
  }

  .reconnecting .status-dot {
    background: #ff9800;
    animation: pulse 1s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }
</style>
