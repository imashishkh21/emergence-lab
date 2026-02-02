<script>
  /**
   * Training control panel.
   *
   * Provides play/pause and basic parameter controls.
   * Will be expanded in US-009 with more controls.
   */
  let { store } = $props();
</script>

<div class="control-panel">
  <h2 class="panel-title">Controls</h2>

  <div class="controls">
    <button
      class="btn-primary"
      onclick={() => store.paused ? store.resume() : store.pause()}
      disabled={!store.connected}
    >
      {store.paused ? "Resume" : "Pause"}
    </button>

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

  .controls {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .btn-primary {
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

  .control-hint {
    font-size: 11px;
    color: #666;
    line-height: 1.4;
  }
</style>
