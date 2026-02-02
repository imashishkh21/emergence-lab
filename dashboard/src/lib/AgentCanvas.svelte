<script>
  /**
   * Agent visualization canvas using Pixi.js v8.
   *
   * Renders agents as colored circles on a grid with smooth position
   * interpolation, a field heatmap background, food dots, and optional
   * agent trails. Uses WebGPU with WebGL fallback.
   */
  import { createRenderer } from "./renderer.js";

  let { store } = $props();

  let containerEl;
  let renderer = null;
  let animFrameId;
  let showTrails = $state(false);

  function renderLoop() {
    if (renderer) {
      renderer.update(store);
    }
    animFrameId = requestAnimationFrame(renderLoop);
  }

  function resizeCanvas() {
    if (!containerEl || !renderer) return;
    const size = Math.min(
      containerEl.clientWidth - 24,
      containerEl.clientHeight - 24,
      700
    );
    if (size > 0) {
      renderer.resize(size);
    }
  }

  function toggleTrails() {
    showTrails = !showTrails;
    if (renderer) {
      renderer.setShowTrails(showTrails);
    }
  }

  $effect(() => {
    if (!containerEl) return;

    let destroyed = false;

    createRenderer(containerEl, {
      size: Math.min(
        containerEl.clientWidth - 24,
        containerEl.clientHeight - 24,
        700
      ),
      gridSize: 20,
      showTrails,
    }).then((r) => {
      if (destroyed) {
        r.destroy();
        return;
      }
      renderer = r;
      renderLoop();
    });

    const onResize = () => resizeCanvas();
    window.addEventListener("resize", onResize);

    return () => {
      destroyed = true;
      cancelAnimationFrame(animFrameId);
      window.removeEventListener("resize", onResize);
      if (renderer) {
        renderer.destroy();
        renderer = null;
      }
    };
  });
</script>

<div class="canvas-container" bind:this={containerEl}>
  {#if !store.connected && !store.positions}
    <div class="status-overlay">
      <p class="status-text">Waiting for server connection...</p>
      <p class="status-hint">Start server: python -m src.server.main --mock</p>
    </div>
  {/if}
</div>

<div class="canvas-controls">
  <button class="trail-toggle" onclick={toggleTrails} title="Toggle agent movement trails">
    {showTrails ? "Hide Trails" : "Show Trails"}
  </button>
</div>

<style>
  .canvas-container {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 100%;
    padding: 12px;
    position: relative;
  }

  .status-overlay {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: rgba(10, 10, 26, 0.85);
    border-radius: 8px;
    z-index: 10;
    pointer-events: none;
  }

  .status-text {
    color: #888;
    font-size: 14px;
    font-family: "Inter", sans-serif;
    margin: 0 0 8px;
  }

  .status-hint {
    color: #555;
    font-size: 11px;
    font-family: "Inter", sans-serif;
    margin: 0;
  }

  .canvas-controls {
    position: absolute;
    bottom: 20px;
    left: 20px;
    z-index: 5;
  }

  .trail-toggle {
    background: rgba(17, 17, 40, 0.8);
    color: #aaa;
    border: 1px solid #1a1a3e;
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 11px;
    cursor: pointer;
    font-family: "Inter", sans-serif;
    transition: color 0.15s, border-color 0.15s;
  }

  .trail-toggle:hover {
    color: #e0e0e0;
    border-color: #333366;
  }
</style>
