<script>
  /**
   * Agent visualization canvas.
   *
   * Renders agents as colored circles on a grid, with food dots and
   * a field heatmap background. Uses Canvas 2D for now; will be
   * upgraded to Pixi.js in US-007.
   */
  let { store } = $props();

  let canvasEl;
  let animFrameId;

  // Grid config (matches default server config)
  const GRID_SIZE = 20;
  const PADDING = 20;

  // Colors for cluster groups
  const CLUSTER_COLORS = [
    "#e94560", // red
    "#0f3460", // blue
    "#533483", // purple
    "#16c79a", // teal
    "#f5a623", // orange
    "#11999e", // cyan
    "#e84545", // coral
    "#903749", // maroon
  ];

  const FOOD_COLOR = "#4caf50";
  const DEAD_COLOR = "#333";

  function draw() {
    if (!canvasEl) return;
    const ctx = canvasEl.getContext("2d");
    const w = canvasEl.width;
    const h = canvasEl.height;
    const cellW = (w - PADDING * 2) / GRID_SIZE;
    const cellH = (h - PADDING * 2) / GRID_SIZE;

    // Clear
    ctx.fillStyle = "#0d0d22";
    ctx.fillRect(0, 0, w, h);

    // Draw field heatmap (if available)
    if (store.field && store.fieldShape) {
      const fieldH = store.fieldShape[0] || GRID_SIZE;
      const fieldW = store.fieldShape[1] || GRID_SIZE;
      const fieldC = store.fieldShape[2] || 4;

      for (let y = 0; y < fieldH; y++) {
        for (let x = 0; x < fieldW; x++) {
          // Sum across channels for intensity
          let intensity = 0;
          for (let c = 0; c < fieldC; c++) {
            intensity += store.field[y * fieldW * fieldC + x * fieldC + c] || 0;
          }
          intensity = Math.min(intensity / fieldC, 1.0);

          if (intensity > 0.01) {
            const px = PADDING + x * cellW;
            const py = PADDING + y * cellH;
            const alpha = intensity * 0.6;
            ctx.fillStyle = `rgba(30, 60, 120, ${alpha})`;
            ctx.fillRect(px, py, cellW, cellH);
          }
        }
      }
    }

    // Draw grid lines (subtle)
    ctx.strokeStyle = "rgba(255, 255, 255, 0.03)";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= GRID_SIZE; i++) {
      const x = PADDING + i * cellW;
      const y = PADDING + i * cellH;
      ctx.beginPath();
      ctx.moveTo(x, PADDING);
      ctx.lineTo(x, h - PADDING);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(PADDING, y);
      ctx.lineTo(w - PADDING, y);
      ctx.stroke();
    }

    // Draw food
    if (store.foodPositions && store.foodCollected) {
      const numFood = store.foodShape[0] || 0;
      for (let i = 0; i < numFood; i++) {
        if (store.foodCollected[i]) continue; // skip collected food
        const fx = store.foodPositions[i * 2];
        const fy = store.foodPositions[i * 2 + 1];
        const px = PADDING + fx * cellW + cellW / 2;
        const py = PADDING + fy * cellH + cellH / 2;

        ctx.fillStyle = FOOD_COLOR;
        ctx.globalAlpha = 0.7;
        ctx.beginPath();
        ctx.arc(px, py, cellW * 0.2, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1.0;
      }
    }

    // Draw agents
    if (store.positions && store.alive) {
      const numAgents = store.positionsShape[0] || 0;
      for (let i = 0; i < numAgents; i++) {
        if (!store.alive[i]) continue; // skip dead agents

        const ax = store.positions[i * 2];
        const ay = store.positions[i * 2 + 1];
        const px = PADDING + ax * cellW + cellW / 2;
        const py = PADDING + ay * cellH + cellH / 2;

        // Color by cluster if available
        const clusterId = store.clusters ? store.clusters[i] : 0;
        const color = CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length];

        // Size based on energy
        const energyVal = store.energy ? store.energy[i] : 100;
        const radius = cellW * 0.3 + (energyVal / 200) * cellW * 0.15;

        // Glow effect
        ctx.shadowColor = color;
        ctx.shadowBlur = 6;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(px, py, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    }

    // Status overlay (when disconnected)
    if (!store.connected && !store.positions) {
      ctx.fillStyle = "rgba(10, 10, 26, 0.8)";
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = "#888";
      ctx.font = "14px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Waiting for server connection...", w / 2, h / 2 - 10);
      ctx.fillStyle = "#555";
      ctx.font = "11px Inter, sans-serif";
      ctx.fillText("Start server: python -m src.server.main --mock", w / 2, h / 2 + 14);
    }
  }

  function renderLoop() {
    draw();
    animFrameId = requestAnimationFrame(renderLoop);
  }

  function resizeCanvas() {
    if (!canvasEl) return;
    const parent = canvasEl.parentElement;
    const size = Math.min(parent.clientWidth - 24, parent.clientHeight - 24, 700);
    canvasEl.width = size;
    canvasEl.height = size;
  }

  $effect(() => {
    resizeCanvas();
    renderLoop();

    const onResize = () => resizeCanvas();
    window.addEventListener("resize", onResize);

    return () => {
      cancelAnimationFrame(animFrameId);
      window.removeEventListener("resize", onResize);
    };
  });
</script>

<div class="canvas-container">
  <canvas bind:this={canvasEl} class="agent-canvas"></canvas>
</div>

<style>
  .canvas-container {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 100%;
    padding: 12px;
  }

  .agent-canvas {
    border-radius: 8px;
    border: 1px solid #1a1a3e;
  }
</style>
