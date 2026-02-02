/**
 * Pixi.js v8 renderer for the agent simulation.
 *
 * Renders agents as colored circles with smooth position interpolation,
 * a field heatmap background, food dots, and optional agent trails.
 * Uses a pre-allocated sprite pool for efficient rendering.
 */
import { Application, Container, Graphics, GraphicsContext } from "pixi.js";

// Cluster colors (matching original Canvas 2D implementation)
const CLUSTER_COLORS = [
  0xe94560, // red
  0x0f3460, // blue
  0x533483, // purple
  0x16c79a, // teal
  0xf5a623, // orange
  0x11999e, // cyan
  0xe84545, // coral
  0x903749, // maroon
];

const FOOD_COLOR = 0x4caf50;
const BG_COLOR = 0x0d0d22;
const GRID_LINE_COLOR = 0xffffff;

const MAX_POOL_SIZE = 64;
const LERP_FACTOR = 0.3;

/**
 * Create and initialize the Pixi.js renderer.
 *
 * @param {HTMLDivElement} container - DOM element to mount the canvas into
 * @param {Object} [options] - Configuration options
 * @param {number} [options.size=600] - Canvas width/height in pixels
 * @param {number} [options.gridSize=20] - Grid dimension
 * @param {boolean} [options.showTrails=false] - Enable agent trails
 * @param {number} [options.trailAlpha=0.03] - Trail fade rate per frame
 * @returns {Promise<Object>} Renderer API with update/resize/destroy methods
 */
export async function createRenderer(container, options = {}) {
  const {
    size: initialSize = 600,
    gridSize = 20,
    showTrails: initialShowTrails = false,
    trailAlpha = 0.03,
  } = options;

  let canvasSize = initialSize;
  let showTrails = initialShowTrails;

  // Initialize Pixi.js application
  const app = new Application();
  await app.init({
    width: canvasSize,
    height: canvasSize,
    backgroundColor: BG_COLOR,
    preference: "webgpu",
    antialias: true,
    resolution: window.devicePixelRatio || 1,
    autoDensity: true,
  });

  // Style the canvas element
  app.canvas.style.borderRadius = "8px";
  app.canvas.style.border = "1px solid #1a1a3e";
  container.appendChild(app.canvas);

  // Scene layers (bottom to top)
  const fieldLayer = new Container();
  const trailLayer = new Container();
  const gridLayer = new Container();
  const foodLayer = new Container();
  const agentLayer = new Container();
  const overlayLayer = new Container();

  app.stage.addChild(fieldLayer, trailLayer, gridLayer, foodLayer, agentLayer, overlayLayer);

  // --- Field heatmap ---
  const fieldCells = [];
  for (let y = 0; y < gridSize; y++) {
    const row = [];
    for (let x = 0; x < gridSize; x++) {
      const cell = new Graphics();
      fieldLayer.addChild(cell);
      row.push(cell);
    }
    fieldCells.push(row);
  }

  // --- Grid lines ---
  const gridGraphics = new Graphics();
  gridLayer.addChild(gridGraphics);

  function drawGrid() {
    const padding = canvasSize * 0.03;
    const cellW = (canvasSize - padding * 2) / gridSize;
    gridGraphics.clear();
    gridGraphics.setStrokeStyle({ width: 0.5, color: GRID_LINE_COLOR, alpha: 0.03 });
    for (let i = 0; i <= gridSize; i++) {
      const x = padding + i * cellW;
      const y = padding + i * cellW;
      gridGraphics.moveTo(x, padding).lineTo(x, canvasSize - padding);
      gridGraphics.moveTo(padding, y).lineTo(canvasSize - padding, y);
    }
    gridGraphics.stroke();
  }

  // --- Food pool ---
  const foodDots = [];
  const foodContext = new GraphicsContext().circle(0, 0, 1).fill({ color: FOOD_COLOR, alpha: 0.7 });

  function ensureFoodPool(count) {
    while (foodDots.length < count) {
      const dot = new Graphics(foodContext);
      dot.visible = false;
      foodLayer.addChild(dot);
      foodDots.push(dot);
    }
  }

  // --- Agent pool ---
  const agentPool = [];
  // Interpolated positions for smooth movement
  const interpX = new Float32Array(MAX_POOL_SIZE);
  const interpY = new Float32Array(MAX_POOL_SIZE);
  // Previous target positions (to detect changes)
  const prevTargetX = new Float32Array(MAX_POOL_SIZE).fill(-1);
  const prevTargetY = new Float32Array(MAX_POOL_SIZE).fill(-1);

  for (let i = 0; i < MAX_POOL_SIZE; i++) {
    const gfx = new Graphics();
    gfx.visible = false;
    agentLayer.addChild(gfx);
    agentPool.push(gfx);
  }

  // --- Trail graphics ---
  const trailGraphics = new Graphics();
  trailLayer.addChild(trailGraphics);
  // Trail history: array of { x, y, color, age }
  let trailPoints = [];
  const MAX_TRAIL_POINTS = 2000;
  const TRAIL_MAX_AGE = 60; // frames

  // --- Overlay (disconnected state) ---
  const overlayBg = new Graphics();
  const overlayText = null; // Pixi.js Text is heavy; we'll use CSS overlay instead
  overlayLayer.addChild(overlayBg);
  overlayLayer.visible = false;

  // Draw initial grid
  drawGrid();

  /**
   * Compute layout metrics from current canvas size.
   */
  function getLayout() {
    const padding = canvasSize * 0.03;
    const cellW = (canvasSize - padding * 2) / gridSize;
    return { padding, cellW };
  }

  /**
   * Update the field heatmap layer.
   */
  function updateField(fieldData, fieldShape) {
    if (!fieldData || !fieldShape) return;

    const { padding, cellW } = getLayout();
    const fieldH = fieldShape[0] || gridSize;
    const fieldW = fieldShape[1] || gridSize;
    const fieldC = fieldShape[2] || 4;

    for (let y = 0; y < Math.min(fieldH, gridSize); y++) {
      for (let x = 0; x < Math.min(fieldW, gridSize); x++) {
        let intensity = 0;
        for (let c = 0; c < fieldC; c++) {
          intensity += fieldData[y * fieldW * fieldC + x * fieldC + c] || 0;
        }
        intensity = Math.min(intensity / fieldC, 1.0);

        const cell = fieldCells[y]?.[x];
        if (!cell) continue;

        cell.clear();
        if (intensity > 0.01) {
          const px = padding + x * cellW;
          const py = padding + y * cellW;
          cell.rect(px, py, cellW, cellW).fill({ color: 0x1e3c78, alpha: intensity * 0.6 });
        }
      }
    }
  }

  /**
   * Update food dots.
   */
  function updateFood(foodPositions, foodCollected, foodShape) {
    if (!foodPositions || !foodCollected || !foodShape) {
      // Hide all food dots
      for (const dot of foodDots) dot.visible = false;
      return;
    }

    const { padding, cellW } = getLayout();
    const numFood = foodShape[0] || 0;
    ensureFoodPool(numFood);

    const foodRadius = cellW * 0.2;

    for (let i = 0; i < foodDots.length; i++) {
      if (i >= numFood) {
        foodDots[i].visible = false;
        continue;
      }
      if (foodCollected[i]) {
        foodDots[i].visible = false;
        continue;
      }

      const fx = foodPositions[i * 2];
      const fy = foodPositions[i * 2 + 1];
      const px = padding + fx * cellW + cellW / 2;
      const py = padding + fy * cellW + cellW / 2;

      foodDots[i].position.set(px, py);
      foodDots[i].scale.set(foodRadius);
      foodDots[i].visible = true;
    }
  }

  /**
   * Update agent positions with lerp interpolation.
   */
  function updateAgents(positions, alive, energy, clusters, positionsShape) {
    if (!positions || !alive) {
      for (const gfx of agentPool) gfx.visible = false;
      return;
    }

    const { padding, cellW } = getLayout();
    const numAgents = positionsShape?.[0] || 0;

    for (let i = 0; i < MAX_POOL_SIZE; i++) {
      const gfx = agentPool[i];
      if (i >= numAgents || !alive[i]) {
        gfx.visible = false;
        continue;
      }

      // Target grid position
      const targetX = positions[i * 2];
      const targetY = positions[i * 2 + 1];

      // Target pixel position (center of cell)
      const targetPx = padding + targetX * cellW + cellW / 2;
      const targetPy = padding + targetY * cellW + cellW / 2;

      // Initialize interpolation if first frame or agent respawned
      if (prevTargetX[i] < 0 || prevTargetY[i] < 0) {
        interpX[i] = targetPx;
        interpY[i] = targetPy;
      } else {
        // Lerp toward target
        interpX[i] += (targetPx - interpX[i]) * LERP_FACTOR;
        interpY[i] += (targetPy - interpY[i]) * LERP_FACTOR;
      }

      prevTargetX[i] = targetX;
      prevTargetY[i] = targetY;

      // Color by cluster
      const clusterId = clusters ? clusters[i] : 0;
      const color = CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length];

      // Radius based on energy
      const energyVal = energy ? energy[i] : 100;
      const radius = cellW * 0.3 + (energyVal / 200) * cellW * 0.15;

      // Redraw agent circle
      gfx.clear();
      // Glow (larger, semi-transparent circle behind)
      gfx.circle(0, 0, radius * 1.5).fill({ color, alpha: 0.15 });
      // Main body
      gfx.circle(0, 0, radius).fill({ color, alpha: 1.0 });

      gfx.position.set(interpX[i], interpY[i]);
      gfx.visible = true;

      // Add trail point
      if (showTrails) {
        trailPoints.push({ x: interpX[i], y: interpY[i], color, age: 0 });
      }
    }
  }

  /**
   * Render trail effect.
   */
  function updateTrails() {
    if (!showTrails) {
      trailGraphics.clear();
      return;
    }

    trailGraphics.clear();

    // Age and prune trail points
    trailPoints = trailPoints.filter((p) => {
      p.age++;
      return p.age < TRAIL_MAX_AGE;
    });

    // Limit total points
    if (trailPoints.length > MAX_TRAIL_POINTS) {
      trailPoints = trailPoints.slice(-MAX_TRAIL_POINTS);
    }

    // Draw trail dots
    for (const p of trailPoints) {
      const alpha = (1 - p.age / TRAIL_MAX_AGE) * trailAlpha;
      if (alpha < 0.002) continue;
      trailGraphics.circle(p.x, p.y, 2).fill({ color: p.color, alpha });
    }
  }

  /**
   * Main update method. Call with store data each frame.
   */
  function update(store) {
    updateField(store.field, store.fieldShape);
    updateFood(store.foodPositions, store.foodCollected, store.foodShape);
    updateAgents(
      store.positions,
      store.alive,
      store.energy,
      store.clusters,
      store.positionsShape
    );
    updateTrails();

    // Show/hide overlay
    const shouldShowOverlay = !store.connected && !store.positions;
    overlayLayer.visible = shouldShowOverlay;
    if (shouldShowOverlay) {
      overlayBg.clear();
      overlayBg.rect(0, 0, canvasSize, canvasSize).fill({ color: 0x0a0a1a, alpha: 0.8 });
    }
  }

  /**
   * Resize the renderer.
   */
  function resize(newSize) {
    canvasSize = newSize;
    app.renderer.resize(canvasSize, canvasSize);
    drawGrid();
    // Reset interpolation so agents snap to new layout
    prevTargetX.fill(-1);
    prevTargetY.fill(-1);
  }

  /**
   * Toggle trail rendering.
   */
  function setShowTrails(enabled) {
    showTrails = enabled;
    if (!enabled) {
      trailPoints = [];
      trailGraphics.clear();
    }
  }

  /**
   * Clean up all Pixi resources.
   */
  function destroy() {
    app.destroy(true, { children: true, texture: true });
  }

  return {
    update,
    resize,
    destroy,
    setShowTrails,
    get canvas() {
      return app.canvas;
    },
    get showTrails() {
      return showTrails;
    },
  };
}
