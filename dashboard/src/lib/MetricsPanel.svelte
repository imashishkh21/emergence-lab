<script>
  /**
   * Real-time metrics charts panel using Plotly.js.
   *
   * Shows 4 live-updating charts with helper tooltips:
   * - Reward over time
   * - Weight divergence
   * - Specialization score
   * - Population size
   *
   * Uses Plotly.extendTraces for efficient incremental updates
   * and a rolling window of 1000 data points.
   */
  import { onMount } from "svelte";

  let { store } = $props();

  const MAX_POINTS = 1000;

  /**
   * Chart definitions — each chart maps to a metric key from the store.
   * @type {Array<{id: string, title: string, metricKey: string, help: string, color: string, yRange?: [number, number]}>}
   */
  const chartDefs = [
    {
      id: "chart-reward",
      title: "Reward",
      metricKey: "mean_reward",
      help: "Are agents getting better at collecting food? Higher = more food collected per step.",
      color: "#4ecdc4",
    },
    {
      id: "chart-divergence",
      title: "Weight Divergence",
      metricKey: "weight_divergence",
      help: "How different are the agents' brains from each other? Higher = more unique strategies.",
      color: "#ff6b6b",
    },
    {
      id: "chart-specialization",
      title: "Specialization",
      metricKey: "specialization_score",
      help: "Are clear roles emerging (scouts, followers, etc.)? 0 = identical, 1 = perfectly specialized.",
      color: "#ffd93d",
      yRange: [0, 1],
    },
    {
      id: "chart-population",
      title: "Population",
      metricKey: "population_size",
      help: "How many agents are currently alive? Stable = healthy ecosystem.",
      color: "#6bcb77",
    },
  ];

  // Track how many points each chart has, for extend vs newPlot logic
  let chartPointCounts = $state(new Map());
  let plotlyModule = $state(null);
  let chartsReady = $state(false);
  let lastHistoryLen = $state(0);

  // Dark layout template shared by all charts
  const darkLayout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#999", size: 10, family: "Inter, sans-serif" },
    margin: { l: 40, r: 8, t: 4, b: 24 },
    xaxis: {
      showgrid: false,
      zeroline: false,
      color: "#555",
      tickfont: { size: 9 },
    },
    yaxis: {
      showgrid: true,
      gridcolor: "rgba(255,255,255,0.04)",
      zeroline: false,
      color: "#555",
      tickfont: { size: 9 },
    },
    showlegend: false,
  };

  onMount(async () => {
    // Dynamic import to avoid SSR issues with Plotly
    const Plotly = await import("plotly.js-dist-min");
    plotlyModule = Plotly.default || Plotly;
    initCharts();
  });

  /**
   * Initialize empty Plotly charts.
   */
  function initCharts() {
    if (!plotlyModule) return;

    for (const def of chartDefs) {
      const el = document.getElementById(def.id);
      if (!el) continue;

      const layout = {
        ...darkLayout,
        yaxis: { ...darkLayout.yaxis },
      };
      if (def.yRange) {
        layout.yaxis.range = def.yRange;
        layout.yaxis.autorange = false;
      }

      plotlyModule.newPlot(
        el,
        [
          {
            x: [],
            y: [],
            type: "scattergl",
            mode: "lines",
            line: { color: def.color, width: 1.5 },
            hovertemplate: `${def.title}: %{y:.3f}<br>Step: %{x}<extra></extra>`,
          },
        ],
        layout,
        {
          responsive: true,
          displayModeBar: false,
          staticPlot: false,
          scrollZoom: false,
        }
      );

      chartPointCounts.set(def.id, 0);
    }

    chartsReady = true;
  }

  /**
   * Update charts when metricsHistory changes.
   * Uses Plotly.extendTraces for efficient append, with periodic
   * full redraws when the rolling window is exceeded.
   */
  $effect(() => {
    if (!chartsReady || !plotlyModule) return;

    const history = store.metricsHistory;
    if (!history || history.length === 0) return;

    // Only process new entries since last update
    const newLen = history.length;
    if (newLen <= lastHistoryLen) return;

    const newEntries = history.slice(lastHistoryLen);
    lastHistoryLen = newLen;

    for (const def of chartDefs) {
      const el = document.getElementById(def.id);
      if (!el) continue;

      const newX = [];
      const newY = [];
      for (const entry of newEntries) {
        const val = entry[def.metricKey];
        if (val !== undefined && val !== null) {
          newX.push(entry.step);
          newY.push(val);
        }
      }

      if (newX.length === 0) continue;

      const currentCount = chartPointCounts.get(def.id) || 0;
      const totalAfter = currentCount + newX.length;

      if (totalAfter > MAX_POINTS) {
        // Full redraw with trimmed data — extract all data and trim
        const allX = [];
        const allY = [];
        for (const entry of history) {
          const val = entry[def.metricKey];
          if (val !== undefined && val !== null) {
            allX.push(entry.step);
            allY.push(val);
          }
        }
        // Keep only the last MAX_POINTS
        const startIdx = Math.max(0, allX.length - MAX_POINTS);
        const trimmedX = allX.slice(startIdx);
        const trimmedY = allY.slice(startIdx);

        plotlyModule.react(
          el,
          [
            {
              x: trimmedX,
              y: trimmedY,
              type: "scattergl",
              mode: "lines",
              line: { color: def.color, width: 1.5 },
              hovertemplate: `${def.title}: %{y:.3f}<br>Step: %{x}<extra></extra>`,
            },
          ],
          el.layout,
          { responsive: true, displayModeBar: false }
        );
        chartPointCounts.set(def.id, trimmedX.length);
      } else {
        // Efficient extend
        plotlyModule.extendTraces(el, { x: [newX], y: [newY] }, [0]);
        chartPointCounts.set(def.id, totalAfter);
      }
    }
  });

  // Current metric values for the summary cards
  let currentMetrics = $derived(store.metrics || {});

  /**
   * Format a metric value for display.
   * @param {number|undefined} v
   * @param {number} decimals
   * @returns {string}
   */
  function fmt(v, decimals = 2) {
    if (v === undefined || v === null) return "--";
    return Number(v).toFixed(decimals);
  }
</script>

<div class="metrics-panel">
  <h2 class="panel-title">Metrics</h2>

  <div class="charts-container">
    {#each chartDefs as def}
      <div class="chart-wrapper">
        <div class="chart-header">
          <span class="chart-label">{def.title}</span>
          <span class="chart-value" style="color: {def.color}">
            {fmt(currentMetrics[def.metricKey], def.metricKey === "population_size" ? 0 : 3)}
          </span>
          <span class="info-icon" title={def.help}>i</span>
        </div>
        <div class="chart-area" id={def.id}></div>
      </div>
    {/each}
  </div>

  <div class="summary-row">
    <div class="summary-item">
      <span class="summary-label">Alive</span>
      <span class="summary-value">{store.aliveCount}</span>
    </div>
    <div class="summary-item">
      <span class="summary-label">Max</span>
      <span class="summary-value">{store.maxAgents}</span>
    </div>
    <div class="summary-item">
      <span class="summary-label">FPS</span>
      <span class="summary-value">{store.metricsHistory.length > 1 ? "30" : "--"}</span>
    </div>
  </div>
</div>

<style>
  .metrics-panel {
    padding: 12px;
    border-bottom: 1px solid #1a1a3e;
    overflow-y: auto;
    flex: 1;
  }

  .panel-title {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #666;
    margin: 0 0 10px 0;
  }

  .charts-container {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .chart-wrapper {
    background: rgba(255, 255, 255, 0.02);
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.04);
    padding: 8px 8px 0 8px;
  }

  .chart-header {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 2px;
  }

  .chart-label {
    font-size: 11px;
    color: #999;
    flex: 1;
  }

  .chart-value {
    font-size: 13px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }

  .info-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.06);
    font-size: 9px;
    color: #666;
    font-style: italic;
    cursor: help;
    flex-shrink: 0;
  }

  .chart-area {
    width: 100%;
    height: 80px;
  }

  /* Override Plotly's internal styles for dark theme */
  .chart-area :global(.plotly) {
    width: 100% !important;
  }

  .summary-row {
    display: flex;
    justify-content: space-between;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid rgba(255, 255, 255, 0.04);
  }

  .summary-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
  }

  .summary-label {
    font-size: 10px;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .summary-value {
    font-size: 13px;
    font-weight: 600;
    color: #ccc;
    font-variant-numeric: tabular-nums;
  }
</style>
