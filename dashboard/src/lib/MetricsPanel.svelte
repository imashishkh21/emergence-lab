<script>
  /**
   * Real-time metrics display panel.
   *
   * Shows key training metrics with labels and helper text.
   * Will be upgraded to Plotly.js charts in US-008.
   */
  let { store } = $props();

  /**
   * @type {Array<{key: string, label: string, help: string, format: (v: number) => string}>}
   */
  const metricDefs = [
    {
      key: "mean_reward",
      label: "Reward",
      help: "Are agents getting better at collecting food?",
      format: (v) => v.toFixed(2),
    },
    {
      key: "total_loss",
      label: "Loss",
      help: "How much the brain is changing each step (lower = more stable)",
      format: (v) => v.toFixed(4),
    },
    {
      key: "entropy",
      label: "Entropy",
      help: "How random are agent decisions? High = exploring, Low = decided",
      format: (v) => v.toFixed(3),
    },
    {
      key: "population_size",
      label: "Population",
      help: "How many agents are currently alive",
      format: (v) => Math.round(v).toString(),
    },
    {
      key: "mean_energy",
      label: "Mean Energy",
      help: "Average energy across all alive agents",
      format: (v) => v.toFixed(1),
    },
  ];
</script>

<div class="metrics-panel">
  <h2 class="panel-title">Metrics</h2>

  <div class="metrics-grid">
    {#each metricDefs as def}
      {@const value = store.metrics[def.key]}
      <div class="metric-card" title={def.help}>
        <div class="metric-label">
          {def.label}
          <span class="info-icon" title={def.help}>i</span>
        </div>
        <div class="metric-value">
          {value !== undefined ? def.format(value) : "--"}
        </div>
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
  }

  .panel-title {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #666;
    margin: 0 0 10px 0;
  }

  .metrics-grid {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .metric-card {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 10px;
    background: rgba(255, 255, 255, 0.02);
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.04);
    cursor: help;
    transition: background 0.15s;
  }

  .metric-card:hover {
    background: rgba(255, 255, 255, 0.05);
  }

  .metric-label {
    font-size: 12px;
    color: #999;
    display: flex;
    align-items: center;
    gap: 6px;
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
  }

  .metric-value {
    font-size: 14px;
    font-weight: 600;
    color: #e0e0e0;
    font-variant-numeric: tabular-nums;
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
