<script>
  /**
   * Agent lineage visualization panel.
   *
   * Shows parent-child relationships as a tree/graph:
   * - Node = agent, edge = parent-child
   * - Color by fitness (energy) or cluster (behavioral group)
   * - Animate births (new nodes) and deaths (faded nodes)
   * - Click agent to highlight in main canvas
   * - Hover shows agent details
   */
  import Tooltip from "./Tooltip.svelte";

  let { store } = $props();

  let colorMode = $state("cluster"); // "cluster" | "fitness"
  let hoveredAgent = $state(null);

  // Cluster colors matching renderer.js / AgentCanvas
  const CLUSTER_COLORS = [
    "#e94560", "#0f3460", "#533483", "#16c79a",
    "#f5a623", "#11999e", "#e84545", "#903749",
  ];

  /**
   * Build lineage tree data from current frame.
   * Returns array of agent nodes with computed positions.
   */
  let agents = $derived(buildAgents(store));
  let edges = $derived(buildEdges(agents));
  let dominantLineages = $derived(store.lineageData?.dominant_lineages ?? []);

  function buildAgents(s) {
    if (!s.alive || !s.agentIds || !s.parentIds) return [];

    const maxAgents = s.alive.length;
    const result = [];

    for (let i = 0; i < maxAgents; i++) {
      const isAlive = s.alive[i] > 0;
      const agentId = s.agentIds[i];
      const parentId = s.parentIds[i];
      const birthStep = s.birthSteps ? s.birthSteps[i] : 0;
      const energyVal = s.energy ? s.energy[i] : 0;
      const cluster = s.clusters ? s.clusters[i] : 0;
      const age = s.step - birthStep;

      result.push({
        index: i,
        agentId,
        parentId,
        birthStep,
        energy: energyVal,
        cluster,
        alive: isAlive,
        age,
        selected: s.selectedAgentIndex === i,
      });
    }

    return result;
  }

  function buildEdges(agentsList) {
    if (!agentsList || agentsList.length === 0) return [];

    // Map agentId → index for quick lookup
    const idToIndex = new Map();
    for (const a of agentsList) {
      idToIndex.set(a.agentId, a.index);
    }

    const result = [];
    for (const a of agentsList) {
      if (a.parentId >= 0 && idToIndex.has(a.parentId)) {
        result.push({
          parentIndex: idToIndex.get(a.parentId),
          childIndex: a.index,
          parentAlive: agentsList[idToIndex.get(a.parentId)]?.alive ?? false,
          childAlive: a.alive,
        });
      }
    }
    return result;
  }

  /**
   * Compute a simple tree layout. Groups agents by generation depth.
   * Returns Map<index, {x, y}> positions.
   */
  let layout = $derived(computeLayout(agents, edges));

  function computeLayout(agentsList, edgesList) {
    if (!agentsList || agentsList.length === 0) return new Map();

    // Find roots (parentId == -1) and alive agents
    const aliveAgents = agentsList.filter((a) => a.alive);
    if (aliveAgents.length === 0) return new Map();

    // Group by generation depth using BFS from roots
    const childrenMap = new Map();
    const parentMap = new Map();
    for (const a of aliveAgents) {
      parentMap.set(a.index, a.parentId);
    }
    for (const e of edgesList) {
      if (!childrenMap.has(e.parentIndex)) childrenMap.set(e.parentIndex, []);
      childrenMap.get(e.parentIndex).push(e.childIndex);
    }

    // Compute depth for each alive agent
    const aliveSet = new Set(aliveAgents.map((a) => a.index));
    const depths = new Map();
    const idToIndex = new Map();
    for (const a of agentsList) {
      idToIndex.set(a.agentId, a.index);
    }

    for (const a of aliveAgents) {
      let depth = 0;
      let current = a;
      while (current && current.parentId >= 0 && idToIndex.has(current.parentId)) {
        depth++;
        const pidx = idToIndex.get(current.parentId);
        current = agentsList[pidx];
        if (depth > 20) break; // Safety
      }
      depths.set(a.index, depth);
    }

    // Group by depth
    const depthGroups = new Map();
    for (const a of aliveAgents) {
      const d = depths.get(a.index) ?? 0;
      if (!depthGroups.has(d)) depthGroups.set(d, []);
      depthGroups.get(d).push(a.index);
    }

    // Assign x,y positions
    const positions = new Map();
    const maxDepth = Math.max(...depthGroups.keys(), 0);
    const svgWidth = 300;
    const svgHeight = Math.max(160, (maxDepth + 1) * 50 + 20);

    for (const [depth, indices] of depthGroups) {
      const count = indices.length;
      for (let i = 0; i < count; i++) {
        const x = (svgWidth / (count + 1)) * (i + 1);
        const y = 20 + depth * 45;
        positions.set(indices[i], { x, y });
      }
    }

    return positions;
  }

  let svgHeight = $derived(
    layout.size > 0
      ? Math.max(160, Math.max(...Array.from(layout.values()).map((p) => p.y)) + 30)
      : 160
  );

  function getAgentColor(agent) {
    if (!agent.alive) return "#333";
    if (colorMode === "fitness") {
      const t = Math.min(agent.energy / 200, 1);
      const r = Math.round(255 * (1 - t));
      const g = Math.round(255 * t);
      return `rgb(${r}, ${g}, 50)`;
    }
    return CLUSTER_COLORS[agent.cluster % CLUSTER_COLORS.length];
  }

  function getAgentRadius(agent) {
    if (!agent.alive) return 4;
    return 5 + (agent.energy / 200) * 4;
  }

  function handleAgentClick(agent) {
    store.selectAgent(agent.selected ? -1 : agent.index);
  }

  function handleAgentHover(agent) {
    hoveredAgent = agent;
  }

  function handleAgentLeave() {
    hoveredAgent = null;
  }

  function formatAge(steps) {
    if (steps < 1000) return `${steps} steps`;
    return `${(steps / 1000).toFixed(1)}k steps`;
  }
</script>

<div class="lineage-panel">
  <div class="panel-header">
    <span class="panel-title">Lineage</span>
    <Tooltip text="Shows the family tree of agents. Lines connect parents to children. Click an agent to highlight it. Thick families = successful lineages." />
    <div class="color-toggle">
      <button
        class="toggle-btn"
        class:active={colorMode === "cluster"}
        onclick={() => (colorMode = "cluster")}
      >
        Cluster
      </button>
      <button
        class="toggle-btn"
        class:active={colorMode === "fitness"}
        onclick={() => (colorMode = "fitness")}
      >
        Fitness
      </button>
    </div>
  </div>

  <!-- Dominant Lineages Summary -->
  {#if dominantLineages.length > 0}
    <div class="dominant-lineages">
      <span class="section-label">Top Lineages</span>
      {#each dominantLineages.slice(0, 3) as lin}
        <div class="lineage-row">
          <span class="lineage-id">#{lin.ancestor_id}</span>
          <div class="lineage-bar-bg">
            <div
              class="lineage-bar"
              style="width: {Math.min(100, (lin.descendants / Math.max(1, store.maxAgents)) * 100)}%"
            ></div>
          </div>
          <span class="lineage-count">{lin.descendants}</span>
        </div>
      {/each}
    </div>
  {/if}

  <!-- Tree Visualization -->
  <div class="tree-container">
    {#if agents.length === 0 || layout.size === 0}
      <div class="empty-state">
        <span>No lineage data yet</span>
        <span class="empty-hint">Waiting for reproduction events...</span>
      </div>
    {:else}
      <svg
        class="tree-svg"
        width="100%"
        viewBox="0 0 300 {svgHeight}"
        preserveAspectRatio="xMidYMin meet"
      >
        <!-- Edges (parent-child lines) -->
        {#each edges as edge}
          {@const parentPos = layout.get(edge.parentIndex)}
          {@const childPos = layout.get(edge.childIndex)}
          {#if parentPos && childPos}
            <line
              x1={parentPos.x}
              y1={parentPos.y}
              x2={childPos.x}
              y2={childPos.y}
              class="edge-line"
              class:edge-dead={!edge.parentAlive || !edge.childAlive}
            />
          {/if}
        {/each}

        <!-- Agent nodes -->
        {#each agents as agent}
          {#if agent.alive}
            {@const pos = layout.get(agent.index)}
            {#if pos}
              <g
                transform="translate({pos.x}, {pos.y})"
                class="agent-node"
                class:agent-selected={agent.selected}
                onclick={() => handleAgentClick(agent)}
                onmouseenter={() => handleAgentHover(agent)}
                onmouseleave={handleAgentLeave}
                role="button"
                tabindex="0"
                onkeydown={(e) => e.key === "Enter" && handleAgentClick(agent)}
              >
                <!-- Glow for selected -->
                {#if agent.selected}
                  <circle
                    r={getAgentRadius(agent) + 4}
                    fill="none"
                    stroke="#fff"
                    stroke-width="1.5"
                    opacity="0.5"
                  />
                {/if}
                <!-- Main circle -->
                <circle
                  r={getAgentRadius(agent)}
                  fill={getAgentColor(agent)}
                  opacity={agent.alive ? 1 : 0.3}
                />
                <!-- Agent ID label -->
                <text
                  y={getAgentRadius(agent) + 10}
                  text-anchor="middle"
                  class="agent-label"
                >
                  #{agent.agentId}
                </text>
              </g>
            {/if}
          {/if}
        {/each}
      </svg>
    {/if}
  </div>

  <!-- Hover tooltip -->
  {#if hoveredAgent}
    <div class="agent-tooltip">
      <div class="tooltip-row"><strong>Agent #{hoveredAgent.agentId}</strong></div>
      {#if hoveredAgent.parentId >= 0}
        <div class="tooltip-row">Child of #{hoveredAgent.parentId}</div>
      {:else}
        <div class="tooltip-row">Original agent</div>
      {/if}
      <div class="tooltip-row">Alive for {formatAge(hoveredAgent.age)}</div>
      <div class="tooltip-row">Energy: {hoveredAgent.energy.toFixed(0)}</div>
      {#if store.clusters}
        <div class="tooltip-row">Group: {hoveredAgent.cluster}</div>
      {/if}
    </div>
  {/if}

  <!-- Stats footer -->
  <div class="lineage-stats">
    {#if store.lineageData}
      <span class="stat">Depth: {store.lineageData.max_depth ?? 0}</span>
      <span class="stat">Births: {store.lineageData.total_births ?? 0}</span>
    {/if}
    <span class="stat">Alive: {store.aliveCount}</span>
  </div>
</div>

<style>
  .lineage-panel {
    padding: 8px 12px;
    border-bottom: 1px solid #1a1a3e;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .panel-header {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .panel-title {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #888;
    flex: 1;
  }

  .color-toggle {
    display: flex;
    gap: 2px;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 4px;
    padding: 1px;
  }

  .toggle-btn {
    font-size: 9px;
    padding: 2px 6px;
    border: none;
    border-radius: 3px;
    background: transparent;
    color: #666;
    cursor: pointer;
    font-family: inherit;
    transition: all 0.15s;
  }

  .toggle-btn.active {
    background: rgba(233, 69, 96, 0.15);
    color: #e94560;
  }

  .toggle-btn:hover:not(.active) {
    color: #aaa;
  }

  /* Dominant lineages */
  .dominant-lineages {
    display: flex;
    flex-direction: column;
    gap: 3px;
  }

  .section-label {
    font-size: 9px;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .lineage-row {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .lineage-id {
    font-size: 10px;
    color: #999;
    width: 28px;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }

  .lineage-bar-bg {
    flex: 1;
    height: 4px;
    background: rgba(255, 255, 255, 0.04);
    border-radius: 2px;
    overflow: hidden;
  }

  .lineage-bar {
    height: 100%;
    background: linear-gradient(90deg, #e94560, #f5a623);
    border-radius: 2px;
    transition: width 0.3s ease;
  }

  .lineage-count {
    font-size: 10px;
    color: #888;
    width: 20px;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }

  /* Tree visualization */
  .tree-container {
    min-height: 120px;
    max-height: 250px;
    overflow-y: auto;
    overflow-x: hidden;
    border-radius: 6px;
    background: rgba(0, 0, 0, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.03);
  }

  .tree-container::-webkit-scrollbar {
    width: 4px;
  }

  .tree-container::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 2px;
  }

  .tree-svg {
    display: block;
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 120px;
    color: #555;
    font-size: 12px;
    gap: 4px;
  }

  .empty-hint {
    font-size: 10px;
    color: #444;
  }

  /* SVG elements */
  .edge-line {
    stroke: rgba(255, 255, 255, 0.12);
    stroke-width: 1;
  }

  .edge-dead {
    stroke: rgba(255, 255, 255, 0.04);
    stroke-dasharray: 2 2;
  }

  .agent-node {
    cursor: pointer;
    transition: opacity 0.15s;
  }

  .agent-node:hover circle {
    filter: brightness(1.3);
  }

  .agent-selected circle {
    filter: brightness(1.3);
  }

  .agent-label {
    font-size: 7px;
    fill: #888;
    font-family: inherit;
  }

  /* Hover tooltip */
  .agent-tooltip {
    position: absolute;
    bottom: 40px;
    left: 12px;
    right: 12px;
    background: rgba(21, 21, 48, 0.95);
    border: 1px solid #2a2a5e;
    border-radius: 6px;
    padding: 6px 8px;
    pointer-events: none;
    z-index: 10;
  }

  .tooltip-row {
    font-size: 11px;
    color: #bbb;
    line-height: 1.4;
  }

  .tooltip-row strong {
    color: #e94560;
  }

  /* Stats footer */
  .lineage-stats {
    display: flex;
    gap: 12px;
    justify-content: center;
  }

  .stat {
    font-size: 10px;
    color: #666;
    font-variant-numeric: tabular-nums;
  }
</style>
