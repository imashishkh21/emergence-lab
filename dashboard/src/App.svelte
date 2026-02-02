<script>
  import { createTrainingStore } from "./stores/training.svelte.js";
  import Header from "./lib/Header.svelte";
  import AgentCanvas from "./lib/AgentCanvas.svelte";
  import MetricsPanel from "./lib/MetricsPanel.svelte";
  import ControlPanel from "./lib/ControlPanel.svelte";
  import LineagePanel from "./lib/LineagePanel.svelte";
  import ReplayControls from "./lib/ReplayControls.svelte";
  import HelpSystem from "./lib/HelpSystem.svelte";
  import GlossaryPanel from "./lib/GlossaryPanel.svelte";
  import StatusBar from "./lib/StatusBar.svelte";
  import AlertSystem from "./lib/AlertSystem.svelte";

  const store = createTrainingStore();
  let glossaryOpen = $state(false);

  // Auto-connect on mount
  $effect(() => {
    store.connect();
    return () => store.disconnect();
  });
</script>

<div class="app">
  <Header {store} />
  <StatusBar {store} />

  <main class="content">
    <div class="canvas-area">
      <AgentCanvas {store} />
    </div>

    <div class="sidebar">
      <HelpSystem {store} showGlossary={() => glossaryOpen = true} />
      <MetricsPanel {store} />
      <LineagePanel {store} />
      <ControlPanel {store} />
      <ReplayControls {store} />
    </div>
  </main>
</div>

<GlossaryPanel open={glossaryOpen} onclose={() => glossaryOpen = false} />
<AlertSystem {store} />

<style>
  :global(body) {
    margin: 0;
    padding: 0;
    background: #0a0a1a;
    color: #e0e0e0;
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
      sans-serif;
    overflow: hidden;
    height: 100vh;
  }

  :global(*) {
    box-sizing: border-box;
  }

  :global(#app) {
    height: 100vh;
    display: flex;
    flex-direction: column;
  }

  .app {
    display: flex;
    flex-direction: column;
    height: 100vh;
  }

  .content {
    display: flex;
    flex: 1;
    overflow: hidden;
    gap: 0;
  }

  .canvas-area {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #0d0d22;
    position: relative;
    min-width: 0;
  }

  .sidebar {
    width: 340px;
    min-width: 340px;
    display: flex;
    flex-direction: column;
    background: #111128;
    border-left: 1px solid #1a1a3e;
    overflow-y: auto;
  }
</style>
