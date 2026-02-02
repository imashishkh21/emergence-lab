<script>
  /**
   * Help system with two features:
   *
   * 1. "What's happening?" panel — plain-English summary of the current
   *    simulation state, generated from live metrics.
   *
   * 2. Onboarding tour — first-time users get a guided walkthrough
   *    of the dashboard elements.
   */
  let { store, showGlossary } = $props();

  let tourActive = $state(false);
  let tourStep = $state(0);
  let dismissed = $state(false);

  // Show onboarding only once (check localStorage)
  $effect(() => {
    try {
      if (typeof localStorage !== "undefined") {
        const seen = localStorage.getItem("emergence-lab-onboarded");
        if (!seen) {
          tourActive = true;
        } else {
          dismissed = true;
        }
      }
    } catch {
      // localStorage may not be available
    }
  });

  const tourSteps = [
    {
      title: "Welcome to Emergence Lab!",
      text: "This dashboard lets you watch artificial creatures learn, evolve, and develop specialized roles in real-time. Let's take a quick tour.",
      target: null,
    },
    {
      title: "The Canvas",
      text: "This is where the action happens. Colored dots are agents (creatures). Green dots are food. The blue heatmap shows the shared field — a kind of 'pheromone' trail agents can read and write.",
      target: "canvas-area",
    },
    {
      title: "Metrics Charts",
      text: "These charts track how the simulation is going: Are agents getting better? Are they developing different brains? Are roles emerging? Hover the (i) icons for explanations.",
      target: "metrics-panel",
    },
    {
      title: "Controls",
      text: "Pause, resume, change speed, and tweak parameters here. Try increasing the Diversity Bonus to push agents toward different strategies!",
      target: "control-panel",
    },
    {
      title: "You're ready!",
      text: "Click the (?) button anytime for help, or the book icon to open the Glossary. Watch for color changes in the agents — that means different behavioral groups are forming!",
      target: null,
    },
  ];

  function nextTourStep() {
    if (tourStep < tourSteps.length - 1) {
      tourStep++;
    } else {
      closeTour();
    }
  }

  function prevTourStep() {
    if (tourStep > 0) {
      tourStep--;
    }
  }

  function closeTour() {
    tourActive = false;
    dismissed = true;
    try {
      if (typeof localStorage !== "undefined") {
        localStorage.setItem("emergence-lab-onboarded", "true");
      }
    } catch {
      // ignore
    }
  }

  function restartTour() {
    tourStep = 0;
    tourActive = true;
  }

  // --- "What's happening?" summary ---

  let summary = $derived(buildSummary(store));

  /**
   * Generate a plain-English summary of the current simulation state.
   * @param {object} s - The training store
   * @returns {string}
   */
  function buildSummary(s) {
    const parts = [];
    const alive = s.aliveCount;
    const max = s.maxAgents;
    const m = s.metrics || {};

    if (!s.connected) {
      return "Not connected to the simulation server. Start the server with: python -m src.server.main --mock";
    }

    if (s.step === 0) {
      return "Simulation is starting up. Agents are being initialized...";
    }

    // Population status
    if (alive > 0 && max > 0) {
      const pct = Math.round((alive / max) * 100);
      if (pct >= 90) {
        parts.push(`${alive} agents are alive (thriving population).`);
      } else if (pct >= 50) {
        parts.push(`${alive} of ${max} agents are alive (stable population).`);
      } else if (pct >= 25) {
        parts.push(`${alive} of ${max} agents are alive (declining — food may be scarce).`);
      } else {
        parts.push(`Only ${alive} agents survive out of ${max} (population struggling!).`);
      }
    }

    // Specialization
    const spec = m.specialization_score;
    if (spec !== undefined && spec !== null) {
      if (spec >= 0.7) {
        parts.push("Clear behavioral roles have emerged — agents are specialized!");
      } else if (spec >= 0.4) {
        parts.push("Some role differentiation is developing. Groups are forming.");
      } else if (spec > 0) {
        parts.push("Agents are still mostly similar. Specialization hasn't emerged yet.");
      }
    }

    // Divergence
    const div = m.weight_divergence;
    if (div !== undefined && div !== null) {
      if (div >= 0.1) {
        parts.push("Agent brains have diverged significantly — different strategies exist.");
      } else if (div >= 0.03) {
        parts.push("Agent brains are starting to differ from each other.");
      }
    }

    // Transfer entropy
    const te = m.transfer_entropy;
    if (te !== undefined && te !== null && te > 0.05) {
      parts.push("Agents appear to be influencing each other's behavior (coordination detected).");
    }

    // Division of labor
    const dol = m.division_of_labor;
    if (dol !== undefined && dol !== null && dol > 0.3) {
      parts.push("Division of labor is increasing — agents are splitting into different task groups.");
    }

    // Phase transition
    const pt = m.phase_transition;
    if (pt !== undefined && pt !== null && pt > 0) {
      parts.push("A phase transition was just detected! The system may be reorganizing.");
    }

    // Training mode
    if (s.trainingMode === "evolve") {
      parts.push("Currently in evolution mode — natural selection is driving changes.");
    } else if (s.paused) {
      parts.push("Simulation is paused.");
    }

    return parts.length > 0
      ? parts.join(" ")
      : "Training is running. Watching for emergence...";
  }
</script>

<!-- What's Happening Panel -->
<div class="whats-happening">
  <div class="wh-header">
    <span class="wh-label">What's happening?</span>
    <button
      class="wh-tour-btn"
      onclick={restartTour}
      title="Restart the guided tour"
    >
      ?
    </button>
    <button
      class="wh-glossary-btn"
      onclick={() => showGlossary?.()}
      title="Open glossary — look up any term"
    >
      &#x1F4D6;
    </button>
  </div>
  <p class="wh-summary">{summary}</p>
</div>

<!-- Onboarding Tour Overlay -->
{#if tourActive}
  <div class="tour-overlay">
    <div class="tour-card">
      <div class="tour-step-indicator">
        {tourStep + 1} / {tourSteps.length}
      </div>
      <h3 class="tour-title">{tourSteps[tourStep].title}</h3>
      <p class="tour-text">{tourSteps[tourStep].text}</p>
      <div class="tour-actions">
        {#if tourStep > 0}
          <button class="tour-btn tour-btn-secondary" onclick={prevTourStep}>
            Back
          </button>
        {/if}
        <button class="tour-btn tour-btn-skip" onclick={closeTour}>
          Skip
        </button>
        <button class="tour-btn tour-btn-primary" onclick={nextTourStep}>
          {tourStep < tourSteps.length - 1 ? "Next" : "Got it!"}
        </button>
      </div>
    </div>
  </div>
{/if}

<style>
  /* What's happening panel */
  .whats-happening {
    padding: 10px 12px;
    border-bottom: 1px solid #1a1a3e;
    background: rgba(233, 69, 96, 0.03);
  }

  .wh-header {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 6px;
  }

  .wh-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #e94560;
    flex: 1;
  }

  .wh-tour-btn,
  .wh-glossary-btn {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    border: 1px solid #2a2a4e;
    background: rgba(255, 255, 255, 0.04);
    color: #888;
    font-size: 12px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.15s;
    padding: 0;
    font-family: inherit;
  }

  .wh-tour-btn:hover,
  .wh-glossary-btn:hover {
    background: rgba(255, 255, 255, 0.1);
    color: #ccc;
    border-color: rgba(233, 69, 96, 0.3);
  }

  .wh-summary {
    font-size: 12px;
    color: #aaa;
    line-height: 1.5;
    margin: 0;
  }

  /* Tour overlay */
  .tour-overlay {
    position: fixed;
    inset: 0;
    z-index: 9500;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    animation: fade-in 0.2s ease-out;
  }

  @keyframes fade-in {
    from { opacity: 0; }
    to { opacity: 1; }
  }

  .tour-card {
    background: #151530;
    border: 1px solid #2a2a5e;
    border-radius: 12px;
    padding: 24px;
    max-width: 420px;
    width: 90vw;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    animation: card-in 0.2s ease-out;
  }

  @keyframes card-in {
    from {
      opacity: 0;
      transform: scale(0.95) translateY(10px);
    }
    to {
      opacity: 1;
      transform: scale(1) translateY(0);
    }
  }

  .tour-step-indicator {
    font-size: 11px;
    color: #666;
    margin-bottom: 8px;
    font-variant-numeric: tabular-nums;
  }

  .tour-title {
    font-size: 16px;
    font-weight: 700;
    color: #e94560;
    margin: 0 0 8px 0;
  }

  .tour-text {
    font-size: 13px;
    color: #bbb;
    line-height: 1.6;
    margin: 0 0 16px 0;
  }

  .tour-actions {
    display: flex;
    gap: 8px;
    justify-content: flex-end;
  }

  .tour-btn {
    padding: 6px 16px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    border: 1px solid transparent;
    font-family: inherit;
    transition: all 0.15s;
  }

  .tour-btn-primary {
    background: #e94560;
    color: white;
    border-color: #e94560;
  }

  .tour-btn-primary:hover {
    background: #d63e56;
  }

  .tour-btn-secondary {
    background: rgba(255, 255, 255, 0.05);
    color: #aaa;
    border-color: #2a2a4e;
  }

  .tour-btn-secondary:hover {
    background: rgba(255, 255, 255, 0.1);
    color: #ddd;
  }

  .tour-btn-skip {
    background: none;
    color: #666;
    border: none;
    margin-right: auto;
  }

  .tour-btn-skip:hover {
    color: #aaa;
  }
</style>
