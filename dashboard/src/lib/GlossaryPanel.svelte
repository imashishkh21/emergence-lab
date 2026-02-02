<script>
  /**
   * Glossary panel — opens as a slide-out panel with all terms explained.
   *
   * Loads terms from static/glossary.json and displays them in a
   * searchable, scrollable list. Each term has a simple explanation
   * and an analogy to make it accessible to non-technical users.
   */
  let { open = false, onclose } = $props();

  let terms = $state([]);
  let searchQuery = $state("");
  let loaded = $state(false);

  // Load glossary terms
  $effect(() => {
    if (open && !loaded) {
      fetch("/glossary.json")
        .then((r) => r.json())
        .then((data) => {
          terms = data.terms || [];
          loaded = true;
        })
        .catch(() => {
          // Fallback inline terms if fetch fails
          terms = [];
          loaded = true;
        });
    }
  });

  let filteredTerms = $derived(
    searchQuery.trim() === ""
      ? terms
      : terms.filter(
          (t) =>
            t.term.toLowerCase().includes(searchQuery.toLowerCase()) ||
            t.simple.toLowerCase().includes(searchQuery.toLowerCase())
        )
  );

  function handleKeydown(e) {
    if (e.key === "Escape") {
      onclose?.();
    }
  }
</script>

{#if open}
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div class="glossary-backdrop" onclick={() => onclose?.()} onkeydown={handleKeydown}>
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div class="glossary-panel" onclick={(e) => e.stopPropagation()} onkeydown={handleKeydown}>
      <div class="panel-header">
        <h2 class="panel-title">Glossary</h2>
        <span class="panel-subtitle">Click any term to learn more</span>
        <button class="close-btn" onclick={() => onclose?.()} aria-label="Close glossary">
          &times;
        </button>
      </div>

      <div class="search-box">
        <input
          type="text"
          placeholder="Search terms..."
          bind:value={searchQuery}
          class="search-input"
        />
      </div>

      <div class="terms-list">
        {#each filteredTerms as t (t.term)}
          <div class="term-card">
            <h3 class="term-name">{t.term}</h3>
            <p class="term-explanation">{t.simple}</p>
            {#if t.analogy}
              <p class="term-analogy">{t.analogy}</p>
            {/if}
          </div>
        {:else}
          <div class="empty-state">
            {#if !loaded}
              Loading glossary...
            {:else if searchQuery}
              No terms match "{searchQuery}"
            {:else}
              No glossary terms available.
            {/if}
          </div>
        {/each}
      </div>
    </div>
  </div>
{/if}

<style>
  .glossary-backdrop {
    position: fixed;
    inset: 0;
    z-index: 9000;
    background: rgba(0, 0, 0, 0.5);
    animation: fade-in 0.15s ease-out;
  }

  @keyframes fade-in {
    from { opacity: 0; }
    to { opacity: 1; }
  }

  .glossary-panel {
    position: fixed;
    top: 0;
    right: 0;
    width: 380px;
    max-width: 90vw;
    height: 100vh;
    background: #111128;
    border-left: 1px solid #2a2a5e;
    display: flex;
    flex-direction: column;
    animation: slide-in 0.2s ease-out;
    box-shadow: -8px 0 32px rgba(0, 0, 0, 0.4);
  }

  @keyframes slide-in {
    from { transform: translateX(100%); }
    to { transform: translateX(0); }
  }

  .panel-header {
    display: flex;
    align-items: baseline;
    gap: 8px;
    padding: 16px 16px 12px;
    border-bottom: 1px solid #1a1a3e;
    flex-wrap: wrap;
  }

  .panel-title {
    font-size: 14px;
    font-weight: 700;
    color: #e94560;
    margin: 0;
  }

  .panel-subtitle {
    font-size: 11px;
    color: #666;
    flex: 1;
  }

  .close-btn {
    background: none;
    border: none;
    color: #666;
    font-size: 20px;
    cursor: pointer;
    padding: 0 4px;
    line-height: 1;
    transition: color 0.15s;
  }

  .close-btn:hover {
    color: #ccc;
  }

  .search-box {
    padding: 12px 16px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
  }

  .search-input {
    width: 100%;
    padding: 8px 12px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid #2a2a4e;
    border-radius: 6px;
    color: #ddd;
    font-size: 13px;
    font-family: inherit;
    outline: none;
    transition: border-color 0.15s;
  }

  .search-input::placeholder {
    color: #555;
  }

  .search-input:focus {
    border-color: rgba(233, 69, 96, 0.4);
  }

  .terms-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px 16px;
  }

  .term-card {
    padding: 12px;
    margin-bottom: 8px;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 6px;
    transition: border-color 0.15s;
  }

  .term-card:hover {
    border-color: rgba(233, 69, 96, 0.2);
  }

  .term-name {
    font-size: 13px;
    font-weight: 600;
    color: #e0e0e0;
    margin: 0 0 4px 0;
  }

  .term-explanation {
    font-size: 12px;
    color: #aaa;
    line-height: 1.5;
    margin: 0 0 4px 0;
  }

  .term-analogy {
    font-size: 11px;
    color: #888;
    font-style: italic;
    margin: 0;
    padding-left: 8px;
    border-left: 2px solid rgba(233, 69, 96, 0.3);
  }

  .empty-state {
    text-align: center;
    color: #666;
    font-size: 13px;
    padding: 32px 16px;
  }
</style>
