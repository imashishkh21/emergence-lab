<script>
  /**
   * Reusable tooltip component.
   *
   * Shows a styled popup on hover with explanatory text.
   * Positions itself above or below the trigger element based on
   * available viewport space.
   *
   * Usage:
   *   <Tooltip text="Explanation here">
   *     <span class="info-icon">i</span>
   *   </Tooltip>
   */
  let { text = "", position = "above", children } = $props();

  let visible = $state(false);
  let triggerEl = $state(null);
  let tooltipEl = $state(null);
  let posStyle = $state("");

  function show() {
    visible = true;
    requestAnimationFrame(updatePosition);
  }

  function hide() {
    visible = false;
  }

  function updatePosition() {
    if (!triggerEl || !tooltipEl) return;
    const triggerRect = triggerEl.getBoundingClientRect();
    const tooltipRect = tooltipEl.getBoundingClientRect();

    // Determine if tooltip should be above or below
    const spaceAbove = triggerRect.top;
    const spaceBelow = window.innerHeight - triggerRect.bottom;
    const preferAbove = position === "above" ? spaceAbove > tooltipRect.height + 8 : false;
    const actualAbove = preferAbove || (position === "above" && spaceAbove > spaceBelow);

    // Center horizontally on trigger, clamp to viewport
    let left = triggerRect.left + triggerRect.width / 2 - tooltipRect.width / 2;
    left = Math.max(8, Math.min(left, window.innerWidth - tooltipRect.width - 8));

    if (actualAbove) {
      const top = triggerRect.top - tooltipRect.height - 6;
      posStyle = `top: ${top}px; left: ${left}px;`;
    } else {
      const top = triggerRect.bottom + 6;
      posStyle = `top: ${top}px; left: ${left}px;`;
    }
  }
</script>

<span
  class="tooltip-trigger"
  bind:this={triggerEl}
  onmouseenter={show}
  onmouseleave={hide}
  onfocusin={show}
  onfocusout={hide}
  role="button"
  tabindex="0"
>
  {#if children}
    {@render children()}
  {:else}
    <span class="default-icon">i</span>
  {/if}
</span>

{#if visible && text}
  <div
    class="tooltip-popup"
    bind:this={tooltipEl}
    style={posStyle}
    role="tooltip"
  >
    {text}
  </div>
{/if}

<style>
  .tooltip-trigger {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    cursor: help;
  }

  .default-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.08);
    color: #666;
    font-size: 10px;
    font-style: italic;
    flex-shrink: 0;
    transition: background 0.15s, color 0.15s;
  }

  .tooltip-trigger:hover .default-icon {
    background: rgba(255, 255, 255, 0.15);
    color: #aaa;
  }

  .tooltip-popup {
    position: fixed;
    z-index: 10000;
    max-width: 280px;
    padding: 8px 12px;
    background: #1e1e3a;
    border: 1px solid #2a2a5e;
    border-radius: 6px;
    color: #ccc;
    font-size: 12px;
    line-height: 1.5;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
    pointer-events: none;
    animation: tooltip-in 0.12s ease-out;
  }

  @keyframes tooltip-in {
    from {
      opacity: 0;
      transform: translateY(4px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
</style>
