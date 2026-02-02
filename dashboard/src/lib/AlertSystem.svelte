<!--
  AlertSystem.svelte

  Toast notification system for important simulation events.
  Alerts appear in the bottom-right corner and auto-dismiss after 8 seconds.

  Alert types:
  - info: Phase transitions, general notifications
  - success: Milestones (specialization > 0.7, divergence > 0.1)
  - warning: Population declining, divergence dropping
  - danger: Population collapsing
-->
<script>
  let { store } = $props();

  const ICONS = {
    info: "\u26A1",      // Lightning bolt
    success: "\uD83C\uDF89", // Party popper
    warning: "\u26A0\uFE0F", // Warning sign
    danger: "\uD83D\uDEA8",  // Rotating light
  };
</script>

<div class="alert-container" role="log" aria-live="polite" aria-label="Notifications">
  {#each store.alerts as alert (alert.id)}
    <div
      class="alert-toast alert-{alert.type}"
      role="status"
    >
      <span class="alert-icon">{ICONS[alert.type] || ICONS.info}</span>
      <span class="alert-message">{alert.message}</span>
      <button
        class="alert-dismiss"
        onclick={() => store.dismissAlert(alert.id)}
        aria-label="Dismiss notification"
      >&times;</button>
    </div>
  {/each}
</div>

<style>
  .alert-container {
    position: fixed;
    bottom: 16px;
    right: 16px;
    z-index: 9000;
    display: flex;
    flex-direction: column-reverse;
    gap: 8px;
    pointer-events: none;
    max-height: 50vh;
    overflow: hidden;
  }

  .alert-toast {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 13px;
    line-height: 1.4;
    max-width: 380px;
    min-width: 260px;
    pointer-events: auto;
    animation: slide-in 0.3s ease-out;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
  }

  .alert-icon {
    font-size: 16px;
    flex-shrink: 0;
  }

  .alert-message {
    flex: 1;
    color: #e0e0e0;
  }

  .alert-dismiss {
    background: none;
    border: none;
    color: #999;
    cursor: pointer;
    font-size: 18px;
    padding: 0 2px;
    line-height: 1;
    flex-shrink: 0;
    transition: color 0.15s;
  }

  .alert-dismiss:hover {
    color: #fff;
  }

  /* Alert type styles */
  .alert-info {
    background: rgba(78, 205, 196, 0.12);
    border: 1px solid rgba(78, 205, 196, 0.35);
    border-left: 3px solid #4ecdc4;
  }

  .alert-success {
    background: rgba(76, 175, 80, 0.12);
    border: 1px solid rgba(76, 175, 80, 0.35);
    border-left: 3px solid #4caf50;
  }

  .alert-warning {
    background: rgba(255, 152, 0, 0.12);
    border: 1px solid rgba(255, 152, 0, 0.35);
    border-left: 3px solid #ff9800;
  }

  .alert-danger {
    background: rgba(233, 69, 96, 0.12);
    border: 1px solid rgba(233, 69, 96, 0.35);
    border-left: 3px solid #e94560;
  }

  @keyframes slide-in {
    from {
      opacity: 0;
      transform: translateX(60px);
    }
    to {
      opacity: 1;
      transform: translateX(0);
    }
  }
</style>
