# Emergence Lab Dashboard

Real-time visualization dashboard for the Emergence Lab multi-agent simulation.

## Setup

```bash
npm install
npm run dev
# Opens http://localhost:5173
```

## Connecting to Training

The dashboard connects to a FastAPI WebSocket server that streams training data.

```bash
# Terminal 1: Start the server (with mock data for development)
python -m src.server.main --mock

# Terminal 2: Start the dashboard
cd dashboard && npm run dev
```

For real training data, start the server without `--mock` and run training separately.

## Architecture

- **Svelte 5** with runes (`$state`, `$derived`, `$effect`) for reactive state
- **Pixi.js v8** (WebGPU with WebGL fallback) for agent canvas rendering
- **Plotly.js** (WebGL `scattergl`) for live metric charts
- **msgpack-lite** for decoding binary WebSocket frames

### Key Files

| File | Description |
|------|-------------|
| `src/stores/training.svelte.js` | WebSocket client, MessagePack decoder, reactive state store |
| `src/lib/renderer.js` | Pixi.js rendering engine (field heatmap, food, agents, trails) |
| `src/lib/AgentCanvas.svelte` | Canvas wrapper with trail toggle and color legend |
| `src/lib/MetricsPanel.svelte` | 4 live Plotly.js charts (reward, divergence, specialization, population) |
| `src/lib/ControlPanel.svelte` | Pause/resume, speed, mutation rate, diversity bonus controls |
| `src/lib/LineagePanel.svelte` | Family tree SVG visualization |
| `src/lib/HelpSystem.svelte` | "What's happening?" summary + onboarding tour |
| `src/lib/GlossaryPanel.svelte` | Searchable glossary of all technical terms |
| `src/lib/StatusBar.svelte` | Connection status, training mode, health indicators |
| `src/lib/AlertSystem.svelte` | Toast notifications for important events |
| `src/lib/ReplayControls.svelte` | Session playback with timeline scrubber |
| `src/lib/Tooltip.svelte` | Reusable tooltip with viewport-aware positioning |
| `static/glossary.json` | Glossary term definitions |

## Building for Production

```bash
npm run build
# Output in dist/
```

## WebSocket Protocol

The dashboard communicates with the server via binary MessagePack frames at ~30fps:

```
Frame {
  step: int,
  positions: Float32Array (max_agents * 2),
  alive: Uint8Array (max_agents),
  energy: Float32Array (max_agents),
  food: Float32Array (num_food * 2),
  field: Float32Array (H * W * C),
  clusters: Int8Array (max_agents),
  training_mode: string,
  metrics: { reward, total_loss, entropy, population, ... }
}
```

Commands are sent as JSON text messages: `{ type: "pause" | "resume" | "set_speed" | "set_param", ... }`.
