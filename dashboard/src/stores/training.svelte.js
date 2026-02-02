/**
 * Reactive state store for training data.
 *
 * Uses Svelte 5 runes ($state) for reactive updates.
 * Manages WebSocket connection with auto-reconnect,
 * MessagePack decoding, and frame buffering.
 */
import msgpack from "msgpack-lite";

/**
 * Unpack a serialized numpy array from the server's format.
 * Server sends: { shape: number[], dtype: string, data: Uint8Array }
 *
 * @param {Object} packed - The packed array object
 * @returns {{ data: Float32Array|Int8Array|Uint8Array, shape: number[] }}
 */
function unpackArray(packed) {
  if (!packed || !packed.data || !packed.shape) {
    return { data: new Float32Array(0), shape: [0] };
  }

  const buffer = packed.data.buffer
    ? packed.data.buffer
    : new Uint8Array(packed.data).buffer;
  const offset = packed.data.byteOffset || 0;
  const dtype = packed.dtype || "<f4";
  const shape = packed.shape;

  let data;
  if (dtype === "<f4" || dtype === "float32") {
    data = new Float32Array(buffer, offset, shape.reduce((a, b) => a * b, 1));
  } else if (dtype === "<i4" || dtype === "int32") {
    data = new Int32Array(buffer, offset, shape.reduce((a, b) => a * b, 1));
  } else if (dtype === "<i1" || dtype === "int8") {
    data = new Int8Array(buffer, offset, shape.reduce((a, b) => a * b, 1));
  } else if (dtype === "|u1" || dtype === "uint8" || dtype === "<u1") {
    data = new Uint8Array(buffer, offset, shape.reduce((a, b) => a * b, 1));
  } else {
    // Fallback: treat as float32
    data = new Float32Array(buffer, offset, shape.reduce((a, b) => a * b, 1));
  }

  return { data, shape };
}

/**
 * Create the training data store.
 *
 * @param {string} serverUrl - WebSocket server URL (default: ws://localhost:8765/ws/training)
 * @returns {Object} Reactive store with training state and connection methods
 */
export function createTrainingStore(serverUrl = "ws://localhost:8765/ws/training") {
  // Connection state
  let connected = $state(false);
  let reconnecting = $state(false);
  let error = $state(null);

  // Frame data
  let step = $state(0);
  let positions = $state(null); // Float32Array (max_agents * 2)
  let alive = $state(null); // Uint8Array (max_agents)
  let energy = $state(null); // Float32Array (max_agents)
  let foodPositions = $state(null); // Float32Array (num_food * 2)
  let foodCollected = $state(null); // Uint8Array (num_food)
  let field = $state(null); // Float32Array (H * W * C)
  let clusters = $state(null); // Int8Array (max_agents)
  let metrics = $state({});
  let timestamp = $state(0);

  // Lineage data
  let agentIds = $state(null); // Int32Array (max_agents)
  let parentIds = $state(null); // Int32Array (max_agents)
  let birthSteps = $state(null); // Int32Array (max_agents)
  let lineageData = $state(null); // { dominant_lineages, max_depth, total_births }
  let selectedAgentIndex = $state(-1); // Index of agent selected in canvas/lineage panel

  // Shapes for interpreting flat arrays
  let positionsShape = $state([0, 2]);
  let fieldShape = $state([20, 20, 4]);
  let foodShape = $state([0, 2]);

  // Metrics history for charts (rolling window)
  let metricsHistory = $state([]);
  const MAX_HISTORY = 1000;

  // Training control state
  let paused = $state(false);
  let speedMultiplier = $state(1);
  let trainingMode = $state("gradient"); // "gradient" | "evolve" | "paused"

  // Connection internals
  let ws = null;
  let reconnectTimer = null;
  let reconnectAttempts = 0;
  const MAX_RECONNECT_DELAY = 10000;

  /**
   * Process a decoded MessagePack frame from the server.
   */
  function processFrame(frame) {
    step = frame.step || 0;
    timestamp = frame.timestamp || 0;

    if (frame.training_mode) {
      trainingMode = frame.training_mode;
    }

    if (frame.positions) {
      const unpacked = unpackArray(frame.positions);
      positions = unpacked.data;
      positionsShape = unpacked.shape;
    }

    if (frame.alive) {
      alive = unpackArray(frame.alive).data;
    }

    if (frame.energy) {
      energy = unpackArray(frame.energy).data;
    }

    if (frame.food_positions) {
      const unpacked = unpackArray(frame.food_positions);
      foodPositions = unpacked.data;
      foodShape = unpacked.shape;
    }

    if (frame.food_collected) {
      foodCollected = unpackArray(frame.food_collected).data;
    }

    if (frame.field) {
      const unpacked = unpackArray(frame.field);
      field = unpacked.data;
      fieldShape = unpacked.shape;
    }

    if (frame.clusters) {
      clusters = unpackArray(frame.clusters).data;
    }

    if (frame.agent_ids) {
      agentIds = unpackArray(frame.agent_ids).data;
    }

    if (frame.parent_ids) {
      parentIds = unpackArray(frame.parent_ids).data;
    }

    if (frame.birth_steps) {
      birthSteps = unpackArray(frame.birth_steps).data;
    }

    if (frame.lineage_data) {
      lineageData = frame.lineage_data;
    }

    if (frame.metrics) {
      metrics = frame.metrics;
      // Append to history
      const entry = { step: frame.step, ...frame.metrics };
      metricsHistory = [...metricsHistory.slice(-(MAX_HISTORY - 1)), entry];
    }
  }

  /**
   * Connect to the WebSocket server.
   */
  function connect() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    try {
      ws = new WebSocket(serverUrl);
      ws.binaryType = "arraybuffer";

      ws.onopen = () => {
        connected = true;
        reconnecting = false;
        reconnectAttempts = 0;
        error = null;
        console.log("[Dashboard] Connected to server");
      };

      ws.onmessage = (event) => {
        try {
          const packed = new Uint8Array(event.data);
          const frame = msgpack.decode(packed);
          processFrame(frame);
        } catch (err) {
          console.error("[Dashboard] Frame decode error:", err);
        }
      };

      ws.onclose = (event) => {
        connected = false;
        ws = null;
        if (!event.wasClean) {
          scheduleReconnect();
        }
      };

      ws.onerror = (event) => {
        error = "Connection failed";
        console.error("[Dashboard] WebSocket error");
      };
    } catch (err) {
      error = err.message;
      scheduleReconnect();
    }
  }

  /**
   * Schedule a reconnection attempt with exponential backoff.
   */
  function scheduleReconnect() {
    if (reconnectTimer) return;
    reconnecting = true;
    reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(1.5, reconnectAttempts - 1), MAX_RECONNECT_DELAY);
    console.log(`[Dashboard] Reconnecting in ${Math.round(delay)}ms (attempt ${reconnectAttempts})`);
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      connect();
    }, delay);
  }

  /**
   * Disconnect from the server.
   */
  function disconnect() {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    reconnecting = false;
    if (ws) {
      ws.close(1000, "Dashboard closed");
      ws = null;
    }
    connected = false;
  }

  /**
   * Send a command to the server.
   * @param {Object} command - JSON command object, e.g. { type: "pause" }
   */
  function sendCommand(command) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(command));
    }
  }

  /**
   * Pause training.
   */
  function pause() {
    paused = true;
    sendCommand({ type: "pause" });
  }

  /**
   * Resume training.
   */
  function resume() {
    paused = false;
    sendCommand({ type: "resume" });
  }

  /**
   * Set simulation speed multiplier.
   * @param {number} value - Speed multiplier (0.25 to 16)
   */
  function setSpeed(value) {
    speedMultiplier = value;
    sendCommand({ type: "set_speed", value });
  }

  /**
   * Set a training parameter.
   * @param {string} key - Parameter name
   * @param {number} value - Parameter value
   */
  function setParam(key, value) {
    sendCommand({ type: "set_param", key, value });
  }

  // ---------------------------------------------------------------
  // Replay WebSocket (separate connection for /ws/replay)
  // ---------------------------------------------------------------

  let replayWs = null;
  let replayConnected = $state(false);
  let replayStatus = $state(null); // { position, total, step, playing, speed, progress, bookmarks, metadata }

  const replayUrl = serverUrl.replace("/ws/training", "/ws/replay");

  /**
   * Connect to the replay WebSocket.
   */
  function connectReplay() {
    if (replayWs && (replayWs.readyState === WebSocket.OPEN || replayWs.readyState === WebSocket.CONNECTING)) {
      return;
    }

    try {
      replayWs = new WebSocket(replayUrl);
      replayWs.binaryType = "arraybuffer";

      replayWs.onopen = () => {
        replayConnected = true;
        console.log("[Dashboard] Replay connected");
      };

      replayWs.onmessage = (event) => {
        try {
          if (typeof event.data === "string") {
            // JSON status message
            const msg = JSON.parse(event.data);
            if (msg.type === "status") {
              replayStatus = msg;
            } else if (msg.type === "error") {
              console.error("[Dashboard] Replay error:", msg.message);
            }
          } else {
            // Binary MessagePack frame — process like a live frame
            const packed = new Uint8Array(event.data);
            const frame = msgpack.decode(packed);
            processFrame(frame);
          }
        } catch (err) {
          console.error("[Dashboard] Replay frame decode error:", err);
        }
      };

      replayWs.onclose = () => {
        replayConnected = false;
        replayWs = null;
        replayStatus = null;
      };

      replayWs.onerror = () => {
        console.error("[Dashboard] Replay WebSocket error");
      };
    } catch (err) {
      console.error("[Dashboard] Replay connection failed:", err);
    }
  }

  /**
   * Disconnect the replay WebSocket.
   */
  function disconnectReplay() {
    if (replayWs) {
      replayWs.close(1000, "Replay closed");
      replayWs = null;
    }
    replayConnected = false;
    replayStatus = null;
  }

  /**
   * Send a command to the replay WebSocket.
   * @param {Object} command - JSON command, e.g. { type: "play" }, { type: "seek", position: 100 }
   */
  function sendReplayCommand(command) {
    if (replayWs && replayWs.readyState === WebSocket.OPEN) {
      replayWs.send(JSON.stringify(command));
    }
  }

  // ---------------------------------------------------------------
  // Derived values
  // ---------------------------------------------------------------

  let aliveCount = $derived(
    alive ? Array.from(alive).filter((v) => v > 0).length : 0
  );

  let maxAgents = $derived(alive ? alive.length : 0);

  return {
    // Connection state
    get connected() { return connected; },
    get reconnecting() { return reconnecting; },
    get error() { return error; },

    // Frame data
    get step() { return step; },
    get positions() { return positions; },
    get positionsShape() { return positionsShape; },
    get alive() { return alive; },
    get energy() { return energy; },
    get foodPositions() { return foodPositions; },
    get foodShape() { return foodShape; },
    get foodCollected() { return foodCollected; },
    get field() { return field; },
    get fieldShape() { return fieldShape; },
    get clusters() { return clusters; },
    get metrics() { return metrics; },
    get timestamp() { return timestamp; },
    get metricsHistory() { return metricsHistory; },
    get agentIds() { return agentIds; },
    get parentIds() { return parentIds; },
    get birthSteps() { return birthSteps; },
    get lineageData() { return lineageData; },
    get selectedAgentIndex() { return selectedAgentIndex; },

    // Derived values
    get aliveCount() { return aliveCount; },
    get maxAgents() { return maxAgents; },

    // Training state
    get paused() { return paused; },
    get speedMultiplier() { return speedMultiplier; },
    get trainingMode() { return trainingMode; },

    // Replay state
    get replayConnected() { return replayConnected; },
    get replayStatus() { return replayStatus; },

    // Methods
    connect,
    disconnect,
    sendCommand,
    pause,
    resume,
    setSpeed,
    setParam,
    selectAgent(index) { selectedAgentIndex = index; },

    // Replay methods
    connectReplay,
    disconnectReplay,
    sendReplayCommand,
  };
}
