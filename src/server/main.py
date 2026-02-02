"""FastAPI WebSocket server for real-time training visualization.

Usage:
    # Start server standalone (streams mock data for testing):
    python -m src.server.main

    # Start server with training (integrated mode):
    # (See integration in src/training/train.py)

The server exposes:
    - GET  /           — Health check
    - GET  /config     — Current training config
    - WS   /ws/training — Binary MessagePack stream of training state at 30Hz
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState

from src.server.streaming import Frame, TrainingBridge, pack_frame

logger = logging.getLogger(__name__)

# Global bridge instance — set by the training loop or by start_server()
_bridge: TrainingBridge | None = None


def get_bridge() -> TrainingBridge:
    """Get the global TrainingBridge instance, creating one if needed."""
    global _bridge
    if _bridge is None:
        _bridge = TrainingBridge()
    return _bridge


def set_bridge(bridge: TrainingBridge) -> None:
    """Set the global TrainingBridge instance (called by training loop)."""
    global _bridge
    _bridge = bridge


def create_app(bridge: TrainingBridge | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        bridge: Optional TrainingBridge to use. If None, uses the global one.

    Returns:
        Configured FastAPI app with WebSocket endpoint.
    """
    if bridge is not None:
        set_bridge(bridge)

    app = FastAPI(
        title="Emergence Lab Dashboard Server",
        description="Real-time training visualization via WebSocket",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok", "service": "emergence-lab-server"}

    @app.get("/config")
    async def get_config() -> dict[str, Any]:
        """Return current server state info."""
        b = get_bridge()
        return {
            "target_fps": b.target_fps,
            "frame_count": b.frame_count,
            "paused": b.paused,
            "speed_multiplier": b.speed_multiplier,
            "training_mode": b.training_mode,
        }

    @app.websocket("/ws/training")
    async def training_stream(websocket: WebSocket) -> None:
        """Stream training state to dashboard clients.

        Protocol:
            - Server sends binary MessagePack frames at target_fps.
            - Client can send JSON commands:
              {"type": "pause"}, {"type": "resume"},
              {"type": "set_param", "key": "...", "value": ...}
        """
        await websocket.accept()
        b = get_bridge()
        interval = 1.0 / b.target_fps if b.target_fps > 0 else 1.0 / 30.0
        last_sent_step = -1

        logger.info("Dashboard client connected")

        try:
            while True:
                # Check for incoming commands (non-blocking)
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_text(), timeout=interval
                    )
                    try:
                        command = json.loads(data)
                        _handle_command(b, command)
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON command from client: %s", data)
                except asyncio.TimeoutError:
                    pass

                # Send latest frame if available and new
                frame = b.get_latest_frame()
                if frame is not None and frame.step != last_sent_step:
                    packed = pack_frame(frame)
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_bytes(packed)
                        last_sent_step = frame.step

        except WebSocketDisconnect:
            logger.info("Dashboard client disconnected")
        except Exception as e:
            logger.error("WebSocket error: %s", e)
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close(code=1011)

    return app


def _handle_command(bridge: TrainingBridge, command: dict[str, Any]) -> None:
    """Process a command from the dashboard client."""
    cmd_type = command.get("type")
    if cmd_type == "pause":
        bridge.paused = True
        bridge.push_command(command)
        logger.info("Training paused by dashboard")
    elif cmd_type == "resume":
        bridge.paused = False
        bridge.push_command(command)
        logger.info("Training resumed by dashboard")
    elif cmd_type == "set_speed":
        value = command.get("value", 1.0)
        bridge.speed_multiplier = float(value)
        bridge.push_command(command)
        logger.info("Speed multiplier set to %sx", bridge.speed_multiplier)
    elif cmd_type == "set_param":
        bridge.push_command(command)
        logger.info(
            "Parameter change requested: %s = %s",
            command.get("key"),
            command.get("value"),
        )
    else:
        bridge.push_command(command)
        logger.info("Unknown command forwarded: %s", cmd_type)


def _generate_mock_frame(
    step: int, max_agents: int = 32, training_mode: str = "gradient"
) -> Frame:
    """Generate a mock frame for testing the server without training."""
    rng = np.random.RandomState(step % 1000)
    grid_size = 20

    positions = rng.randint(0, grid_size, size=(max_agents, 2)).astype(np.float32)
    alive = np.zeros(max_agents, dtype=bool)
    alive[: max_agents // 2 + rng.randint(0, max_agents // 2)] = True
    energy = rng.uniform(10, 200, size=max_agents).astype(np.float32) * alive
    food_pos = rng.randint(0, grid_size, size=(10, 2)).astype(np.float32)
    food_col = rng.random(10) < 0.3
    field_vals = rng.random((grid_size, grid_size, 4)).astype(np.float32) * 0.5

    cluster_labels = np.zeros(max_agents, dtype=np.int8)
    cluster_labels[: max_agents // 2] = 0
    cluster_labels[max_agents // 2 :] = 1

    # Simulate gradually improving metrics over time
    progress = min(step / 500.0, 1.0)  # 0→1 over 500 steps
    metrics = {
        "mean_reward": float(rng.uniform(0, 3) + progress * 5),
        "total_loss": float(rng.uniform(0, 0.5) + (1 - progress) * 0.5),
        "entropy": float(rng.uniform(0.5, 1.5) + (1 - progress) * 0.5),
        "population_size": float(alive.sum()),
        "mean_energy": float(energy[alive].mean()) if alive.any() else 0.0,
        "weight_divergence": float(
            rng.uniform(0, 0.02) + progress * 0.12
        ),
        "specialization_score": float(
            max(0, rng.uniform(-0.05, 0.05) + progress * 0.75)
        ),
        "transfer_entropy": float(
            max(0, rng.uniform(-0.01, 0.01) + progress * 0.15)
        ),
        "te_density": float(
            max(0, min(1.0, rng.uniform(-0.05, 0.05) + progress * 0.6))
        ),
    }

    return Frame(
        step=step,
        positions=positions,
        alive=alive,
        energy=energy,
        food_positions=food_pos,
        food_collected=food_col,
        field_values=field_vals,
        cluster_labels=cluster_labels,
        metrics=metrics,
        training_mode=training_mode,
    )


async def _mock_training_loop(bridge: TrainingBridge) -> None:
    """Simulate a training loop by publishing mock frames."""
    step = 0
    # Simulate freeze-evolve: cycle between gradient and evolve modes
    gradient_steps = 200
    evolve_steps = 50
    cycle_length = gradient_steps + evolve_steps

    while True:
        if not bridge.paused:
            # Determine mock training mode based on step cycle
            cycle_pos = step % cycle_length
            mode = "evolve" if cycle_pos >= gradient_steps else "gradient"
            bridge.training_mode = mode

            frame = _generate_mock_frame(step, training_mode=mode)
            bridge.publish_frame(frame)
            step += max(1, int(bridge.speed_multiplier))
        else:
            bridge.training_mode = "paused"
        await asyncio.sleep(1.0 / 30.0)


def start_server(
    host: str = "0.0.0.0",
    port: int = 8765,
    bridge: TrainingBridge | None = None,
    mock: bool = False,
) -> None:
    """Start the FastAPI server (blocking).

    Args:
        host: Bind address.
        port: Port number.
        bridge: TrainingBridge instance. Created if None.
        mock: If True, generate mock training data.
    """
    import uvicorn

    if bridge is None:
        bridge = TrainingBridge()
    app = create_app(bridge)

    if mock:
        @app.on_event("startup")
        async def start_mock() -> None:
            asyncio.create_task(_mock_training_loop(bridge))

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Emergence Lab Dashboard Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8765, help="Port")
    parser.add_argument(
        "--mock", action="store_true", help="Generate mock data for testing"
    )
    args = parser.parse_args()

    print(f"Starting Emergence Lab server on {args.host}:{args.port}")
    if args.mock:
        print("Mock mode: generating fake training data")
    start_server(host=args.host, port=args.port, mock=args.mock)
