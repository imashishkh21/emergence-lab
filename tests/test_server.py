"""Tests for the WebSocket server and TrainingBridge.

Tests cover:
- Frame creation and packing (MessagePack serialization)
- TrainingBridge publish/consume with rate limiting
- TrainingBridge command queue (bidirectional communication)
- FastAPI app creation and endpoints
- WebSocket streaming of frames
- Mock data generation
"""

import asyncio
import json
import time

import msgpack
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.testclient import TestClient as StarletteTestClient

from src.server.main import (
    _generate_mock_frame,
    _handle_command,
    create_app,
    get_bridge,
    set_bridge,
)
from src.server.streaming import (
    Frame,
    TrainingBridge,
    _pack_array,
    pack_frame,
)


# ---------------------------------------------------------------------------
# Frame tests
# ---------------------------------------------------------------------------


class TestFrame:
    """Tests for the Frame dataclass."""

    def test_frame_creation(self) -> None:
        frame = Frame(
            step=100,
            positions=np.zeros((32, 2), dtype=np.float32),
            alive=np.ones(32, dtype=bool),
            energy=np.full(32, 100.0, dtype=np.float32),
            food_positions=np.zeros((10, 2), dtype=np.float32),
            food_collected=np.zeros(10, dtype=bool),
            field_values=np.zeros((20, 20, 4), dtype=np.float32),
            cluster_labels=None,
            metrics={"mean_reward": 1.5},
        )
        assert frame.step == 100
        assert frame.timestamp > 0
        assert frame.cluster_labels is None

    def test_frame_with_clusters(self) -> None:
        frame = Frame(
            step=0,
            positions=np.zeros((8, 2), dtype=np.float32),
            alive=np.ones(8, dtype=bool),
            energy=np.zeros(8, dtype=np.float32),
            food_positions=np.zeros((5, 2), dtype=np.float32),
            food_collected=np.zeros(5, dtype=bool),
            field_values=np.zeros((10, 10, 4), dtype=np.float32),
            cluster_labels=np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int8),
            metrics={},
        )
        assert frame.cluster_labels is not None
        assert frame.cluster_labels.shape == (8,)

    def test_frame_custom_timestamp(self) -> None:
        frame = Frame(
            step=0,
            positions=np.zeros((2, 2), dtype=np.float32),
            alive=np.ones(2, dtype=bool),
            energy=np.zeros(2, dtype=np.float32),
            food_positions=np.zeros((1, 2), dtype=np.float32),
            food_collected=np.zeros(1, dtype=bool),
            field_values=np.zeros((5, 5, 1), dtype=np.float32),
            cluster_labels=None,
            metrics={},
            timestamp=12345.0,
        )
        assert frame.timestamp == 12345.0


# ---------------------------------------------------------------------------
# Packing tests
# ---------------------------------------------------------------------------


class TestPackFrame:
    """Tests for MessagePack serialization."""

    def _make_frame(self, max_agents: int = 8) -> Frame:
        return Frame(
            step=42,
            positions=np.random.rand(max_agents, 2).astype(np.float32),
            alive=np.ones(max_agents, dtype=bool),
            energy=np.full(max_agents, 100.0, dtype=np.float32),
            food_positions=np.random.rand(5, 2).astype(np.float32),
            food_collected=np.zeros(5, dtype=bool),
            field_values=np.random.rand(10, 10, 4).astype(np.float32),
            cluster_labels=np.zeros(max_agents, dtype=np.int8),
            metrics={"mean_reward": 5.0, "population_size": 8.0},
        )

    def test_pack_returns_bytes(self) -> None:
        frame = self._make_frame()
        packed = pack_frame(frame)
        assert isinstance(packed, bytes)
        assert len(packed) > 0

    def test_packed_is_valid_msgpack(self) -> None:
        frame = self._make_frame()
        packed = pack_frame(frame)
        data = msgpack.unpackb(packed, raw=False)
        assert isinstance(data, dict)

    def test_packed_has_required_keys(self) -> None:
        frame = self._make_frame()
        packed = pack_frame(frame)
        data = msgpack.unpackb(packed, raw=False)
        required = {
            "step",
            "timestamp",
            "positions",
            "alive",
            "energy",
            "food_positions",
            "food_collected",
            "field",
            "metrics",
            "clusters",
        }
        assert required.issubset(set(data.keys()))

    def test_packed_without_clusters(self) -> None:
        frame = self._make_frame()
        frame.cluster_labels = None
        packed = pack_frame(frame)
        data = msgpack.unpackb(packed, raw=False)
        assert "clusters" not in data

    def test_packed_step_matches(self) -> None:
        frame = self._make_frame()
        packed = pack_frame(frame)
        data = msgpack.unpackb(packed, raw=False)
        assert data["step"] == 42

    def test_packed_metrics_preserved(self) -> None:
        frame = self._make_frame()
        packed = pack_frame(frame)
        data = msgpack.unpackb(packed, raw=False)
        assert data["metrics"]["mean_reward"] == 5.0
        assert data["metrics"]["population_size"] == 8.0

    def test_packed_array_reconstructable(self) -> None:
        """Verify packed arrays can be reconstructed to numpy."""
        frame = self._make_frame()
        packed = pack_frame(frame)
        data = msgpack.unpackb(packed, raw=False)

        pos_data = data["positions"]
        arr = np.frombuffer(pos_data["data"], dtype=pos_data["dtype"]).reshape(
            pos_data["shape"]
        )
        assert arr.shape == (8, 2)
        assert arr.dtype == np.float32
        np.testing.assert_array_almost_equal(arr, frame.positions, decimal=5)

    def test_pack_array_helper(self) -> None:
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        packed = _pack_array(arr)
        assert packed["shape"] == [3]
        assert packed["dtype"] == "<f4"  # float32
        assert isinstance(packed["data"], bytes)


# ---------------------------------------------------------------------------
# TrainingBridge tests
# ---------------------------------------------------------------------------


class TestTrainingBridge:
    """Tests for the TrainingBridge class."""

    def _make_frame(self, step: int = 0) -> Frame:
        return Frame(
            step=step,
            positions=np.zeros((8, 2), dtype=np.float32),
            alive=np.ones(8, dtype=bool),
            energy=np.zeros(8, dtype=np.float32),
            food_positions=np.zeros((5, 2), dtype=np.float32),
            food_collected=np.zeros(5, dtype=bool),
            field_values=np.zeros((5, 5, 1), dtype=np.float32),
            cluster_labels=None,
            metrics={"mean_reward": 1.0},
        )

    def test_init_defaults(self) -> None:
        bridge = TrainingBridge()
        assert bridge.target_fps == 30.0
        assert bridge.frame_count == 0
        assert bridge.paused is False
        assert bridge.get_latest_frame() is None

    def test_publish_and_get(self) -> None:
        bridge = TrainingBridge(target_fps=1000)  # High fps to avoid rate limiting
        frame = self._make_frame(step=10)
        accepted = bridge.publish_frame(frame)
        assert accepted is True
        assert bridge.frame_count == 1
        latest = bridge.get_latest_frame()
        assert latest is not None
        assert latest.step == 10

    def test_rate_limiting(self) -> None:
        bridge = TrainingBridge(target_fps=10)  # 100ms interval
        frame1 = self._make_frame(step=1)
        frame2 = self._make_frame(step=2)
        bridge.publish_frame(frame1)
        accepted = bridge.publish_frame(frame2)
        # Second frame should be rejected (too fast)
        assert accepted is False
        assert bridge.frame_count == 1

    def test_rate_limiting_allows_after_interval(self) -> None:
        bridge = TrainingBridge(target_fps=100)  # 10ms interval
        frame1 = self._make_frame(step=1)
        frame2 = self._make_frame(step=2)
        bridge.publish_frame(frame1)
        time.sleep(0.015)  # Wait > 10ms
        accepted = bridge.publish_frame(frame2)
        assert accepted is True
        assert bridge.frame_count == 2

    def test_latest_frame_replaces(self) -> None:
        bridge = TrainingBridge(target_fps=1000)
        bridge.publish_frame(self._make_frame(step=1))
        time.sleep(0.002)
        bridge.publish_frame(self._make_frame(step=2))
        latest = bridge.get_latest_frame()
        assert latest is not None
        assert latest.step == 2

    def test_get_latest_packed(self) -> None:
        bridge = TrainingBridge(target_fps=1000)
        assert bridge.get_latest_packed() is None
        bridge.publish_frame(self._make_frame(step=5))
        packed = bridge.get_latest_packed()
        assert packed is not None
        assert isinstance(packed, bytes)
        data = msgpack.unpackb(packed, raw=False)
        assert data["step"] == 5

    def test_pause_state(self) -> None:
        bridge = TrainingBridge()
        assert bridge.paused is False
        bridge.paused = True
        assert bridge.paused is True
        bridge.paused = False
        assert bridge.paused is False


# ---------------------------------------------------------------------------
# Command queue tests
# ---------------------------------------------------------------------------


class TestCommandQueue:
    """Tests for the bidirectional command queue."""

    def test_push_and_pop(self) -> None:
        bridge = TrainingBridge()
        bridge.push_command({"type": "pause"})
        bridge.push_command({"type": "set_param", "key": "mutation_std", "value": 0.05})
        commands = bridge.pop_commands()
        assert len(commands) == 2
        assert commands[0]["type"] == "pause"
        assert commands[1]["key"] == "mutation_std"

    def test_pop_clears_queue(self) -> None:
        bridge = TrainingBridge()
        bridge.push_command({"type": "pause"})
        bridge.pop_commands()
        assert bridge.pop_commands() == []

    def test_empty_pop(self) -> None:
        bridge = TrainingBridge()
        assert bridge.pop_commands() == []

    def test_handle_pause_command(self) -> None:
        bridge = TrainingBridge()
        _handle_command(bridge, {"type": "pause"})
        assert bridge.paused is True
        cmds = bridge.pop_commands()
        assert len(cmds) == 1

    def test_handle_resume_command(self) -> None:
        bridge = TrainingBridge()
        bridge.paused = True
        _handle_command(bridge, {"type": "resume"})
        assert bridge.paused is False

    def test_handle_set_param_command(self) -> None:
        bridge = TrainingBridge()
        _handle_command(
            bridge, {"type": "set_param", "key": "mutation_std", "value": 0.1}
        )
        cmds = bridge.pop_commands()
        assert len(cmds) == 1
        assert cmds[0]["value"] == 0.1

    def test_handle_unknown_command(self) -> None:
        bridge = TrainingBridge()
        _handle_command(bridge, {"type": "unknown_cmd"})
        cmds = bridge.pop_commands()
        assert len(cmds) == 1


# ---------------------------------------------------------------------------
# FastAPI app tests
# ---------------------------------------------------------------------------


class TestFastAPIApp:
    """Tests for FastAPI HTTP endpoints."""

    def test_health_endpoint(self) -> None:
        bridge = TrainingBridge()
        app = create_app(bridge)
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "emergence-lab-server"

    def test_config_endpoint(self) -> None:
        bridge = TrainingBridge(target_fps=15.0)
        app = create_app(bridge)
        client = TestClient(app)
        response = client.get("/config")
        assert response.status_code == 200
        data = response.json()
        assert data["target_fps"] == 15.0
        assert data["frame_count"] == 0
        assert data["paused"] is False

    def test_cors_headers(self) -> None:
        bridge = TrainingBridge()
        app = create_app(bridge)
        client = TestClient(app)
        response = client.get("/", headers={"Origin": "http://localhost:5173"})
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers


# ---------------------------------------------------------------------------
# WebSocket tests
# ---------------------------------------------------------------------------


class TestWebSocket:
    """Tests for the WebSocket training stream."""

    def test_websocket_connect(self) -> None:
        bridge = TrainingBridge(target_fps=1000)
        app = create_app(bridge)
        client = TestClient(app)

        # Publish a frame before connecting
        frame = Frame(
            step=99,
            positions=np.zeros((4, 2), dtype=np.float32),
            alive=np.ones(4, dtype=bool),
            energy=np.zeros(4, dtype=np.float32),
            food_positions=np.zeros((2, 2), dtype=np.float32),
            food_collected=np.zeros(2, dtype=bool),
            field_values=np.zeros((5, 5, 1), dtype=np.float32),
            cluster_labels=None,
            metrics={"mean_reward": 3.0},
        )
        bridge.publish_frame(frame)

        with client.websocket_connect("/ws/training") as ws:
            data = ws.receive_bytes()
            decoded = msgpack.unpackb(data, raw=False)
            assert decoded["step"] == 99
            assert decoded["metrics"]["mean_reward"] == 3.0

    def test_websocket_receives_commands(self) -> None:
        bridge = TrainingBridge(target_fps=1000)
        app = create_app(bridge)
        client = TestClient(app)

        # Publish a frame so there's data to send
        frame = Frame(
            step=1,
            positions=np.zeros((4, 2), dtype=np.float32),
            alive=np.ones(4, dtype=bool),
            energy=np.zeros(4, dtype=np.float32),
            food_positions=np.zeros((2, 2), dtype=np.float32),
            food_collected=np.zeros(2, dtype=bool),
            field_values=np.zeros((5, 5, 1), dtype=np.float32),
            cluster_labels=None,
            metrics={},
        )
        bridge.publish_frame(frame)

        with client.websocket_connect("/ws/training") as ws:
            # Receive the initial frame
            ws.receive_bytes()
            # Send a pause command
            ws.send_text(json.dumps({"type": "pause"}))
            # Give the server a moment to process
            time.sleep(0.1)
            assert bridge.paused is True


# ---------------------------------------------------------------------------
# Mock data tests
# ---------------------------------------------------------------------------


class TestMockData:
    """Tests for mock data generation."""

    def test_mock_frame_creation(self) -> None:
        frame = _generate_mock_frame(step=0)
        assert isinstance(frame, Frame)
        assert frame.step == 0

    def test_mock_frame_shapes(self) -> None:
        frame = _generate_mock_frame(step=10, max_agents=16)
        assert frame.positions.shape == (16, 2)
        assert frame.alive.shape == (16,)
        assert frame.energy.shape == (16,)
        assert frame.food_positions.shape == (10, 2)
        assert frame.food_collected.shape == (10,)
        assert frame.field_values.shape == (20, 20, 4)
        assert frame.cluster_labels is not None
        assert frame.cluster_labels.shape == (16,)

    def test_mock_frame_different_steps(self) -> None:
        frame1 = _generate_mock_frame(step=0)
        frame2 = _generate_mock_frame(step=100)
        # Different seeds should produce different positions
        assert not np.array_equal(frame1.positions, frame2.positions)

    def test_mock_frame_metrics(self) -> None:
        frame = _generate_mock_frame(step=0)
        assert "mean_reward" in frame.metrics
        assert "population_size" in frame.metrics
        assert isinstance(frame.metrics["mean_reward"], float)

    def test_mock_frame_packable(self) -> None:
        frame = _generate_mock_frame(step=5)
        packed = pack_frame(frame)
        data = msgpack.unpackb(packed, raw=False)
        assert data["step"] == 5


# ---------------------------------------------------------------------------
# TrainingBridge.create_frame_from_state tests
# ---------------------------------------------------------------------------


class TestCreateFrameFromState:
    """Tests for creating frames from JAX-like state objects."""

    def test_create_frame_from_mock_state(self) -> None:
        """Test frame creation from a mock state that mimics batched EnvState."""

        class MockFieldState:
            def __init__(self) -> None:
                self.values = np.random.rand(4, 10, 10, 2).astype(np.float32)

        class MockState:
            def __init__(self) -> None:
                self.agent_positions = np.random.randint(
                    0, 20, size=(4, 8, 2)
                ).astype(np.int32)
                self.agent_alive = np.ones((4, 8), dtype=bool)
                self.agent_energy = np.full((4, 8), 100.0, dtype=np.float32)
                self.food_positions = np.random.randint(
                    0, 20, size=(4, 5, 2)
                ).astype(np.int32)
                self.food_collected = np.zeros((4, 5), dtype=bool)
                self.field_state = MockFieldState()

        bridge = TrainingBridge()
        state = MockState()
        metrics = {"mean_reward": 2.5, "total_loss": 0.1}
        frame = bridge.create_frame_from_state(state, metrics, step=500)

        assert frame.step == 500
        assert frame.positions.shape == (8, 2)
        assert frame.alive.shape == (8,)
        assert frame.energy.shape == (8,)
        assert frame.food_positions.shape == (5, 2)
        assert frame.food_collected.shape == (5,)
        assert frame.field_values.shape == (10, 10, 2)
        assert frame.metrics["mean_reward"] == 2.5
        assert frame.cluster_labels is None

    def test_create_frame_with_clusters(self) -> None:
        class MockFieldState:
            def __init__(self) -> None:
                self.values = np.zeros((1, 5, 5, 1), dtype=np.float32)

        class MockState:
            def __init__(self) -> None:
                self.agent_positions = np.zeros((1, 4, 2), dtype=np.int32)
                self.agent_alive = np.ones((1, 4), dtype=bool)
                self.agent_energy = np.zeros((1, 4), dtype=np.float32)
                self.food_positions = np.zeros((1, 2, 2), dtype=np.int32)
                self.food_collected = np.zeros((1, 2), dtype=bool)
                self.field_state = MockFieldState()

        bridge = TrainingBridge()
        labels = np.array([0, 1, 0, 1], dtype=np.int8)
        frame = bridge.create_frame_from_state(
            MockState(), {"x": 1.0}, step=10, cluster_labels=labels
        )
        assert frame.cluster_labels is not None
        np.testing.assert_array_equal(frame.cluster_labels, labels)

    def test_create_frame_handles_non_float_metrics(self) -> None:
        class MockFieldState:
            def __init__(self) -> None:
                self.values = np.zeros((1, 5, 5, 1), dtype=np.float32)

        class MockState:
            def __init__(self) -> None:
                self.agent_positions = np.zeros((1, 4, 2), dtype=np.int32)
                self.agent_alive = np.ones((1, 4), dtype=bool)
                self.agent_energy = np.zeros((1, 4), dtype=np.float32)
                self.food_positions = np.zeros((1, 2, 2), dtype=np.int32)
                self.food_collected = np.zeros((1, 2), dtype=bool)
                self.field_state = MockFieldState()

        bridge = TrainingBridge()
        # Metrics with a non-convertible value
        metrics = {"good": 1.5, "bad": "not_a_number"}
        frame = bridge.create_frame_from_state(MockState(), metrics, step=0)
        assert "good" in frame.metrics
        assert "bad" not in frame.metrics
