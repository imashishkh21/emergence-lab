"""Integration tests for the dashboard pipeline: training → server → WebSocket.

Tests the complete end-to-end flow:
1. Training loop publishes frames to TrainingBridge
2. FastAPI server streams frames over WebSocket
3. WebSocket client receives valid frame data
4. Commands from client affect training state
5. Help system data (glossary) is available

These tests do NOT require a browser (Playwright). They test the Python-side
integration using FastAPI's TestClient with WebSocket support.
"""

import json
import os
import threading
import time

import msgpack
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.server.main import _generate_mock_frame, create_app, _handle_command
from src.server.streaming import Frame, TrainingBridge, pack_frame


def _small_config():
    """Create a small config suitable for fast integration testing."""
    from src.configs import Config

    config = Config()
    config.train.num_envs = 2
    config.train.num_steps = 16
    config.train.total_steps = 500
    config.train.num_epochs = 2
    config.train.minibatch_size = 32
    config.env.num_agents = 4
    config.env.grid_size = 10
    config.env.num_food = 5
    config.env.max_steps = 20
    config.field.num_channels = 2
    config.agent.hidden_dims = (32, 32)
    config.log.wandb = False
    return config


@pytest.mark.timeout(120)
class TestTrainingToServer:
    """Test that training data flows correctly through bridge to WebSocket."""

    def test_training_publishes_to_bridge(self):
        """Train for a few steps and verify frames appear on the bridge."""
        from src.training.train import create_train_state, train_step

        config = _small_config()
        key = __import__("jax").random.PRNGKey(42)

        runner_state = create_train_state(config, key)
        bridge = TrainingBridge(target_fps=1000)  # High fps to avoid rate limiting

        # Run a few training updates, publishing frames after each
        for i in range(3):
            runner_state, metrics = train_step(runner_state, config)
            frame = bridge.create_frame_from_state(
                runner_state.env_state, metrics, step=i
            )
            bridge.publish_frame(frame)

        # Bridge should have the latest frame
        latest = bridge.get_latest_frame()
        assert latest is not None, "Bridge should have a frame after training"
        assert latest.step == 2  # 0-indexed, last was step=2
        assert latest.positions.shape[1] == 2  # (max_agents, 2)
        assert latest.alive.ndim == 1
        assert latest.energy.ndim == 1
        assert len(latest.metrics) > 0
        assert bridge.frame_count == 3

    def test_training_frames_stream_via_websocket(self):
        """Train, publish to bridge, and verify WebSocket receives the frame."""
        from src.training.train import create_train_state, train_step

        config = _small_config()
        key = __import__("jax").random.PRNGKey(99)

        runner_state = create_train_state(config, key)
        bridge = TrainingBridge(target_fps=1000)

        # Train and publish one frame
        runner_state, metrics = train_step(runner_state, config)
        frame = bridge.create_frame_from_state(
            runner_state.env_state, metrics, step=100
        )
        bridge.publish_frame(frame)

        # Create app and connect via WebSocket
        app = create_app(bridge)
        client = TestClient(app)

        with client.websocket_connect("/ws/training") as ws:
            data = ws.receive_bytes()
            decoded = msgpack.unpackb(data, raw=False)

            # Verify frame structure from real training data
            assert decoded["step"] == 100
            assert "positions" in decoded
            assert "alive" in decoded
            assert "energy" in decoded
            assert "food_positions" in decoded
            assert "food_collected" in decoded
            assert "field" in decoded
            assert "metrics" in decoded
            assert "timestamp" in decoded

            # Verify positions can be reconstructed
            pos_data = decoded["positions"]
            positions = np.frombuffer(
                pos_data["data"], dtype=pos_data["dtype"]
            ).reshape(pos_data["shape"])
            assert positions.ndim == 2
            assert positions.shape[1] == 2

            # Verify metrics contain training data
            m = decoded["metrics"]
            assert "mean_reward" in m or "total_loss" in m

    def test_training_metrics_are_real(self):
        """Verify that metrics from actual training are real numbers."""
        from src.training.train import create_train_state, train_step
        import jax.numpy as jnp

        config = _small_config()
        key = __import__("jax").random.PRNGKey(7)

        runner_state = create_train_state(config, key)
        bridge = TrainingBridge(target_fps=1000)

        runner_state, metrics = train_step(runner_state, config)
        frame = bridge.create_frame_from_state(
            runner_state.env_state, metrics, step=0
        )

        # All metrics should be finite floats
        for k, v in frame.metrics.items():
            assert isinstance(v, float), f"Metric {k} should be float, got {type(v)}"
            assert np.isfinite(v), f"Metric {k} is not finite: {v}"

        # Population metrics should be present
        assert "population_size" in frame.metrics
        assert frame.metrics["population_size"] >= 0

        # Alive mask should match population
        alive_count = int(frame.alive.sum())
        assert alive_count == int(frame.metrics["population_size"])


@pytest.mark.timeout(120)
class TestServerEndpoints:
    """Test HTTP and WebSocket endpoints with real training data."""

    def test_health_endpoint(self):
        """Health endpoint returns ok."""
        bridge = TrainingBridge()
        app = create_app(bridge)
        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_config_endpoint_reflects_state(self):
        """Config endpoint reflects bridge state changes."""
        bridge = TrainingBridge(target_fps=15.0)
        bridge.paused = True
        bridge.speed_multiplier = 4.0
        bridge.training_mode = "evolve"

        app = create_app(bridge)
        client = TestClient(app)
        resp = client.get("/config")
        data = resp.json()

        assert data["target_fps"] == 15.0
        assert data["paused"] is True
        assert data["speed_multiplier"] == 4.0
        assert data["training_mode"] == "evolve"

    def test_cors_allows_dashboard_origin(self):
        """CORS headers allow dashboard at localhost:5173."""
        bridge = TrainingBridge()
        app = create_app(bridge)
        client = TestClient(app)
        resp = client.get("/", headers={"Origin": "http://localhost:5173"})
        assert "access-control-allow-origin" in resp.headers


@pytest.mark.timeout(120)
class TestControlsAffectTraining:
    """Test that dashboard controls affect training state via WebSocket."""

    def test_pause_command_pauses_bridge(self):
        """Sending pause command via WebSocket pauses the bridge."""
        bridge = TrainingBridge(target_fps=1000)
        # Publish a frame so there's data
        bridge.publish_frame(_generate_mock_frame(step=0))

        app = create_app(bridge)
        client = TestClient(app)

        assert bridge.paused is False
        with client.websocket_connect("/ws/training") as ws:
            ws.receive_bytes()  # consume initial frame
            ws.send_text(json.dumps({"type": "pause"}))
            time.sleep(0.1)
            assert bridge.paused is True

    def test_resume_command_resumes_bridge(self):
        """Sending resume command via WebSocket resumes the bridge."""
        bridge = TrainingBridge(target_fps=1000)
        bridge.paused = True
        bridge.publish_frame(_generate_mock_frame(step=0))

        app = create_app(bridge)
        client = TestClient(app)

        with client.websocket_connect("/ws/training") as ws:
            ws.receive_bytes()
            ws.send_text(json.dumps({"type": "resume"}))
            time.sleep(0.1)
            assert bridge.paused is False

    def test_speed_command_changes_speed(self):
        """Sending set_speed command changes speed multiplier."""
        bridge = TrainingBridge(target_fps=1000)
        bridge.publish_frame(_generate_mock_frame(step=0))

        app = create_app(bridge)
        client = TestClient(app)

        with client.websocket_connect("/ws/training") as ws:
            ws.receive_bytes()
            ws.send_text(json.dumps({"type": "set_speed", "value": 4.0}))
            time.sleep(0.1)
            assert bridge.speed_multiplier == 4.0

    def test_set_param_command_queued(self):
        """Sending set_param command queues it for the training loop."""
        bridge = TrainingBridge(target_fps=1000)
        bridge.publish_frame(_generate_mock_frame(step=0))

        app = create_app(bridge)
        client = TestClient(app)

        with client.websocket_connect("/ws/training") as ws:
            ws.receive_bytes()
            ws.send_text(json.dumps({
                "type": "set_param",
                "key": "mutation_std",
                "value": 0.05,
            }))
            time.sleep(0.1)

        # Command should be in the queue
        cmds = bridge.pop_commands()
        assert any(
            c.get("type") == "set_param" and c.get("key") == "mutation_std"
            for c in cmds
        )

    def test_controls_pipeline_with_real_training(self):
        """Full pipeline: train → publish → WebSocket → command → verify."""
        from src.training.train import create_train_state, train_step

        config = _small_config()
        key = __import__("jax").random.PRNGKey(42)

        runner_state = create_train_state(config, key)
        bridge = TrainingBridge(target_fps=1000)

        # Train and publish
        runner_state, metrics = train_step(runner_state, config)
        frame = bridge.create_frame_from_state(
            runner_state.env_state, metrics, step=50
        )
        bridge.publish_frame(frame)

        app = create_app(bridge)
        client = TestClient(app)

        with client.websocket_connect("/ws/training") as ws:
            # Receive the real training frame
            data = ws.receive_bytes()
            decoded = msgpack.unpackb(data, raw=False)
            assert decoded["step"] == 50
            assert "metrics" in decoded
            assert len(decoded["metrics"]) > 0

            # Send pause
            ws.send_text(json.dumps({"type": "pause"}))
            time.sleep(0.1)
            assert bridge.paused is True

            # Send speed change
            ws.send_text(json.dumps({"type": "set_speed", "value": 2.0}))
            time.sleep(0.1)
            assert bridge.speed_multiplier == 2.0

            # Resume
            ws.send_text(json.dumps({"type": "resume"}))
            time.sleep(0.1)
            assert bridge.paused is False


@pytest.mark.timeout(60)
class TestMockDataPipeline:
    """Test the mock data pipeline used for dashboard development."""

    def test_mock_frames_stream_continuously(self):
        """Mock frames can be streamed and decoded in sequence."""
        bridge = TrainingBridge(target_fps=1000)
        app = create_app(bridge)
        client = TestClient(app)

        # Publish multiple mock frames
        for step in range(5):
            frame = _generate_mock_frame(step=step)
            bridge.publish_frame(frame)
            time.sleep(0.002)  # Small delay to satisfy rate limiter

        with client.websocket_connect("/ws/training") as ws:
            data = ws.receive_bytes()
            decoded = msgpack.unpackb(data, raw=False)

            # Should get the latest frame
            assert decoded["step"] == 4  # Last published was step=4
            assert "positions" in decoded
            assert "metrics" in decoded

    def test_mock_metrics_improve_over_time(self):
        """Mock metrics simulate gradual improvement."""
        frame_0 = _generate_mock_frame(step=0)
        frame_500 = _generate_mock_frame(step=499)

        # Specialization should increase from ~0 to ~0.75
        assert frame_500.metrics["specialization_score"] > frame_0.metrics["specialization_score"]

        # Weight divergence should increase
        assert frame_500.metrics["weight_divergence"] > frame_0.metrics["weight_divergence"]

    def test_mock_frame_all_arrays_valid(self):
        """Mock frame arrays are valid numpy arrays with correct dtypes."""
        frame = _generate_mock_frame(step=42, max_agents=16)

        assert frame.positions.dtype == np.float32
        assert frame.positions.shape == (16, 2)
        assert frame.alive.dtype == bool
        assert frame.alive.shape == (16,)
        assert frame.energy.dtype == np.float32
        assert frame.energy.shape == (16,)
        assert frame.food_positions.dtype == np.float32
        assert frame.food_collected.shape == (10,)
        assert frame.field_values.dtype == np.float32
        assert frame.field_values.shape == (20, 20, 4)
        assert frame.cluster_labels is not None
        assert frame.cluster_labels.shape == (16,)

    def test_mock_training_mode_cycling(self):
        """Mock frames correctly cycle between gradient and evolve modes."""
        # Step in gradient phase (< 200)
        frame_grad = _generate_mock_frame(step=50, training_mode="gradient")
        assert frame_grad.training_mode == "gradient"

        # Step in evolve phase (>= 200)
        frame_evol = _generate_mock_frame(step=210, training_mode="evolve")
        assert frame_evol.training_mode == "evolve"


@pytest.mark.timeout(60)
class TestFrameSerialization:
    """Test that frame serialization round-trips correctly."""

    def test_full_frame_roundtrip(self):
        """Pack and unpack a complete frame, verify all fields survive."""
        frame = Frame(
            step=999,
            positions=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            alive=np.array([True, False], dtype=bool),
            energy=np.array([100.0, 0.0], dtype=np.float32),
            food_positions=np.array([[5.0, 6.0]], dtype=np.float32),
            food_collected=np.array([False], dtype=bool),
            field_values=np.ones((10, 10, 4), dtype=np.float32) * 0.5,
            cluster_labels=np.array([0, 1], dtype=np.int8),
            metrics={"reward": 3.14, "loss": 0.01},
            training_mode="evolve",
        )

        packed = pack_frame(frame)
        data = msgpack.unpackb(packed, raw=False)

        assert data["step"] == 999
        assert data["training_mode"] == "evolve"
        assert abs(data["metrics"]["reward"] - 3.14) < 1e-6
        assert abs(data["metrics"]["loss"] - 0.01) < 1e-6

        # Reconstruct positions
        pos = np.frombuffer(
            data["positions"]["data"], dtype=data["positions"]["dtype"]
        ).reshape(data["positions"]["shape"])
        np.testing.assert_array_almost_equal(pos, frame.positions)

        # Reconstruct alive
        alive = np.frombuffer(
            data["alive"]["data"], dtype=data["alive"]["dtype"]
        ).reshape(data["alive"]["shape"])
        assert alive[0] == 1  # True → uint8(1)
        assert alive[1] == 0  # False → uint8(0)

        # Clusters present
        assert "clusters" in data
        clusters = np.frombuffer(
            data["clusters"]["data"], dtype=data["clusters"]["dtype"]
        ).reshape(data["clusters"]["shape"])
        np.testing.assert_array_equal(clusters, frame.cluster_labels)

    def test_frame_without_clusters_roundtrip(self):
        """Frame without cluster labels serializes correctly."""
        frame = Frame(
            step=0,
            positions=np.zeros((4, 2), dtype=np.float32),
            alive=np.ones(4, dtype=bool),
            energy=np.zeros(4, dtype=np.float32),
            food_positions=np.zeros((2, 2), dtype=np.float32),
            food_collected=np.zeros(2, dtype=bool),
            field_values=np.zeros((5, 5, 1), dtype=np.float32),
            cluster_labels=None,
            metrics={},
        )

        packed = pack_frame(frame)
        data = msgpack.unpackb(packed, raw=False)
        assert "clusters" not in data


@pytest.mark.timeout(60)
class TestHelpSystemData:
    """Test that help system data (glossary) is available for the dashboard."""

    def test_glossary_json_exists(self):
        """Glossary JSON file exists in the dashboard static directory."""
        glossary_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "dashboard",
            "static",
            "glossary.json",
        )
        assert os.path.exists(glossary_path), (
            f"Glossary file should exist at {glossary_path}"
        )

    def test_glossary_json_valid(self):
        """Glossary JSON is valid and contains expected terms."""
        glossary_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "dashboard",
            "static",
            "glossary.json",
        )
        with open(glossary_path) as f:
            glossary_data = json.load(f)

        # Glossary is wrapped in {"terms": [...]}
        assert isinstance(glossary_data, dict), "Glossary should be a dict"
        assert "terms" in glossary_data, "Glossary should have 'terms' key"
        terms = glossary_data["terms"]
        assert isinstance(terms, list), "Glossary terms should be a list"
        assert len(terms) >= 10, f"Expected at least 10 terms, got {len(terms)}"

        # Check structure of each entry
        for entry in terms:
            assert "term" in entry, f"Entry missing 'term': {entry}"
            assert "simple" in entry, f"Entry missing 'simple': {entry}"

        # Key terms that should be in the glossary
        term_names = {e["term"].lower() for e in terms}
        expected_terms = {"agent", "emergence", "specialization", "evolution"}
        for term in expected_terms:
            assert term in term_names, f"Missing glossary term: {term}"

    def test_glossary_terms_have_analogies(self):
        """Core glossary terms include analogies for laypeople."""
        glossary_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "dashboard",
            "static",
            "glossary.json",
        )
        with open(glossary_path) as f:
            glossary_data = json.load(f)

        terms = glossary_data["terms"]
        # At least some entries should have analogies
        entries_with_analogy = [
            e for e in terms if e.get("analogy")
        ]
        assert len(entries_with_analogy) > 0, (
            "At least some glossary entries should have analogies"
        )


@pytest.mark.timeout(120)
class TestBackgroundTrainingIntegration:
    """Test training running in a background thread with live WebSocket streaming."""

    def test_background_training_publishes_frames(self):
        """Training in background thread publishes frames that WebSocket receives."""
        from src.training.train import create_train_state, train_step

        config = _small_config()
        key = __import__("jax").random.PRNGKey(42)

        runner_state = create_train_state(config, key)
        bridge = TrainingBridge(target_fps=1000)

        # Track frames published
        frames_published = []
        stop_event = threading.Event()

        def training_thread():
            nonlocal runner_state
            step = 0
            while not stop_event.is_set() and step < 3:
                runner_state, metrics = train_step(runner_state, config)
                frame = bridge.create_frame_from_state(
                    runner_state.env_state, metrics, step=step
                )
                if bridge.publish_frame(frame):
                    frames_published.append(step)
                step += 1
                time.sleep(0.01)  # Small delay between steps

        # Start training in background
        t = threading.Thread(target=training_thread, daemon=True)
        t.start()

        # Wait for training to produce some frames
        t.join(timeout=60)
        stop_event.set()

        assert len(frames_published) > 0, "Training should have published frames"

        # Verify the last frame is accessible
        latest = bridge.get_latest_frame()
        assert latest is not None
        assert latest.positions.ndim == 2
        assert latest.alive.ndim == 1
        assert len(latest.metrics) > 0

    def test_background_training_with_websocket_client(self):
        """Full end-to-end: background training → bridge → WebSocket → client."""
        from src.training.train import create_train_state, train_step

        config = _small_config()
        key = __import__("jax").random.PRNGKey(42)

        runner_state = create_train_state(config, key)
        bridge = TrainingBridge(target_fps=1000)
        stop_event = threading.Event()

        def training_thread():
            nonlocal runner_state
            step = 0
            while not stop_event.is_set() and step < 5:
                runner_state, metrics = train_step(runner_state, config)
                frame = bridge.create_frame_from_state(
                    runner_state.env_state, metrics, step=step
                )
                bridge.publish_frame(frame)
                step += 1
                time.sleep(0.01)

        t = threading.Thread(target=training_thread, daemon=True)
        t.start()

        # Wait for at least one frame
        deadline = time.time() + 60
        while bridge.get_latest_frame() is None and time.time() < deadline:
            time.sleep(0.1)

        assert bridge.get_latest_frame() is not None, "Training should have produced a frame"

        # Connect WebSocket and verify
        app = create_app(bridge)
        client = TestClient(app)

        with client.websocket_connect("/ws/training") as ws:
            data = ws.receive_bytes()
            decoded = msgpack.unpackb(data, raw=False)

            assert "step" in decoded
            assert "positions" in decoded
            assert "alive" in decoded
            assert "energy" in decoded
            assert "metrics" in decoded

            # Metrics should be real training metrics
            m = decoded["metrics"]
            for k, v in m.items():
                assert isinstance(v, (int, float)), f"Metric {k} should be numeric"

        stop_event.set()
        t.join(timeout=10)

    def test_commands_reach_training_thread(self):
        """Commands sent via WebSocket are consumable by the training thread."""
        bridge = TrainingBridge(target_fps=1000)
        bridge.publish_frame(_generate_mock_frame(step=0))

        app = create_app(bridge)
        client = TestClient(app)

        with client.websocket_connect("/ws/training") as ws:
            ws.receive_bytes()
            # Simulate dashboard sending parameter change
            ws.send_text(json.dumps({
                "type": "set_param",
                "key": "diversity_bonus",
                "value": 0.1,
            }))
            time.sleep(0.1)

        # Training thread would consume commands like this
        cmds = bridge.pop_commands()
        found = False
        for cmd in cmds:
            if cmd.get("type") == "set_param" and cmd.get("key") == "diversity_bonus":
                assert cmd["value"] == 0.1
                found = True
        assert found, "diversity_bonus command should be in the queue"


@pytest.mark.timeout(60)
class TestDashboardBuildArtifacts:
    """Test that dashboard build artifacts and structure exist."""

    def test_dashboard_package_json_exists(self):
        """Dashboard package.json exists with correct dependencies."""
        pkg_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "dashboard",
            "package.json",
        )
        assert os.path.exists(pkg_path), "dashboard/package.json should exist"
        with open(pkg_path) as f:
            pkg = json.load(f)
        assert "dependencies" in pkg
        deps = pkg["dependencies"]
        dev_deps = pkg.get("devDependencies", {})
        all_deps = {**deps, **dev_deps}
        assert "svelte" in all_deps, "Svelte should be a dependency"
        assert "pixi.js" in all_deps, "Pixi.js should be a dependency"
        assert "plotly.js-dist-min" in all_deps, "Plotly.js should be a dependency"
        assert "msgpack-lite" in all_deps, "msgpack-lite should be a dependency"

    def test_dashboard_source_files_exist(self):
        """Key dashboard source files exist."""
        dashboard_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "dashboard",
            "src",
        )
        expected_files = [
            "App.svelte",
            "main.js",
            "lib/AgentCanvas.svelte",
            "lib/MetricsPanel.svelte",
            "lib/ControlPanel.svelte",
            "lib/renderer.js",
            "lib/Tooltip.svelte",
            "lib/GlossaryPanel.svelte",
            "lib/HelpSystem.svelte",
            "lib/Header.svelte",
            "stores/training.svelte.js",
        ]
        for f in expected_files:
            path = os.path.join(dashboard_dir, f)
            assert os.path.exists(path), f"Dashboard file should exist: src/{f}"

    def test_vite_config_exists(self):
        """Vite configuration exists for dashboard build."""
        vite_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "dashboard",
            "vite.config.js",
        )
        assert os.path.exists(vite_path), "dashboard/vite.config.js should exist"

    def test_index_html_exists(self):
        """Dashboard index.html entry point exists."""
        index_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "dashboard",
            "index.html",
        )
        assert os.path.exists(index_path), "dashboard/index.html should exist"
