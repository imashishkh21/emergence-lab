"""Tests for the recording and replay system.

Tests cover:
- Bookmark creation and serialization
- SessionRecorder: recording, sampling, bookmarks, save/load
- SessionPlayer: loading, playback, seeking, speed control
- Frame serialization for replay
- Server recording/session endpoints
- Replay WebSocket streaming
"""

import gzip
import json
import os
import time

import msgpack
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.server.main import (
    DEFAULT_RECORDINGS_DIR,
    _generate_mock_frame,
    create_app,
)
from src.server.replay import (
    Bookmark,
    SessionPlayer,
    SessionRecorder,
    _serialize_frame,
    list_sessions,
)
from src.server.streaming import Frame, TrainingBridge, pack_frame


# ---------------------------------------------------------------------------
# Helper to create a simple Frame
# ---------------------------------------------------------------------------


def _make_frame(step: int = 0, max_agents: int = 8) -> Frame:
    return Frame(
        step=step,
        positions=np.random.rand(max_agents, 2).astype(np.float32),
        alive=np.ones(max_agents, dtype=bool),
        energy=np.full(max_agents, 100.0, dtype=np.float32),
        food_positions=np.random.rand(5, 2).astype(np.float32),
        food_collected=np.zeros(5, dtype=bool),
        field_values=np.random.rand(10, 10, 2).astype(np.float32),
        cluster_labels=np.zeros(max_agents, dtype=np.int8),
        metrics={"mean_reward": float(step * 0.1)},
    )


# ---------------------------------------------------------------------------
# Bookmark tests
# ---------------------------------------------------------------------------


class TestBookmark:
    """Tests for the Bookmark dataclass."""

    def test_bookmark_creation(self) -> None:
        bm = Bookmark(step=100, frame_index=10, label="Phase transition")
        assert bm.step == 100
        assert bm.frame_index == 10
        assert bm.label == "Phase transition"
        assert bm.timestamp > 0

    def test_bookmark_custom_timestamp(self) -> None:
        bm = Bookmark(step=0, frame_index=0, label="start", timestamp=12345.0)
        assert bm.timestamp == 12345.0

    def test_bookmark_to_dict(self) -> None:
        bm = Bookmark(step=50, frame_index=5, label="interesting")
        d = bm.to_dict()
        assert d["step"] == 50
        assert d["frame_index"] == 5
        assert d["label"] == "interesting"
        assert "timestamp" in d

    def test_bookmark_from_dict(self) -> None:
        d = {"step": 200, "frame_index": 20, "label": "emergence", "timestamp": 99.0}
        bm = Bookmark.from_dict(d)
        assert bm.step == 200
        assert bm.frame_index == 20
        assert bm.label == "emergence"
        assert bm.timestamp == 99.0

    def test_bookmark_round_trip(self) -> None:
        original = Bookmark(step=300, frame_index=30, label="test")
        restored = Bookmark.from_dict(original.to_dict())
        assert restored.step == original.step
        assert restored.frame_index == original.frame_index
        assert restored.label == original.label
        assert restored.timestamp == original.timestamp


# ---------------------------------------------------------------------------
# Frame serialization tests
# ---------------------------------------------------------------------------


class TestFrameSerialization:
    """Tests for frame serialization used by the recorder."""

    def test_serialize_frame_has_required_keys(self) -> None:
        frame = _make_frame(step=42)
        data = _serialize_frame(frame)
        required = {"step", "timestamp", "positions", "alive", "energy",
                     "food_positions", "food_collected", "field", "metrics",
                     "training_mode"}
        assert required.issubset(set(data.keys()))

    def test_serialize_frame_step_matches(self) -> None:
        frame = _make_frame(step=99)
        data = _serialize_frame(frame)
        assert data["step"] == 99

    def test_serialize_frame_metrics_preserved(self) -> None:
        frame = _make_frame(step=10)
        data = _serialize_frame(frame)
        assert "mean_reward" in data["metrics"]

    def test_serialize_frame_includes_clusters(self) -> None:
        frame = _make_frame()
        data = _serialize_frame(frame)
        assert "clusters" in data

    def test_serialize_frame_without_clusters(self) -> None:
        frame = _make_frame()
        frame.cluster_labels = None
        data = _serialize_frame(frame)
        assert "clusters" not in data

    def test_serialize_frame_with_lineage(self) -> None:
        frame = _make_frame()
        frame.agent_ids = np.arange(8, dtype=np.int32)
        frame.parent_ids = np.full(8, -1, dtype=np.int32)
        frame.birth_steps = np.zeros(8, dtype=np.int32)
        frame.lineage_data = {"max_depth": 0}
        data = _serialize_frame(frame)
        assert "agent_ids" in data
        assert "parent_ids" in data
        assert "birth_steps" in data
        assert "lineage_data" in data

    def test_serialized_frame_is_msgpackable(self) -> None:
        frame = _make_frame(step=5)
        data = _serialize_frame(frame)
        packed = msgpack.packb(data, use_bin_type=True)
        decoded = msgpack.unpackb(packed, raw=False)
        assert decoded["step"] == 5


# ---------------------------------------------------------------------------
# SessionRecorder tests
# ---------------------------------------------------------------------------


class TestSessionRecorder:
    """Tests for recording training sessions."""

    def test_init(self, tmp_path: object) -> None:
        rec = SessionRecorder(str(tmp_path) + "/test_session")
        assert rec.frame_count == 0
        assert rec.recording is False

    def test_start_stop(self, tmp_path: object) -> None:
        rec = SessionRecorder(str(tmp_path) + "/test_session")
        rec.start()
        assert rec.recording is True
        rec.stop()
        assert rec.recording is False

    def test_record_frame(self, tmp_path: object) -> None:
        rec = SessionRecorder(str(tmp_path) + "/test_session")
        rec.start()
        result = rec.record(_make_frame(step=0))
        assert result is True
        assert rec.frame_count == 1

    def test_record_skips_when_not_recording(self, tmp_path: object) -> None:
        rec = SessionRecorder(str(tmp_path) + "/test_session")
        result = rec.record(_make_frame(step=0))
        assert result is False
        assert rec.frame_count == 0

    def test_record_multiple_frames(self, tmp_path: object) -> None:
        rec = SessionRecorder(str(tmp_path) + "/test_session")
        rec.start()
        for i in range(10):
            rec.record(_make_frame(step=i * 100))
        assert rec.frame_count == 10

    def test_sample_rate(self, tmp_path: object) -> None:
        rec = SessionRecorder(str(tmp_path) + "/test_session", sample_rate=3)
        rec.start()
        for i in range(9):
            rec.record(_make_frame(step=i))
        # With sample_rate=3, only frames at counter 3, 6, 9 are recorded
        assert rec.frame_count == 3

    def test_max_frames(self, tmp_path: object) -> None:
        rec = SessionRecorder(str(tmp_path) + "/test_session", max_frames=5)
        rec.start()
        for i in range(10):
            rec.record(_make_frame(step=i))
        assert rec.frame_count == 5

    def test_add_bookmark(self, tmp_path: object) -> None:
        rec = SessionRecorder(str(tmp_path) + "/test_session")
        rec.start()
        rec.record(_make_frame(step=100))
        rec.record(_make_frame(step=200))
        bm = rec.add_bookmark("Phase transition")
        assert bm.step == 200
        assert bm.label == "Phase transition"
        assert len(rec.bookmarks) == 1

    def test_add_bookmark_with_step(self, tmp_path: object) -> None:
        rec = SessionRecorder(str(tmp_path) + "/test_session")
        rec.start()
        rec.record(_make_frame(step=100))
        rec.record(_make_frame(step=200))
        bm = rec.add_bookmark("Custom", step=150)
        assert bm.step == 150

    def test_save_creates_files(self, tmp_path: object) -> None:
        output_dir = str(tmp_path) + "/test_session"
        rec = SessionRecorder(output_dir)
        rec.start()
        rec.record(_make_frame(step=0))
        rec.record(_make_frame(step=100))
        rec.add_bookmark("test")
        path = rec.save()

        assert os.path.exists(os.path.join(str(path), "metadata.json"))
        assert os.path.exists(os.path.join(str(path), "frames.msgpack.gz"))
        assert os.path.exists(os.path.join(str(path), "bookmarks.json"))

    def test_save_metadata_content(self, tmp_path: object) -> None:
        output_dir = str(tmp_path) + "/test_session"
        rec = SessionRecorder(output_dir)
        rec.start()
        rec.record(_make_frame(step=0))
        rec.record(_make_frame(step=500))
        rec.save()

        with open(os.path.join(output_dir, "metadata.json")) as f:
            meta = json.load(f)
        assert meta["start_step"] == 0
        assert meta["end_step"] == 500
        assert meta["frame_count"] == 2
        assert meta["sample_rate"] == 1

    def test_save_bookmarks_content(self, tmp_path: object) -> None:
        output_dir = str(tmp_path) + "/test_session"
        rec = SessionRecorder(output_dir)
        rec.start()
        rec.record(_make_frame(step=100))
        rec.add_bookmark("mark1")
        rec.save()

        with open(os.path.join(output_dir, "bookmarks.json")) as f:
            bookmarks = json.load(f)
        assert len(bookmarks) == 1
        assert bookmarks[0]["label"] == "mark1"

    def test_save_frames_loadable(self, tmp_path: object) -> None:
        output_dir = str(tmp_path) + "/test_session"
        rec = SessionRecorder(output_dir)
        rec.start()
        rec.record(_make_frame(step=42))
        rec.save()

        with gzip.open(os.path.join(output_dir, "frames.msgpack.gz"), "rb") as f:
            raw = f.read()
        frames = msgpack.unpackb(raw, raw=False)
        assert len(frames) == 1
        assert frames[0]["step"] == 42

    def test_context_manager(self, tmp_path: object) -> None:
        output_dir = str(tmp_path) + "/test_session"
        with SessionRecorder(output_dir) as rec:
            rec.record(_make_frame(step=0))
            rec.record(_make_frame(step=100))
        # After context manager, should be saved
        assert os.path.exists(os.path.join(output_dir, "frames.msgpack.gz"))


# ---------------------------------------------------------------------------
# SessionPlayer tests
# ---------------------------------------------------------------------------


class TestSessionPlayer:
    """Tests for playing back recorded sessions."""

    def _record_session(self, output_dir: str, num_frames: int = 10) -> str:
        """Helper to create a recorded session for testing."""
        rec = SessionRecorder(output_dir)
        rec.start()
        for i in range(num_frames):
            rec.record(_make_frame(step=i * 100))
        rec.add_bookmark("start", step=0)
        rec.add_bookmark("middle", step=500)
        rec.save()
        return output_dir

    def test_init(self, tmp_path: object) -> None:
        player = SessionPlayer(str(tmp_path))
        assert player.loaded is False
        assert player.frame_count == 0
        assert player.playing is False

    def test_load(self, tmp_path: object) -> None:
        path = self._record_session(str(tmp_path) + "/session")
        player = SessionPlayer(path)
        player.load()
        assert player.loaded is True
        assert player.frame_count == 10

    def test_load_missing_dir(self) -> None:
        player = SessionPlayer("/nonexistent/path")
        with pytest.raises(FileNotFoundError):
            player.load()

    def test_load_missing_frames(self, tmp_path: object) -> None:
        # Create dir with metadata but no frames file
        session_dir = str(tmp_path) + "/bad_session"
        os.makedirs(session_dir)
        with open(os.path.join(session_dir, "metadata.json"), "w") as f:
            json.dump({}, f)
        player = SessionPlayer(session_dir)
        with pytest.raises(ValueError):
            player.load()

    def test_metadata(self, tmp_path: object) -> None:
        path = self._record_session(str(tmp_path) + "/session")
        player = SessionPlayer(path)
        player.load()
        meta = player.metadata
        assert meta["frame_count"] == 10
        assert meta["start_step"] == 0
        assert meta["end_step"] == 900

    def test_bookmarks_loaded(self, tmp_path: object) -> None:
        path = self._record_session(str(tmp_path) + "/session")
        player = SessionPlayer(path)
        player.load()
        assert len(player.bookmarks) == 2
        assert player.bookmarks[0].label == "start"
        assert player.bookmarks[1].label == "middle"

    def test_play_pause(self, tmp_path: object) -> None:
        path = self._record_session(str(tmp_path) + "/session")
        player = SessionPlayer(path)
        player.load()
        assert player.playing is False
        player.play()
        assert player.playing is True
        player.pause()
        assert player.playing is False

    def test_advance(self, tmp_path: object) -> None:
        path = self._record_session(str(tmp_path) + "/session")
        player = SessionPlayer(path)
        player.load()
        player.play()
        assert player.position == 0
        assert player.advance() is True
        assert player.position == 1

    def test_advance_stops_at_end(self, tmp_path: object) -> None:
        path = self._record_session(str(tmp_path) + "/session", num_frames=3)
        player = SessionPlayer(path)
        player.load()
        player.play()
        player.advance()  # 0 -> 1
        player.advance()  # 1 -> 2
        result = player.advance()  # 2 -> end
        assert result is False
        assert player.playing is False

    def test_advance_when_paused(self, tmp_path: object) -> None:
        path = self._record_session(str(tmp_path) + "/session")
        player = SessionPlayer(path)
        player.load()
        # Don't call play()
        assert player.advance() is False
        assert player.position == 0

    def test_seek(self, tmp_path: object) -> None:
        path = self._record_session(str(tmp_path) + "/session")
        player = SessionPlayer(path)
        player.load()
        player.seek(5)
        assert player.position == 5

    def test_seek_clamps(self, tmp_path: object) -> None:
        path = self._record_session(str(tmp_path) + "/session")
        player = SessionPlayer(path)
        player.load()
        player.seek(-10)
        assert player.position == 0
        player.seek(99999)
        assert player.position == 9  # Last frame index

    def test_seek_to_step(self, tmp_path: object) -> None:
        path = self._record_session(str(tmp_path) + "/session")
        player = SessionPlayer(path)
        player.load()
        player.seek_to_step(500)
        # Step 500 = frame index 5 (step = i * 100)
        assert player.position == 5

    def test_seek_to_bookmark(self, tmp_path: object) -> None:
        path = self._record_session(str(tmp_path) + "/session")
        player = SessionPlayer(path)
        player.load()
        player.seek_to_bookmark(1)  # "middle" bookmark at step 500
        assert player.position == 5

    def test_seek_to_invalid_bookmark(self, tmp_path: object) -> None:
        path = self._record_session(str(tmp_path) + "/session")
        player = SessionPlayer(path)
        player.load()
        player.seek(5)
        player.seek_to_bookmark(99)  # Invalid index — no change
        assert player.position == 5

    def test_current_frame(self, tmp_path: object) -> None:
        path = self._record_session(str(tmp_path) + "/session")
        player = SessionPlayer(path)
        player.load()
        frame = player.current_frame()
        assert frame is not None
        assert frame["step"] == 0

    def test_current_frame_after_seek(self, tmp_path: object) -> None:
        path = self._record_session(str(tmp_path) + "/session")
        player = SessionPlayer(path)
        player.load()
        player.seek(3)
        frame = player.current_frame()
        assert frame is not None
        assert frame["step"] == 300

    def test_current_step(self, tmp_path: object) -> None:
        path = self._record_session(str(tmp_path) + "/session")
        player = SessionPlayer(path)
        player.load()
        assert player.current_step() == 0
        player.seek(5)
        assert player.current_step() == 500

    def test_speed(self, tmp_path: object) -> None:
        player = SessionPlayer(str(tmp_path))
        assert player.speed == 1.0
        player.speed = 4.0
        assert player.speed == 4.0

    def test_speed_clamped(self, tmp_path: object) -> None:
        player = SessionPlayer(str(tmp_path))
        player.speed = 0.01
        assert player.speed == 0.25
        player.speed = 999.0
        assert player.speed == 16.0

    def test_progress(self, tmp_path: object) -> None:
        path = self._record_session(str(tmp_path) + "/session")
        player = SessionPlayer(path)
        player.load()
        assert player.progress == 0.0
        player.seek(9)
        assert player.progress == 1.0
        player.seek(4)
        assert 0.0 < player.progress < 1.0

    def test_progress_empty(self, tmp_path: object) -> None:
        player = SessionPlayer(str(tmp_path))
        assert player.progress == 0.0

    def test_current_frame_empty(self, tmp_path: object) -> None:
        player = SessionPlayer(str(tmp_path))
        assert player.current_frame() is None

    def test_full_playback(self, tmp_path: object) -> None:
        path = self._record_session(str(tmp_path) + "/session", num_frames=5)
        player = SessionPlayer(path)
        player.load()
        player.play()

        steps_seen = []
        while player.playing or player.position < player.frame_count:
            frame = player.current_frame()
            if frame is not None:
                steps_seen.append(frame["step"])
            if not player.advance():
                break

        assert len(steps_seen) == 5
        assert steps_seen == [0, 100, 200, 300, 400]


# ---------------------------------------------------------------------------
# list_sessions tests
# ---------------------------------------------------------------------------


class TestListSessions:
    """Tests for listing recorded sessions."""

    def test_empty_dir(self, tmp_path: object) -> None:
        sessions = list_sessions(str(tmp_path))
        assert sessions == []

    def test_nonexistent_dir(self) -> None:
        sessions = list_sessions("/nonexistent/path")
        assert sessions == []

    def test_lists_sessions(self, tmp_path: object) -> None:
        # Create two session directories
        for name in ["session_1", "session_2"]:
            session_dir = os.path.join(str(tmp_path), name)
            rec = SessionRecorder(session_dir)
            rec.start()
            rec.record(_make_frame(step=0))
            rec.save()

        sessions = list_sessions(str(tmp_path))
        assert len(sessions) == 2
        names = [s["name"] for s in sessions]
        assert "session_1" in names
        assert "session_2" in names

    def test_session_has_metadata(self, tmp_path: object) -> None:
        session_dir = os.path.join(str(tmp_path), "test")
        rec = SessionRecorder(session_dir)
        rec.start()
        rec.record(_make_frame(step=0))
        rec.record(_make_frame(step=100))
        rec.save()

        sessions = list_sessions(str(tmp_path))
        assert len(sessions) == 1
        assert sessions[0]["metadata"]["frame_count"] == 2

    def test_skips_non_session_dirs(self, tmp_path: object) -> None:
        # Create a directory without frames.msgpack.gz
        os.makedirs(os.path.join(str(tmp_path), "not_a_session"))
        sessions = list_sessions(str(tmp_path))
        assert sessions == []


# ---------------------------------------------------------------------------
# Server recording endpoint tests
# ---------------------------------------------------------------------------


class TestServerRecording:
    """Tests for the recording HTTP endpoints."""

    def test_sessions_endpoint_empty(self, tmp_path: object, monkeypatch: object) -> None:
        monkeypatch.setattr(
            "src.server.main.DEFAULT_RECORDINGS_DIR",
            str(tmp_path) + "/empty",
        )
        bridge = TrainingBridge()
        app = create_app(bridge)
        client = TestClient(app)
        response = client.get("/sessions")
        assert response.status_code == 200
        assert response.json() == []

    def test_sessions_endpoint_with_data(self, tmp_path: object, monkeypatch: object) -> None:
        recordings_dir = str(tmp_path)
        monkeypatch.setattr(
            "src.server.main.DEFAULT_RECORDINGS_DIR", recordings_dir
        )
        # Create a session
        rec = SessionRecorder(os.path.join(recordings_dir, "test_session"))
        rec.start()
        rec.record(_make_frame(step=0))
        rec.save()

        bridge = TrainingBridge()
        app = create_app(bridge)
        client = TestClient(app)
        response = client.get("/sessions")
        assert response.status_code == 200
        sessions = response.json()
        assert len(sessions) == 1
        assert sessions[0]["name"] == "test_session"

    def test_start_recording(self, tmp_path: object, monkeypatch: object) -> None:
        monkeypatch.setattr(
            "src.server.main.DEFAULT_RECORDINGS_DIR", str(tmp_path)
        )
        monkeypatch.setattr("src.server.main._recorder", None)
        bridge = TrainingBridge()
        app = create_app(bridge)
        client = TestClient(app)
        response = client.post("/record/start?name=test&sample_rate=2")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "test" in data["session_name"]
        # Clean up
        import src.server.main as main_mod
        if main_mod._recorder is not None:
            main_mod._recorder.stop()
            main_mod._recorder = None

    def test_stop_recording_not_recording(self, monkeypatch: object) -> None:
        monkeypatch.setattr("src.server.main._recorder", None)
        bridge = TrainingBridge()
        app = create_app(bridge)
        client = TestClient(app)
        response = client.post("/record/stop")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"

    def test_start_stop_recording_cycle(
        self, tmp_path: object, monkeypatch: object
    ) -> None:
        monkeypatch.setattr(
            "src.server.main.DEFAULT_RECORDINGS_DIR", str(tmp_path)
        )
        monkeypatch.setattr("src.server.main._recorder", None)
        bridge = TrainingBridge(target_fps=1000)
        app = create_app(bridge)
        client = TestClient(app)

        # Start recording
        resp = client.post("/record/start?name=cycle_test")
        assert resp.json()["status"] == "ok"

        # Simulate publishing a frame through the bridge
        import src.server.main as main_mod
        rec = main_mod._recorder
        assert rec is not None and rec.recording

        # Manually record some frames
        for i in range(5):
            rec.record(_make_frame(step=i * 100))

        # Stop and save
        resp = client.post("/record/stop")
        data = resp.json()
        assert data["status"] == "ok"
        assert data["frame_count"] == 5

        # Verify session appears in list
        resp = client.get("/sessions")
        sessions = resp.json()
        assert len(sessions) == 1

    def test_add_bookmark_not_recording(self, monkeypatch: object) -> None:
        monkeypatch.setattr("src.server.main._recorder", None)
        bridge = TrainingBridge()
        app = create_app(bridge)
        client = TestClient(app)
        response = client.post("/record/bookmark?label=test")
        data = response.json()
        assert data["status"] == "error"


# ---------------------------------------------------------------------------
# Replay WebSocket tests
# ---------------------------------------------------------------------------


class TestReplayWebSocket:
    """Tests for the replay WebSocket endpoint."""

    def _create_session(self, tmp_path: str, num_frames: int = 5) -> str:
        """Create a recorded session and return its path."""
        session_dir = os.path.join(tmp_path, "replay_test")
        rec = SessionRecorder(session_dir)
        rec.start()
        for i in range(num_frames):
            rec.record(_make_frame(step=i * 100))
        rec.add_bookmark("test_mark", step=200)
        rec.save()
        return session_dir

    def test_replay_connect(self, tmp_path: object) -> None:
        bridge = TrainingBridge()
        app = create_app(bridge)
        client = TestClient(app)

        with client.websocket_connect("/ws/replay") as ws:
            # Just verify connection works
            pass  # Closes cleanly

    def test_replay_load_session(self, tmp_path: object) -> None:
        session_path = self._create_session(str(tmp_path))
        bridge = TrainingBridge()
        app = create_app(bridge)
        client = TestClient(app)

        with client.websocket_connect("/ws/replay") as ws:
            ws.send_text(json.dumps({"type": "load", "session": session_path}))
            # Should receive a status message
            response = ws.receive_text()
            status = json.loads(response)
            assert status["type"] == "status"
            assert status["total"] == 5
            assert status["position"] == 0
            assert status["playing"] is False

    def test_replay_load_invalid_session(self, tmp_path: object) -> None:
        bridge = TrainingBridge()
        app = create_app(bridge)
        client = TestClient(app)

        with client.websocket_connect("/ws/replay") as ws:
            ws.send_text(json.dumps({"type": "load", "session": "/nonexistent"}))
            response = ws.receive_text()
            data = json.loads(response)
            assert data["type"] == "error"

    def test_replay_play_streams_frames(self, tmp_path: object) -> None:
        session_path = self._create_session(str(tmp_path), num_frames=3)
        bridge = TrainingBridge()
        app = create_app(bridge)
        client = TestClient(app)

        with client.websocket_connect("/ws/replay") as ws:
            # Load session
            ws.send_text(json.dumps({"type": "load", "session": session_path}))
            ws.receive_text()  # status

            # Play
            ws.send_text(json.dumps({"type": "play"}))
            # Should receive binary frames
            data = ws.receive_bytes()
            decoded = msgpack.unpackb(data, raw=False)
            assert decoded["step"] == 0

    def test_replay_seek(self, tmp_path: object) -> None:
        session_path = self._create_session(str(tmp_path))
        bridge = TrainingBridge()
        app = create_app(bridge)
        client = TestClient(app)

        with client.websocket_connect("/ws/replay") as ws:
            ws.send_text(json.dumps({"type": "load", "session": session_path}))
            ws.receive_text()  # initial status

            ws.send_text(json.dumps({"type": "seek", "position": 3}))
            response = ws.receive_text()
            status = json.loads(response)
            assert status["position"] == 3
            assert status["step"] == 300

    def test_replay_seek_step(self, tmp_path: object) -> None:
        session_path = self._create_session(str(tmp_path))
        bridge = TrainingBridge()
        app = create_app(bridge)
        client = TestClient(app)

        with client.websocket_connect("/ws/replay") as ws:
            ws.send_text(json.dumps({"type": "load", "session": session_path}))
            ws.receive_text()

            ws.send_text(json.dumps({"type": "seek_step", "step": 200}))
            response = ws.receive_text()
            status = json.loads(response)
            assert status["step"] == 200

    def test_replay_seek_bookmark(self, tmp_path: object) -> None:
        session_path = self._create_session(str(tmp_path))
        bridge = TrainingBridge()
        app = create_app(bridge)
        client = TestClient(app)

        with client.websocket_connect("/ws/replay") as ws:
            ws.send_text(json.dumps({"type": "load", "session": session_path}))
            response = ws.receive_text()
            status = json.loads(response)
            assert len(status["bookmarks"]) == 1

            ws.send_text(json.dumps({"type": "seek_bookmark", "index": 0}))
            response = ws.receive_text()
            status = json.loads(response)
            assert status["step"] == 200

    def test_replay_set_speed(self, tmp_path: object) -> None:
        session_path = self._create_session(str(tmp_path))
        bridge = TrainingBridge()
        app = create_app(bridge)
        client = TestClient(app)

        with client.websocket_connect("/ws/replay") as ws:
            ws.send_text(json.dumps({"type": "load", "session": session_path}))
            ws.receive_text()

            ws.send_text(json.dumps({"type": "set_speed", "value": 4}))
            response = ws.receive_text()
            status = json.loads(response)
            assert status["speed"] == 4.0

    def test_replay_status_has_metadata(self, tmp_path: object) -> None:
        session_path = self._create_session(str(tmp_path))
        bridge = TrainingBridge()
        app = create_app(bridge)
        client = TestClient(app)

        with client.websocket_connect("/ws/replay") as ws:
            ws.send_text(json.dumps({"type": "load", "session": session_path}))
            response = ws.receive_text()
            status = json.loads(response)
            assert "metadata" in status
            assert status["metadata"]["frame_count"] == 5


# ---------------------------------------------------------------------------
# Record → Replay round-trip test
# ---------------------------------------------------------------------------


class TestRecordReplayRoundTrip:
    """End-to-end test: record a session, then replay it."""

    def test_record_then_replay(self, tmp_path: object) -> None:
        output_dir = os.path.join(str(tmp_path), "roundtrip")

        # Record
        frames_recorded = []
        with SessionRecorder(output_dir) as rec:
            for i in range(5):
                frame = _make_frame(step=i * 100)
                rec.record(frame)
                frames_recorded.append(frame.step)
            rec.add_bookmark("important", step=200)

        # Replay
        player = SessionPlayer(output_dir)
        player.load()
        assert player.frame_count == 5
        assert len(player.bookmarks) == 1
        assert player.bookmarks[0].label == "important"

        # Verify all frames accessible
        steps_replayed = []
        for i in range(player.frame_count):
            player.seek(i)
            frame = player.current_frame()
            assert frame is not None
            steps_replayed.append(frame["step"])

        assert steps_replayed == frames_recorded

    def test_record_replay_preserves_metrics(self, tmp_path: object) -> None:
        output_dir = os.path.join(str(tmp_path), "metrics_test")

        with SessionRecorder(output_dir) as rec:
            frame = _make_frame(step=42)
            rec.record(frame)

        player = SessionPlayer(output_dir)
        player.load()
        replayed = player.current_frame()
        assert replayed is not None
        assert replayed["metrics"]["mean_reward"] == pytest.approx(4.2, abs=0.01)
