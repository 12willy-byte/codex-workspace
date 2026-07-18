from threading import Event

import pytest

from aquaguard.runtime import RuntimeStep, VideoRuntimeManager, VideoWorldRuntime
from aquaguard.evidence import EvidenceRecorder
from aquaguard.video import OpenCVFrameSource, VideoFrame
from aquaguard.vision import (
    CalibratedObservationAdapter,
    HomographyProjector,
    PixelTrackObservation,
    ScriptedFrameAnalyzer,
)
from aquaguard.world import TemporalFusionCoordinator, WorldModelPipeline


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeCapture:
    def __init__(self, opened: bool, frames: list[object] | None = None) -> None:
        self.opened = opened
        self.frames = list(frames or [])
        self.released = False

    def isOpened(self) -> bool:
        return self.opened and not self.released

    def read(self) -> tuple[bool, object | None]:
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self) -> None:
        self.released = True


def test_video_source_timestamps_and_sequences_frames() -> None:
    clock = FakeClock()
    source = OpenCVFrameSource(
        "cam-a",
        "0",
        capture_factory=lambda _: FakeCapture(True, ["frame-a", "frame-b"]),
        clock=clock,
        reconnect=False,
    )
    first = source.read()
    clock.now = 0.04
    second = source.read()
    assert first.frame == VideoFrame("cam-a", 0, 0.0, "frame-a")
    assert second.frame == VideoFrame("cam-a", 1, 0.04, "frame-b")
    assert source.read().error == "end_of_stream"


def test_video_source_applies_bounded_reconnect_backoff() -> None:
    clock = FakeClock()
    attempts: list[FakeCapture] = []

    def factory(_: object) -> FakeCapture:
        capture = FakeCapture(opened=len(attempts) >= 2, frames=["ready"])
        attempts.append(capture)
        return capture

    source = OpenCVFrameSource(
        "cam-a",
        "rtsp://camera/stream",
        capture_factory=factory,
        clock=clock,
        initial_delay=1,
        max_delay=2,
    )
    assert source.read().retry_at == 1
    clock.now = 0.5
    assert source.read().error == "source_unavailable"
    clock.now = 1
    assert source.read().retry_at == 3
    clock.now = 3
    assert source.read().ok is True
    assert len(attempts) == 3


def test_runtime_connects_video_frame_to_world_model() -> None:
    clock = FakeClock()
    source = OpenCVFrameSource(
        "cam-a",
        "recording.mp4",
        capture_factory=lambda _: FakeCapture(True, ["pixels"]),
        clock=clock,
        reconnect=False,
    )

    def analyze(frame: VideoFrame) -> list[PixelTrackObservation]:
        assert frame.image == "pixels"
        return [
            PixelTrackObservation(
                camera_id=frame.camera_id,
                local_track_id="track-1",
                timestamp=frame.timestamp,
                anchor_x=50,
                anchor_y=40,
                confidence=0.9,
                motion=0.8,
            )
        ]

    evidence = EvidenceRecorder()
    runtime = VideoWorldRuntime(
        source,
        ScriptedFrameAnalyzer(analyze),
        CalibratedObservationAdapter(
            {"cam-a": HomographyProjector("cam-a", (0.1, 0, 0, 0, 0.1, 0, 0, 0, 1))}
        ),
        TemporalFusionCoordinator(WorldModelPipeline()),
        observers=(evidence,),
    )
    step = runtime.step()
    assert step.ok is True
    assert step.result["tracks"][0]["position"] == {"x": 5.0, "y": 4.0}
    assert evidence.buffer.bounds("cam-a") == (0.0, 0.0)


class ManagedRuntime:
    def __init__(self, steps: list[RuntimeStep | Exception]) -> None:
        self.steps = list(steps)
        self.closed = False
        self.called = Event()
        self.terminal = Event()

    def step(self) -> RuntimeStep:
        item = self.steps.pop(0)
        self.called.set()
        if isinstance(item, Exception):
            self.terminal.set()
            raise item
        if not item.ok and item.error == "end_of_stream":
            self.terminal.set()
        return item

    def close(self) -> None:
        self.closed = True


class BlockingRuntime:
    def __init__(self) -> None:
        self.entered = Event()
        self.release = Event()

    def step(self) -> RuntimeStep:
        self.entered.set()
        self.release.wait(2)
        return RuntimeStep(False, error="end_of_stream")

    def close(self) -> None:
        pass


def test_runtime_manager_isolates_camera_failures() -> None:
    healthy = ManagedRuntime(
        [RuntimeStep(True, result={"tracks": []}), RuntimeStep(False, error="end_of_stream")]
    )
    failed = ManagedRuntime([RuntimeError("decoder crashed")])
    manager = VideoRuntimeManager(idle_wait_seconds=0, join_timeout_seconds=1)
    manager.add("cam-healthy", healthy)
    manager.add("cam-failed", failed)

    manager.start()
    assert healthy.called.wait(1)
    assert failed.called.wait(1)
    assert healthy.terminal.wait(1)
    assert failed.terminal.wait(1)
    manager.stop()
    statuses = {item["camera_id"]: item for item in manager.statuses()}

    assert statuses["cam-healthy"]["frames_processed"] == 1
    assert statuses["cam-healthy"]["last_error"] == "end_of_stream"
    assert statuses["cam-failed"]["last_error"] == "runtime_failure: decoder crashed"
    assert all(item["running"] is False for item in statuses.values())
    assert healthy.closed is True
    assert failed.closed is True


def test_runtime_manager_rejects_registration_after_start() -> None:
    runtime = ManagedRuntime([RuntimeStep(False, error="end_of_stream")])
    manager = VideoRuntimeManager()
    manager.start()

    try:
        with pytest.raises(RuntimeError, match="after the manager has started"):
            manager.add("cam-a", runtime)
    finally:
        manager.stop()


def test_runtime_manager_reports_shutdown_timeout_truthfully() -> None:
    runtime = BlockingRuntime()
    manager = VideoRuntimeManager(join_timeout_seconds=0)
    manager.add("cam-blocked", runtime)
    manager.start()
    assert runtime.entered.wait(1)

    manager.stop()

    assert manager.statuses()[0]["running"] is True
    assert manager.statuses()[0]["last_error"] == "shutdown_timeout"
    runtime.release.set()
    manager.join_timeout_seconds = 1
    manager.stop()


def test_runtime_status_never_equates_thread_health_with_protection_capability() -> None:
    runtime = ManagedRuntime([RuntimeStep(False, error="end_of_stream")])
    runtime.capabilities = ("person_detection", "local_tracking")
    manager = VideoRuntimeManager()
    manager.add("cam-a", runtime)

    status = manager.statuses()[0]

    assert status["protection_level"] == "tracking_only"
    assert status["capabilities"] == ("person_detection", "local_tracking")
