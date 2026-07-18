from aquaguard.runtime import VideoWorldRuntime
from aquaguard.video import OpenCVFrameSource, VideoFrame
from aquaguard.vision import (
    CalibratedObservationAdapter,
    HomographyProjector,
    PixelTrackObservation,
    ScriptedFrameAnalyzer,
)
from aquaguard.world import WorldModelPipeline


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

    runtime = VideoWorldRuntime(
        source,
        ScriptedFrameAnalyzer(analyze),
        CalibratedObservationAdapter(
            {"cam-a": HomographyProjector("cam-a", (0.1, 0, 0, 0, 0.1, 0, 0, 0, 1))}
        ),
        WorldModelPipeline(),
    )
    step = runtime.step()
    assert step.ok is True
    assert step.result["tracks"][0]["position"] == {"x": 5.0, "y": 4.0}
