from dataclasses import dataclass
from typing import Protocol

from aquaguard.video.models import FrameRead, VideoFrame
from aquaguard.vision.adapter import CalibratedObservationAdapter
from aquaguard.vision.models import PixelTrackObservation
from aquaguard.world.pipeline import WorldModelPipeline


class VideoSource(Protocol):
    def read(self) -> FrameRead: ...

    def close(self) -> None: ...


class FrameAnalyzer(Protocol):
    def analyze(self, frame: VideoFrame) -> list[PixelTrackObservation]: ...


@dataclass(frozen=True, slots=True)
class RuntimeStep:
    ok: bool
    result: dict | None = None
    error: str | None = None
    retry_at: float | None = None


class VideoWorldRuntime:
    """One-step orchestration from video capture to the pool world model."""

    def __init__(
        self,
        source: VideoSource,
        analyzer: FrameAnalyzer,
        adapter: CalibratedObservationAdapter,
        pipeline: WorldModelPipeline,
    ):
        self.source = source
        self.analyzer = analyzer
        self.adapter = adapter
        self.pipeline = pipeline

    def step(self) -> RuntimeStep:
        read = self.source.read()
        if not read.ok or read.frame is None:
            return RuntimeStep(False, error=read.error, retry_at=read.retry_at)
        pixels = self.analyzer.analyze(read.frame)
        observations = self.adapter.convert(pixels)
        return RuntimeStep(True, result=self.pipeline.process(observations))

    def close(self) -> None:
        self.source.close()
