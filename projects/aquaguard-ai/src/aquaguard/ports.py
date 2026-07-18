from typing import Protocol

from aquaguard.domain import AlarmEvent
from aquaguard.video.models import FrameRead, VideoFrame
from aquaguard.vision.models import PixelTrackObservation


class FrameSource(Protocol):
    def read(self) -> FrameRead: ...
    def close(self) -> None: ...


class VisionAnalyzer(Protocol):
    def analyze(self, frame: VideoFrame) -> list[PixelTrackObservation]: ...


class AlarmSink(Protocol):
    def publish(self, event: AlarmEvent) -> None: ...
