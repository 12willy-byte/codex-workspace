from collections.abc import Callable

from aquaguard.video.models import VideoFrame
from aquaguard.vision.models import PixelTrackObservation


class ScriptedFrameAnalyzer:
    """Test/demo adapter; production analyzers must implement the same method."""

    def __init__(self, script: Callable[[VideoFrame], list[PixelTrackObservation]]):
        self.script = script

    def analyze(self, frame: VideoFrame) -> list[PixelTrackObservation]:
        return self.script(frame)
