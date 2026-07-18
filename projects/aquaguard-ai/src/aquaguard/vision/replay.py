from collections.abc import Iterable, Iterator

from aquaguard.vision.adapter import CalibratedObservationAdapter
from aquaguard.vision.models import PixelTrackObservation
from aquaguard.world.pipeline import WorldModelPipeline


class ObservationReplay:
    """Deterministic replay harness for detector/tracker outputs from recorded frames."""

    def __init__(
        self,
        frames: Iterable[list[PixelTrackObservation]],
        adapter: CalibratedObservationAdapter,
        pipeline: WorldModelPipeline,
    ):
        self.frames = frames
        self.adapter = adapter
        self.pipeline = pipeline

    def run(self) -> Iterator[dict]:
        for frame in self.frames:
            yield self.pipeline.process(self.adapter.convert(frame))
