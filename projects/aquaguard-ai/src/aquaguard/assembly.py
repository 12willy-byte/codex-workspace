from collections.abc import Callable
from typing import Protocol

from pydantic import BaseModel, Field

from aquaguard.evidence.recorder import EvidenceRecorder
from aquaguard.runtime import FrameAnalyzer, VideoRuntimeManager, VideoWorldRuntime, WorldFrameProcessor
from aquaguard.video.source import CaptureFactory, OpenCVFrameSource
from aquaguard.vision.adapter import CalibratedObservationAdapter
from aquaguard.vision.calibration import HomographyProjector


class CameraRuntimeConfig(BaseModel):
    camera_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    homography: tuple[float, float, float, float, float, float, float, float, float]
    enabled: bool = False
    reconnect: bool = True


class FrameAnalyzerFactory(Protocol):
    def __call__(self, config: CameraRuntimeConfig) -> FrameAnalyzer: ...


class UnavailableFrameAnalyzerFactory:
    def __call__(self, config: CameraRuntimeConfig) -> FrameAnalyzer:
        raise RuntimeError(
            f"Camera {config.camera_id} is enabled but no production frame analyzer is configured"
        )


class VideoRuntimeAssembler:
    """Build configured camera runtimes without silently substituting fake vision."""

    def __init__(
        self,
        analyzer_factory: FrameAnalyzerFactory,
        processor: WorldFrameProcessor,
        evidence: EvidenceRecorder,
        *,
        capture_factory: CaptureFactory | None = None,
        clock: Callable[[], float] | None = None,
    ):
        self.analyzer_factory = analyzer_factory
        self.processor = processor
        self.evidence = evidence
        self.capture_factory = capture_factory
        self.clock = clock

    def build(self, configs: tuple[CameraRuntimeConfig, ...]) -> VideoRuntimeManager:
        manager = VideoRuntimeManager()
        seen: set[str] = set()
        for config in configs:
            if config.camera_id in seen:
                raise ValueError(f"Duplicate camera configuration: {config.camera_id}")
            seen.add(config.camera_id)
            if not config.enabled:
                continue
            analyzer = self.analyzer_factory(config)
            source_options = {
                "capture_factory": self.capture_factory,
                "reconnect": config.reconnect,
            }
            if self.clock is not None:
                source_options["clock"] = self.clock
            source = OpenCVFrameSource(
                config.camera_id,
                config.source,
                **source_options,
            )
            adapter = CalibratedObservationAdapter(
                {
                    config.camera_id: HomographyProjector(
                        config.camera_id, config.homography
                    )
                }
            )
            manager.add(
                config.camera_id,
                VideoWorldRuntime(
                    source,
                    analyzer,
                    adapter,
                    self.processor,
                    observers=(self.evidence,),
                ),
            )
        return manager
