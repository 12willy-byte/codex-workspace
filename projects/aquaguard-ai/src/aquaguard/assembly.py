from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from aquaguard.evidence.recorder import EvidenceRecorder
from aquaguard.runtime import FrameAnalyzer, VideoRuntimeManager, VideoWorldRuntime, WorldFrameProcessor
from aquaguard.video.source import CaptureFactory, OpenCVFrameSource
from aquaguard.vision.adapter import CalibratedObservationAdapter
from aquaguard.vision.camera_calibration import CameraCalibrationArtifact
from aquaguard.vision.calibration import HomographyProjector
from aquaguard.vision.regions import PolygonRegion
from aquaguard.vision.ultralytics import (
    UltralyticsPoseTrackAnalyzer,
    UltralyticsTrackAnalyzer,
)


class CameraRuntimeConfig(BaseModel):
    camera_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    homography: tuple[float, float, float, float, float, float, float, float, float]
    enabled: bool = False
    reconnect: bool = True
    analyzer: Literal["none", "ultralytics_tracking", "ultralytics_pose_tracking"] = "none"
    model_path: str | None = None
    confidence: float = Field(default=0.5, ge=0, le=1)
    device: str | None = None
    keypoint_confidence: float = Field(default=0.3, ge=0, le=1)
    water_roi: tuple[tuple[float, float], ...] | None = None
    track_ttl_seconds: float = Field(default=30, gt=0)
    calibration_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def from_calibration(
        cls,
        calibration: CameraCalibrationArtifact,
        *,
        source: str,
        **options: object,
    ) -> CameraRuntimeConfig:
        return cls(
            camera_id=calibration.camera_id,
            source=source,
            homography=calibration.homography,
            water_roi=calibration.water_roi,
            calibration_sha256=calibration.sha256(),
            **options,
        )


class FrameAnalyzerFactory(Protocol):
    def __call__(self, config: CameraRuntimeConfig) -> FrameAnalyzer: ...


class UnavailableFrameAnalyzerFactory:
    def __call__(self, config: CameraRuntimeConfig) -> FrameAnalyzer:
        raise RuntimeError(
            f"Camera {config.camera_id} is enabled but no production frame analyzer is configured"
        )


class ConfiguredFrameAnalyzerFactory:
    def __call__(self, config: CameraRuntimeConfig) -> FrameAnalyzer:
        if config.analyzer == "none":
            return UnavailableFrameAnalyzerFactory()(config)
        if not config.model_path:
            raise ValueError(f"Camera {config.camera_id} requires a local model_path")
        analyzer_type = (
            UltralyticsPoseTrackAnalyzer
            if config.analyzer == "ultralytics_pose_tracking"
            else UltralyticsTrackAnalyzer
        )
        options = {
            "confidence": config.confidence,
            "device": config.device,
            "track_ttl_seconds": config.track_ttl_seconds,
        }
        if analyzer_type is UltralyticsPoseTrackAnalyzer:
            options["keypoint_confidence"] = config.keypoint_confidence
            options["water_region"] = (
                PolygonRegion(config.water_roi) if config.water_roi is not None else None
            )
        return analyzer_type(config.model_path, **options)


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
