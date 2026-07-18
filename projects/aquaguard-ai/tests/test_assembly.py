import pytest
from pydantic import ValidationError

from aquaguard.assembly import (
    CameraRuntimeConfig,
    UnavailableFrameAnalyzerFactory,
    VideoRuntimeAssembler,
)
from aquaguard.evidence import EvidenceRecorder
from aquaguard.vision import ScriptedFrameAnalyzer
from aquaguard.world import TemporalFusionCoordinator, WorldModelPipeline


IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def assembler(analyzer_factory=UnavailableFrameAnalyzerFactory()) -> VideoRuntimeAssembler:
    return VideoRuntimeAssembler(
        analyzer_factory,
        TemporalFusionCoordinator(WorldModelPipeline()),
        EvidenceRecorder(),
    )


def test_disabled_camera_does_not_require_analyzer() -> None:
    config = CameraRuntimeConfig(
        camera_id="cam-a", source="rtsp://camera/stream", homography=IDENTITY
    )

    manager = assembler().build((config,))

    assert manager.statuses() == []


def test_enabled_camera_fails_loudly_without_production_analyzer() -> None:
    config = CameraRuntimeConfig(
        camera_id="cam-a",
        source="rtsp://camera/stream",
        homography=IDENTITY,
        enabled=True,
    )

    with pytest.raises(RuntimeError, match="no production frame analyzer"):
        assembler().build((config,))


def test_enabled_camera_is_registered_with_explicit_analyzer() -> None:
    config = CameraRuntimeConfig(
        camera_id="cam-a", source="recording.mp4", homography=IDENTITY, enabled=True
    )
    def factory(_: CameraRuntimeConfig) -> ScriptedFrameAnalyzer:
        return ScriptedFrameAnalyzer(lambda frame: [])

    manager = assembler(factory).build((config,))

    assert manager.statuses()[0]["camera_id"] == "cam-a"


def test_duplicate_camera_configuration_is_rejected() -> None:
    config = CameraRuntimeConfig(camera_id="cam-a", source="0", homography=IDENTITY)

    with pytest.raises(ValueError, match="Duplicate camera"):
        assembler().build((config, config))


def test_homography_requires_exactly_nine_values() -> None:
    with pytest.raises(ValidationError):
        CameraRuntimeConfig(camera_id="cam-a", source="0", homography=(1.0, 2.0))
