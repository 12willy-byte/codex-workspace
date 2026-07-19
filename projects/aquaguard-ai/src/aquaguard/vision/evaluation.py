from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from aquaguard.vision.adapter import CalibratedObservationAdapter
from aquaguard.vision.benchmark import (
    BinaryRiskBenchmark,
    FrameRiskLabels,
    RiskBenchmarkManifest,
    RiskBenchmarkReport,
)
from aquaguard.vision.calibration import HomographyProjector
from aquaguard.vision.models import PixelTrackObservation
from aquaguard.vision.replay import ObservationReplay
from aquaguard.world.models import PoolGeometry
from aquaguard.world.pipeline import WorldModelPipeline


class EvaluationProvenance(BaseModel):
    source_recording_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_version: str = Field(min_length=1)
    configuration_version: str = Field(min_length=1)
    annotation_version: str = Field(min_length=1)

    @field_validator("model_version", "configuration_version", "annotation_version")
    @classmethod
    def normalize_version(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("provenance versions must not be blank")
        return value


class CameraCalibrationRecord(BaseModel):
    camera_id: str = Field(min_length=1)
    homography: tuple[float, float, float, float, float, float, float, float, float]

    @field_validator("camera_id")
    @classmethod
    def normalize_camera_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("camera_id must not be blank")
        return value

    def projector(self) -> HomographyProjector:
        return HomographyProjector(self.camera_id, self.homography)


class PixelObservationRecord(BaseModel):
    camera_id: str = Field(min_length=1)
    local_track_id: str = Field(min_length=1)
    timestamp: float = Field(ge=0)
    anchor_x: float
    anchor_y: float
    global_track_id: str | None = None
    confidence: float = Field(default=1.0, ge=0, le=1)
    head_submerged: float = Field(default=0.0, ge=0, le=1)
    body_vertical: float = Field(default=0.0, ge=0, le=1)
    struggle: float = Field(default=0.0, ge=0, le=1)
    motion: float = Field(default=0.0, ge=0, le=1)
    occlusion: float = Field(default=0.0, ge=0, le=1)
    head_in_water_region: float = Field(default=0.0, ge=0, le=1)
    water_relation_confidence: float = Field(default=0.0, ge=0, le=1)
    wrist_motion: float = Field(default=0.0, ge=0, le=1)
    wrist_motion_confidence: float = Field(default=0.0, ge=0, le=1)

    @field_validator("camera_id", "local_track_id")
    @classmethod
    def normalize_required_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("observation identifiers must not be blank")
        return value

    @field_validator("global_track_id")
    @classmethod
    def normalize_optional_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("global_track_id must not be blank")
        return value

    def observation(self) -> PixelTrackObservation:
        return PixelTrackObservation(**self.model_dump())


class ReplayEvaluationFrame(BaseModel):
    timestamp: float = Field(ge=0)
    observations: tuple[PixelObservationRecord, ...]
    labels: FrameRiskLabels

    @model_validator(mode="after")
    def require_aligned_timestamps(self) -> ReplayEvaluationFrame:
        if self.labels.timestamp != self.timestamp:
            raise ValueError("frame and label timestamps must match")
        if any(item.timestamp != self.timestamp for item in self.observations):
            raise ValueError("frame and observation timestamps must match")
        return self


class ReplayEvaluationBundle(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    name: str = Field(min_length=1)
    provenance: EvaluationProvenance
    geometry: PoolGeometry = PoolGeometry()
    calibrations: tuple[CameraCalibrationRecord, ...]
    frames: tuple[ReplayEvaluationFrame, ...]

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("bundle name must not be blank")
        return value

    @model_validator(mode="after")
    def validate_bundle(self) -> ReplayEvaluationBundle:
        self.geometry.validate()
        camera_ids = [item.camera_id for item in self.calibrations]
        if len(camera_ids) != len(set(camera_ids)):
            raise ValueError("camera calibrations must be unique")
        timestamps = [item.timestamp for item in self.frames]
        if timestamps != sorted(timestamps) or len(timestamps) != len(set(timestamps)):
            raise ValueError("bundle frame timestamps must be unique and increasing")
        calibrated = set(camera_ids)
        observed = {
            observation.camera_id for frame in self.frames for observation in frame.observations
        }
        missing = observed.difference(calibrated)
        if missing:
            raise ValueError(f"missing camera calibrations: {', '.join(sorted(missing))}")
        return self

    @classmethod
    def load(cls, path: Path) -> ReplayEvaluationBundle:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.canonical_bytes(pretty=True))

    def canonical_bytes(self, *, pretty: bool = False) -> bytes:
        options = {"ensure_ascii": False, "sort_keys": True}
        if pretty:
            options["indent"] = 2
        serialized = json.dumps(self.model_dump(mode="json"), **options)
        return (serialized + "\n").encode()

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class ReplayEvaluationResult:
    bundle_sha256: str
    provenance: dict
    report: RiskBenchmarkReport

    def to_dict(self) -> dict:
        return {
            "bundle_sha256": self.bundle_sha256,
            "provenance": self.provenance,
            "report": asdict(self.report),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class ReplayEvaluationRunner:
    def run(self, bundle: ReplayEvaluationBundle) -> ReplayEvaluationResult:
        projectors = {
            calibration.camera_id: calibration.projector() for calibration in bundle.calibrations
        }
        frames = [
            [record.observation() for record in frame.observations] for frame in bundle.frames
        ]
        manifest = RiskBenchmarkManifest(
            name=bundle.name,
            source=bundle.provenance.source_recording_sha256,
            frames=tuple(frame.labels for frame in bundle.frames),
        )
        replay = ObservationReplay(
            frames,
            CalibratedObservationAdapter(projectors),
            WorldModelPipeline(bundle.geometry),
        )
        report = BinaryRiskBenchmark().evaluate(replay.run(), manifest)
        return ReplayEvaluationResult(
            bundle_sha256=bundle.sha256(),
            provenance=bundle.provenance.model_dump(mode="json"),
            report=report,
        )
