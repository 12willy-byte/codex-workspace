from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from aquaguard.vision.annotation_governance import _require_aware
from aquaguard.vision.calibration import HomographyProjector
from aquaguard.vision.regions import PolygonRegion


class CalibrationCorrespondence(BaseModel):
    image_x: float
    image_y: float
    pool_x: float
    pool_y: float

    @field_validator("image_x", "image_y", "pool_x", "pool_y")
    @classmethod
    def require_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("calibration correspondence values must be finite")
        return value


class CameraCalibrationArtifact(BaseModel):
    """Traceable, validated image-to-pool calibration for one physical camera."""

    model_config = {"frozen": True}

    schema_version: Literal["1.0"] = "1.0"
    camera_id: str = Field(min_length=1)
    calibration_version: str = Field(min_length=1)
    source_frame_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)
    homography: tuple[float, float, float, float, float, float, float, float, float]
    water_roi: tuple[tuple[float, float], ...] = Field(min_length=3)
    correspondences: tuple[CalibrationCorrespondence, ...] = Field(min_length=4)
    maximum_reprojection_error_meters: float = Field(gt=0)
    calibrated_at: datetime
    calibrated_by: str = Field(min_length=1)

    @field_validator("camera_id", "calibration_version", "calibrated_by")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("calibration text fields must not be blank")
        return value

    @field_validator("calibrated_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        return _require_aware(value)

    @model_validator(mode="after")
    def validate_geometry(self) -> CameraCalibrationArtifact:
        self.projector()
        region = self.water_region()
        for x, y in region.points:
            self._require_image_point(x, y, "water ROI")

        image_points = [(item.image_x, item.image_y) for item in self.correspondences]
        pool_points = [(item.pool_x, item.pool_y) for item in self.correspondences]
        if len(set(image_points)) != len(image_points):
            raise ValueError("calibration image correspondences must be unique")
        if len(set(pool_points)) != len(pool_points):
            raise ValueError("calibration pool correspondences must be unique")
        if not self._has_two_dimensional_coverage(image_points):
            raise ValueError("calibration image correspondences must span two dimensions")
        if not self._has_two_dimensional_coverage(pool_points):
            raise ValueError("calibration pool correspondences must span two dimensions")
        for item in self.correspondences:
            self._require_image_point(item.image_x, item.image_y, "calibration correspondence")

        if self.max_reprojection_error_meters() > self.maximum_reprojection_error_meters:
            raise ValueError("calibration exceeds maximum reprojection error")
        return self

    def _require_image_point(self, x: float, y: float, label: str) -> None:
        if not (0 <= x < self.frame_width and 0 <= y < self.frame_height):
            raise ValueError(f"{label} must stay within the calibrated frame")

    @staticmethod
    def _has_two_dimensional_coverage(points: list[tuple[float, float]]) -> bool:
        first = points[0]
        for second_index in range(1, len(points) - 1):
            second = points[second_index]
            for third in points[second_index + 1 :]:
                area_twice = (second[0] - first[0]) * (third[1] - first[1]) - (
                    second[1] - first[1]
                ) * (third[0] - first[0])
                if abs(area_twice) > 1e-9:
                    return True
        return False

    def projector(self) -> HomographyProjector:
        return HomographyProjector(self.camera_id, self.homography)

    def water_region(self) -> PolygonRegion:
        return PolygonRegion(self.water_roi)

    def reprojection_errors_meters(self) -> tuple[float, ...]:
        projector = self.projector()
        errors = []
        for item in self.correspondences:
            pool_x, pool_y = projector.project(item.image_x, item.image_y)
            errors.append(math.hypot(pool_x - item.pool_x, pool_y - item.pool_y))
        return tuple(errors)

    def mean_reprojection_error_meters(self) -> float:
        errors = self.reprojection_errors_meters()
        return sum(errors) / len(errors)

    def max_reprojection_error_meters(self) -> float:
        return max(self.reprojection_errors_meters())

    def canonical_bytes(self, *, pretty: bool = False) -> bytes:
        options = {"ensure_ascii": False, "sort_keys": True}
        if pretty:
            options["indent"] = 2
        return (json.dumps(self.model_dump(mode="json"), **options) + "\n").encode()

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.canonical_bytes(pretty=True))

    @classmethod
    def load(cls, path: Path) -> CameraCalibrationArtifact:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class CameraCalibrationVerificationReport(BaseModel):
    camera_id: str
    calibration_version: str
    calibration_sha256: str
    control_points: int
    water_roi_area_pixels: float
    mean_reprojection_error_meters: float
    max_reprojection_error_meters: float
    allowed_maximum_reprojection_error_meters: float
    valid: Literal[True] = True


class CameraCalibrationVerifier:
    def verify(self, artifact: CameraCalibrationArtifact) -> CameraCalibrationVerificationReport:
        return CameraCalibrationVerificationReport(
            camera_id=artifact.camera_id,
            calibration_version=artifact.calibration_version,
            calibration_sha256=artifact.sha256(),
            control_points=len(artifact.correspondences),
            water_roi_area_pixels=artifact.water_region().area(),
            mean_reprojection_error_meters=artifact.mean_reprojection_error_meters(),
            max_reprojection_error_meters=artifact.max_reprojection_error_meters(),
            allowed_maximum_reprojection_error_meters=(
                artifact.maximum_reprojection_error_meters
            ),
        )
