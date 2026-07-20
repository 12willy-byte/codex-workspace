from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field, field_validator, model_validator

from aquaguard.world.models import FusedTrack, RiskForecast


TEMPORAL_FEATURE_SCHEMA_V1 = (
    "elapsed_seconds",
    "pool_x",
    "pool_y",
    "confidence",
    "head_submerged",
    "body_vertical",
    "struggle",
    "motion",
    "occlusion",
    "head_in_water_region",
    "water_relation_confidence",
    "wrist_motion",
    "wrist_motion_confidence",
)


class TemporalRiskModelManifest(BaseModel):
    model_config = {"frozen": True}

    schema_version: Literal["1.0"] = "1.0"
    model_version: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_schema: Literal["aquaguard.temporal_features.v1"]
    minimum_samples: int = Field(ge=2)
    maximum_samples: int = Field(ge=2)
    minimum_duration_seconds: float = Field(gt=0)
    maximum_gap_seconds: float = Field(gt=0)
    validated_for_assistive_alerting: bool = False
    validation_bundle_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    validation_report_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    approval_artifact_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )

    @field_validator("model_version")
    @classmethod
    def normalize_version(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("model_version must not be blank")
        return value

    @model_validator(mode="after")
    def validate_limits_and_approval(self) -> TemporalRiskModelManifest:
        if self.maximum_samples < self.minimum_samples:
            raise ValueError("maximum_samples cannot be below minimum_samples")
        evidence = (
            self.validation_bundle_sha256,
            self.validation_report_sha256,
            self.approval_artifact_sha256,
        )
        if self.validated_for_assistive_alerting and any(item is None for item in evidence):
            raise ValueError("validated model requires bundle, report, and approval digests")
        if not self.validated_for_assistive_alerting and any(item is not None for item in evidence):
            raise ValueError("unvalidated model cannot claim validation or approval digests")
        return self

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
    def load(cls, path: Path) -> TemporalRiskModelManifest:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class TemporalRiskModelOutput(BaseModel):
    current_risk: float = Field(ge=0, le=1)
    future_risk: float = Field(ge=0, le=1)
    time_to_critical_seconds: float | None = Field(default=None, ge=0)
    uncertainty: float = Field(ge=0, le=1)
    reasons: tuple[str, ...] = ()

    @field_validator("reasons")
    @classmethod
    def validate_reasons(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError("temporal model reasons must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("temporal model reasons must be unique")
        return normalized


class TemporalRiskModel(Protocol):
    def predict(self, sequence: tuple[tuple[float, ...], ...]) -> object: ...


class TemporalModelPackageVerificationReport(BaseModel):
    model_version: str
    manifest_sha256: str
    artifact_sha256: str
    feature_schema: str
    manifest_declares_assistive_validation: bool
    validation_bundle_sha256: str | None
    validation_report_sha256: str | None
    approval_artifact_sha256: str | None
    valid: Literal[True] = True


class TemporalModelPackageVerifier:
    def verify(
        self,
        manifest: TemporalRiskModelManifest,
        model_artifact_path: Path,
    ) -> TemporalModelPackageVerificationReport:
        if not model_artifact_path.is_file():
            raise ValueError("temporal model artifact must be an existing local file")
        artifact_sha256 = _sha256_file(model_artifact_path)
        if artifact_sha256 != manifest.artifact_sha256:
            raise ValueError("temporal model artifact SHA-256 does not match manifest")
        return TemporalModelPackageVerificationReport(
            model_version=manifest.model_version,
            manifest_sha256=manifest.sha256(),
            artifact_sha256=artifact_sha256,
            feature_schema=manifest.feature_schema,
            manifest_declares_assistive_validation=manifest.validated_for_assistive_alerting,
            validation_bundle_sha256=manifest.validation_bundle_sha256,
            validation_report_sha256=manifest.validation_report_sha256,
            approval_artifact_sha256=manifest.approval_artifact_sha256,
        )


class ValidatedTemporalRiskPredictor:
    """Validate model provenance and fail closed on unusable sequences or inference."""

    def __init__(
        self,
        model: TemporalRiskModel,
        manifest: TemporalRiskModelManifest,
        model_artifact_path: Path,
        *,
        approval_verified: bool = False,
    ) -> None:
        TemporalModelPackageVerifier().verify(manifest, model_artifact_path)
        if approval_verified and not manifest.validated_for_assistive_alerting:
            raise ValueError("cannot verify approval for a manifest without validation evidence")
        self.model = model
        self.manifest = manifest
        self.model_artifact_path = model_artifact_path
        self.approval_verified = approval_verified

    def predict(self, history: tuple[FusedTrack, ...]) -> RiskForecast:
        if not history:
            raise ValueError("History must not be empty")
        current = history[-1]
        failure = self._history_failure(history)
        if failure is not None:
            return self._unavailable(current, failure)
        selected = history[-self.manifest.maximum_samples :]
        sequence = self._features(selected)
        try:
            output = TemporalRiskModelOutput.model_validate(self.model.predict(sequence))
        except Exception:
            return self._unavailable(current, "temporal_model_inference_failed")
        sensor_uncertainty = min(
            1.0,
            0.5 * current.occlusion + 0.3 * (1.0 - current.confidence),
        )
        return RiskForecast(
            track_id=current.track_id,
            current_risk=output.current_risk,
            future_risk=output.future_risk,
            time_to_critical_seconds=output.time_to_critical_seconds,
            uncertainty=max(output.uncertainty, sensor_uncertainty),
            reasons=output.reasons,
            predictor_kind="learned_temporal",
            model_version=self.manifest.model_version,
            assistive_alerting_eligible=(
                self.manifest.validated_for_assistive_alerting and self.approval_verified
            ),
        )

    def _history_failure(self, history: tuple[FusedTrack, ...]) -> str | None:
        track_ids = {item.track_id for item in history}
        if len(track_ids) != 1:
            return "temporal_model_mixed_track_history"
        timestamps = [item.timestamp for item in history]
        if any(not math.isfinite(value) for value in timestamps):
            return "temporal_model_non_finite_timestamp"
        if any(second <= first for first, second in zip(timestamps, timestamps[1:])):
            return "temporal_model_non_increasing_timestamps"
        selected = history[-self.manifest.maximum_samples :]
        if len(selected) < self.manifest.minimum_samples:
            return "temporal_model_insufficient_samples"
        if selected[-1].timestamp - selected[0].timestamp < self.manifest.minimum_duration_seconds:
            return "temporal_model_insufficient_duration"
        if any(
            second.timestamp - first.timestamp > self.manifest.maximum_gap_seconds
            for first, second in zip(selected, selected[1:])
        ):
            return "temporal_model_sequence_gap"
        return None

    @staticmethod
    def _features(history: tuple[FusedTrack, ...]) -> tuple[tuple[float, ...], ...]:
        started_at = history[0].timestamp
        return tuple(
            (
                item.timestamp - started_at,
                item.pool_x,
                item.pool_y,
                item.confidence,
                item.head_submerged,
                item.body_vertical,
                item.struggle,
                item.motion,
                item.occlusion,
                item.head_in_water_region,
                item.water_relation_confidence,
                item.wrist_motion,
                item.wrist_motion_confidence,
            )
            for item in history
        )

    def _unavailable(self, current: FusedTrack, reason: str) -> RiskForecast:
        return RiskForecast(
            track_id=current.track_id,
            current_risk=0.0,
            future_risk=0.0,
            time_to_critical_seconds=None,
            uncertainty=1.0,
            reasons=(reason,),
            predictor_kind="learned_temporal",
            model_version=self.manifest.model_version,
            assistive_alerting_eligible=False,
        )


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
