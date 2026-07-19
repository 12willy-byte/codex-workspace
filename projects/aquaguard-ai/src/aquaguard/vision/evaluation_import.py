from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from aquaguard.vision.benchmark import FrameRiskLabels
from aquaguard.vision.evaluation import (
    CameraCalibrationRecord,
    EvaluationProvenance,
    PixelObservationRecord,
    ReplayEvaluationBundle,
    ReplayEvaluationFrame,
)
from aquaguard.world.models import PoolGeometry

ModelT = TypeVar("ModelT", bound=BaseModel)


class EvaluationCalibrationFile(BaseModel):
    geometry: PoolGeometry = PoolGeometry()
    calibrations: tuple[CameraCalibrationRecord, ...]

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def load(cls, path: Path) -> EvaluationCalibrationFile:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class ReplayEvaluationImporter:
    def build(
        self,
        *,
        name: str,
        source_recording_path: Path,
        observations_path: Path,
        labels_path: Path,
        calibration_path: Path,
        model_version: str,
        configuration_version: str,
        annotation_version: str,
        expected_source_sha256: str | None = None,
    ) -> ReplayEvaluationBundle:
        source_sha256 = sha256_file(source_recording_path)
        if expected_source_sha256 is not None:
            if not re.fullmatch(r"[0-9a-f]{64}", expected_source_sha256):
                raise ValueError("expected source SHA-256 must be 64 lowercase hex characters")
            if source_sha256 != expected_source_sha256:
                raise ValueError("source recording SHA-256 does not match expected digest")

        observations = _load_jsonl(observations_path, PixelObservationRecord)
        labels = _load_jsonl(labels_path, FrameRiskLabels)
        calibration = EvaluationCalibrationFile.load(calibration_path)

        observations_by_time: dict[float, list[PixelObservationRecord]] = defaultdict(list)
        for observation in observations:
            observations_by_time[observation.timestamp].append(observation)

        label_timestamps = [frame.timestamp for frame in labels]
        if label_timestamps != sorted(label_timestamps) or len(label_timestamps) != len(
            set(label_timestamps)
        ):
            raise ValueError("label timestamps must be unique and increasing")
        unlabelled_timestamps = set(observations_by_time).difference(label_timestamps)
        if unlabelled_timestamps:
            formatted = ", ".join(str(value) for value in sorted(unlabelled_timestamps))
            raise ValueError(f"observations contain unlabelled timestamps: {formatted}")

        frames = tuple(
            ReplayEvaluationFrame(
                timestamp=frame_labels.timestamp,
                observations=tuple(observations_by_time.get(frame_labels.timestamp, ())),
                labels=frame_labels,
            )
            for frame_labels in labels
        )
        return ReplayEvaluationBundle(
            name=name,
            provenance=EvaluationProvenance(
                source_recording_sha256=source_sha256,
                model_version=model_version,
                configuration_version=configuration_version,
                annotation_version=annotation_version,
            ),
            geometry=calibration.geometry,
            calibrations=calibration.calibrations,
            frames=frames,
        )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path, model: type[ModelT]) -> list[ModelT]:
    records: list[ModelT] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
            records.append(model.model_validate(payload))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid {path.name} line {line_number}: {exc}") from exc
    return records
