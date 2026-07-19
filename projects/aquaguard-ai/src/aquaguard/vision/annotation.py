from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from aquaguard.vision.benchmark import FrameRiskLabels, TrackRiskLabel


class RiskJudgment(StrEnum):
    SAFE = "safe"
    DANGEROUS = "dangerous"
    UNCERTAIN = "uncertain"


class AnnotationProtocol(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    protocol_id: str = Field(min_length=1)
    annotation_version: str = Field(min_length=1)
    effective_date: date
    language: str = Field(min_length=1)
    temporal_context_before_seconds: float = Field(ge=0)
    temporal_context_after_seconds: float = Field(ge=0)
    dangerous_observable_criteria: tuple[str, ...] = Field(min_length=1)
    known_non_dangerous_contexts: tuple[str, ...] = Field(min_length=1)
    uncertainty_triggers: tuple[str, ...] = Field(min_length=1)
    occlusion_policy: str = Field(min_length=1)
    identity_discontinuity_policy: str = Field(min_length=1)
    reviewer_training_requirements: tuple[str, ...] = Field(min_length=1)
    privacy_requirements: tuple[str, ...] = Field(min_length=1)
    approved_by: tuple[str, ...] = Field(min_length=1)

    @field_validator(
        "protocol_id",
        "annotation_version",
        "language",
        "occlusion_policy",
        "identity_discontinuity_policy",
    )
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("annotation protocol fields must not be blank")
        return value

    @field_validator(
        "dangerous_observable_criteria",
        "known_non_dangerous_contexts",
        "uncertainty_triggers",
        "reviewer_training_requirements",
        "privacy_requirements",
        "approved_by",
    )
    @classmethod
    def normalize_list(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError("annotation protocol lists must not contain blank values")
        if len(normalized) != len(set(normalized)):
            raise ValueError("annotation protocol lists must not contain duplicates")
        return normalized

    @model_validator(mode="after")
    def require_temporal_context(self) -> AnnotationProtocol:
        if self.temporal_context_before_seconds + self.temporal_context_after_seconds <= 0:
            raise ValueError("annotation protocol must define non-zero temporal context")
        return self

    @classmethod
    def load(cls, path: Path) -> AnnotationProtocol:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def canonical_bytes(self, *, pretty: bool = False) -> bytes:
        options = {"ensure_ascii": False, "sort_keys": True}
        if pretty:
            options["indent"] = 2
        serialized = json.dumps(self.model_dump(mode="json"), **options)
        return (serialized + "\n").encode()

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.canonical_bytes(pretty=True))


class AnnotationReview(BaseModel):
    protocol_id: str = Field(min_length=1)
    annotation_version: str = Field(min_length=1)
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    recording_id: str = Field(min_length=1)
    venue_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    timestamp: float = Field(ge=0)
    track_id: str = Field(min_length=1)
    reviewer_id: str = Field(min_length=1)
    judgment: RiskJudgment

    @field_validator(
        "protocol_id",
        "annotation_version",
        "recording_id",
        "venue_id",
        "session_id",
        "track_id",
        "reviewer_id",
    )
    @classmethod
    def normalize_identifier(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("annotation identifiers must not be blank")
        return value


class AnnotationAdjudication(BaseModel):
    protocol_id: str = Field(min_length=1)
    annotation_version: str = Field(min_length=1)
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    recording_id: str = Field(min_length=1)
    timestamp: float = Field(ge=0)
    track_id: str = Field(min_length=1)
    adjudicator_id: str = Field(min_length=1)
    dangerous: bool
    reason: str = Field(min_length=1)

    @field_validator(
        "protocol_id",
        "annotation_version",
        "recording_id",
        "track_id",
        "adjudicator_id",
        "reason",
    )
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("adjudication fields must not be blank")
        return value


class ResolvedAnnotations(BaseModel):
    protocol_id: str
    annotation_version: str
    protocol_sha256: str
    recording_id: str
    venue_id: str
    session_id: str
    frames: tuple[FrameRiskLabels, ...]
    reviewed_items: int
    adjudicated_items: int


AnnotationKey = tuple[str, float, str]


class DualReviewResolver:
    """Resolve exactly two independent reviews, requiring third-party adjudication on ambiguity."""

    def resolve(
        self,
        reviews: list[AnnotationReview],
        adjudications: list[AnnotationAdjudication],
    ) -> ResolvedAnnotations:
        if not reviews:
            raise ValueError("at least one annotation review is required")
        recording_ids = {item.recording_id for item in reviews}
        venue_ids = {item.venue_id for item in reviews}
        session_ids = {item.session_id for item in reviews}
        if len(recording_ids) != 1 or len(venue_ids) != 1 or len(session_ids) != 1:
            raise ValueError("reviews must belong to one recording, venue, and session")
        protocol_references = {
            (item.protocol_id, item.annotation_version, item.protocol_sha256) for item in reviews
        }
        protocol_references.update(
            (item.protocol_id, item.annotation_version, item.protocol_sha256)
            for item in adjudications
        )
        if len(protocol_references) != 1:
            raise ValueError("reviews and adjudications must use one annotation protocol")

        grouped_reviews: dict[AnnotationKey, list[AnnotationReview]] = defaultdict(list)
        for review in reviews:
            grouped_reviews[self._review_key(review)].append(review)
        adjudications_by_key: dict[AnnotationKey, AnnotationAdjudication] = {}
        for adjudication in adjudications:
            key = self._adjudication_key(adjudication)
            if key in adjudications_by_key:
                raise ValueError("annotation item has duplicate adjudications")
            adjudications_by_key[key] = adjudication

        resolved_by_time: dict[float, list[TrackRiskLabel]] = defaultdict(list)
        used_adjudications: set[AnnotationKey] = set()
        for key in sorted(grouped_reviews, key=lambda item: (item[1], item[2])):
            item_reviews = grouped_reviews[key]
            reviewer_ids = {item.reviewer_id for item in item_reviews}
            if len(item_reviews) != 2 or len(reviewer_ids) != 2:
                raise ValueError("each annotation item requires exactly two independent reviewers")
            judgments = {item.judgment for item in item_reviews}
            requires_adjudication = len(judgments) != 1 or RiskJudgment.UNCERTAIN in judgments
            if requires_adjudication:
                adjudication = adjudications_by_key.get(key)
                if adjudication is None:
                    raise ValueError("disputed or uncertain annotation requires adjudication")
                if adjudication.adjudicator_id in reviewer_ids:
                    raise ValueError("adjudicator must be independent of both reviewers")
                dangerous = adjudication.dangerous
                used_adjudications.add(key)
            else:
                dangerous = next(iter(judgments)) == RiskJudgment.DANGEROUS
                if key in adjudications_by_key:
                    raise ValueError("agreed annotation must not have redundant adjudication")
            resolved_by_time[key[1]].append(TrackRiskLabel(track_id=key[2], dangerous=dangerous))

        orphaned = set(adjudications_by_key).difference(used_adjudications)
        if orphaned:
            raise ValueError("adjudication does not match a disputed annotation item")
        frames = tuple(
            FrameRiskLabels(
                timestamp=timestamp,
                tracks=tuple(sorted(labels, key=lambda item: item.track_id)),
            )
            for timestamp, labels in sorted(resolved_by_time.items())
        )
        protocol_id, annotation_version, protocol_sha256 = next(iter(protocol_references))
        return ResolvedAnnotations(
            protocol_id=protocol_id,
            annotation_version=annotation_version,
            protocol_sha256=protocol_sha256,
            recording_id=next(iter(recording_ids)),
            venue_id=next(iter(venue_ids)),
            session_id=next(iter(session_ids)),
            frames=frames,
            reviewed_items=len(grouped_reviews),
            adjudicated_items=len(used_adjudications),
        )

    @staticmethod
    def _review_key(review: AnnotationReview) -> AnnotationKey:
        return review.recording_id, review.timestamp, review.track_id

    @staticmethod
    def _adjudication_key(adjudication: AnnotationAdjudication) -> AnnotationKey:
        return adjudication.recording_id, adjudication.timestamp, adjudication.track_id


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class DatasetRecording(BaseModel):
    recording_id: str = Field(min_length=1)
    venue_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)

    @field_validator("recording_id", "venue_id", "session_id")
    @classmethod
    def normalize_identifier(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("dataset identifiers must not be blank")
        return value


class DatasetSplitAssignment(BaseModel):
    recording_id: str
    venue_id: str
    session_id: str
    split: DatasetSplit


class VenueGroupedSplitter:
    """Deterministically assign whole venues to one split to prevent visual leakage."""

    def __init__(
        self,
        seed: str,
        train_fraction: float = 0.7,
        validation_fraction: float = 0.15,
    ) -> None:
        if not seed.strip():
            raise ValueError("split seed must not be blank")
        if not 0 < train_fraction < 1:
            raise ValueError("train_fraction must be in (0, 1)")
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1)")
        if train_fraction + validation_fraction >= 1:
            raise ValueError("train and validation fractions must leave a test partition")
        self.seed = seed
        self.train_fraction = train_fraction
        self.validation_fraction = validation_fraction

    def assign(self, recordings: list[DatasetRecording]) -> list[DatasetSplitAssignment]:
        recording_ids = [item.recording_id for item in recordings]
        if len(recording_ids) != len(set(recording_ids)):
            raise ValueError("recording ids must be unique")
        venues = sorted({item.venue_id for item in recordings}, key=self._venue_digest)
        if len(venues) < 3:
            raise ValueError("at least three venues are required for leakage-safe three-way split")
        train_count = max(1, int(len(venues) * self.train_fraction))
        validation_count = max(1, int(len(venues) * self.validation_fraction))
        if train_count + validation_count >= len(venues):
            train_count = len(venues) - validation_count - 1
        venue_splits = {
            venue_id: (
                DatasetSplit.TRAIN
                if index < train_count
                else DatasetSplit.VALIDATION
                if index < train_count + validation_count
                else DatasetSplit.TEST
            )
            for index, venue_id in enumerate(venues)
        }
        return [
            DatasetSplitAssignment(
                recording_id=item.recording_id,
                venue_id=item.venue_id,
                session_id=item.session_id,
                split=venue_splits[item.venue_id],
            )
            for item in sorted(recordings, key=lambda value: value.recording_id)
        ]

    def _venue_digest(self, venue_id: str) -> bytes:
        return hashlib.sha256(f"{self.seed}\0{venue_id}".encode()).digest()
