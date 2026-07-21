from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from aquaguard.vision.annotation import (
    DatasetRecording,
    DatasetSplit,
    RiskJudgment,
    VenueGroupedSplitter,
)
from aquaguard.vision.evaluation_import import sha256_file


def _canonical_bytes(value: BaseModel, *, pretty: bool = False) -> bytes:
    options: dict[str, object] = {"ensure_ascii": False, "sort_keys": True}
    if pretty:
        options["indent"] = 2
    return (json.dumps(value.model_dump(mode="json"), **options) + "\n").encode()


class TrainingRecordingSource(BaseModel):
    """A local recording and the evidence required to admit it for model training."""

    schema_version: Literal["1.0"] = "1.0"
    recording_id: str = Field(min_length=1)
    venue_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    camera_id: str = Field(min_length=1)
    source_path: Path
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    duration_seconds: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    captured_at: datetime
    calibration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rights_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    consent_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    commercial_training_permitted: bool
    contains_minors: bool = False
    guardian_consent_artifact_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )

    @field_validator("recording_id", "venue_id", "session_id", "camera_id")
    @classmethod
    def normalize_identifier(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("recording identifiers must not be blank")
        return value

    @field_validator("captured_at")
    @classmethod
    def require_aware_capture_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("captured_at must include a timezone")
        return value

    @model_validator(mode="after")
    def require_guardian_consent_for_minors(self) -> TrainingRecordingSource:
        if self.contains_minors and self.guardian_consent_artifact_sha256 is None:
            raise ValueError("recordings containing minors require guardian consent evidence")
        if not self.contains_minors and self.guardian_consent_artifact_sha256 is not None:
            raise ValueError("guardian consent evidence is only valid when contains_minors is true")
        return self


class BehaviorEpisode(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    episode_id: str = Field(min_length=1)
    recording_id: str = Field(min_length=1)
    track_id: str = Field(min_length=1)
    participant_group_id: str = Field(min_length=1)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    judgment: RiskJudgment
    scenario_code: str = Field(min_length=1)
    annotation_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    resolved_annotation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator(
        "episode_id", "recording_id", "track_id", "participant_group_id", "scenario_code"
    )
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("episode fields must not be blank")
        return value

    @model_validator(mode="after")
    def require_positive_interval(self) -> BehaviorEpisode:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("episode end_seconds must be greater than start_seconds")
        return self


class ClipPlanningPolicy(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    context_before_seconds: float = Field(ge=0)
    context_after_seconds: float = Field(ge=0)
    minimum_clip_seconds: float = Field(gt=0)
    maximum_clip_seconds: float = Field(gt=0)
    train_fraction: float = Field(gt=0, lt=1, default=0.7)
    validation_fraction: float = Field(gt=0, lt=1, default=0.15)

    @model_validator(mode="after")
    def validate_policy(self) -> ClipPlanningPolicy:
        if self.maximum_clip_seconds < self.minimum_clip_seconds:
            raise ValueError("maximum_clip_seconds must be at least minimum_clip_seconds")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("split fractions must leave a test partition")
        return self


class TrainingRecordingProvenance(BaseModel):
    recording_id: str
    venue_id: str
    session_id: str
    camera_id: str
    source_sha256: str
    duration_seconds: float
    width: int
    height: int
    fps: float
    captured_at: datetime
    calibration_sha256: str
    rights_artifact_sha256: str
    consent_artifact_sha256: str
    guardian_consent_artifact_sha256: str | None


class TrainingClipRecord(BaseModel):
    clip_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    recording_id: str
    venue_id: str
    session_id: str
    camera_id: str
    source_sha256: str
    calibration_sha256: str
    episode_id: str
    track_id: str
    participant_group_id: str
    start_seconds: float
    end_seconds: float
    judgment: Literal[RiskJudgment.SAFE, RiskJudgment.DANGEROUS]
    scenario_code: str
    annotation_protocol_sha256: str
    resolved_annotation_sha256: str
    split: DatasetSplit


class TemporalTrainingDatasetManifest(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset_name: str = Field(min_length=1)
    created_at: datetime
    split_seed: str = Field(min_length=1)
    policy: ClipPlanningPolicy
    recordings: tuple[TrainingRecordingProvenance, ...] = Field(min_length=1)
    clips: tuple[TrainingClipRecord, ...] = Field(min_length=1)

    @field_validator("dataset_name", "split_seed")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("manifest text fields must not be blank")
        return value

    @field_validator("created_at")
    @classmethod
    def require_aware_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_isolation(self) -> TemporalTrainingDatasetManifest:
        clip_ids = [clip.clip_id for clip in self.clips]
        if len(clip_ids) != len(set(clip_ids)):
            raise ValueError("clip ids must be unique")
        recording_ids = {recording.recording_id for recording in self.recordings}
        if {clip.recording_id for clip in self.clips} != recording_ids:
            raise ValueError("every admitted recording must contribute at least one clip")
        venue_splits: dict[str, set[DatasetSplit]] = defaultdict(set)
        recording_splits: dict[str, set[DatasetSplit]] = defaultdict(set)
        participant_splits: dict[str, set[DatasetSplit]] = defaultdict(set)
        for clip in self.clips:
            venue_splits[clip.venue_id].add(clip.split)
            recording_splits[clip.recording_id].add(clip.split)
            participant_splits[clip.participant_group_id].add(clip.split)
        if any(len(splits) != 1 for splits in venue_splits.values()):
            raise ValueError("a venue must not cross dataset splits")
        if any(len(splits) != 1 for splits in recording_splits.values()):
            raise ValueError("a recording must not cross dataset splits")
        if any(len(splits) != 1 for splits in participant_splits.values()):
            raise ValueError("a participant group must not cross dataset splits")
        return self

    @classmethod
    def load(cls, path: Path) -> TemporalTrainingDatasetManifest:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def canonical_bytes(self, *, pretty: bool = False) -> bytes:
        return _canonical_bytes(self, pretty=pretty)

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.canonical_bytes(pretty=True))

    def summary(self) -> dict[str, object]:
        split_counts = Counter(clip.split.value for clip in self.clips)
        judgment_counts = Counter(clip.judgment.value for clip in self.clips)
        return {
            "recordings": len(self.recordings),
            "clips": len(self.clips),
            "splits": dict(sorted(split_counts.items())),
            "judgments": dict(sorted(judgment_counts.items())),
        }


class TemporalTrainingDatasetBuilder:
    def build(
        self,
        *,
        dataset_name: str,
        recordings: list[TrainingRecordingSource],
        episodes: list[BehaviorEpisode],
        policy: ClipPlanningPolicy,
        split_seed: str,
        created_at: datetime,
    ) -> TemporalTrainingDatasetManifest:
        recording_ids = [recording.recording_id for recording in recordings]
        if len(recording_ids) != len(set(recording_ids)):
            raise ValueError("recording ids must be unique")
        episode_ids = [episode.episode_id for episode in episodes]
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError("episode ids must be unique")
        sources = {recording.recording_id: recording for recording in recordings}
        if not sources or not episodes:
            raise ValueError("recordings and episodes must not be empty")
        for source in recordings:
            self._verify_source(source)

        assignments = VenueGroupedSplitter(
            split_seed,
            train_fraction=policy.train_fraction,
            validation_fraction=policy.validation_fraction,
        ).assign(
            [
                DatasetRecording(
                    recording_id=source.recording_id,
                    venue_id=source.venue_id,
                    session_id=source.session_id,
                )
                for source in recordings
            ]
        )
        split_by_recording = {item.recording_id: item.split for item in assignments}
        clips = tuple(
            self._plan_clip(episode, sources, policy, split_by_recording)
            for episode in sorted(episodes, key=lambda item: item.episode_id)
        )
        used_recordings = {clip.recording_id for clip in clips}
        if used_recordings != set(sources):
            raise ValueError("every recording must have at least one resolved episode")
        provenance = tuple(
            TrainingRecordingProvenance.model_validate(
                source.model_dump(exclude={"source_path", "commercial_training_permitted", "contains_minors"})
            )
            for source in sorted(recordings, key=lambda item: item.recording_id)
        )
        return TemporalTrainingDatasetManifest(
            dataset_name=dataset_name,
            created_at=created_at,
            split_seed=split_seed,
            policy=policy,
            recordings=provenance,
            clips=clips,
        )

    @staticmethod
    def _verify_source(source: TrainingRecordingSource) -> None:
        if not source.commercial_training_permitted:
            raise ValueError(f"recording {source.recording_id} is not permitted for commercial training")
        if not source.source_path.is_file():
            raise ValueError(f"recording source does not exist: {source.source_path}")
        if sha256_file(source.source_path) != source.source_sha256:
            raise ValueError(f"recording source hash mismatch: {source.recording_id}")

    @staticmethod
    def _plan_clip(
        episode: BehaviorEpisode,
        sources: dict[str, TrainingRecordingSource],
        policy: ClipPlanningPolicy,
        split_by_recording: dict[str, DatasetSplit],
    ) -> TrainingClipRecord:
        source = sources.get(episode.recording_id)
        if source is None:
            raise ValueError(f"episode references unknown recording: {episode.recording_id}")
        if episode.judgment == RiskJudgment.UNCERTAIN:
            raise ValueError(f"uncertain episode is not eligible for training: {episode.episode_id}")
        if episode.end_seconds > source.duration_seconds:
            raise ValueError(f"episode exceeds recording duration: {episode.episode_id}")
        start = max(0.0, episode.start_seconds - policy.context_before_seconds)
        end = min(source.duration_seconds, episode.end_seconds + policy.context_after_seconds)
        duration = end - start
        if duration < policy.minimum_clip_seconds:
            raise ValueError(f"planned clip is shorter than policy minimum: {episode.episode_id}")
        if duration > policy.maximum_clip_seconds:
            raise ValueError(f"planned clip exceeds policy maximum: {episode.episode_id}")
        identity = {
            "recording_id": source.recording_id,
            "source_sha256": source.source_sha256,
            "episode_id": episode.episode_id,
            "start_seconds": start,
            "end_seconds": end,
            "resolved_annotation_sha256": episode.resolved_annotation_sha256,
        }
        clip_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return TrainingClipRecord(
            clip_id=clip_id,
            recording_id=source.recording_id,
            venue_id=source.venue_id,
            session_id=source.session_id,
            camera_id=source.camera_id,
            source_sha256=source.source_sha256,
            calibration_sha256=source.calibration_sha256,
            episode_id=episode.episode_id,
            track_id=episode.track_id,
            participant_group_id=episode.participant_group_id,
            start_seconds=start,
            end_seconds=end,
            judgment=episode.judgment,
            scenario_code=episode.scenario_code,
            annotation_protocol_sha256=episode.annotation_protocol_sha256,
            resolved_annotation_sha256=episode.resolved_annotation_sha256,
            split=split_by_recording[source.recording_id],
        )
