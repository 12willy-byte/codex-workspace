from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from aquaguard.training_dataset_cli import main
from aquaguard.vision.annotation import DatasetSplit, RiskJudgment
from aquaguard.vision.evaluation_import import sha256_file
from aquaguard.vision.training_dataset import (
    BehaviorEpisode,
    ClipPlanningPolicy,
    TemporalTrainingDatasetBuilder,
    TemporalTrainingDatasetManifest,
    TrainingRecordingSource,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
DIGEST_D = "d" * 64


def _recording(
    tmp_path: Path,
    index: int,
    *,
    commercial_training_permitted: bool = True,
    contains_minors: bool = False,
) -> TrainingRecordingSource:
    source = tmp_path / f"recording-{index}.mp4"
    source.write_bytes(f"synthetic-video-{index}".encode())
    return TrainingRecordingSource(
        recording_id=f"recording-{index}",
        venue_id=f"venue-{index}",
        session_id=f"session-{index}",
        camera_id=f"camera-{index}",
        source_path=source,
        source_sha256=sha256_file(source),
        duration_seconds=30,
        width=1920,
        height=1080,
        fps=25,
        captured_at=datetime(2026, 7, index, 12, tzinfo=UTC),
        calibration_sha256=DIGEST_A,
        rights_artifact_sha256=DIGEST_B,
        consent_artifact_sha256=DIGEST_C,
        commercial_training_permitted=commercial_training_permitted,
        contains_minors=contains_minors,
        guardian_consent_artifact_sha256=DIGEST_D if contains_minors else None,
    )


def _episode(
    index: int,
    *,
    judgment: RiskJudgment | str = RiskJudgment.DANGEROUS,
    participant_group_id: str | None = None,
    end_seconds: float = 10,
) -> BehaviorEpisode:
    return BehaviorEpisode(
        episode_id=f"episode-{index}",
        recording_id=f"recording-{index}",
        track_id=f"track-{index}",
        participant_group_id=participant_group_id or f"participant-{index}",
        start_seconds=5,
        end_seconds=end_seconds,
        judgment=judgment,
        scenario_code="simulated_struggle" if index % 2 else "normal_swim",
        annotation_protocol_sha256=DIGEST_A,
        resolved_annotation_sha256=DIGEST_B,
    )


def _policy(**changes: float) -> ClipPlanningPolicy:
    values = {
        "context_before_seconds": 2,
        "context_after_seconds": 3,
        "minimum_clip_seconds": 5,
        "maximum_clip_seconds": 20,
    }
    values.update(changes)
    return ClipPlanningPolicy(**values)


def _build(
    tmp_path: Path,
    *,
    recordings: list[TrainingRecordingSource] | None = None,
    episodes: list[BehaviorEpisode] | None = None,
    policy: ClipPlanningPolicy | None = None,
) -> TemporalTrainingDatasetManifest:
    return TemporalTrainingDatasetBuilder().build(
        dataset_name="pilot-temporal-v1",
        recordings=recordings or [_recording(tmp_path, index) for index in range(1, 4)],
        episodes=episodes or [_episode(index) for index in range(1, 4)],
        policy=policy or _policy(),
        split_seed="frozen-pilot-split-v1",
        created_at=datetime(2026, 7, 21, tzinfo=UTC),
    )


def test_builds_stable_leakage_safe_clip_plan(tmp_path: Path) -> None:
    manifest = _build(tmp_path)

    assert len(manifest.recordings) == 3
    assert len(manifest.clips) == 3
    assert {clip.split for clip in manifest.clips} == set(DatasetSplit)
    assert {(clip.start_seconds, clip.end_seconds) for clip in manifest.clips} == {(3, 13)}
    assert {clip.venue_id for clip in manifest.clips} == {
        "venue-1",
        "venue-2",
        "venue-3",
    }
    assert manifest.summary() == {
        "recordings": 3,
        "clips": 3,
        "splits": {"test": 1, "train": 1, "validation": 1},
        "judgments": {"dangerous": 3},
    }

    output = tmp_path / "manifest.json"
    manifest.save(output)
    loaded = TemporalTrainingDatasetManifest.load(output)
    assert loaded == manifest
    assert loaded.sha256() == manifest.sha256()
    assert all("source_path" not in item for item in json.loads(output.read_text())["recordings"])


def test_rejects_source_hash_mismatch(tmp_path: Path) -> None:
    recordings = [_recording(tmp_path, index) for index in range(1, 4)]
    recordings[0].source_path.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="source hash mismatch"):
        _build(tmp_path, recordings=recordings)


def test_rejects_recording_without_commercial_training_permission(tmp_path: Path) -> None:
    recordings = [_recording(tmp_path, 1, commercial_training_permitted=False)] + [
        _recording(tmp_path, index) for index in range(2, 4)
    ]

    with pytest.raises(ValueError, match="not permitted for commercial training"):
        _build(tmp_path, recordings=recordings)


def test_requires_guardian_consent_for_recording_with_minors(tmp_path: Path) -> None:
    recording = _recording(tmp_path, 1, contains_minors=True)
    data = recording.model_dump()
    data["guardian_consent_artifact_sha256"] = None

    with pytest.raises(ValidationError, match="guardian consent"):
        TrainingRecordingSource.model_validate(data)


@pytest.mark.parametrize(
    ("episode", "message"),
    [
        (_episode(1, judgment=RiskJudgment.UNCERTAIN), "uncertain episode"),
        (_episode(1, end_seconds=31), "exceeds recording duration"),
    ],
)
def test_rejects_ineligible_episode(
    tmp_path: Path, episode: BehaviorEpisode, message: str
) -> None:
    recordings = [_recording(tmp_path, index) for index in range(1, 4)]
    episodes = [episode, _episode(2), _episode(3)]

    with pytest.raises(ValueError, match=message):
        _build(tmp_path, recordings=recordings, episodes=episodes)


def test_rejects_clip_exceeding_policy_maximum(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exceeds policy maximum"):
        _build(tmp_path, policy=_policy(maximum_clip_seconds=9))


def test_rejects_participant_group_crossing_venue_splits(tmp_path: Path) -> None:
    episodes = [
        _episode(index, participant_group_id="same-participant") for index in range(1, 4)
    ]

    with pytest.raises(ValueError, match="participant group must not cross"):
        _build(tmp_path, episodes=episodes)


def test_rejects_unused_recording(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="every recording"):
        _build(tmp_path, episodes=[_episode(1), _episode(2)])


def test_cli_builds_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recordings = [_recording(tmp_path, index) for index in range(1, 4)]
    episodes = [_episode(index) for index in range(1, 4)]
    recordings_path = tmp_path / "recordings.jsonl"
    episodes_path = tmp_path / "episodes.jsonl"
    policy_path = tmp_path / "policy.json"
    output_path = tmp_path / "dataset.json"
    recordings_path.write_text(
        "\n".join(item.model_dump_json() for item in recordings) + "\n", encoding="utf-8"
    )
    episodes_path.write_text(
        "\n".join(item.model_dump_json() for item in episodes) + "\n", encoding="utf-8"
    )
    policy_path.write_text(_policy().model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aquaguard-build-training-dataset",
            str(recordings_path),
            str(episodes_path),
            str(policy_path),
            str(output_path),
            "--name",
            "pilot-temporal-v1",
            "--split-seed",
            "frozen-pilot-split-v1",
            "--created-at",
            "2026-07-21T00:00:00+00:00",
        ],
    )

    main()

    assert TemporalTrainingDatasetManifest.load(output_path).summary()["clips"] == 3
