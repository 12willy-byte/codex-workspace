import json

import pytest

from aquaguard.domain import AlarmEvent, AlarmLevel, RiskAssessment
from aquaguard.evidence import (
    EvidenceRecorder,
    EvidenceRetentionPolicy,
    EventEvidenceService,
    FileEvidenceRepository,
)
from aquaguard.video import VideoFrame


def alarm_event() -> AlarmEvent:
    return AlarmEvent(
        camera_id="cam-a",
        track_id="person-1",
        area="deep-pool",
        assessment=RiskAssessment(score=92, level=AlarmLevel.EMERGENCY, confirmed=True),
    )


def test_repository_persists_manifest_and_verifies_integrity(tmp_path) -> None:
    recorder = EvidenceRecorder()
    event = alarm_event()
    service = EventEvidenceService(recorder, FileEvidenceRepository(tmp_path))
    recorder.ingest(VideoFrame("cam-a", 0, 4, "pixels-a"))
    recorder.ingest(VideoFrame("cam-a", 1, 5, "pixels-b"))
    service.request(event, 5, pre_seconds=1, post_seconds=0)
    artifact = service.persist_ready(5)[0]

    assert artifact.path.parent == tmp_path
    assert artifact.path.name.startswith(".") is False
    assert artifact.media_type == "application/json"
    assert service.repository.verify(artifact) is True
    manifest = json.loads(artifact.path.read_text(encoding="utf-8"))
    assert manifest["event_id"] == str(event.id)
    assert len(manifest["frames"]) == 2


def test_repository_detects_tampering(tmp_path) -> None:
    recorder = EvidenceRecorder()
    event = alarm_event()
    service = EventEvidenceService(recorder, FileEvidenceRepository(tmp_path))
    recorder.ingest(VideoFrame("cam-a", 0, 5, "pixels"))
    service.request(event, 5, pre_seconds=0, post_seconds=0)
    artifact = service.persist_ready(5)[0]
    artifact.path.write_text("tampered", encoding="utf-8")
    assert service.repository.verify(artifact) is False


def test_event_id_cannot_escape_storage_directory(tmp_path) -> None:
    recorder = EvidenceRecorder()
    recorder.ingest(VideoFrame("cam-a", 0, 5, "pixels"))
    recorder.request("../../outside", "cam-a", 5, pre_seconds=0, post_seconds=0)
    clip = recorder.finalize_ready(5)[0]
    artifact = FileEvidenceRepository(tmp_path).persist(clip)
    assert artifact.path.parent == tmp_path
    assert "outside" not in artifact.path.name


def test_cleanup_removes_expired_then_oldest_until_under_capacity(tmp_path) -> None:
    times = iter([0.0, 5.0, 9.0])
    recorder = EvidenceRecorder()
    repository = FileEvidenceRepository(tmp_path, clock=lambda: next(times))
    service = EventEvidenceService(recorder, repository)

    for sequence, event_id in enumerate(("old", "middle", "new")):
        recorder.ingest(VideoFrame("cam-a", sequence, float(sequence), "pixels"))
        recorder.request(event_id, "cam-a", float(sequence), pre_seconds=0, post_seconds=0)
        service.persist_ready(float(sequence))

    artifacts = dict(service.artifacts)
    newest_size = artifacts["new"].size_bytes
    plan = service.cleanup(EvidenceRetentionPolicy(newest_size, max_age_seconds=8), now=10)

    assert plan.remove_event_ids == ("old", "middle")
    assert plan.retained_event_ids == ("new",)
    assert plan.retained_bytes == newest_size
    assert artifacts["old"].path.exists() is False
    assert artifacts["middle"].checksum_path.exists() is False
    assert repository.verify(artifacts["new"]) is True


def test_repository_refuses_to_delete_paths_outside_root(tmp_path) -> None:
    recorder = EvidenceRecorder()
    event = alarm_event()
    repository = FileEvidenceRepository(tmp_path)
    service = EventEvidenceService(recorder, repository)
    recorder.ingest(VideoFrame("cam-a", 0, 5, "pixels"))
    service.request(event, 5, pre_seconds=0, post_seconds=0)
    artifact = service.persist_ready(5)[0]
    unsafe = artifact.__class__(
        artifact.event_id,
        tmp_path.parent / artifact.path.name,
        artifact.checksum_path,
        artifact.sha256,
        artifact.media_type,
        artifact.frame_count,
        artifact.complete,
        artifact.stored_at,
        artifact.size_bytes,
    )
    with pytest.raises(ValueError, match="outside"):
        repository.delete(unsafe)
