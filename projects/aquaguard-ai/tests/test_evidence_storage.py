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
    assert artifact.metadata_path.is_file()
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


def test_service_recovers_persisted_evidence_after_restart(tmp_path) -> None:
    recorder = EvidenceRecorder()
    event = alarm_event()
    repository = FileEvidenceRepository(tmp_path, clock=lambda: 12.5)
    service = EventEvidenceService(recorder, repository)
    recorder.ingest(VideoFrame("cam-a", 0, 5, "pixels"))
    service.request(event, 5, pre_seconds=0, post_seconds=0)
    original = service.persist_ready(5)[0]

    restarted = EventEvidenceService(EvidenceRecorder(), FileEvidenceRepository(tmp_path))
    recovered = restarted.recover()

    assert recovered == (original,)
    assert restarted.status(str(event.id)) == {
        "event_id": str(event.id),
        "status": "stored",
        "media_type": "application/json",
        "frame_count": 1,
        "complete": True,
        "stored_at": 12.5,
        "size_bytes": original.size_bytes,
        "integrity": "verified",
    }


def test_recovery_rejects_tampered_and_unsafe_metadata(tmp_path) -> None:
    recorder = EvidenceRecorder()
    repository = FileEvidenceRepository(tmp_path)
    recorder.ingest(VideoFrame("cam-a", 0, 5, "pixels"))
    recorder.request("event-1", "cam-a", 5, pre_seconds=0, post_seconds=0)
    artifact = repository.persist(recorder.finalize_ready(5)[0])
    artifact.path.write_text("tampered", encoding="utf-8")
    unsafe = tmp_path / "unsafe.evidence.json"
    unsafe.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event_id": "unsafe",
                "artifact_name": "../outside",
                "checksum_name": "outside.sha256",
                "sha256": "bad",
                "media_type": "application/json",
                "frame_count": 1,
                "complete": True,
                "stored_at": 1,
            }
        ),
        encoding="utf-8",
    )

    recovery = repository.recover()

    assert recovery.artifacts == ()
    assert len(recovery.rejected_metadata) == 2


def test_recovery_rejects_metadata_that_claims_another_event(tmp_path) -> None:
    recorder = EvidenceRecorder()
    repository = FileEvidenceRepository(tmp_path)
    recorder.ingest(VideoFrame("cam-a", 0, 5, "pixels"))
    recorder.request("event-1", "cam-a", 5, pre_seconds=0, post_seconds=0)
    artifact = repository.persist(recorder.finalize_ready(5)[0])
    metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
    metadata["event_id"] = "event-2"
    artifact.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    recovery = repository.recover()

    assert recovery.artifacts == ()
    assert "does not match event id" in recovery.rejected_metadata[0]


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
        event_id=artifact.event_id,
        path=tmp_path.parent / artifact.path.name,
        checksum_path=artifact.checksum_path,
        sha256=artifact.sha256,
        media_type=artifact.media_type,
        frame_count=artifact.frame_count,
        complete=artifact.complete,
        stored_at=artifact.stored_at,
        size_bytes=artifact.size_bytes,
        metadata_path=artifact.metadata_path,
    )
    with pytest.raises(ValueError, match="outside"):
        repository.delete(unsafe)
