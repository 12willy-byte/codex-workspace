import json

from aquaguard.domain import AlarmEvent, AlarmLevel, RiskAssessment
from aquaguard.evidence import (
    EvidenceRecorder,
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
