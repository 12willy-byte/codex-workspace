from aquaguard.consistency import EvidenceConsistencyService
from aquaguard.domain import AlarmEvent, AlarmLevel, RiskAssessment
from aquaguard.events import InMemoryAlarmEventRepository
from aquaguard.evidence import EvidenceRecorder, EventEvidenceService, FileEvidenceRepository


def event() -> AlarmEvent:
    return AlarmEvent(
        camera_id="cam-a",
        track_id="track-1",
        area="pool",
        assessment=RiskAssessment(score=90, level=AlarmLevel.EMERGENCY, confirmed=True),
    )


def test_consistency_report_identifies_missing_and_orphaned_evidence(tmp_path) -> None:
    events = InMemoryAlarmEventRepository()
    evidence = EventEvidenceService(EvidenceRecorder(), FileEvidenceRepository(tmp_path))
    registered = event()
    events.append_if_allowed(registered, 0)
    evidence.recorder.request("orphan", "cam-a", 10, pre_seconds=1, post_seconds=1)

    report = EvidenceConsistencyService(events, evidence).inspect()

    assert report.events[0].event_id == str(registered.id)
    assert report.events[0].evidence_status == "missing"
    assert report.orphaned_evidence_ids == ["orphan"]
    assert report.counts == {
        "missing": 1,
        "orphaned": 1,
        "rejected_metadata": 0,
    }


def test_consistency_report_links_pending_evidence(tmp_path) -> None:
    events = InMemoryAlarmEventRepository()
    evidence = EventEvidenceService(EvidenceRecorder(), FileEvidenceRepository(tmp_path))
    registered = event()
    events.append_if_allowed(registered, 0)
    evidence.request(registered, 10, pre_seconds=1, post_seconds=1)

    report = EvidenceConsistencyService(events, evidence).inspect()

    assert report.events[0].evidence_status == "pending"
    assert report.orphaned_evidence_ids == []


def test_consistency_report_links_finalized_unpersisted_evidence(tmp_path) -> None:
    events = InMemoryAlarmEventRepository()
    evidence = EventEvidenceService(EvidenceRecorder(), FileEvidenceRepository(tmp_path))
    registered = event()
    events.append_if_allowed(registered, 0)
    evidence.request(registered, 10, pre_seconds=0, post_seconds=0)
    evidence.recorder.force_finalize(str(registered.id))

    report = EvidenceConsistencyService(events, evidence).inspect()

    assert report.events[0].evidence_status == "ready"
    assert report.orphaned_evidence_ids == []
